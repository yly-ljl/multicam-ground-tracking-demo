"""CPU-only recording of the unmodified Norfair greedy baseline, without GT."""
import json
from pathlib import Path
import numpy as np
from norfair import Tracker
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from .tracking import GroundTracker,MetricFilterFactory
from .identity_evaluation import read_csv,save_csv


class RecordedGreedyTracker(Tracker):
    def __init__(self,config):
        super().__init__(distance_function='euclidean',distance_threshold=config['distance_threshold_m'],
                         hit_counter_max=config['hit_counter_max'],initialization_delay=config['initialization_delay'],
                         filter_factory=MetricFilterFactory(config),past_detections_length=0)
        self.records=[];self.frame=None;self.stage=0;self.context={}

    def _update_objects_in_place(self,distance_function,distance_threshold,objects,candidates,period):
        self.context={'frame':self.frame,'stage':'mature' if self.stage==0 else 'initializing',
                      'objects':[{'track_id':obj.id,'predicted_xy':obj.estimate[0].tolist(),
                                  'previous_obs_id':obj.last_detection.data['obs_id'],
                                  'previous_frame':int(round(obj.last_detection.data['tick']*5)),
                                  'previous_raw_xy':obj.last_detection.points[0].tolist(),
                                  'hit_counter_before_match':obj.hit_counter} for obj in objects],
                      'candidates':[{'obs_id':c.data['obs_id'],'xy':c.points[0].tolist()} for c in (candidates or [])]}
        self.stage+=1
        return super()._update_objects_in_place(distance_function,distance_threshold,objects,candidates,period)

    def match_dets_and_objs(self,distance_matrix,distance_threshold):
        # Only call the original greedy implementation; never apply a different assignment.
        rows,cols=super().match_dets_and_objs(distance_matrix,distance_threshold)
        valid=np.isfinite(distance_matrix)&(distance_matrix<distance_threshold)
        maximum=int(np.count_nonzero(maximum_bipartite_matching(csr_matrix(valid),perm_type='column')>=0)) if valid.size else 0
        self.records.append({**self.context,'distance_matrix_rows_candidates_cols_tracks':distance_matrix.tolist(),
                             'threshold_m':distance_threshold,'edge_rule':'finite and strictly < threshold',
                             'greedy_candidate_indices':[int(v) for v in rows],'greedy_track_indices':[int(v) for v in cols],
                             'maximum_feasible_cardinality_diagnostic':maximum,'greedy_cardinality':len(rows)})
        return rows,cols


def trace_baselines(root,config):
    tracker_config=json.loads((root/'configs/multicam_baseline.json').read_text(encoding='utf-8'))['tracker']
    output=root/config['output']/'association_replay';output.mkdir(parents=True,exist_ok=True);summary=[]
    for segment,sc in config['segments'].items():
        source=root/sc['track_sources']['B3'];groups=read_csv(source.parent/'fused_observations.csv');expected=read_csv(source)
        wrapper=GroundTracker(tracker_config);recorder=RecordedGreedyTracker(tracker_config);wrapper.tracker=recorder;states=[]
        for frame in range(sc['frame_start'],sc['frame_stop'],sc['frame_step']):
            recorder.frame=frame;recorder.stage=0
            obs=[{'obs_id':g['fusion_id'],'x_m':float(g['x_m']),'y_m':float(g['y_m'])} for g in groups if int(g['frame'])==frame]
            current,_=wrapper.update(frame/10,obs)
            for s in current:s['frame']=frame
            states.extend(current)
        assert len(states)==len(expected)
        for a,b in zip(states,expected):
            for key,value in a.items():
                if isinstance(value,float):assert abs(value-float(b[key]))<1e-12,(segment,key)
                else:assert str(value)==b[key],(segment,key,value,b[key])
        folder=output/segment;folder.mkdir(exist_ok=True)
        save_csv(folder/'tracks.csv',states)
        with (folder/'associations.jsonl').open('w',encoding='utf-8') as handle:
            for record in recorder.records:handle.write(json.dumps(record,allow_nan=False)+'\n')
        competing=[r for r in recorder.records if r['maximum_feasible_cardinality_diagnostic']>r['greedy_cardinality']]
        (folder/'cardinality_competition.json').write_text(json.dumps(competing,indent=2),encoding='utf-8')
        summary.append({'segment':segment,'source':str(source.relative_to(root)),'records':len(recorder.records),'replayed_track_rows':len(states),
                        'baseline_equal_within_1e_12':True,'cardinality_competition_ticks':len(competing),'frames':[r['frame'] for r in competing]})
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
