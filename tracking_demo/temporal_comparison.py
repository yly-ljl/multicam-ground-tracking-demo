"""E005 CPU prediction from fixed fused observations. No GT/evaluation imports."""
import csv
from datetime import datetime
import json,os,sys,time
from .geometry import GroundCamera
from .tracking import GroundTracker
from .multicam import write_csv


def read_csv(path):
    with path.open(encoding='utf-8') as handle:return list(csv.DictReader(handle))


def segment_config(config,name):
    segment=config['segments'][name]
    return {**config,**{k:segment[k] for k in ('frame_start','frame_stop','frame_step')},
            'segment':name,'output':f'{config["output"]}/{name}',
            'local_comparison_sources':True,
            'prediction_provenance':f'{config.get("experiment","E005")} {name}: fixed original B3 fusion inputs; independent empty tracker for each association condition.'}


def replay(root,config):
    output=root/config['output'];output.mkdir(parents=True,exist_ok=True)
    camera=GroundCamera(root/config['data_root'],config['calibrations']['C1'],config['roi_m'])
    regressions=[]
    for name,segment in config['segments'].items():
        cache_begin=time.perf_counter()
        groups=read_csv(root/segment['group_source']);members=read_csv(root/segment['member_source'])
        expected=read_csv(root/segment['baseline_tracks']) if 'B3' in config['association_methods'] else None
        cache_seconds=time.perf_counter()-cache_begin
        frames=list(range(segment['frame_start'],segment['frame_stop'],segment['frame_step']))
        for variant,method in config['association_methods'].items():
            folder=output/name/variant;folder.mkdir(parents=True,exist_ok=True)
            tracker=GroundTracker({**config['tracker'],'association_method':method})
            all_states=[];all_ticks=[];timings=[];mapping={};started=datetime.now().astimezone().isoformat();begin=time.perf_counter()
            for frame in frames:
                frame_start=time.perf_counter();current=[g for g in groups if int(g['frame'])==frame]
                states,ticks=tracker.update(frame/10,[{'obs_id':g['fusion_id'],'x_m':float(g['x_m']),'y_m':float(g['y_m'])} for g in current])
                observed={s['obs_id']:s['track_id'] for s in states if s['state']=='observed'}
                assert set(observed)=={g['fusion_id'] for g in current}
                assert len({s['track_id'] for s in states})==len(states)
                mapping.update(observed)
                for s in states:s.update(frame=frame,variant=variant,in_roi=camera.in_roi([s['x_m'],s['y_m']]))
                all_states.extend(states);all_ticks.extend(ticks)
                timings.append({'frame':frame,'time_s':frame/10,'fixed_input_groups':len(current),
                                'observed_states':len(observed),'predicted_states':len(states)-len(observed),
                                'tracking_ms':(time.perf_counter()-frame_start)*1000})
            seconds=time.perf_counter()-begin
            if variant=='B3':
                assert len(all_states)==len(expected)
                for a,b in zip(all_states,expected):
                    for key,value in a.items():
                        if isinstance(value,float):assert abs(value-float(b[key]))<1e-12,(name,key)
                        else:assert str(value)==b[key],(name,key,value,b[key])
                regressions.append({'segment':name,'baseline_states':len(all_states),'all_fields_equal_within_1e_12':True})
            # Keep source fusion IDs, members and coordinates exactly; only remap track ID.
            output_groups=[{**g,'track_id':mapping[g['fusion_id']]} for g in groups]
            output_members=[{**m,'track_id':mapping[m['fusion_id']]} for m in members]
            for filename,rows in [('fused_observations.csv',output_groups),('fusion_members.csv',output_members),
                                  ('tracks.csv',all_states),('ticks.csv',all_ticks),('frame_timings.csv',timings)]:write_csv(folder/filename,rows)
            track_summary=[]
            for tid in sorted({s['track_id'] for s in all_states}):
                rows=[s for s in all_states if s['track_id']==tid]
                track_summary.append({'track_id':tid,'first_time_s':rows[0]['time_s'],'last_time_s':rows[-1]['time_s'],
                                      'observed_states':sum(s['state']=='observed' for s in rows),'predicted_states':sum(s['state']=='predicted' for s in rows)})
            write_csv(folder/'track_summary.csv',track_summary)
            report={'status':'completed','segment':name,'variant':variant,'association_method':method,'frames':len(frames),
                    'started_at':started,'finished_at':datetime.now().astimezone().isoformat(),'pid':os.getpid(),
                    'command_argv':[sys.executable,*sys.argv],'tracker_config':tracker.config,
                    'cache_seconds':cache_seconds,'cached_tracking_seconds':seconds,'timing_scope':'After input CSV load; excludes CSV serialization, inference, fusion and rendering.',
                    'source_groups':segment['group_source'],'source_members':segment['member_source'],'input_groups':len(groups),'input_members':len(members),
                    'track_ids':len(track_summary),'single_observation_ids':sum(s['observed_states']==1 for s in track_summary),
                    'observed_states':sum(s['state']=='observed' for s in all_states),'predicted_states':sum(s['state']=='predicted' for s in all_states),
                    'initialization':'Empty state for each segment and condition; no warmup.','prediction_inputs':'Fixed image-derived fused groups and tracker config only; no GT.'}
            (folder/'run.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(f'{name} {variant}: {len(groups)} unchanged groups; {len(track_summary)} IDs; {seconds:.3f}s',flush=True)
    if regressions:
        (output/'baseline_regression.json').write_text(json.dumps(regressions,indent=2),encoding='utf-8')
