"""E005 independent post-run evaluation and predeclared threshold decisions."""
import json
import numpy as np
from .analysis import compare_multicam
from .identity_evaluation import evaluate_identities,read_csv,save_csv
from .temporal_comparison import segment_config


def evaluate(root,config):
    output=root/config['output'];input_checks=[]
    for name,segment in config['segments'].items():
        folder=output/name
        for filename in ('fused_observations.csv','fusion_members.csv'):
            source=read_csv(root/segment['group_source' if filename.startswith('fused') else 'member_source'])
            base=read_csv(folder/'B3'/filename);candidate=read_csv(folder/'B3_GLOBAL'/filename)
            def stripped(rows):return [{k:v for k,v in r.items() if k!='track_id'} for r in rows]
            assert stripped(source)==stripped(base)==stripped(candidate),(name,filename)
            input_checks.append({'segment':name,'file':filename,'rows':len(source),'all_fields_except_track_id_identical':True})
        compare_multicam(root,segment_config(config,name))
    evaluate_identities(root,config)
    identity=read_csv(output/'identity_summary.csv');spatial=read_csv(output/'spatial_summary.csv')
    old_identity=read_csv(root/'outputs/E004_identity/identity_summary.csv')
    old_spatial=read_csv(root/'outputs/E004_identity/spatial_summary.csv')
    regressions=[];raw_checks=[];changes=[];change_summary=[];thresholds=[];deltas=[]
    limits=config['adoption_thresholds']
    for name,segment in config['segments'].items():
        b=next(r for r in identity if (r['segment'],r['variant'])==(name,'B3'))
        c=next(r for r in identity if (r['segment'],r['variant'])==(name,'B3_GLOBAL'))
        original=next(r for r in old_identity if (r['segment'],r['variant'])==(name,'B3'))
        for key in ('idf1','idp','idr','idtp','idfp','idfn','num_objects','num_predictions','num_switches','num_fragmentations'):
            assert abs(float(b[key])-float(original[key]))<1e-12,(name,key)
        regressions.append({'segment':name,'baseline_identity_equals_E004':True})
        gain=(float(c['idf1'])-float(b['idf1']))*100
        minimum=limits['reference_idf1_gain_pp'] if name=='seen_reference' else limits['development_min_idf1_gain_pp']
        thresholds.append({'segment':name,'criterion':'IDF1 gain pp','actual':gain,'relation':'>=','threshold':minimum,'passed':gain>=minimum})
        sb=next(r for r in spatial if (r['segment'],r['variant'],r['region'],r['pool'])==(name,'B3','full_roi','filtered_observed'))
        sc=next(r for r in spatial if (r['segment'],r['variant'],r['region'],r['pool'])==(name,'B3_GLOBAL','full_roi','filtered_observed'))
        precision_gain=(float(sc['spatial_precision'])-float(sb['spatial_precision']))*100
        recall_gain=(float(sc['spatial_recall'])-float(sb['spatial_recall']))*100
        p90_change=float(sc['distance_p90_m'])-float(sb['distance_p90_m'])
        for criterion,actual,limit in [('Spatial precision gain pp',precision_gain,limits['min_spatial_precision_gain_pp']),
                                       ('Spatial recall gain pp',recall_gain,limits['min_spatial_recall_gain_pp'])]:
            thresholds.append({'segment':name,'criterion':criterion,'actual':actual,'relation':'>=','threshold':limit,'passed':actual>=limit})
        thresholds.append({'segment':name,'criterion':'Matched distance P90 increase m','actual':p90_change,'relation':'<=','threshold':limits['max_p90_increase_m'],'passed':p90_change<=limits['max_p90_increase_m']})
        deltas.append({'segment':name,'IDF1_gain_pp':gain,'spatial_precision_gain_pp':precision_gain,'spatial_recall_gain_pp':recall_gain,'P90_change_m':p90_change})
        for region in config['evaluation']['regions']:
            raw=[r for r in spatial if r['segment']==name and r['region']==region and r['pool']=='raw_observations']
            assert len(raw)==2
            for key in raw[0]:
                if key not in ('variant','source'):assert raw[0][key]==raw[1][key],(name,region,key)
            raw_checks.append({'segment':name,'region':region,'raw_space_identical':True})
        for r in [r for r in spatial if r['segment']==name and r['variant']=='B3']:
            previous=next(v for v in old_spatial if (v['segment'],v['variant'],v['region'],v['pool'])==(name,'B3',r['region'],r['pool']))
            for key in ('gt_count','prediction_count','matched','unmatched_predictions','unmatched_gt','distance_mean_m','distance_median_m','distance_p90_m','distance_p95_m'):
                assert abs(float(r[key])-float(previous[key]))<1e-12,(name,key)
        records={f:json.loads((root/config['data_root']/f'annotations_positions/{f:08d}.json').read_text(encoding='utf-8'))
                 for f in range(segment['frame_start'],segment['frame_stop'],segment['frame_step'])}
        matches=read_csv(output/name/'evaluation_matches.csv')
        for region in config['evaluation']['regions']:
            maps={variant:{(int(r['frame']),records[int(r['frame'])][int(r['gt_index'])]['personID']):r for r in matches if r['variant']==variant and r['region']==region and r['pool']=='filtered_observed'} for variant in ('B3','B3_GLOBAL')}
            base,candidate=maps['B3'],maps['B3_GLOBAL'];shared=set(base)&set(candidate);added=set(candidate)-set(base);lost=set(base)-set(candidate)
            for key in sorted(set(base)|set(candidate)):
                a=base.get(key);b_=candidate.get(key)
                changes.append({'segment':name,'region':region,'frame':key[0],'personID':key[1],
                                'change':'shared' if a and b_ else 'lost' if a else 'added',
                                'baseline_prediction_key':a['prediction_key'] if a else '',
                                'candidate_prediction_key':b_['prediction_key'] if b_ else '',
                                'baseline_distance_m':a['distance_m'] if a else '',
                                'candidate_distance_m':b_['distance_m'] if b_ else '',
                                'candidate_minus_baseline_m':float(b_['distance_m'])-float(a['distance_m']) if a and b_ else ''})
            error_changes=[float(candidate[k]['distance_m'])-float(base[k]['distance_m']) for k in shared]
            change_summary.append({'segment':name,'region':region,'shared_GT_frames':len(shared),'newly_matched_GT_frames':len(added),
                                   'lost_GT_frames':len(lost),'shared_different_prediction_keys':sum(base[k]['prediction_key']!=candidate[k]['prediction_key'] for k in shared),
                                   'mean_distance_change_on_shared_GT_m':float(np.mean(error_changes)) if error_changes else None,
                                   'scope':'Same GT-frame subset; does not hide added/lost coverage or certify GT identity.'})
    save_csv(output/'thresholds.csv',thresholds);save_csv(output/'metric_deltas.csv',deltas)
    save_csv(output/'spatial_match_changes.csv',changes);save_csv(output/'spatial_change_summary.csv',change_summary)
    decision={'nonheldout_thresholds_passed':all(r['passed'] for r in thresholds),'criteria_count':len(thresholds),
              'failed_criteria':[r for r in thresholds if not r['passed']],
              'next_step':'Submit to controller for review; no automatic freeze or heldout execution.','default':'B3 remains unchanged',
              'candidate_count':1,'parameters_changed_after_results':False,'heldout_used':False}
    (output/'decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    (output/'comparison_checks.json').write_text(json.dumps({'fusion_inputs':input_checks,'raw_space':raw_checks,'baseline_identity_and_spatial':regressions},indent=2),encoding='utf-8')
    print(json.dumps({'deltas':deltas,'decision':decision},indent=2))
