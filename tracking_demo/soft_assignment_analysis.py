"""E006-only evaluation: explicit candidate and historical metric/match sources."""
import json,math
import numpy as np
from .analysis import compare_multicam
from .identity_evaluation import evaluate_identities,read_csv,save_csv
from .temporal_comparison import segment_config


def evaluate(root,config):
    output=root/config['output'];checks=[]
    for name,sc in config['segments'].items():
        for filename,key in [('fused_observations.csv','group_source'),('fusion_members.csv','member_source')]:
            source=read_csv(root/sc[key]);candidate=read_csv(output/name/'B3_SOFT'/filename)
            def strip(rows):return [{k:v for k,v in r.items() if k!='track_id'} for r in rows]
            assert strip(source)==strip(candidate),(name,filename)
            checks.append({'segment':name,'file':filename,'source':sc[key],'rows':len(source),'all_fields_except_track_id_identical':True})
        compare_multicam(root,segment_config(config,name))
    evaluate_identities(root,config)
    current_i=read_csv(output/'identity_summary.csv');current_s=read_csv(output/'spatial_summary.csv')
    history_i=read_csv(root/config['historical_identity_csv']);history_s=read_csv(root/config['historical_spatial_csv'])
    identities=[];spatial=[];thresholds=[];deltas=[];changes=[];change_summary=[];raw_checks=[]
    limits=config['adoption_thresholds']
    for name,sc in config['segments'].items():
        candidate_i=next(r for r in current_i if r['segment']==name and r['variant']=='B3_SOFT')
        b=next(r for r in history_i if r['segment']==name and r['variant']=='B3')
        for variant,rows,source in [('B3',history_i,config['historical_identity_csv']),('B3_SOFT',current_i,f'{config["output"]}/identity_summary.csv'),('B3_GLOBAL',history_i,config['historical_identity_csv'])]:
            row=next(r for r in rows if r['segment']==name and r['variant']==variant)
            identities.append({**row,'metric_source':source,'role':'candidate' if variant=='B3_SOFT' else 'main_baseline' if variant=='B3' else 'historical_negative_control'})
        for rows,source in [(history_s,config['historical_spatial_csv']),(current_s,f'{config["output"]}/spatial_summary.csv')]:
            spatial.extend({**r,'summary_source':source,'role':'candidate' if r['variant']=='B3_SOFT' else 'main_baseline' if r['variant']=='B3' else 'historical_negative_control'} for r in rows if r['segment']==name and r['variant'] in ('B3','B3_SOFT','B3_GLOBAL'))
        cb=next(r for r in history_s if (r['segment'],r['variant'],r['region'],r['pool'])==(name,'B3','full_roi','filtered_observed'))
        cc=next(r for r in current_s if (r['segment'],r['variant'],r['region'],r['pool'])==(name,'B3_SOFT','full_roi','filtered_observed'))
        gains={'IDF1_gain_pp':(float(candidate_i['idf1'])-float(b['idf1']))*100,
               'spatial_precision_gain_pp':(float(cc['spatial_precision'])-float(cb['spatial_precision']))*100,
               'spatial_recall_gain_pp':(float(cc['spatial_recall'])-float(cb['spatial_recall']))*100,
               'P90_change_m':float(cc['distance_p90_m'])-float(cb['distance_p90_m'])}
        deltas.append({'segment':name,**gains})
        criteria=[('IDF1 gain pp',gains['IDF1_gain_pp'],'>=',limits['reference_idf1_gain_pp'] if name=='seen_reference' else limits['development_min_idf1_gain_pp']),
                  ('Spatial precision gain pp',gains['spatial_precision_gain_pp'],'>=',limits['min_spatial_precision_gain_pp']),
                  ('Spatial recall gain pp',gains['spatial_recall_gain_pp'],'>=',limits['min_spatial_recall_gain_pp']),
                  ('Matched distance P90 increase m',gains['P90_change_m'],'<=',limits['max_p90_increase_m'])]
        for label,actual,relation,limit in criteria:
            defined=math.isfinite(actual)
            thresholds.append({'segment':name,'criterion':label,'actual':actual if defined else None,'relation':relation,'threshold':limit,
                               'defined':defined,'passed':defined and (actual>=limit if relation=='>=' else actual<=limit)})
        for region in config['evaluation']['regions']:
            x=next(r for r in history_s if (r['segment'],r['variant'],r['region'],r['pool'])==(name,'B3',region,'raw_observations'))
            y=next(r for r in current_s if (r['segment'],r['variant'],r['region'],r['pool'])==(name,'B3_SOFT',region,'raw_observations'))
            keys=[k for k in x if k not in ('variant','source')]
            assert all(x[k]==y[k] for k in keys),(name,region)
            raw_checks.append({'segment':name,'region':region,'raw_spatial_identical_to_historical_B3':True})
        records={f:json.loads((root/config['data_root']/f'annotations_positions/{f:08d}.json').read_text(encoding='utf-8')) for f in range(sc['frame_start'],sc['frame_stop'],sc['frame_step'])}
        baseline_matches=read_csv(root/sc['historical_matches']);candidate_matches=read_csv(output/name/'evaluation_matches.csv')
        for region in config['evaluation']['regions']:
            def matched_map(rows,variant):
                return {(int(r['frame']),records[int(r['frame'])][int(r['gt_index'])]['personID']):r for r in rows if r['variant']==variant and r['pool']=='filtered_observed' and r['region']==region}
            bm=matched_map(baseline_matches,'B3');cm=matched_map(candidate_matches,'B3_SOFT');shared=set(bm)&set(cm)
            for key in sorted(set(bm)|set(cm)):
                a,z=bm.get(key),cm.get(key)
                changes.append({'segment':name,'region':region,'frame':key[0],'personID':key[1],'change':'shared' if a and z else 'lost' if a else 'added',
                                'baseline_prediction_key':a['prediction_key'] if a else '', 'candidate_prediction_key':z['prediction_key'] if z else '',
                                'baseline_distance_m':a['distance_m'] if a else '', 'candidate_distance_m':z['distance_m'] if z else '',
                                'candidate_minus_baseline_m':float(z['distance_m'])-float(a['distance_m']) if a and z else ''})
            change_summary.append({'segment':name,'region':region,'shared_GT_frames':len(shared),'newly_matched_GT_frames':len(set(cm)-set(bm)),
                                   'lost_GT_frames':len(set(bm)-set(cm)),'mean_distance_change_shared_GT_m':float(np.mean([float(cm[k]['distance_m'])-float(bm[k]['distance_m']) for k in shared])) if shared else None})
    for filename,rows in [('identity_comparison.csv',identities),('spatial_comparison.csv',spatial),('thresholds.csv',thresholds),('metric_deltas.csv',deltas),('spatial_match_changes.csv',changes),('spatial_change_summary.csv',change_summary)]:save_csv(output/filename,rows)
    decision={'twelve_seen_criteria_passed':all(r['passed'] for r in thresholds),'criteria_count':len(thresholds),'failed_criteria':[r for r in thresholds if not r['passed']],
              'default':'B3','candidate_count':1,'heldout_used':False,'controller_freeze_recorded':False,
              'next_step':'If all pass, hand off for controller freeze review before heldout; otherwise finish negative-result delivery.','post_result_parameter_changes':False}
    (output/'decision.json').write_text(json.dumps(decision,indent=2,allow_nan=False),encoding='utf-8')
    (output/'comparison_checks.json').write_text(json.dumps({'fixed_inputs':checks,'raw_space':raw_checks,'historical_identity_source':config['historical_identity_csv'],'historical_spatial_source':config['historical_spatial_csv'],'candidate_spatial_sources':{n:s['spatial_source'] for n,s in config['segments'].items()}},indent=2),encoding='utf-8')
    print(json.dumps({'deltas':deltas,'decision':decision},indent=2))
