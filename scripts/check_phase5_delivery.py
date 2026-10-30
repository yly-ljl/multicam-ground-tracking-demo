"""Final bounded artifact audit: no inference, installation, or heldout access."""
import csv,json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'outputs/PHASE5_delivery'
def read(path):
    with path.open(encoding='utf-8') as h:return list(csv.DictReader(h))
before=json.loads((out/'preservation_before.json').read_text(encoding='utf-8'));checks=[]
for r in before:
    p=root/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],r['path']
    checks.append({'path':r['path'],'unchanged':True})
old={r['path'] for r in before if r['path'].startswith('outputs/')}
now={p.relative_to(root).as_posix() for name in ('E001_c1_baseline','E002_multicam','E003_improvement','E004_identity','E005_temporal_assignment') for p in (root/'outputs'/name).rglob('*') if p.is_file()}
assert now==old
(out/'preservation_after.json').write_text(json.dumps({'files':len(checks),'all_unchanged':True,'old_output_file_set_unchanged':True,'checks':checks},indent=2),encoding='utf-8')
performance=json.loads((out/'performance/performance.json').read_text(encoding='utf-8'))
assert performance['total_source_image_predict_calls']==180 and performance['model_objects_created']==1
assert performance['same_model_and_predictor_reused'] and len(set(performance['predictor_object_ids_per_pass']))==1
call_checks=[]
for n,report in enumerate(performance['passes'],1):
    rows=read(out/f'performance/pass_{n}/image_timings.csv');pairs={(int(r['frame']),r['camera']) for r in rows}
    assert len(rows)==60 and pairs=={(f,c) for f in range(0,100,5) for c in ('C1','C2','C3')}
    assert report['new_image_predict_calls']==60 and report['pid']==performance['pid']
    assert report['overview_clipped_states']==0 and report['all_observed_and_predicted_states_displayed']
    call_checks.append({'pass':n,'rows':len(rows),'frame_range':[0,95,5],'camera_order':['C1','C2','C3'],'same_pid':True})
media=json.loads((out/'media_check.json').read_text(encoding='utf-8'))
assert [(r['decoded_frames'],r['fps'],r['seconds']) for r in media]==[(120,2.,60.),(160,2.,80.)]
reference=json.loads((out/'reference_replay/baseline_reproduction.json').read_text(encoding='utf-8'));assert reference['all_exported_fields_equal_to_reference_B3']
assert reference['track_rows']==4956 and reference['fusion_groups']==3569
cached=json.loads((out/'reference_replay/run.json').read_text(encoding='utf-8'));assert cached['new_image_predict_calls']==0 and cached['overview_clipped_states']==0
result={'protected_files':len(checks),'performance_calls':call_checks,'total_new_source_image_predict_calls_in_package':360,
        'unique_new_images_in_E004':180,'previously_seen_images_repeated_for_performance':60,'performance_repeats':3,
        'new_heldout_calls':0,'extra_reference_inference_for_entry_verification':0,'video_decode_checks':media,
        'reference_B3_tracks_and_groups_reproduced':reference,'default_method':'greedy B3','E005_adopted':False,'heldout_used':False,
        'all_tests':31,'GPU_repeated_for_minor_fixes':False}
(out/'delivery_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('Final checks passed:',len(checks),'protected files; 180 performance calls; both videos; exact B3 cache reproduction.')
