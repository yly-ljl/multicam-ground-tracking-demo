"""Single approved 100-tick default-B3 regression; no candidate/GPU/GT execution."""
import json,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from tracking_demo.temporal_comparison import read_csv
from tracking_demo.tracking import GroundTracker
from tracking_demo.geometry import GroundCamera
config=json.loads((root/'configs/e006_soft_assignment.json').read_text(encoding='utf-8'));sc=config['segments']['old_development']
groups=read_csv(root/sc['group_source']);expected=read_csv(root/sc['baseline_tracks']);tracker=GroundTracker(config['tracker'])
camera=GroundCamera(root/config['data_root'],config['calibrations']['C1'],config['roi_m']);states=[];started=time.perf_counter();ticks=0
for f in range(sc['frame_start'],sc['frame_stop'],sc['frame_step']):
    current,_=tracker.update(f/10,[{'obs_id':g['fusion_id'],'x_m':float(g['x_m']),'y_m':float(g['y_m'])} for g in groups if int(g['frame'])==f])
    for s in current:s.update(frame=f,variant='B3',in_roi=camera.in_roi([s['x_m'],s['y_m']]))
    states.extend(current);ticks+=1
assert ticks==100 and len(states)==len(expected)==3670
for a,b in zip(states,expected):
    for k,v in a.items():
        if isinstance(v,float):assert abs(v-float(b[k]))<1e-12,(k,v,b[k])
        else:assert str(v)==b[k],(k,v,b[k])
out=root/'outputs/E006_soft_assignment';out.mkdir(exist_ok=True)
record={'condition':'B3 default, E006 disabled','segment':'old_development','ticks':ticks,'states':len(states),'all_fields_equal_within_1e_12':True,'seconds':time.perf_counter()-started,'other_baseline_or_GLOBAL_predictions_rerun':False}
(out/'baseline_regression.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record,indent=2))
