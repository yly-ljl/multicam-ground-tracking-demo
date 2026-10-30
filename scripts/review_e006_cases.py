"""Bounded E006 case windows, explicit sources, one from-start CPU diagnostic replay."""
import argparse,csv,json,sys
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from tracking_demo.tracking import GroundTracker
from tracking_demo.association import SoftAssignmentTracker,gated_global_pairs
from tracking_demo.association_diagnostics import RecordedGreedyTracker
from tracking_demo.geometry import grid_to_world

class RecordedSoftTracker(RecordedGreedyTracker,SoftAssignmentTracker):pass

parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/e006_soft_assignment.json');args=parser.parse_args()
config=json.loads((root/args.config).read_text(encoding='utf-8'));out=root/config['output'];case_folder=out/'cases';case_folder.mkdir(parents=True,exist_ok=True)
def read(path):
    with path.open(encoding='utf-8') as h:return list(csv.DictReader(h))
def box(person,camera):
    v=next(v for v in person['views'] if v['viewNum']==int(camera[-1])-1);return [v[k] for k in ('xmin','ymin','xmax','ymax')]
def iou(a,b):
    if all(v==-1 for v in b):return 0.
    b=[max(0,b[0]),max(0,b[1]),min(1920,b[2]),min(1080,b[3])]
    if b[2]<=b[0] or b[3]<=b[1]:return 0.
    i=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i) if i else 0.
data={};search=[]
for order,(name,sc) in enumerate(config['segments'].items()):
    groups={g['fusion_id']:g for g in read(root/sc['group_source'])}
    obs={r['obs_id']:r for path in sc['observation_sources'].values() for r in read(root/path)}
    gt={f:json.loads((root/config['data_root']/f'annotations_positions/{f:08d}.json').read_text(encoding='utf-8')) for f in range(sc['frame_start'],sc['frame_stop'],5)}
    labels={}
    for key,g in groups.items():
        votes=[];f=int(g['frame'])
        for oid in g['member_obs_ids'].split(';'):
            o=obs[oid];a=[float(o[k]) for k in ('x1','y1','x2','y2')]
            scores=sorted([(iou(a,box(p,o['camera'])),p['personID']) for p in gt[f]],reverse=True)
            score,pid=scores[0];margin=score-(scores[1][0] if len(scores)>1 else 0)
            votes.append({'obs_id':oid,'camera':o['camera'],'personID':pid,'iou':score,'strong':score>=.5 and margin>=.1,
                          'clipped_detection':min(a[:2])<=1 or a[2]>=1919 or a[3]>=1079,'partial_negative_GT_box':min(box(next(p for p in gt[f] if p['personID']==pid),o['camera']))<0})
        strong=[v for v in votes if v['strong']];pids={v['personID'] for v in strong}
        labels[key]={'personID':next(iter(pids)) if len(pids)==1 else None,'votes':votes,'quality':min((v['iou'] for v in strong),default=0.),
                     'clear_support':any(not v['clipped_detection'] and not v['partial_negative_GT_box'] for v in strong)}
    transitions={};track_tables={}
    sources={**sc['historical_tracks'],'B3_SOFT':sc['track_sources']['B3_SOFT']}
    for variant,path in sources.items():
        table=read(root/path);track_tables[variant]=table;previous={};mapping={}
        for row in table:
            if row['state']!='observed':continue
            tid=row['track_id'];old=previous.get(tid);key=row['obs_id']
            mapping[key]={'track_id':int(tid),'frame':int(row['frame']),'previous_group':old['obs_id'] if old else None,'previous_frame':int(old['frame']) if old else None,
                          'current_label':labels[key],'previous_label':labels[old['obs_id']] if old else None}
            previous[tid]=row
        transitions[variant]=mapping
    def identity_tp(path):
        return {int(float(r['personID'])):int(r['valid_cooccurrences']) if r['track_id'] else 0 for r in read(path) if r['personID']}
    baseline_tp=identity_tp(root/sc['historical_identity_mapping']);soft_tp=identity_tp(out/name/'B3_SOFT/identity_mapping.csv')
    data[name]={'groups':groups,'observations':obs,'gt':gt,'labels':labels,'transitions':transitions,'tracks':track_tables,'order':order}
    for key,a in transitions['B3'].items():
        b=transitions['B3_SOFT'][key];pid=a['current_label']['personID'];f=a['frame']
        if pid is None or pid==344 or a['previous_frame']!=f-5 or b['previous_frame']!=f-5:continue
        pa=a['previous_label']['personID'];pb=b['previous_label']['personID']
        if pa is None or pb is None or 344 in (pa,pb):continue
        quality=min(a['current_label']['quality'],a['previous_label']['quality'],b['previous_label']['quality'])
        if not all(v['clear_support'] for v in (a['current_label'],a['previous_label'],b['previous_label'])):continue
        kind='improved' if pa!=pid and pb==pid else 'worsened' if pa==pid and pb!=pid else 'unchanged' if pa==pb==pid and a['previous_group']==b['previous_group'] else None
        if kind:
            delta=soft_tp.get(pid,0)-baseline_tp.get(pid,0)
            search.append({'segment':name,'frame':f,'group_id':key,'kind':kind,'personID':pid,'person_identity_TP_change':delta,'box_quality':quality,'segment_order':order,
                           'baseline_transition':a,'candidate_transition':b})
# Mandatory windows are registered in advance, not selected by favorable results.
windows=[]
for name,f,key,pid in [('old_development',255,'B3_00000255_010',187),('seen_reference',840,'B3_00000840_012',193)]:
    windows.append({'segment':name,'frame':f,'group_id':key,'personID':pid,'kind':'mandatory','baseline_transition':data[name]['transitions']['B3'][key],'candidate_transition':data[name]['transitions']['B3_SOFT'][key]})
used={(r['segment'],r['frame']) for r in windows}
for kind in ('improved','worsened','unchanged'):
    candidates=[r for r in search if r['kind']==kind]
    candidates.sort(key=lambda r:((-abs(r['person_identity_TP_change']) if kind!='unchanged' else abs(r['person_identity_TP_change'])),-r['box_quality'],r['segment_order'],r['frame'],r['group_id']))
    count=0
    for r in candidates:
        if (r['segment'],r['frame']) in used:continue
        sc=config['segments'][r['segment']]
        if r['frame']-5<sc['frame_start'] or r['frame']+5>=sc['frame_stop']:continue
        windows.append(r);used.add((r['segment'],r['frame']));count+=1
        if count==2:break
assert len(windows)<=8
for r in windows:
    sc=config['segments'][r['segment']];r['window_frames']=[f for f in (r['frame']-5,r['frame'],r['frame']+5) if sc['frame_start']<=f<sc['frame_stop']]
    r['historical_GLOBAL_transition']=data[r['segment']]['transitions']['B3_GLOBAL'].get(r['group_id'])
(out/'case_selection.json').write_text(json.dumps({'selection':'registered priorities; finite purposive sample, not error frequencies','windows':windows,'automatic_transition_clue_counts':{k:sum(r['kind']==k for r in search) for k in ('improved','worsened','unchanged')}},indent=2),encoding='utf-8')
# One from-start replay per segment, ending at the latest selected window, never window-only initialization.
diag_summary=[];matrix_by_frame={}
for name,sc in config['segments'].items():
    chosen={f for r in windows if r['segment']==name for f in r['window_frames']}
    if not chosen:continue
    end=max(chosen);wrapper=GroundTracker({**config['tracker'],'association_method':'soft_global'});recorder=RecordedSoftTracker(config['tracker']);wrapper.tracker=recorder
    states=[];tick_count=0
    for f in range(sc['frame_start'],end+1,5):
        recorder.frame=f;recorder.stage=0
        current,_=wrapper.update(f/10,[{'obs_id':g['fusion_id'],'x_m':float(g['x_m']),'y_m':float(g['y_m'])} for g in data[name]['groups'].values() if int(g['frame'])==f])
        for s in current:s['frame']=f
        states.extend(current);tick_count+=1
    expected=[r for r in data[name]['tracks']['B3_SOFT'] if int(r['frame'])<=end];assert len(states)==len(expected)
    for a,b in zip(states,expected):
        for k,v in a.items():
            if isinstance(v,float):assert abs(v-float(b[k]))<1e-12,(name,k)
            else:assert str(v)==b[k],(name,k)
    keep=[]
    for r in recorder.records:
        if r['frame'] not in chosen:continue
        r['selected_candidate_indices']=r.pop('greedy_candidate_indices');r['selected_track_indices']=r.pop('greedy_track_indices');r['selected_cardinality']=r.pop('greedy_cardinality')
        d=np.array(r['distance_matrix_rows_candidates_cols_tracks']);rr,cc=gated_global_pairs(d,1.5) if d.ndim==2 else ([],[])
        r['shadow_GLOBAL_indices']={'rows':rr,'cols':cc,'scope':'SOFT own current predicted state, diagnostic only, not applied'}
        r['SOFT_objective']=sum(d[i,j]/1.5 for i,j in zip(r['selected_candidate_indices'],r['selected_track_indices']))+len(r['candidates'])-r['selected_cardinality']
        keep.append(r);matrix_by_frame[(name,r['frame'],r['stage'])]=r
    path=case_folder/f'{name}_selected_matrices.json';path.write_text(json.dumps(keep,indent=2),encoding='utf-8')
    diag_summary.append({'segment':name,'from_frame':sc['frame_start'],'through_frame':end,'additional_CPU_ticks':tick_count,'formal_state_prefix_equal_within_1e_12':True,'exported_only_selected_window_frames':sorted(chosen),'matrix_file':path.relative_to(out).as_posix()})
assert sum(r['additional_CPU_ticks'] for r in diag_summary)<=280
(out/'diagnostic_replay.json').write_text(json.dumps(diag_summary,indent=2),encoding='utf-8')
def crop(canvas,x,y,d,f,pid,key,caption,camera=None):
    cv2.putText(canvas,caption,(x+7,y+20),0,.48,(255,255,255),1)
    if f is None or pid is None:
        cv2.putText(canvas,'No inherited observation / ambiguous label',(x+7,y+55),0,.4,(170,170,250),1);return
    person=next((p for p in d['gt'][f] if p['personID']==pid),None)
    if person is None:return
    votes=[v for v in d['labels'][key]['votes'] if v['personID']==pid] if key else []
    if camera:votes=[v for v in votes if v['camera']==camera]
    chosen=max(votes,key=lambda v:v['iou']) if votes else None;camera=camera or (chosen['camera'] if chosen else 'C1');b=box(person,camera)
    cv2.putText(canvas,f'{camera} frame{f} GT{pid}',(x+7,y+42),0,.43,(160,230,160),1)
    if all(v==-1 for v in b):return
    im=cv2.imread(str(root/config['data_root']/f'Image_subsets/{camera}/{f:08d}.png'));b=[max(0,b[0]),max(0,b[1]),min(1919,b[2]),min(1079,b[3])];cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(0,255,0),2)
    if chosen:
        o=d['observations'][chosen['obs_id']];a=[round(float(o[k])) for k in ('x1','y1','x2','y2')];cv2.rectangle(im,tuple(a[:2]),tuple(a[2:]),(0,150,255),2)
    x1=max(0,b[0]-35);x2=min(1920,b[2]+35);y1=max(0,b[1]-20);y2=min(1080,b[3]+35)
    if x2<=x1 or y2<=y1:return
    im=im[y1:y2,x1:x2];s=min(480/im.shape[1],385/im.shape[0]);im=cv2.resize(im,None,fx=s,fy=s);canvas[y+55:y+55+im.shape[0],x:x+im.shape[1]]=im
for n,r in enumerate(windows,1):
    name=r['segment'];d=data[name];f=r['frame'];a=r['baseline_transition'];b=r['candidate_transition'];pid=r['personID']
    ids=[(a['previous_group'],a['previous_label']['personID'] if a['previous_label'] else None),(r['group_id'],pid),(b['previous_group'],b['previous_label']['personID'] if b['previous_label'] else None)]
    cameras=[{v['camera'] for v in d['labels'][key]['votes'] if v['personID']==p} if key and p is not None else set() for key,p in ids]
    common=set.intersection(*cameras) if all(cameras) else set();cam=sorted(common)[0] if common else None
    image=np.full((730,1500,3),24,np.uint8);cv2.putText(image,f'E006 EVAL ONLY {r["kind"]} {name} frame{f}; window {r["window_frames"]}',(10,25),0,.55,(255,255,255),1)
    crop(image,0,45,d,a['previous_frame'],ids[0][1],ids[0][0],f'B3 previous track {a["track_id"]}',cam)
    crop(image,500,45,d,f,pid,r['group_id'],f'Current fixed group, GT {pid}',cam)
    crop(image,1000,45,d,b['previous_frame'],ids[2][1],ids[2][0],f'SOFT previous track {b["track_id"]}',cam)
    cv2.putText(image,f'B3 previous GT {ids[0][1]} -> {pid}; SOFT previous GT {ids[2][1]} -> {pid}',(10,555),0,.58,(255,255,255),1)
    cv2.putText(image,'No shared-state assumption across full runs. Green GT / orange detector; GT used only after prediction.',(10,595),0,.5,(225,225,225),1)
    cv2.putText(image,f'Raw group {r["group_id"]}; uncertain/clipped examples require explicit review.',(10,635),0,.49,(225,225,225),1)
    filename=f'{n:02d}_{r["kind"]}_{f}.jpg';cv2.imwrite(str(case_folder/filename),image);r['image']=f'cases/{filename}';r['same_camera_used']=cam
    rec=matrix_by_frame.get((name,f,'mature'));r['candidate_matrix_record']='mature in '+f'cases/{name}_selected_matrices.json' if rec else None
    if rec:
        i=next((i for i,c in enumerate(rec['candidates']) if c['obs_id']==r['group_id']),None)
        if i is not None:
            j=next((j for ri,j in zip(rec['selected_candidate_indices'],rec['selected_track_indices']) if ri==i),None)
            r['actual_candidate_selection']={'candidate_row':i,'track_column':j,'old_track':rec['objects'][j] if j is not None else None,'new_track_if_unmatched':b['track_id'] if j is None else None}
(out/'case_index.json').write_text(json.dumps(windows,indent=2),encoding='utf-8')
print(json.dumps({'windows':[{k:r.get(k) for k in ('kind','segment','frame','personID','person_identity_TP_change','image')} for r in windows],
                  'additional_diagnostic_CPU_ticks':sum(r['additional_CPU_ticks'] for r in diag_summary)},indent=2))
