"""E005 post-hoc CPU association records and appearance-grounded case search."""
import csv,json,sys
from pathlib import Path
import numpy as np
import cv2
from norfair import Tracker
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from tracking_demo.tracking import GroundTracker
from tracking_demo.association import GatedGlobalTracker
from tracking_demo.association_diagnostics import RecordedGreedyTracker

class RecordedGlobalTracker(RecordedGreedyTracker,GatedGlobalTracker):
    pass

out=root/'outputs/E005_temporal_assignment';config=json.loads((root/'configs/e005_temporal.json').read_text(encoding='utf-8'))
def read(path):
    with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))
def gtbox(person,camera):
    view=next(v for v in person['views'] if v['viewNum']==int(camera[-1])-1)
    return [view[k] for k in ('xmin','ymin','xmax','ymax')]
def iou(a,b):
    if all(v==-1 for v in b):return 0.
    b=[max(0,b[0]),max(0,b[1]),min(1920,b[2]),min(1080,b[3])]
    if b[2]<=b[0] or b[3]<=b[1]:return 0.
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.
data={};transitions={};trace_summary=[]
for name,sc in config['segments'].items():
    groups={g['fusion_id']:g for g in read(root/sc['group_source'])}
    sources=({'C1':'outputs/E001_c1_baseline/observations.csv','C2':'outputs/E002_multicam/views/C2/observations.csv','C3':'outputs/E002_multicam/views/C3/observations.csv'} if name=='old_development' else
             {c:f'outputs/E004_identity/new_development/views/{c}/observations.csv' for c in ('C1','C2','C3')} if name=='new_development' else
             {c:f'outputs/E003_improvement/validation/views/{c}/observations.csv' for c in ('C1','C2','C3')})
    obs={r['obs_id']:r for p in sources.values() for r in read(root/p)}
    gt={f:json.loads((root/config['data_root']/f'annotations_positions/{f:08d}.json').read_text(encoding='utf-8')) for f in range(sc['frame_start'],sc['frame_stop'],5)}
    labels={}
    for key,g in groups.items():
        votes=[];f=int(g['frame'])
        for oid in g['member_obs_ids'].split(';'):
            o=obs[oid];a=[float(o[k]) for k in ('x1','y1','x2','y2')]
            scores=sorted([(iou(a,gtbox(person,o['camera'])),person['personID']) for person in gt[f]],reverse=True)
            score,pid=scores[0];margin=score-(scores[1][0] if len(scores)>1 else 0)
            votes.append({'obs_id':oid,'camera':o['camera'],'personID':pid,'iou':score,'margin':margin,'strong':score>=.5 and margin>=.1,
                          'detector_box_touches_edge':min(a[:2])<=1 or a[2]>=1919 or a[3]>=1079})
        strong=[v for v in votes if v['strong']];pids={v['personID'] for v in strong}
        labels[key]={'personID':next(iter(pids)) if len(pids)==1 else None,'votes':votes,'quality':min((v['iou'] for v in strong),default=0),
                     'unclipped_strong_support':any(not v['detector_box_touches_edge'] for v in strong)}
    data[name]={'groups':groups,'observations':obs,'gt':gt,'labels':labels}
    for variant,cls in [('B3',RecordedGreedyTracker),('B3_GLOBAL',RecordedGlobalTracker)]:
        wrapper=GroundTracker(config['tracker']);recorder=cls(config['tracker']);wrapper.tracker=recorder
        states=[]
        for frame in range(sc['frame_start'],sc['frame_stop'],5):
            recorder.frame=frame;recorder.stage=0
            current,_=wrapper.update(frame/10,[{'obs_id':g['fusion_id'],'x_m':float(g['x_m']),'y_m':float(g['y_m'])} for g in groups.values() if int(g['frame'])==frame])
            for s in current:s['frame']=frame
            states.extend(current)
        expected=read(root/sc['track_sources'][variant]);assert len(states)==len(expected)
        for a,b in zip(states,expected):
            for k,v in a.items():
                if isinstance(v,float):assert abs(v-float(b[k]))<1e-12,(name,variant,k)
                else:assert str(v)==b[k],(name,variant,k)
        records=[];mapping={};more=0
        for record in recorder.records:
            record['selected_candidate_indices']=record.pop('greedy_candidate_indices');record['selected_track_indices']=record.pop('greedy_track_indices');record['selected_cardinality']=record.pop('greedy_cardinality')
            record['condition']=variant
            matrix=np.asarray(record['distance_matrix_rows_candidates_cols_tracks'],float)
            shadow_rows,shadow_cols=Tracker.match_dets_and_objs(recorder,matrix,record['threshold_m'])
            record['shadow_greedy_candidate_indices']=[int(v) for v in shadow_rows];record['shadow_greedy_track_indices']=[int(v) for v in shadow_cols]
            record['shadow_scope']='Diagnostic greedy on this condition\'s own predicted state; never applied. Full runs may have different states.'
            if len(record['selected_candidate_indices'])>len(shadow_rows):more+=1
            for i,j in zip(record['selected_candidate_indices'],record['selected_track_indices']):
                candidate=record['candidates'][i];obj=record['objects'][j]
                cur=labels[candidate['obs_id']];prev=labels.get(obj['previous_obs_id'])
                if prev is None or cur['personID'] is None or prev['personID'] is None or obj['previous_frame']!=record['frame']-5:continue
                mapping[candidate['obs_id']]={'frame':record['frame'],'condition':variant,'candidate_index':i,'track_index':j,'track':obj,
                    'current_label':cur,'previous_label':prev,'same_personID':cur['personID']==prev['personID'],
                    'quality':min(cur['quality'],prev['quality']),'selected_distance_m':float(matrix[i,j]),
                    'selected_cardinality':record['selected_cardinality'],'shadow_greedy_cardinality':len(shadow_rows),
                    'shadow_pair_for_candidate':next((int(c) for r,c in zip(shadow_rows,shadow_cols) if r==i),None),
                    'record':record}
            records.append(record)
        folder=out/'association_replay'/name/variant;folder.mkdir(parents=True,exist_ok=True)
        with (folder/'associations.jsonl').open('w',encoding='utf-8') as h:
            for record in records:h.write(json.dumps(record,allow_nan=False)+'\n')
        transitions[(name,variant)]=mapping
        trace_summary.append({'segment':name,'variant':variant,'state_rows':len(states),'recorded_states_match_formal_run_within_1e_12':True,
                              'association_records':len(records),'ticks_more_pairs_than_shadow_greedy_on_own_state':more})
(out/'association_replay/summary.json').write_text(json.dumps(trace_summary,indent=2),encoding='utf-8')
candidates=[]
for name in data:
    baseline=transitions[(name,'B3')];candidate=transitions[(name,'B3_GLOBAL')]
    for key in sorted(set(baseline)&set(candidate)):
        a,b=baseline[key],candidate[key]
        if a['same_personID']==b['same_personID']:continue
        if any(v['personID']==344 for v in (a['current_label'],a['previous_label'],b['previous_label'])):continue
        label='positive' if b['same_personID'] else 'negative'
        candidates.append({'segment':name,'frame':a['frame'],'group_id':key,'kind':label,'current_personID':b['current_label']['personID'],
                           'quality':min(a['quality'],b['quality']),'baseline':a,'candidate':b,
                           'more_pairs_on_candidate_own_state':b['selected_cardinality']>b['shadow_greedy_cardinality'],
                           'scope':'Same raw group, compared through each run\'s own previous observation; box/GT votes require visual review.'})
candidates.sort(key=lambda r:(r['more_pairs_on_candidate_own_state'],r['quality']),reverse=True)
(out/'case_search.json').write_text(json.dumps(candidates,indent=2),encoding='utf-8')
print(json.dumps({'counts':{k:sum(r['kind']==k for r in candidates) for k in ('positive','negative')},'top':[{k:r[k] for k in ('segment','frame','group_id','kind','current_personID','quality','more_pairs_on_candidate_own_state')} for r in candidates[:12]]},indent=2))
if '--images' not in sys.argv:sys.exit(0)
selected=[]
for kind in ('positive','negative'):
    choices=[r for r in candidates if r['kind']==kind and r['baseline']['current_label']['unclipped_strong_support']]
    for r in choices[:3]:selected.append(r)
folder=out/'cases';folder.mkdir(exist_ok=True);index=[]
def cell(canvas,x,y,data_,frame,pid,group,caption,camera=None):
    person=next((p for p in data_['gt'][frame] if p['personID']==pid),None)
    if person is None:return
    votes=[v for v in data_['labels'][group]['votes'] if v['personID']==pid]
    if camera:votes=[v for v in votes if v['camera']==camera]
    chosen=max(votes,key=lambda v:v['iou']) if votes else None
    camera=camera or (chosen['camera'] if chosen else 'C1');box=gtbox(person,camera)
    cv2.putText(canvas,caption,(x+8,y+20),0,.48,(255,255,255),1)
    cv2.putText(canvas,f'{camera} frame{frame} GT{pid}',(x+8,y+40),0,.45,(180,235,180),1)
    if all(v==-1 for v in box):return
    image=cv2.imread(str(root/config['data_root']/f'Image_subsets/{camera}/{frame:08d}.png'))
    box=[max(0,box[0]),max(0,box[1]),min(1919,box[2]),min(1079,box[3])]
    cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),(0,255,0),2)
    if chosen:
        obs=data_['observations'][chosen['obs_id']];b=[round(float(obs[k])) for k in ('x1','y1','x2','y2')];cv2.rectangle(image,tuple(b[:2]),tuple(b[2:]),(0,150,255),2)
    x1=max(0,box[0]-35);x2=min(1920,box[2]+35);y1=max(0,box[1]-15);y2=min(1080,box[3]+35)
    crop=image[y1:y2,x1:x2];scale=min(485/crop.shape[1],395/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale)
    canvas[y+55:y+55+crop.shape[0],x:x+crop.shape[1]]=crop
for n,r in enumerate(selected,1):
    d=data[r['segment']];f=r['frame'];a,b=r['baseline'],r['candidate'];pid=r['current_personID']
    common=set(v['camera'] for v in a['current_label']['votes'] if v['personID']==pid)
    common&=set(v['camera'] for v in a['previous_label']['votes'] if v['personID']==a['previous_label']['personID'])
    common&=set(v['camera'] for v in b['previous_label']['votes'] if v['personID']==b['previous_label']['personID'])
    camera=sorted(common)[0] if common else None
    canvas=np.full((750,1500,3),24,np.uint8)
    cv2.putText(canvas,f'E005 EVAL ONLY {r["kind"]} {r["segment"]} frame{f}; shared group {r["group_id"]}',(10,25),0,.55,(255,255,255),1)
    cell(canvas,0,45,d,f-5,a['previous_label']['personID'],a['track']['previous_obs_id'],f'B3 previous track {a["track"]["track_id"]}',camera)
    cell(canvas,500,45,d,f,pid,r['group_id'],f'Current raw group, GT {pid}',camera)
    cell(canvas,1000,45,d,f-5,b['previous_label']['personID'],b['track']['previous_obs_id'],f'GLOBAL previous track {b["track"]["track_id"]}',camera)
    cv2.putText(canvas,f'B3 transition GT {a["previous_label"]["personID"]}->{pid}; GLOBAL {b["previous_label"]["personID"]}->{pid}',(10,565),0,.6,(255,255,255),1)
    cv2.putText(canvas,f'GLOBAL own-state cardinality {b["selected_cardinality"]}; shadow greedy {b["shadow_greedy_cardinality"]}',(10,605),0,.58,(255,255,255),1)
    cv2.putText(canvas,'Full runs have different predicted states. GT/box identity is post-hoc evidence; no score tuning.',(10,645),0,.52,(225,225,225),1)
    cv2.putText(canvas,'Green GT / orange detector. Track ID numbers are independent between conditions.',(10,685),0,.52,(225,225,225),1)
    filename=f'{r["kind"]}_{n:02d}.jpg';cv2.imwrite(str(folder/filename),canvas)
    index.append({**r,'image':f'cases/{filename}','same_camera_used':camera})
(out/'case_index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
print('Saved',len(index),'example candidates for visual review')
