"""E004 post-hoc image/GT review. Does not modify predictions or choose a candidate."""
import csv,json,sys
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from tracking_demo.geometry import grid_to_world
out=root/'outputs/E004_identity';config=json.loads((root/'configs/e004_identity.json').read_text(encoding='utf-8'))
def read(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))
def box(g,c):
    b=next(v for v in g['views'] if v['viewNum']==int(c[-1])-1)
    return [b[k] for k in ('xmin','ymin','xmax','ymax')]
def overlap(a,b):
    if all(v==-1 for v in b):return 0.
    b=[max(0,b[0]),max(0,b[1]),min(1920,b[2]),min(1080,b[3])]
    if b[2]<=b[0] or b[3]<=b[1]:return 0.
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0.
segments={}
for name,sc in config['segments'].items():
    folder=(root/sc['track_sources']['B3']).parent
    obs_sources=(json.loads((root/'configs/e004_new_development.json').read_text(encoding='utf-8'))['observation_sources'] if name=='new_development' else
                 {'C1':'outputs/E001_c1_baseline/observations.csv','C2':'outputs/E002_multicam/views/C2/observations.csv','C3':'outputs/E002_multicam/views/C3/observations.csv'} if name=='old_development' else
                 {c:f'outputs/E003_improvement/validation/views/{c}/observations.csv' for c in ('C1','C2','C3')})
    observations={r['obs_id']:r for p in obs_sources.values() for r in read(root/p)}
    groups={g['fusion_id']:g for g in read(folder/'fused_observations.csv')}
    gt={f:json.loads((root/f'data/Wildtrack_dataset/annotations_positions/{f:08d}.json').read_text(encoding='utf-8')) for f in range(sc['frame_start'],sc['frame_stop'],5)}
    labels={}
    for key,g in groups.items():
        f=int(g['frame']);votes=[]
        for oid in g['member_obs_ids'].split(';'):
            o=observations[oid];a=[float(o[k]) for k in ('x1','y1','x2','y2')]
            scores=sorted([(overlap(a,box(person,o['camera'])),person['personID']) for person in gt[f]],reverse=True)
            best,pid=scores[0];margin=best-(scores[1][0] if len(scores)>1 else 0)
            votes.append({'obs_id':oid,'camera':o['camera'],'personID':pid,'iou':best,'margin':margin,'strong':best>=.5 and margin>=.1})
        strong=[v for v in votes if v['strong']];pids={v['personID'] for v in strong}
        labels[key]={'personID':next(iter(pids)) if len(pids)==1 else None,'votes':votes,
                     'min_strong_iou':min((v['iou'] for v in strong),default=0)}
    segments[name]={'groups':groups,'observations':observations,'gt':gt,'labels':labels,'tracks':read(folder/'tracks.csv')}
search=[]
for name,data in segments.items():
    records=[json.loads(line) for line in (out/f'association_replay/{name}/associations.jsonl').read_text(encoding='utf-8').splitlines()]
    for record in records:
        d=np.asarray(record['distance_matrix_rows_candidates_cols_tracks'])
        if d.ndim!=2 or not d.size:continue
        used_rows=set(record['greedy_candidate_indices']);used_cols=set(record['greedy_track_indices'])
        for a,j in zip(record['greedy_candidate_indices'],record['greedy_track_indices']):
            ga=data['labels'][record['candidates'][a]['obs_id']]
            pj=data['labels'].get(record['objects'][j]['previous_obs_id'],{})
            if ga.get('personID') is None or pj.get('personID') is None or ga['personID']==pj['personID']:continue
            if record['objects'][j]['previous_frame']!=record['frame']-5:continue
            for b in range(d.shape[0]):
                if b in used_rows or not d[b,j]<1.5:continue
                gb=data['labels'][record['candidates'][b]['obs_id']]
                if gb.get('personID')!=pj['personID']:continue
                for k in range(d.shape[1]):
                    if k in used_cols or not d[a,k]<1.5:continue
                    pk=data['labels'].get(record['objects'][k]['previous_obs_id'],{})
                    if pk.get('personID')!=ga['personID'] or record['objects'][k]['previous_frame']!=record['frame']-5:continue
                    search.append({'segment':name,'frame':record['frame'],'stage':record['stage'],'row_indices':[a,b],'column_indices':[j,k],
                                   'matrix_2x2_rows_A_B_cols_J_K':d[np.ix_([a,b],[j,k])].tolist(),
                                   'A':record['candidates'][a],'B':record['candidates'][b],'J':record['objects'][j],'K':record['objects'][k],
                                   'person_A':ga['personID'],'person_B':gb['personID'],
                                   'vote_quality_min':min(x['min_strong_iou'] for x in (ga,gb,pj,pk)),
                                   'same_frame_box_evidence':{'A':ga,'B':gb,'J_previous':pj,'K_previous':pk},
                                   'greedy':'A->J; B unmatched; K unmatched','feasible_identity_consistent_edges':'A->K and B->J; strictly <1.5m',
                                   'scope':'Frame-local box/ID clue; image confirmation required, not an oracle or E005 benefit claim.'})
search.sort(key=lambda r:r['vote_quality_min'],reverse=True)
(out/'competition_search.json').write_text(json.dumps(search,indent=2),encoding='utf-8')
print(json.dumps({'candidate_count':len(search),'top':[ {k:v for k,v in r.items() if k not in ('same_frame_box_evidence','A','B','J','K')} for r in search[:8]]},indent=2))

if '--images' not in sys.argv:sys.exit(0)
folder=out/'cases';folder.mkdir(exist_ok=True)
def crop_cell(canvas,x,y,width,height,data,f,pid,caption,obs_id=None,camera=None):
    g=next((g for g in data['gt'][f] if g['personID']==pid),None)
    cv2.putText(canvas,caption,(x+8,y+20),0,.48,(255,255,255),1)
    if g is None:
        cv2.putText(canvas,'GT absent',(x+8,y+50),0,.5,(150,150,255),1);return
    chosen=None
    if obs_id and obs_id in data['labels']:
        votes=[v for v in data['labels'][obs_id]['votes'] if v['personID']==pid]
        if camera is not None:votes=[v for v in votes if v['camera']==camera]
        if votes:chosen=max(votes,key=lambda v:v['iou']);camera=chosen['camera']
    if camera is None:
        options=[c for c in ('C1','C2','C3') if not all(v==-1 for v in box(g,c))]
        if not options:return
        camera=max(options,key=lambda c:min(1080,box(g,c)[3])-max(0,box(g,c)[1]))
    bb=box(g,camera)
    if all(v==-1 for v in bb):return
    image=cv2.imread(str(root/f'data/Wildtrack_dataset/Image_subsets/{camera}/{f:08d}.png'))
    b=[max(0,bb[0]),max(0,bb[1]),min(1919,bb[2]),min(1079,bb[3])]
    cv2.rectangle(image,tuple(b[:2]),tuple(b[2:]),(0,255,0),2)
    if chosen:
        o=data['observations'][chosen['obs_id']];a=[round(float(o[k])) for k in ('x1','y1','x2','y2')]
        cv2.rectangle(image,tuple(a[:2]),tuple(a[2:]),(0,150,255),2)
    x1=max(0,b[0]-40);x2=min(1920,b[2]+40);y1=max(0,b[1]-20);y2=min(1080,b[3]+35)
    if x2<=x1 or y2<=y1:return
    crop=image[y1:y2,x1:x2];scale=min((width-15)/crop.shape[1],(height-65)/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale)
    canvas[y+48:y+48+crop.shape[0],x:x+crop.shape[1]]=crop
    cv2.putText(canvas,f'{camera} frame {f} GT {pid} box={bb}',(x+8,y+40),0,.35,(180,230,180),1)
selected=[];used=set()
for r in search:
    key=(r['segment'],r['frame'])
    if key in used:continue
    selected.append(r);used.add(key)
    if len(selected)==4:break
cases=[]
for n,r in enumerate(selected,1):
    data=segments[r['segment']];f=r['frame'];canvas=np.full((850,1500,3),24,np.uint8)
    cv2.putText(canvas,f'E004 EVAL ONLY competition {n}: {r["segment"]} frame {f}; green GT / orange detector',(10,25),0,.6,(255,255,255),1)
    cells=[(r['J']['previous_frame'],r['person_B'],f'Previous J=track {r["J"]["track_id"]}',r['J']['previous_obs_id']),
           (f,r['person_A'],f'Current A -> greedy J={r["J"]["track_id"]}',r['A']['obs_id']),
           (r['K']['previous_frame'],r['person_A'],f'Previous K=track {r["K"]["track_id"]}',r['K']['previous_obs_id']),
           (f,r['person_B'],'Current B unmatched by greedy',r['B']['obs_id'])]
    common_camera='C2' if r['segment']=='old_development' and f==255 else None
    for col,(cf,pid,caption,key) in enumerate(cells):crop_cell(canvas,col*375,45,375,560,data,cf,pid,caption,key,camera=common_camera)
    cv2.putText(canvas,f'Rows [A,B], cols [J,K]; d(m)={np.round(r["matrix_2x2_rows_A_B_cols_J_K"],4).tolist()}',(10,660),0,.57,(255,255,255),1)
    cv2.putText(canvas,'Original greedy: A->J; B unmatched; K unmatched. Feasible consistent: A->K and B->J (<1.5m).',(10,700),0,.55,(255,255,255),1)
    cv2.putText(canvas,'This is a local failure diagnosis, not a candidate run or proof of aggregate benefit.',(10,740),0,.55,(230,230,230),1)
    name=f'competition_{n:02d}.jpg';cv2.imwrite(str(folder/name),canvas)
    cases.append({'case_id':f'competition_{n:02d}','category':'temporal_competition','image':f'cases/{name}',**r})
# Finite review sample: event examples, known GT anomalies, normal and upstream controls.
extra=[('gt_344_anomaly','old_development',344,[445,455,465],'GT position/clothing anomaly; do not edit or exclude primary GT.'),
       ('partial_negative_box','old_development',483,[310,320,335],'C2 ymin=-1 is partial clipping, not all--1 invisibility.'),
       ('normal_adjacent','old_development',122,[0,5,10],'Normal adjacent-frame label/appearance control.'),
       ('foot_offset','old_development',17,[20,25,30],'Known footpoint error, not evidence of greedy competition.'),
       ('upstream_missed_detection','old_development',27,[0,5,10],'Known occlusion/missing-box clue.'),
       ('upstream_neighbor_fusion','old_development',None,[115,120,125],'Known frame120 C1_...006/C2_...004 neighbor fusion; upstream confound.'),
       ('upstream_duplicate','old_development',None,[480,485,490],'Known same-frame duplicate fusion groups at485; upstream confound.')]
for label,segment,pid,frames,note in extra:
    data=segments[segment]
    if label=='upstream_neighbor_fusion':
        group=next(g for g in data['groups'].values() if 'C1_00000120_006' in g['member_obs_ids']);votes=data['labels'][group['fusion_id']]['votes'];pid=votes[0]['personID']
    if label=='upstream_duplicate':pid=data['gt'][485][5]['personID']
    extra_details={'personID':pid}
    if label=='upstream_neighbor_fusion':extra_details['member_vote_evidence']=data['labels'][group['fusion_id']]
    canvas=np.full((530,1500,3),24,np.uint8);cv2.putText(canvas,f'E004 EVAL ONLY {label} GT {pid}',(10,25),0,.6,(255,255,255),1)
    for col,f in enumerate(frames):crop_cell(canvas,col*500,35,500,425,data,f,pid,f'frame {f}',camera='C2' if label in ('partial_negative_box','gt_344_anomaly') else 'C1')
    cv2.putText(canvas,note,(10,505),0,.48,(230,230,230),1);cv2.imwrite(str(folder/f'{label}.jpg'),canvas)
    cases.append({'case_id':label,'category':label,'segment':segment,'frames':frames,'image':f'cases/{label}.jpg','note':note,**extra_details})
for segment,quota in [('old_development',2),('new_development',1),('seen_reference',2)]:
    data=segments[segment];events=read(out/segment/'B3/events.csv');chosen=[]
    for event in events:
        if event['Type']!='SWITCH':continue
        f=int(event['FrameId']);pid=int(float(event['OId']))
        if f-5 not in data['gt'] or f+5 not in data['gt'] or (f,pid) in chosen:continue
        chosen.append((f,pid))
        canvas=np.full((530,1500,3),24,np.uint8);label=f'switch_{segment}_{len(chosen)}'
        cv2.putText(canvas,f'E004 library SWITCH {segment} frame {f} GT {pid} current track {event["HId"]}',(10,25),0,.55,(255,255,255),1)
        for col,cf in enumerate((f-5,f,f+5)):crop_cell(canvas,col*500,35,500,425,data,cf,pid,f'GT frame {cf}')
        cv2.putText(canvas,'Library event is diagnostic; appearance/upstream/GT quality must be reviewed independently.',(10,505),0,.48,(230,230,230),1)
        cv2.imwrite(str(folder/f'{label}.jpg'),canvas)
        states=[r for r in data['tracks'] if int(r['frame'])==f and r['track_id']==str(int(float(event['HId'])))]
        cases.append({'case_id':label,'category':'library_switch','segment':segment,'frame':f,'personID':pid,'event':event,'track_state':states,'image':f'cases/{label}.jpg',
                      'scope':'Library SWITCH is not an independently confirmed real identity swap.'})
        if len(chosen)>=quota:break
(out/'case_index.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
print('Saved',len(cases),'finite review cases')
