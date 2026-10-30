"""E007 read-only cache diagnosis. No prediction, optimization, or semantic classifier."""
import csv
import hashlib
import json
import math
import platform
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .geometry import GroundCamera, grid_to_world


def rows(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        yield from csv.DictReader(f)


def write_csv(path, values, fields=None):
    values=list(values);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:fields=list(dict.fromkeys(k for r in values for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values)


def save_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def unique_index(values,key):
    result={}
    for row in values:
        k=key(row)
        if k in result:raise ValueError(f'Duplicate key: {k}')
        result[k]=row
    return result


def signature(members):
    atoms=[(int(m['frame']),m['camera'],m['obs_id']) for m in members]
    if len(set(atoms))!=len(atoms):raise ValueError('Duplicate member atom')
    return tuple(sorted(atoms))


def exact_parents(b2,b3):
    """Maps group ID to (parent ID or None, structure), without coordinate guesses."""
    index=unique_index([{'id':g,'sig':signature(ms)} for g,ms in b2.items()],lambda r:r['sig'])
    used=[];result={}
    for g,ms in b3.items():
        signature(ms)
        remaining=[m for m in ms if m['camera']!='C3']
        if not remaining:result[g]=(None,'C3_new');continue
        sig=signature(remaining)
        if sig not in index:raise ValueError(f'Missing exact parent: {g}')
        parent=index[sig]['id'];used.append(parent)
        result[g]=(parent,'C3_joined' if any(m['camera']=='C3' for m in ms) else 'no_C3')
    if Counter(used)!=Counter({g:1 for g in b2}):raise ValueError('Every B2 parent must occur exactly once')
    return result


def person_at_json_index(records,index):
    return int(records[int(index)]['personID'])


def identity_contributions(mapping,pairs):
    valid_mapping={(int(r['personID']),int(r['track_id'])) for r in mapping if r['personID']!='' and r['track_id']!=''}
    result={}
    for r in pairs:
        p,t=int(r['personID']),int(r['track_id'])
        if (p,t) in valid_mapping and r['valid_le_1m']=='True':
            assert float(r['distance_m'])<=1.0
            key=(int(r['frame']),p)
            if key in result:raise ValueError('Multiple IDTP contributions for one GT tick')
            result[key]={'track_id':t,'distance_m':float(r['distance_m'])}
    return result


def four_sets(universe,b2,b3):
    u,a,b=set(universe),set(b2),set(b3)
    assert a<=u and b<=u
    groups={'both':a&b,'B2_only':a-b,'B3_only':b-a,'neither':u-(a|b)}
    assert sum(map(len,groups.values()))==len(u)
    return groups


def stratified_sample(group_rows,config):
    rng=random.Random(config['seed']);chosen=[];strata=[]
    for segment in config['segment_order']:
        for structure in config['structure_order']:
            population=sorted([r for r in group_rows if r['segment']==segment and r['structure']==structure],key=lambda r:(int(r['frame']),r['B3_group']))
            n=min(config['random_per_stratum'],len(population))
            ids=sorted(rng.sample(range(len(population)),n))
            strata.append({'segment':segment,'structure':structure,'N_h':len(population),'n_h':n,'inclusion_probability':n/len(population) if population else None})
            for i in ids:chosen.append({**population[i],'selection':'random','N_h':len(population),'n_h':n,'inclusion_probability':n/len(population)})
    return chosen,strata


def retain_review(samples,reviews):
    """A review can be uncertain; it never changes or replaces a selected unit."""
    assert {r['case_id'] for r in reviews}=={s['case_id'] for s in samples}
    return [{**s,'review':next(r for r in reviews if r['case_id']==s['case_id'])} for s in samples]


def condition_member_roles(members_by_condition,camera):
    """A shared panel is a union; each atom retains its condition membership."""
    roles=defaultdict(list)
    for variant,members in members_by_condition.items():
        for member in members:
            if member['camera']==camera:roles[member['obs_id']].append(variant)
    return dict(roles)


def analyze(root,config):
    root=Path(root);out=root/config['output'];out.mkdir(parents=True,exist_ok=True)
    assert config['heldout_enabled'] is False
    registered=json.loads((out/'config_registered.json').read_text())
    assert config==registered
    inputs=set();groups_all=[];members_all=[];gt_all=[];summary=[];checks=[];contexts={};purpose=[]
    cameras={c:GroundCamera(root/config['data_root'],cal,config['roi_m']) for c,cal in config['calibrations'].items()}
    def read(path):inputs.add(path);return list(rows(root/path))
    def region(xy):return 'common_projected' if all(c.sees_ground(xy) for c in cameras.values()) else 'remaining_roi'
    for segment in config['segment_order']:
        sc=config['segments'][segment];assert 0<=sc['frame_start']<sc['frame_stop']<=1400
        annotations={};gt={}
        for frame in range(sc['frame_start'],sc['frame_stop'],5):
            path=f'{config["data_root"]}/annotations_positions/{frame:08d}.json';inputs.add(path)
            annotations[frame]=json.loads((root/path).read_text())
            for obj in annotations[frame]:
                xy=grid_to_world(obj['positionID'])
                if cameras['C1'].in_roi(xy):gt[(frame,int(obj['personID']))]={'xy':xy.tolist(),'region':region(xy)}
        observations={}
        for camera,path in sc['observation_sources'].items():
            observations.update(unique_index(read(path),lambda r:(int(r['frame']),camera,r['obs_id'])))
        conditions={}
        for variant,source in sc['conditions'].items():
            g=unique_index(read(source['groups']),lambda r:r['fusion_id'])
            ms=read(source['members']);unique_index(ms,lambda r:(int(r['frame']),r['camera'],r['obs_id']))
            bygroup=defaultdict(list)
            for m in ms:
                assert m['fusion_id'] in g
                key=(int(m['frame']),m['camera'],m['obs_id']);assert key in observations
                assert int(m['frame'])==int(g[m['fusion_id']]['frame'])
                bygroup[m['fusion_id']].append(m)
            assert set(g)==set(bygroup)
            for gid in g:assert int(g[gid]['member_count'])==len(bygroup[gid])
            tracks=read(source['tracks']);observed=unique_index([t for t in tracks if t['state']=='observed'],lambda r:r['obs_id'])
            assert set(g)==set(observed)
            assert all(t['obs_id'] in g for t in observed.values())
            mapping=read(source['identity_root']+'/identity_mapping.csv')
            frame_counts=read(source['identity_root']+'/frame_counts.csv')
            pairpath=source['identity_root']+'/distance_pairs.csv';inputs.add(pairpath)
            idtp=identity_contributions(mapping,rows(root/pairpath))
            metrics_path=source['identity_root']+'/metrics.json';inputs.add(metrics_path);metric=json.loads((root/metrics_path).read_text())
            bytrack=unique_index(observed.values(),lambda r:(int(r['frame']),int(r['track_id'])))
            for key,value in idtp.items():value['group']=bytrack[(key[0],value['track_id'])]['obs_id']
            inside=[t for t in observed.values() if cameras['C1'].in_roi((float(t['x_m']),float(t['y_m'])))]
            assert len(idtp)==metric['idtp']==sum(int(m['valid_cooccurrences']) for m in mapping)
            assert len(gt)==metric['num_objects']==sum(int(f['GT']) for f in frame_counts)
            assert len(inside)==metric['num_predictions']==sum(int(f['observed']) for f in frame_counts)
            assert len(inside)-len(idtp)==metric['idfp'] and len(gt)-len(idtp)==metric['idfn']
            prev={};last={}
            for t in sorted(observed.values(),key=lambda r:(int(r['frame']),int(r['track_id']))):
                tid=int(t['track_id']);prev[t['obs_id']]=last.get(tid);last[tid]=t['obs_id']
            conditions[variant]={'groups':g,'members':bygroup,'observed':observed,'idtp':idtp,'previous':prev,'metric':metric}
            checks.append({'segment':segment,'variant':variant,'raw_groups':len(g),'observed_total':len(observed),'observed_in_roi':len(inside),'observed_excluded':len(observed)-len(inside),'pure_prediction_total':sum(t['state']=='predicted' for t in tracks),'IDTP':len(idtp),'IDFP':metric['idfp'],'IDFN':metric['idfn'],'GT':len(gt),'identity_matches_history':True})
        matches=read(sc['spatial_matches']);spatial_summary=read(sc['spatial_summary']);read(sc['spatial_per_frame'])
        spatial={}
        for pool in ['raw_observations','filtered_observed']:
            for variant in ['B2','B3']:
                selected=[r for r in matches if r['region']=='full_roi' and r['pool']==pool and r['variant']==variant]
                table=unique_index(selected,lambda r:(int(r['frame']),person_at_json_index(annotations[int(r['frame'])],r['gt_index'])))
                for key,r in table.items():
                    assert key in gt and float(r['distance_m'])<=1.0
                    assert r['prediction_key'] in conditions[variant]['groups']
                hist=next(r for r in spatial_summary if (r['region'],r['pool'],r['variant'])==('full_roi',pool,variant))
                assert len(table)==int(hist['matched']) and int(hist['gt_count'])==len(gt)
                expected=len(conditions[variant]['groups']) if pool=='raw_observations' else next(c['observed_in_roi'] for c in checks if c['segment']==segment and c['variant']==variant)
                assert int(hist['prediction_count'])==expected
                spatial[(pool,variant)]=table
                checks.append({'segment':segment,'variant':variant,'pool':pool,'matched':len(table),'prediction_count':expected,'unmatched_predictions':expected-len(table),'unmatched_gt':len(gt)-len(table),'spatial_matches_history':True})
        parents=exact_parents(conditions['B2']['members'],conditions['B3']['members'])
        maps={}
        for definition in ['raw_observations','filtered_observed','identity']:
            a=conditions['B2']['idtp'] if definition=='identity' else spatial[(definition,'B2')]
            b=conditions['B3']['idtp'] if definition=='identity' else spatial[(definition,'B3')]
            fs=four_sets(gt,a,b);maps[definition]=(a,b,fs)
            for reg in ['full_roi','common_projected','remaining_roi']:
                domain=set(gt) if reg=='full_roi' else {k for k,v in gt.items() if v['region']==reg}
                counts={k:len(v&domain) for k,v in fs.items()}
                summary.append({'segment':segment,'definition':definition,'region':reg,'GT':len(domain),**counts,'B2_total':len(set(a)&domain),'B3_total':len(set(b)&domain),'net_gain':len(set(b)&domain)-len(set(a)&domain)})
        for key,data in sorted(gt.items()):
            r={'segment':segment,'frame':key[0],'personID':key[1],'x_m':data['xy'][0],'y_m':data['xy'][1],'GT_region':data['region']}
            for definition,(a,b,fs) in maps.items():
                r[definition+'_class']=next(k for k,v in fs.items() if key in v)
                for variant,table in [('B2',a),('B3',b)]:
                    item=table.get(key,{})
                    r[definition+'_'+variant+'_group']=item.get('group',item.get('prediction_key',''))
                    r[definition+'_'+variant+'_distance_m']=item.get('distance_m','')
            gt_all.append(r)
        reverse={}
        for definition,(a,b,_) in maps.items():
            for variant,table in [('B2',a),('B3',b)]:
                reverse[(definition,variant)]={v.get('group',v.get('prediction_key')):key[1] for key,v in table.items()}
        segment_rows=[]
        for gid,g in conditions['B3']['groups'].items():
            parent,structure=parents[gid];pg=conditions['B2']['groups'].get(parent)
            xy=np.array([float(g['x_m']),float(g['y_m'])]);p_xy=np.array([float(pg['x_m']),float(pg['y_m'])]) if pg else None
            conf=[];edges=[];points=[];c3conf=None
            for member in conditions['B3']['members'][gid]:
                obs=observations[(int(member['frame']),member['camera'],member['obs_id'])]
                confidence=float(obs['confidence']);u,v=float(obs['foot_u']),float(obs['foot_v']);edge=min(u/1920,(1920-u)/1920,v/1080,(1080-v)/1080)
                conf.append(confidence);edges.append(edge);points.append([float(member['raw_x_m']),float(member['raw_y_m'])])
                if member['camera']=='C3':c3conf=confidence
                members_all.append({'segment':segment,'B3_group':gid,'B2_parent':parent or '',**member,'confidence':confidence,'foot_u':u,'foot_v':v,'edge_distance_signed':edge,'x1':obs['x1'],'y1':obs['y1'],'x2':obs['x2'],'y2':obs['y2']})
            frame=int(g['frame']);near=[pid for (f,pid),v in gt.items() if f==frame and np.linalg.norm(xy-v['xy'])<=1]
            r={'segment':segment,'frame':frame,'B3_group':gid,'B2_parent':parent or '','structure':structure,'member_count':len(points),'members':g['member_obs_ids'],'cameras':g['member_cameras'],'B3_raw_x_m':float(xy[0]),'B3_raw_y_m':float(xy[1]),'B2_raw_x_m':float(p_xy[0]) if pg else None,'B2_raw_y_m':float(p_xy[1]) if pg else None,'confidence_min':min(conf),'confidence_mean':float(np.mean(conf)),'C3_confidence':c3conf,'edge_min_signed':min(edges),'dispersion_m':max(float(np.linalg.norm(np.array(a)-b)) for a in points for b in points),'parent_child_displacement_m':float(np.linalg.norm(xy-p_xy)) if pg else None,'GT_near_count':len(near),'GT_near_bucket':str(len(near)) if len(near)<2 else '2+','GT_near_personIDs':'|'.join(map(str,near)),'B3_region':region(xy),'B2_region':region(p_xy) if pg else None,'region_crossed':region(xy)!=region(p_xy) if pg else None,'fully_connected':True}
            for variant,key in [('B2',parent),('B3',gid)]:
                t=conditions[variant]['observed'].get(key,{})
                r[variant+'_track_id']=t.get('track_id','');r[variant+'_observed_in_roi']=cameras['C1'].in_roi((float(t['x_m']),float(t['y_m']))) if t else None
                r[variant+'_previous_group']=conditions[variant]['previous'].get(key) or ''
                for definition in maps:r[variant+'_'+definition+'_personID']=reverse[(definition,variant)].get(key)
            segment_rows.append(r)
        groups_all.extend(segment_rows)
        byid={r['B3_group']:r for r in segment_rows};parent_to_child={r['B2_parent']:r['B3_group'] for r in segment_rows if r['B2_parent']}
        used=set()
        def add(slot,candidates):
            for frame,pid,gid,reason in candidates:
                unit=(frame,gid or f'GT:{pid}')
                if unit in used:continue
                used.add(unit);base=byid.get(gid,{'segment':segment,'frame':frame,'B3_group':'','B2_parent':'','structure':'no_B3_group'})
                purpose.append({**base,'selection':'purpose','slot':slot,'target_personID':pid,'selection_reason':reason});return
        a,b,fs=maps['raw_observations']
        candidates=[(k[0],k[1],b[k]['prediction_key'],'raw B3-only; C3-new first, then frame/personID/group') for k in fs['B3_only']]
        add('new_coverage',sorted(candidates,key=lambda x:(byid[x[2]]['structure']!='C3_new',x[0],x[1],x[2])))
        a,b,fs=maps['identity']
        candidates=[]
        for k in sorted(fs['B2_only']):
            gid=parent_to_child[a[k]['group']]
            candidates.append((k[0],k[1],gid,'earliest B2-only global IDTP; exact child of B2 contribution'))
        add('identity_loss',candidates)
        add('success_control',[(k[0],k[1],b[k]['group'],'both global IDTP and exact parent equals B2 contribution') for k in sorted(fs['both']) if parents[b[k]['group']][0]==a[k]['group']])
        add('largest_displacement',[(r['frame'],None,r['B3_group'],'largest displacement, then frame/group') for r in sorted([r for r in segment_rows if r['structure']=='C3_joined'],key=lambda r:(-r['parent_child_displacement_m'],r['frame'],r['B3_group']))])
        contexts[segment]={'conditions':conditions,'annotations':annotations,'gt':gt,'observations':observations}
    write_csv(out/'gt_outcomes.csv',gt_all);write_csv(out/'four_sets.csv',summary);write_csv(out/'group_relations.csv',groups_all);write_csv(out/'member_features.csv',members_all);save_json(out/'consistency_checks.json',checks)
    feature_rows=[];margins=[]
    for segment in config['segment_order']:
        for structure in config['structure_order']:
            population=[r for r in groups_all if r['segment']==segment and r['structure']==structure]
            margins.append({'segment':segment,'structure':structure,'N':len(population),'fully_connected':sum(r['fully_connected'] for r in population),'B3_raw_matched':sum(r['B3_raw_observations_personID'] is not None for r in population),'B3_IDTP':sum(r['B3_identity_personID'] is not None for r in population),'B2_parent_IDTP':sum(r['B2_identity_personID'] is not None for r in population)})
            for feature in ['confidence_min','confidence_mean','C3_confidence','edge_min_signed','dispersion_m','parent_child_displacement_m']:
                values=[r[feature] for r in population if r[feature] is not None]
                feature_rows.append({'segment':segment,'structure':structure,'feature':feature,'N':len(population),'available':len(values),'missing':len(population)-len(values),**{f'q{int(q*100)}':float(np.quantile(values,q)) if values else None for q in config['quantiles']}})
    write_csv(out/'structure_summary.csv',margins);write_csv(out/'feature_quantiles.csv',feature_rows)
    supplement=[]
    for segment in config['segment_order']:
        for dimension in ['member_count','B3_region','B2_region','region_crossed','GT_near_bucket']:
            counter=Counter(str(r[dimension]) for r in groups_all if r['segment']==segment)
            supplement.extend({'segment':segment,'dimension':dimension,'value':v,'N':n} for v,n in sorted(counter.items()))
    write_csv(out/'supplementary_margins.csv',supplement)
    random_samples,strata=stratified_sample(groups_all,config)
    cases=[];key_to_id={}
    for sample in random_samples+purpose:
        key=(sample['segment'],sample['frame'],sample['B3_group'] or f'GT:{sample["target_personID"]}')
        if key not in key_to_id:
            cid=f'C{len(cases)+1:02d}';key_to_id[key]=cid
            sc=config['segments'][sample['segment']]
            times=[f for f in [sample['frame']-5,sample['frame'],sample['frame']+5] if sc['frame_start']<=f<sc['frame_stop']]
            cases.append({**sample,'case_id':cid,'times':times,'selection_sources':[sample['selection']],'semantic_review_status':'not_reviewed'})
        else:next(c for c in cases if c['case_id']==key_to_id[key])['selection_sources'].append(sample['selection'])
        sample['case_id']=key_to_id[key]
    write_csv(out/'random_samples.csv',random_samples);write_csv(out/'random_strata.csv',strata);write_csv(out/'purpose_samples.csv',purpose);save_json(out/'cases_frozen.json',cases)
    for camera,cal in config['calibrations'].items():
        for path in [f'{config["data_root"]}/calibrations/intrinsic_zero/intr_{cal}.xml',f'{config["data_root"]}/calibrations/extrinsic/extr_{cal}.xml']:inputs.add(path)
    save_json(out/'input_manifest.json',[{'path':p,'bytes':(root/p).stat().st_size,'sha256':sha(root/p)} for p in sorted(inputs)])
    save_json(out/'mechanical_and_sample_freeze.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'sampling':'Python random.Random.sample; one seeded instance; sorted index sample restored to stratum order','seed':config['seed'],'images_reviewed':False,'semantic_auto_labels_generated':False,'files':{p:sha(out/p) for p in ['config_registered.json','gt_outcomes.csv','four_sets.csv','group_relations.csv','member_features.csv','random_samples.csv','purpose_samples.csv','cases_frozen.json']},'random_units':len(random_samples),'purpose_slots':len(purpose),'unique_windows':len(cases),'new_tracking_replay_ticks':0})
    return contexts,cases


def render_cases(root,config):
    """Render only frozen windows, all three cameras; no automatic semantic verdict."""
    import cv2
    root=Path(root);out=root/config['output'];cases=json.loads((out/'cases_frozen.json').read_text())
    freeze=json.loads((out/'mechanical_and_sample_freeze.json').read_text())
    assert all(sha(out/name)==value for name,value in freeze['files'].items())
    dest=out/'cases';dest.mkdir(exist_ok=True)
    cameras={c:GroundCamera(root/config['data_root'],cal,config['roi_m']) for c,cal in config['calibrations'].items()}
    image_sources=set();index=[];details=[];log_frames=defaultdict(set)
    def text(im,value,xy,scale=.55,color=(245,245,245)):
        cv2.putText(im,str(value),xy,cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(im,str(value),xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
    for segment in config['segment_order']:
        sc=config['segments'][segment];obs={}
        for camera,path in sc['observation_sources'].items():
            obs.update({(int(r['frame']),camera,r['obs_id']):r for r in rows(root/path)})
        conditions={}
        for variant,sources in sc['conditions'].items():
            gs={r['fusion_id']:r for r in rows(root/sources['groups'])};ms=defaultdict(list)
            for m in rows(root/sources['members']):ms[m['fusion_id']].append(m)
            tracks=[r for r in rows(root/sources['tracks']) if r['state']=='observed']
            ts={r['obs_id']:r for r in tracks};bytf={(int(r['track_id']),int(r['frame'])):r['obs_id'] for r in tracks}
            prev={};last={}
            for t in sorted(tracks,key=lambda r:(int(r['frame']),int(r['track_id']))):
                prev[t['obs_id']]=last.get(t['track_id']);last[t['track_id']]=t['obs_id']
            conditions[variant]={'groups':gs,'members':ms,'tracks':ts,'bytf':bytf,'previous':prev}
        for case in [c for c in cases if c['segment']==segment]:
            cid=case['case_id'];center=case['frame'];log_frames[segment].add(center)
            center_groups={'B2':case['B2_parent'],'B3':case['B3_group']}
            ids=set()
            if case.get('target_personID') is not None:ids.add(int(case['target_personID']))
            for k in ['GT_near_personIDs']:
                ids.update(int(v) for v in case.get(k,'').split('|') if v)
            for variant in ['B2','B3']:
                for definition in ['raw_observations','filtered_observed','identity']:
                    p=case.get(variant+'_'+definition+'_personID')
                    if p is not None and p!='':ids.add(int(p))
            canvas=np.full((90+len(case['times'])*440+390,1680,3),28,np.uint8)
            text(canvas,f'{cid} {segment} f={center} {case["structure"]} | {case["selection_sources"]}',(15,28),.7)
            text(canvas,f'B3 {center_groups["B3"]} | exact B2 {center_groups["B2"] or "NONE"} | EVAL ONLY; IDs are GT hints, not visual verdicts',(15,55),.57)
            text(canvas,'GREEN=source detection (B2/B3 roles labelled), RED=foot, BLUE=GT, ORANGE=B3 raw; panels show union of conditions',(15,79),.57)
            case_detail={'case_id':cid,'segment':segment,'frame':center,'target_personIDs_geometry_or_mapping_hints':sorted(ids),'timepoints':[],'center_previous_observed_groups':{v:conditions[v]['previous'].get(center_groups[v]) for v in ['B2','B3']},'GT_is_posthoc_only':True}
            for ti,frame in enumerate(case['times']):
                records=json.loads((root/config['data_root']/f'annotations_positions/{frame:08d}.json').read_text());selected={}
                for variant in ['B2','B3']:
                    data=conditions[variant];cg=center_groups[variant]
                    if not cg:selected[variant]=None;continue
                    tid=int(data['tracks'][cg]['track_id']);selected[variant]=data['bytf'].get((tid,frame))
                time_detail={'frame':frame,'groups':selected,'members_by_condition':{v:conditions[v]['members'][gid] if gid else [] for v,gid in selected.items()},'members':[],'nearby_GT':[]}
                points=[]
                for variant,gid in selected.items():
                    if gid:
                        points.append([float(conditions[variant]['groups'][gid]['x_m']),float(conditions[variant]['groups'][gid]['y_m'])])
                frameids=set(ids)
                for person in records:
                    xy=grid_to_world(person['positionID'])
                    if any(np.linalg.norm(xy-p)<=1.5 for p in points):frameids.add(int(person['personID']))
                time_detail['nearby_GT']=[{'personID':int(p['personID']),'xy':grid_to_world(p['positionID']).tolist(),'views':p['views']} for p in records if int(p['personID']) in frameids]
                for ci,camera in enumerate(['C1','C2','C3']):
                    imagepath=f'{config["data_root"]}/Image_subsets/{camera}/{frame:08d}.png';image_sources.add(imagepath)
                    im=cv2.imread(str(root/imagepath));assert im is not None,imagepath
                    sourceobs={};roles=condition_member_roles(time_detail['members_by_condition'],camera)
                    for variant,gid in selected.items():
                        if gid:
                            for m in conditions[variant]['members'][gid]:
                                if m['camera']==camera:
                                    sourceobs[m['obs_id']]=obs[(frame,camera,m['obs_id'])]
                    boxes=[]
                    for ob in sourceobs.values():boxes.append([float(ob[k]) for k in ['x1','y1','x2','y2']])
                    gtboxes=[]
                    for person in records:
                        if int(person['personID']) not in frameids:continue
                        view=next(v for v in person['views'] if int(v['viewNum'])==ci)
                        box=[int(view[k]) for k in ['xmin','ymin','xmax','ymax']]
                        if box[2]>box[0] and box[3]>box[1] and box[2]>0 and box[3]>0 and box[0]<1920 and box[1]<1080:
                            gtboxes.append((int(person['personID']),box));boxes.append(box)
                    if boxes:
                        x0=max(0,int(min(b[0] for b in boxes))-55);y0=max(0,int(min(b[1] for b in boxes))-45)
                        x1=min(1920,int(max(b[2] for b in boxes))+55);y1=min(1080,int(max(b[3] for b in boxes))+45)
                    else:x0,y0,x1,y1=0,0,1920,1080
                    for pid,box in gtboxes:
                        cv2.rectangle(im,(box[0],box[1]),(box[2],box[3]),(255,160,50),2)
                        text(im,f'GT {pid}',(max(0,box[0]),max(18,box[1]-8)),.65,(255,180,75))
                        person=next(p for p in records if int(p['personID'])==pid);uv=cameras[camera].to_image(grid_to_world(person['positionID']))
                        if np.isfinite(uv).all() and np.max(np.abs(uv))<1e5:cv2.drawMarker(im,tuple(np.int32(uv)),(255,170,80),cv2.MARKER_CROSS,15,2)
                    for ob in sourceobs.values():
                        b=[int(float(ob[k])) for k in ['x1','y1','x2','y2']];uv=(int(float(ob['foot_u'])),int(float(ob['foot_v'])))
                        cv2.rectangle(im,(b[0],b[1]),(b[2],b[3]),(30,245,60),2);cv2.circle(im,uv,5,(40,40,250),-1)
                        text(im,'+'.join(roles[ob['obs_id']])+' '+ob['obs_id'].split('_')[-1]+f' c={float(ob["confidence"]):.2f}',(b[0],min(1070,b[3]+20)),.6,(50,250,80))
                        time_detail['members'].append({'camera':camera,**ob,'condition_roles':roles[ob['obs_id']]})
                    if selected['B3']:
                        g=conditions['B3']['groups'][selected['B3']];uv=cameras[camera].to_image([float(g['x_m']),float(g['y_m'])])
                        if np.isfinite(uv).all() and np.max(np.abs(uv))<1e5:cv2.drawMarker(im,tuple(np.int32(uv)),(0,170,255),cv2.MARKER_DIAMOND,17,2)
                    crop=im[y0:y1,x0:x1];scale=min(550/crop.shape[1],402/crop.shape[0]);resized=cv2.resize(crop,(int(crop.shape[1]*scale),int(crop.shape[0]*scale)))
                    panel=np.full((440,560,3),18,np.uint8);px=(560-resized.shape[1])//2;py=35+(402-resized.shape[0])//2
                    panel[py:py+resized.shape[0],px:px+resized.shape[1]]=resized
                    text(panel,f'{camera} f={frame} dets={len(sourceobs)} GT hints={len(gtboxes)}',(8,23),.56)
                    canvas[90+ti*440:90+(ti+1)*440,ci*560:(ci+1)*560]=panel
                case_detail['timepoints'].append(time_detail)
            # Same scale on X/Y, 50 px/m, 6x6 m view around current B3 raw position.
            bottom=90+len(case['times'])*440;bev=np.full((360,360,3),240,np.uint8)
            xy=np.array([float(case['B3_raw_x_m']),float(case['B3_raw_y_m'])]);origin=xy-3
            def pix(p):return (int(30+50*(p[0]-origin[0])),int(330-50*(p[1]-origin[1])))
            for k in range(7):
                cv2.line(bev,(30+50*k,30),(30+50*k,330),(210,210,210),1);cv2.line(bev,(30,30+50*k),(330,30+50*k),(210,210,210),1)
            records=json.loads((root/config['data_root']/f'annotations_positions/{center:08d}.json').read_text())
            for person in records:
                p=grid_to_world(person['positionID']);q=pix(p)
                if 5<q[0]<350 and 5<q[1]<350:cv2.circle(bev,q,4,(130,130,130),-1);text(bev,str(person['personID']),(q[0]+4,q[1]-3),.45,(30,30,30))
            for variant,color in [('B2',(220,80,0)),('B3',(0,100,240))]:
                gid=center_groups[variant]
                if not gid:continue
                g=conditions[variant]['groups'][gid];p=[float(g['x_m']),float(g['y_m'])];cv2.drawMarker(bev,pix(p),color,cv2.MARKER_DIAMOND,15,2)
                for m in conditions[variant]['members'][gid]:
                    p=[float(m['raw_x_m']),float(m['raw_y_m'])];cv2.circle(bev,pix(p),5,color,1);text(bev,m['camera'],(pix(p)[0]+4,pix(p)[1]+13),.4,color)
            canvas[bottom+5:bottom+365,10:370]=bev
            lines=[f'BEV x,y in meters; equal scale 50 px/m; grid 1m. x=[{origin[0]:.2f},{origin[0]+6:.2f}] y=[{origin[1]:.2f},{origin[1]+6:.2f}]',
                   f'BLUE diamond=B2 raw; ORANGE diamond=B3 raw; small rings=source ground points; gray=GT EVAL ONLY',
                   f'Parent displacement={case["parent_child_displacement_m"]}; dispersion={case["dispersion_m"]:.3f}m; confidence min={case["confidence_min"]:.3f}',
                   f'Geometry <=1m GT hints: {case["GT_near_personIDs"] or "none"}. Hints do not establish identity.',
                   f'Global-IDTP GT: B2={case.get("B2_identity_personID")} B3={case.get("B3_identity_personID")}; whole-segment mapping, not local truth.',
                   f'B2 previous observed: {case_detail["center_previous_observed_groups"]["B2"]}',
                   f'B3 previous observed: {case_detail["center_previous_observed_groups"]["B3"]}',
                   'Panels contain B2/B3 member union: a B2-only box is NOT part of the B3 group. Roles are explicit on every detection.',
                   'No visual labels generated automatically. Manual judgments and uncertainty are stored separately.']
            for i,line in enumerate(lines):text(canvas,line,(390,bottom+30+i*35),.51)
            filename=f'{cid}.jpg';cv2.imwrite(str(dest/filename),canvas,[cv2.IMWRITE_JPEG_QUALITY,92])
            save_json(dest/f'{cid}.json',case_detail);details.append(case_detail)
            index.append({'case_id':cid,'segment':segment,'frame':center,'structure':case['structure'],'selection':'+'.join(case['selection_sources']),'image':f'cases/{filename}','detail':f'cases/{cid}.json','timepoints':'|'.join(map(str,case['times']))})
    # Extract existing actual B3 records only for frozen center frames, no new tracking.
    selected_records=[]
    for segment,frames in log_frames.items():
        with (root/config['segments'][segment]['B3_associations']).open(encoding='utf-8') as f:
            for line in f:
                record=json.loads(line)
                if record['frame'] in frames:selected_records.append({'segment':segment,**record})
    with (out/'selected_B3_associations.jsonl').open('w',encoding='utf-8') as f:
        for record in selected_records:f.write(json.dumps(record,allow_nan=True)+'\n')
    write_csv(out/'case_index.csv',index)
    save_json(out/'image_sources.json',[{'path':p,'bytes':(root/p).stat().st_size,'sha256':sha(root/p)} for p in sorted(image_sources)])
    save_json(out/'render_record.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'windows':len(cases),'unique_source_images':len(image_sources),'max_timepoints':max(len(c['times']) for c in cases),'new_tracking_ticks':0,'reused_actual_B3_records':len(selected_records),'B2_evidence':'saved observed-track predecessor history only; no new actual matrix claims','manual_review_started':(out/'manual_reviews.json').exists()})
    lines=['# E007 frozen visual evidence','', 'EVAL ONLY. All camera panels and at most three registered timepoints per case. Semantic review is separate.','']
    lines.extend(f'- [{r["case_id"]} {r["segment"]} frame {r["frame"]} {r["structure"]}]({r["image"]}) · [source detail]({r["detail"]})' for r in index)
    (out/'case_index.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'rendered_windows':len(cases),'unique_source_images':len(image_sources),'new_tracking_ticks':0}))


def audit_delivery(root,config):
    """Check frozen definitions, used inputs and manual coverage; summarize actual logs."""
    root=Path(root);out=root/config['output'];cases=json.loads((out/'cases_frozen.json').read_text())
    reviews=json.loads((out/'manual_reviews.json').read_text(encoding='utf-8'));retain_review(cases,reviews)
    freeze=json.loads((out/'mechanical_and_sample_freeze.json').read_text())
    assert all(sha(out/name)==value for name,value in freeze['files'].items())
    manifest=json.loads((out/'input_manifest.json').read_text())
    assert all(sha(root/s['path'])==s['sha256'] and (root/s['path']).stat().st_size==s['bytes'] for s in manifest)
    groups=list(rows(out/'group_relations.csv'));all_members=list(rows(out/'member_features.csv'))
    assert len(groups)==7843 and len(all_members)==10515
    structures=[]
    for segment in config['segment_order']:
        for structure in config['structure_order']:
            rr=[r for r in groups if r['segment']==segment and r['structure']==structure]
            summary={'segment':segment,'structure':structure,'N':len(rr),'fully_connected':sum(r['fully_connected']=='True' for r in rr),'B3_observed_in_roi':sum(r['B3_observed_in_roi']=='True' for r in rr)}
            for variant,label in [('B3','B3'),('B2','B2_parent')]:
                for definition,name in [('raw_observations','raw_matched'),('filtered_observed','observed_matched'),('identity','IDTP')]:summary[label+'_'+name]=sum(r[variant+'_'+definition+'_personID']!='' for r in rr)
            for bucket in ['0','1','2+']:summary['GT_near_'+bucket.replace('+','plus')]=sum(r['GT_near_bucket']==bucket for r in rr)
            summary['region_crossings']=sum(r['region_crossed']=='True' for r in rr);structures.append(summary)
    write_csv(out/'structure_summary.csv',structures)
    for sc in config['segments'].values():
        for condition,paths in sc['conditions'].items():
            source_obs={}
            for cam,path in sc['observation_sources'].items():
                source_obs.update({(int(r['frame']),cam,r['obs_id']):r for r in rows(root/path)})
            bygroup=defaultdict(list)
            for m in rows(root/paths['members']):
                ob=source_obs[(int(m['frame']),m['camera'],m['obs_id'])]
                assert ob['reason']=='accepted'
                assert abs(float(m['raw_x_m'])-float(ob['x_m']))<=1e-12
                assert abs(float(m['raw_y_m'])-float(ob['y_m']))<=1e-12
                bygroup[m['fusion_id']].append([float(m['raw_x_m']),float(m['raw_y_m'])])
            for g in rows(root/paths['groups']):
                assert np.max(np.abs(np.mean(bygroup[g['fusion_id']],axis=0)-[float(g['x_m']),float(g['y_m'])]))<=1e-12
    transitions=[];used_logs=[]
    for segment in config['segment_order']:
        sc=config['segments'][segment];chosen=[c for c in cases if c['segment']==segment]
        frames={t for c in chosen for t in c['times']};records={}
        path=sc['B3_associations'];used_logs.append({'path':path,'bytes':(root/path).stat().st_size,'sha256':sha(root/path)})
        with (root/path).open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                if r['frame'] in frames:
                    for i,j in zip(r['greedy_candidate_indices'],r['greedy_track_indices']):
                        gid=r['candidates'][i]['obs_id'];obj=r['objects'][j]
                        assert (r['frame'],gid) not in records
                        records[(r['frame'],gid)]={'stage':r['stage'],'candidate_index':i,'track_index':j,'distance_m':r['distance_matrix_rows_candidates_cols_tracks'][i][j],**obj}
        for case in chosen:
            detail=json.loads((out/'cases'/f'{case["case_id"]}.json').read_text())
            for variant in ['B2','B3']:
                trackrows=[r for r in rows(root/sc['conditions'][variant]['tracks']) if r['state']=='observed']
                history=defaultdict(list)
                for t in sorted(trackrows,key=lambda r:int(r['frame'])):history[t['track_id']].append(t)
                for time in detail['timepoints']:
                    gid=time['groups'][variant]
                    if not gid:continue
                    current=next(t for t in trackrows if t['obs_id']==gid)
                    before=[t for t in history[current['track_id']] if int(t['frame'])<int(current['frame'])]
                    prev=before[-1] if before else None
                    actual=records.get((time['frame'],gid)) if variant=='B3' else None
                    if variant=='B3':
                        assert (actual is None)==(prev is None),(segment,gid)
                        if actual:assert actual['previous_obs_id']==prev['obs_id'] and int(actual['track_id'])==int(current['track_id'])
                    transitions.append({'case_id':case['case_id'],'segment':segment,'variant':variant,'frame':time['frame'],'group':gid,'track_id':current['track_id'],'previous_group':prev['obs_id'] if prev else '',
                        'previous_frame':prev['frame'] if prev else '', 'evidence':'existing actual B3 record + formal states' if variant=='B3' else 'formal states predecessor only; matrix not claimed',
                        'actual_stage':actual['stage'] if actual else '', 'actual_distance_m':actual['distance_m'] if actual else '',
                        'candidate_index':actual['candidate_index'] if actual else '', 'track_index':actual['track_index'] if actual else '',
                        'predicted_xy':json.dumps(actual['predicted_xy']) if actual else '', 'new_track_no_previous':prev is None})
    write_csv(out/'selected_temporal_transitions.csv',transitions);save_json(out/'used_association_sources.json',used_logs)
    summaries=[]
    stages=['detection_or_foot_issue','cross_view_mixed_identity','temporal_identity_change']
    for segment in config['segment_order']:
        for selection,structures in [('random',config['structure_order']),('purpose',['fixed four slots'])]:
            for structure in structures:
                selected=[c for c in cases if c['segment']==segment and selection in c['selection_sources'] and (selection=='purpose' or c['structure']==structure)]
                rr=[next(r for r in reviews if r['case_id']==c['case_id']) for c in selected]
                row={'selection':selection,'segment':segment,'structure':structure,'selected':len(selected),'reviewed':len(rr),'any_issue_supported':sum(any(r[k]=='支持' for k in stages) for r in rr),'any_uncertainty':sum(any(r[k]=='不确定' for k in stages) for r in rr)}
                for stage in stages:
                    for status in ['支持','不支持','不确定','不适用']:row[stage+'_'+status]=sum(r[stage]==status for r in rr)
                summaries.append(row)
    write_csv(out/'manual_review_summary.csv',summaries)
    lines=['# E007逐例视觉判读','', '48个冻结窗口全部已审。支持仅指本例可见证据；不支持指本窗未见证据，不证明该环节全程无错。不确定不换样。检测/足部风险包含可见性和截断，不等同真实接地点误差。','']
    for c in cases:
        r=next(r for r in reviews if r['case_id']==c['case_id'])
        lines += [f'## {c["case_id"]} {c["segment"]} frame {c["frame"]} · {c["structure"]}', '', f'[全部相机/时刻拼图](cases/{c["case_id"]}.jpg) · [来源与上下文](cases/{c["case_id"]}.json)', '', f'检测/可见性风险：{r[stages[0]]}；同帧异人混组：{r[stages[1]]}；时间身份变化：{r[stages[2]]}。', '', r['visible_evidence_and_limits'], '']
    (out/'visual_review.md').write_text('\n'.join(lines),encoding='utf-8')
    save_json(out/'parent_anomalies.json',{'B3_groups':len(groups),'B3_members':len(all_members),'missing_or_duplicate_source_atoms':0,'unrestored_parents':0,'unrepresented_B2_parents':0,'multiply_represented_B2_parents':0,'source_XY_and_equal_weight_means_checked':True,'coordinate_tolerance':1e-12})
    save_json(out/'delivery_checks.json',{'checked_utc':datetime.now(timezone.utc).isoformat(),'frozen_file_hashes_unchanged':True,'used_inputs_unchanged':len(manifest),'reviewed_windows':len(reviews),'random_units':36,'purpose_windows':12,'max_timepoints':max(len(c['times']) for c in cases),'replacements':0,'new_tracking_ticks':0,'B3_actual_selected_transitions_verified':sum(t['variant']=='B3' for t in transitions),'B2_new_matrix_claims':0,'GPU_calls':0,'heldout_used':False,'new_dependencies':0,'downloads':0,'training':0})
    print(json.dumps({'all_reviews_present':len(reviews),'frozen_and_inputs_unchanged':True,'selected_temporal_rows':len(transitions),'new_tracking_ticks':0}))
