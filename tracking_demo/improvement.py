"""E003 controlled sensitivity-weighted fusion; prediction never reads GT."""
import csv
from datetime import datetime
import json,os,sys,time
import cv2
import numpy as np
from .fusion import fuse_observations
from .multicam import cameras_for,read_observations,write_csv,snapshot_sources,detect_views
from .tracking import GroundTracker
from .render import MultiRenderer


def ground_jacobian(camera,uv):
    """Analytic inverse-homography derivative, meters per pixel."""
    a=camera.H_inv
    q=a @ np.array([*uv,1.])
    j=(a[:2,:2]*q[2]-q[:2,None]*a[2,:2])/q[2]**2
    if not np.isfinite(j).all():raise ValueError('Invalid ground Jacobian')
    return j


def weight_group(group,cameras,enabled):
    members=group['members'];sensitivity=[]
    for m in members:
        j=ground_jacobian(cameras[m['camera']],[m['foot_u'],m['foot_v']])
        trace=float(np.sum(j*j))
        if not np.isfinite(trace) or trace<=0:raise ValueError('Nonpositive sensitivity')
        sensitivity.append(trace)
    weights=1/np.asarray(sensitivity) if enabled else np.ones(len(members))
    weights/=weights.sum()
    xy=np.array([[m['x_m'],m['y_m']] for m in members])
    if enabled and len(members)>1:
        group['x_m'],group['y_m']=[float(v) for v in weights @ xy]
    return weights.tolist(),sensitivity


def replay(root,config):
    output=root/config['output'];output.mkdir(parents=True,exist_ok=True)
    snapshot_sources(root,config)
    cameras=cameras_for(root,config)
    cache_start=time.perf_counter()
    views={c:read_observations(root/p) for c,p in config['observation_sources'].items()}
    cache_seconds=time.perf_counter()-cache_start
    frames=range(config['frame_start'],config['frame_stop'],config['frame_step'])
    for variant,names in config['variants'].items():
        folder=output/variant;folder.mkdir(exist_ok=True)
        tracker=GroundTracker(config['tracker'])
        accepted_rows=[];audit=[];members=[];tracks=[];ticks=[];timings=[]
        started=time.perf_counter();stamp=datetime.now().astimezone().isoformat()
        for frame in frames:
            begin=time.perf_counter()
            by_camera={c:[r for r in views[c] if r['frame']==frame and r['reason']=='accepted'] for c in names}
            groups=fuse_observations(by_camera,names,config['fusion_distance_m'],frame,variant)
            for g in groups:
                g['weights'],g['sensitivity']=weight_group(g,cameras,variant in config['weighted_variants'])
                g['reason']='accepted'
            accepted=[g for g in groups if g['reason']=='accepted']
            states,events=tracker.update(frame/config['frame_number_hz'],[{'obs_id':g['fusion_id'],'x_m':g['x_m'],'y_m':g['y_m']} for g in accepted])
            mapping={s['obs_id']:s['track_id'] for s in states if s['state']=='observed'}
            assert set(mapping)=={g['fusion_id'] for g in accepted}
            assert len({s['track_id'] for s in states})==len(states)
            used=set()
            for g in groups:
                key=g['fusion_id'];tid=mapping.get(key,'')
                row={'frame':frame,'time_s':frame/10,'fusion_id':key,'x_m':g['x_m'],'y_m':g['y_m'],'z_m':0.,
                     'member_count':len(g['members']),'member_cameras':';'.join(m['camera'] for m in g['members']),
                     'member_obs_ids':';'.join(m['obs_id'] for m in g['members']),
                     'max_pair_distance_m':g['max_pair_distance_m'],'track_id':tid,'reason':g['reason']}
                audit.append(row)
                if g['reason']=='accepted':accepted_rows.append(row)
                assert len({m['camera'] for m in g['members']})==len(g['members'])
                assert g['max_pair_distance_m']<=config['fusion_distance_m']
                for index,m in enumerate(g['members']):
                    assert m['obs_id'] not in used;used.add(m['obs_id'])
                    members.append({'frame':frame,'time_s':frame/10,'camera':m['camera'],'obs_id':m['obs_id'],
                                    'raw_x_m':m['x_m'],'raw_y_m':m['y_m'],'fusion_id':key,'track_id':tid,'reason':g['reason'],
                                    'normalized_weight':g['weights'][index],'sensitivity_m2_per_px2':g['sensitivity'][index]})
            assert len(used)==sum(len(v) for v in by_camera.values())
            for s in states:s.update({'frame':frame,'variant':variant,'in_roi':cameras['C1'].in_roi([s['x_m'],s['y_m']])})
            tracks.extend(states);ticks.extend(events)
            timings.append({'frame':frame,'input_observations':len(used),'groups_before_policy':len(groups),
                            'accepted_groups':len(accepted),'excluded_groups':len(groups)-len(accepted),
                            'fusion_tracking_ms':(time.perf_counter()-begin)*1000})
        seconds=time.perf_counter()-started
        for filename,rows in [('fused_observations.csv',accepted_rows),('group_audit.csv',audit),('fusion_members.csv',members),
                              ('tracks.csv',tracks),('ticks.csv',ticks),('frame_timings.csv',timings)]:
            write_csv(folder/filename,rows)
        summaries=[]
        for tid in sorted({s['track_id'] for s in tracks}):
            ss=[s for s in tracks if s['track_id']==tid]
            summaries.append({'track_id':tid,'first_time_s':ss[0]['time_s'],'last_time_s':ss[-1]['time_s'],
                              'observed_states':sum(s['state']=='observed' for s in ss),'predicted_states':sum(s['state']=='predicted' for s in ss)})
        write_csv(folder/'track_summary.csv',summaries)
        report={'status':'completed','segment':config['segment'],'variant':variant,'cameras':names,'frames':len(frames),
                'started_at':stamp,'finished_at':datetime.now().astimezone().isoformat(),'pid':os.getpid(),
                'command_argv':[sys.executable,*sys.argv],'cached_fusion_tracking_seconds':seconds,'shared_cache_seconds':cache_seconds,
                'timing_scope':'Cache loading and result serialization excluded; no inference, image reads or rendering.',
                'input_observations':len(members),'groups_before_policy':len(audit),'accepted_groups':len(accepted_rows),
                'excluded_reasons':{},'weighting':'inverse trace(J J.T)' if variant in config['weighted_variants'] else 'arithmetic mean',
                'group_sizes_before_policy':{str(n):sum(g['member_count']==n for g in audit) for n in (1,2,3)},
                'track_ids':len(summaries),'observed_states':sum(s['state']=='observed' for s in tracks),
                'predicted_states':sum(s['state']=='predicted' for s in tracks),
                'id_observation_distribution':{label:sum((s['observed_states']==1 if label=='one' else 2<=s['observed_states']<=5 if label=='two_to_five' else s['observed_states']>=6) for s in summaries) for label in ('one','two_to_five','six_plus')},
                'initial_state':'empty tracker, no warmup; independent segment/condition IDs','source_observations':config['observation_sources'],
                'prediction_inputs':'Image-derived observations, calibration, fixed config only; GT not read.'}
        (folder/'run.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(f'{config["segment"]} {variant}: {len(audit)} groups -> {len(accepted_rows)} accepted; {len(summaries)} IDs',flush=True)


def render(root,config):
    output=root/config['output'];cameras=cameras_for(root,config)
    views={c:read_observations(root/p) for c,p in config['observation_sources'].items()}
    for variant in config['media_variants']:
        folder=output/variant;names=config['variants'][variant]
        def read(name):
            with (folder/name).open() as f:return list(csv.DictReader(f))
        tracks=read('tracks.csv');members=read('fusion_members.csv');groups=read('fused_observations.csv')
        for s in tracks:
            s.update(frame=int(s['frame']),track_id=int(s['track_id']),x_m=float(s['x_m']),y_m=float(s['y_m']),in_roi=s['in_roi']=='True')
        renderer=MultiRenderer({c:cameras[c] for c in names},config['roi_m'],config['video']['trail_seconds'],
                               'sensitivity weights' if variant in config['weighted_variants'] else 'arithmetic mean')
        writer=None
        if config['segment']=='validation' and variant=='B3':
            writer=cv2.VideoWriter(str(folder/'global_tracking.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),2.,(1920,1080))
            if not writer.isOpened():raise RuntimeError('Video writer failed')
        begin=time.perf_counter()
        try:
            for f in range(config['frame_start'],config['frame_stop'],config['frame_step']):
                states=[s for s in tracks if s['frame']==f];renderer.update(f/10,states)
                if writer is None and f not in config['screenshots']:continue
                mapping={m['obs_id']:int(m['track_id']) for m in members if int(m['frame'])==f and m['track_id']}
                observations=[dict(r) for c in names for r in views[c] if r['frame']==f]
                for r in observations:r['track_id']=mapping.get(r['obs_id'],'')
                images={c:cv2.imread(str(root/config['data_root']/f'Image_subsets/{c}/{f:08d}.png')) for c in names}
                counts=[sum(int(g['frame'])==f and int(g['member_count'])==n for g in groups) for n in (1,2,3)]
                canvas=renderer.draw(images,f,observations,states,counts)
                cv2.putText(canvas,f'E003 {config["segment"]} {variant} '+('sensitivity weighted' if variant.startswith('W') else 'baseline'),(20,1064),0,.55,(255,255,255),1)
                if writer is not None:writer.write(canvas)
                if f in config['screenshots']:cv2.imwrite(str(folder/f'global_{f:08d}.jpg'),canvas)
        finally:
            if writer is not None:writer.release()
        (folder/'media_run.json').write_text(json.dumps({'status':'completed','pid':os.getpid(),'seconds':time.perf_counter()-begin,
          'scope':'CPU saved-state rendering only; no inference, fusion or filter updates.','video_frames':len(range(config['frame_start'],config['frame_stop'],config['frame_step'])) if writer else 0},indent=2))


def execute(root,config,mode):
    if mode=='diagnose':raise ValueError('Development evidence scripts are documented in the E003 report; no validation diagnosis before lock.')
    if mode in ('detect','all'):
        if config['segment']!='validation':raise ValueError('Development must reuse E001/E002 detection caches')
        detect_views(root,config)
    if mode in ('run','all'):replay(root,config)
    if mode in ('evaluate','all'):
        from .analysis import compare_multicam
        compare_multicam(root,config)
    if mode in ('render','all'):render(root,config)
