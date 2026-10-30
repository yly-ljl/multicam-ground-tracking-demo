"""E002 image detection cache, then same-time fusion and global temporal tracking."""
import csv
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

from .fusion import fuse_observations
from .geometry import GroundCamera
from .render import MultiRenderer
from .tracking import GroundTracker


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def read_observations(path):
    with path.open(encoding='utf-8') as handle:
        rows=list(csv.DictReader(handle))
    for row in rows:
        row['frame']=int(row['frame'])
        for key in ('time_s','confidence','x1','y1','x2','y2','foot_u','foot_v','x_m','y_m','z_m'):
            row[key]=float(row[key])
        # An E001 local track ID must never become an E002 global ID implicitly.
        row['track_id']=''
    return rows


def cameras_for(root, config):
    return {name:GroundCamera(root/config['data_root'],config['calibrations'][name],config['roi_m'])
            for name in config['cameras']}


def snapshot_sources(root, config):
    output=root/config['output']; output.mkdir(parents=True,exist_ok=True)
    files=[root/'run_demo.py',root/'requirements.txt']
    files+=sorted((root/'tracking_demo').glob('*.py'))+sorted((root/'tests').glob('*.py'))
    files+=[root/'configs/multicam_baseline.json',root/'configs/c1_baseline.json']
    if config.get('experiment')=='E003':
        files+=sorted((root/'configs').glob('e003_*.json'))
    if config.get('experiment')=='E004':
        files+=sorted((root/'configs').glob('e004_*.json'))
    for source in files:
        destination=output/'source'/source.relative_to(root)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
    (output/'config_used.json').write_text(json.dumps(config,indent=2),encoding='utf-8')


def diagnose_views(root, config):
    from .analysis import diagnose
    for name in ('C2','C3'):
        local={**config,'camera':name,'camera_calibration':config['calibrations'][name],
               'output':f'{config["output"]}/diagnostics/{name}'}
        diagnose(root,local)


def detect_views(root, config):
    # This stage only reads new C2/C3 images, fixed weights and calibration; no GT.
    os.environ['YOLO_CONFIG_DIR']=str(root/'.cache/ultralytics')
    os.environ['MPLCONFIGDIR']=str(root/'.cache/matplotlib')
    import torch
    from ultralytics import YOLO, settings
    settings.update({'sync':False})
    snapshot_sources(root,config)
    output=root/config['output']; cameras=cameras_for(root,config)
    started_at=datetime.now().astimezone().isoformat(); started=time.perf_counter()
    model=YOLO(str(root/config['weights']))
    detection_cameras=config.get('detection_cameras',['C2','C3'])
    observations={name:[] for name in detection_cameras}; detections={name:[] for name in observations}
    timings=[]
    frames=list(range(config['frame_start'],config['frame_stop'],config['frame_step']))
    for number,frame in enumerate(frames,1):
        for name in detection_cameras:
            image_start=time.perf_counter()
            image=cv2.imread(str(root/config['data_root']/f'Image_subsets/{name}/{frame:08d}.png'))
            if image is None or image.shape[:2]!=(1080,1920):
                raise ValueError(f'Invalid calibration-size image {name}/{frame}')
            read_end=time.perf_counter(); torch.cuda.synchronize(); predict_start=time.perf_counter()
            result=model.predict(image,verbose=False,**config['detector'])[0]
            torch.cuda.synchronize(); predict_end=time.perf_counter()
            device=str(next(model.model.parameters()).device)
            if not device.startswith('cuda:'):
                raise RuntimeError(device)
            for index,(box,score) in enumerate(zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy())):
                x1,y1,x2,y2=[float(v) for v in box]
                detection={'frame':frame,'time_s':frame/config['frame_number_hz'],'camera':name,
                           'obs_id':f'{name}_{frame:08d}_{index:03d}','class_id':0,'confidence':float(score),
                           'x1':x1,'y1':y1,'x2':x2,'y2':y2,'device':device}
                foot=((x1+x2)/2,y2); xy,reason=cameras[name].to_ground(foot)
                detections[name].append(detection)
                observations[name].append({**detection,'foot_u':foot[0],'foot_v':foot[1],
                                            'x_m':float(xy[0]),'y_m':float(xy[1]),'z_m':0.,
                                            'reason':reason,'track_id':''})
            timings.append({'frame':frame,'camera':name,'time_s':frame/10,'device':device,
                            'read_ms':(read_end-image_start)*1000,'predict_wall_ms':(predict_end-predict_start)*1000,
                            'preprocess_ms':result.speed['preprocess'],'inference_ms':result.speed['inference'],
                            'postprocess_ms':result.speed['postprocess'],'total_image_ms':(time.perf_counter()-image_start)*1000})
        if number%10==0:
            print(f'new detection {number}/{len(frames)} times; {detection_cameras} sequentially',flush=True)
    summary={'started_at':started_at,'finished_at':datetime.now().astimezone().isoformat(),
             'stage_seconds':time.perf_counter()-started,'pid':os.getpid(),'command_argv':[sys.executable,*sys.argv],
             'status':'completed','frames_per_camera':len(frames),'new_inference_images':len(detection_cameras)*len(frames),
             'camera_order_per_time':detection_cameras,'device':device,'dtype':str(next(model.model.parameters()).dtype),
             'config':config,'source_snapshot':str((output/'source').relative_to(root)),
             'versions':{name:importlib.metadata.version(name) for name in ('torch','ultralytics','norfair','opencv-python')},
             'C1_source':f'{config["e001_output"]}/observations.csv','timing_scope':'New C2/C3 detection stage only. C1 reused from E001; not a fresh end-to-end 3-camera GPU throughput.',
             'views':{}}
    for name,rows in observations.items():
        folder=output/'views'/name; folder.mkdir(parents=True,exist_ok=True)
        write_csv(folder/'observations.csv',rows); write_csv(folder/'detections.csv',detections[name])
        summary['views'][name]={'detections':len(rows),'accepted':sum(r['reason']=='accepted' for r in rows),
                                'rejections':{reason:sum(r['reason']==reason for r in rows)
                                              for reason in sorted({r['reason'] for r in rows if r['reason']!='accepted'})},
                                'calibration':cameras[name].describe()}
    if config.get('experiment') in ('E003','E004'):
        summary['C1_source']=f'Fresh {config.get("segment","validation")} detection in views/C1/observations.csv'
        summary['timing_scope']='One sequential C1/C2/C3 GPU detection pass; all conditions reuse these observations. Excludes fusion/tracking/rendering.'
    write_csv(output/'detection_timings.csv',timings)
    (output/'detection_run.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('new_inference_images','stage_seconds','views')},indent=2))


def fuse_and_track(root, config):
    snapshot_sources(root,config)
    output=root/config['output']; cameras=cameras_for(root,config)
    source_paths={'C1':root/config['e001_output']/'observations.csv',
                  **{c:output/'views'/c/'observations.csv' for c in ('C2','C3')}}
    cache_started=time.perf_counter()
    view_rows={name:read_observations(path) for name,path in source_paths.items()}
    cache_load_seconds=time.perf_counter()-cache_started
    frames=list(range(config['frame_start'],config['frame_stop'],config['frame_step']))
    for variant,names in config['variants'].items():
        if len(names)==1:
            continue
        folder=output/variant; folder.mkdir(parents=True,exist_ok=True)
        tracker=GroundTracker(config['tracker']); renderer=MultiRenderer({c:cameras[c] for c in names},config['roi_m'])
        writer=None
        if len(names)==3:
            writer=cv2.VideoWriter(str(folder/'global_tracking.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),2.,(1920,1080))
            if not writer.isOpened(): raise RuntimeError('Video writer did not open')
        all_groups=[]; all_members=[]; all_tracks=[]; frame_rows=[]; all_ticks=[]
        started_at=datetime.now().astimezone().isoformat(); started=time.perf_counter()
        try:
            for frame in frames:
                frame_start=time.perf_counter()
                observations=[dict(r) for name in names for r in view_rows[name] if r['frame']==frame]
                by_camera={name:[r for r in observations if r['camera']==name and r['reason']=='accepted'] for name in names}
                groups=fuse_observations(by_camera,names,config['fusion_distance_m'],frame,variant)
                fused=[{'obs_id':g['fusion_id'],'x_m':g['x_m'],'y_m':g['y_m']} for g in groups]
                states,ticks=tracker.update(frame/10,fused)
                mapping={r['obs_id']:r['track_id'] for r in states if r['state']=='observed'}
                if len(mapping)!=len(groups) or len({s['track_id'] for s in states})!=len(states):
                    raise RuntimeError('Non-unique group/global track assignment')
                observation_ids=set(); member_to_track={}
                for g in groups:
                    key=g['fusion_id']; track_id=mapping[key]
                    all_groups.append({'frame':frame,'time_s':frame/10,'fusion_id':key,'x_m':g['x_m'],'y_m':g['y_m'],'z_m':0.,
                                       'member_count':len(g['members']),'member_cameras':';'.join(m['camera'] for m in g['members']),
                                       'member_obs_ids':';'.join(m['obs_id'] for m in g['members']),
                                       'max_pair_distance_m':g['max_pair_distance_m'],'track_id':track_id})
                    if len({m['camera'] for m in g['members']})!=len(g['members']) or g['max_pair_distance_m']>config['fusion_distance_m']:
                        raise RuntimeError('Invalid fusion group')
                    for m in g['members']:
                        if m['obs_id'] in observation_ids: raise RuntimeError('Observation used twice')
                        observation_ids.add(m['obs_id']); member_to_track[m['obs_id']]=track_id
                        all_members.append({'frame':frame,'time_s':frame/10,'camera':m['camera'],'obs_id':m['obs_id'],
                                            'raw_x_m':m['x_m'],'raw_y_m':m['y_m'],'fusion_id':key,
                                            'fused_x_m':g['x_m'],'fused_y_m':g['y_m'],
                                            'member_distance_to_mean_m':float(np.hypot(m['x_m']-g['x_m'],m['y_m']-g['y_m'])),
                                            'group_max_pair_distance_m':g['max_pair_distance_m'],'track_id':track_id})
                if len(observation_ids)!=sum(len(r) for r in by_camera.values()): raise RuntimeError('Unassigned observation')
                for r in observations: r['track_id']=member_to_track.get(r['obs_id'],'')
                for r in states: r.update({'frame':frame,'variant':variant,'in_roi':cameras['C1'].in_roi([r['x_m'],r['y_m']])})
                group_counts=[sum(len(g['members'])==count for g in groups) for count in (1,2,3)]
                renderer.update(frame/10,states)
                compute_end=time.perf_counter()
                if writer is not None or frame in config['screenshots']:
                    images={name:cv2.imread(str(root/config['data_root']/f'Image_subsets/{name}/{frame:08d}.png')) for name in names}
                    canvas=renderer.draw(images,frame,observations,states,group_counts)
                    if writer is not None: writer.write(canvas)
                    if frame in config['screenshots']: cv2.imwrite(str(folder/f'global_{frame:08d}.jpg'),canvas)
                all_tracks.extend(states); all_ticks.extend(ticks)
                frame_rows.append({'frame':frame,'time_s':frame/10,'input_observations':len(observation_ids),'fused_groups':len(groups),
                                   'groups_1':group_counts[0],'groups_2':group_counts[1],'groups_3':group_counts[2],
                                   'observed_states':len(mapping),'predicted_states':len(states)-len(mapping),
                                   'fusion_tracking_ms':(compute_end-frame_start)*1000,'total_replay_frame_ms':(time.perf_counter()-frame_start)*1000})
        finally:
            if writer is not None: writer.release()
        seconds=time.perf_counter()-started
        for filename,rows in [('fused_observations.csv',all_groups),('fusion_members.csv',all_members),('tracks.csv',all_tracks),('frame_timings.csv',frame_rows),('ticks.csv',all_ticks)]:
            write_csv(folder/filename,rows)
        summaries=[]
        for key in sorted({r['track_id'] for r in all_tracks}):
            rows=[r for r in all_tracks if r['track_id']==key]
            summaries.append({'track_id':key,'first_time_s':rows[0]['time_s'],'last_time_s':rows[-1]['time_s'],
                              'observed_states':sum(r['state']=='observed' for r in rows),'predicted_states':sum(r['state']=='predicted' for r in rows)})
        write_csv(folder/'track_summary.csv',summaries)
        summary={'status':'completed','variant':variant,'cameras':names,'frames':len(frames),'started_at':started_at,
                 'finished_at':datetime.now().astimezone().isoformat(),'pid':os.getpid(),'command_argv':[sys.executable,*sys.argv],
                 'cached_replay_seconds_with_rendering':seconds,'shared_cache_load_seconds':cache_load_seconds,
                 'timing_scope':'Timer starts after shared view CSV loading and per-condition initialization; includes fusion/tracking/image reading/render/video close, excludes CSV result writing. No GPU detection; not end-to-end inference throughput.',
                 'source_observations':{name:str(source_paths[name].relative_to(root)) for name in names},
                 'input_observations':len(all_members),'fused_groups':len(all_groups),
                 'group_sizes':{str(n):sum(r['member_count']==n for r in all_groups) for n in (1,2,3)},
                 'track_ids':len(summaries),'observed_states':sum(r['state']=='observed' for r in all_tracks),
                 'predicted_states':sum(r['state']=='predicted' for r in all_tracks),
                 'id_observation_distribution':{label:sum((r['observed_states']==1 if label=='one' else 2<=r['observed_states']<=5 if label=='two_to_five' else r['observed_states']>=6) for r in summaries) for label in ('one','two_to_five','six_plus')},
                 'tracker_config':config['tracker'],'fusion_gate_m':config['fusion_distance_m'],
                 'video_config':config['video'],
                 'prediction_inputs':'Only image-derived accepted observations and fixed parameters. No GT identities or positions.'}
        (folder/'run.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary,indent=2))


def render_cached(root, config):
    """Rebuild media from saved global states/mappings; no inference or filter update."""
    snapshot_sources(root,config)
    output=root/config['output']; cameras=cameras_for(root,config)
    cache_start=time.perf_counter()
    sources={'C1':root/config['e001_output']/'observations.csv',
             **{c:output/'views'/c/'observations.csv' for c in ('C2','C3')}}
    views={name:read_observations(path) for name,path in sources.items()}
    view_cache_seconds=time.perf_counter()-cache_start
    for variant,names in config['variants'].items():
        if len(names)==1: continue
        folder=output/variant
        with (folder/'tracks.csv').open(encoding='utf-8') as handle: tracks=list(csv.DictReader(handle))
        with (folder/'fusion_members.csv').open(encoding='utf-8') as handle: members=list(csv.DictReader(handle))
        with (folder/'fused_observations.csv').open(encoding='utf-8') as handle: groups=list(csv.DictReader(handle))
        for row in tracks:
            row['frame']=int(row['frame']); row['track_id']=int(row['track_id'])
            row['x_m']=float(row['x_m']); row['y_m']=float(row['y_m']); row['in_roi']=row['in_roi']=='True'
        renderer=MultiRenderer({c:cameras[c] for c in names},config['roi_m'],config['video']['trail_seconds'])
        writer=None
        if len(names)==3:
            writer=cv2.VideoWriter(str(folder/'global_tracking.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),2.,(1920,1080))
            if not writer.isOpened(): raise RuntimeError('Video writer did not open')
        started_at=datetime.now().astimezone().isoformat(); started=time.perf_counter()
        try:
            for frame in range(config['frame_start'],config['frame_stop'],config['frame_step']):
                states=[r for r in tracks if r['frame']==frame]
                renderer.update(frame/10,states)
                if writer is None and frame not in config['screenshots']: continue
                mapping={r['obs_id']:int(r['track_id']) for r in members if int(r['frame'])==frame}
                observations=[dict(r) for c in names for r in views[c] if r['frame']==frame]
                for r in observations: r['track_id']=mapping.get(r['obs_id'],'')
                counts=[sum(int(g['frame'])==frame and int(g['member_count'])==n for g in groups) for n in (1,2,3)]
                images={c:cv2.imread(str(root/config['data_root']/f'Image_subsets/{c}/{frame:08d}.png')) for c in names}
                canvas=renderer.draw(images,frame,observations,states,counts)
                if writer is not None: writer.write(canvas)
                if frame in config['screenshots']: cv2.imwrite(str(folder/f'global_{frame:08d}.jpg'),canvas)
        finally:
            if writer is not None: writer.release()
        report={'variant':variant,'started_at':started_at,'finished_at':datetime.now().astimezone().isoformat(),
                'pid':os.getpid(),'command_argv':[sys.executable,*sys.argv],'media_seconds_after_csv_loading':time.perf_counter()-started,
                'shared_observation_cache_load_seconds':view_cache_seconds,
                'scope':'Media-only CPU replay using saved tracks and members; no GPU, fusion or tracking recomputation.',
                'fixes':['History advanced every physical tick and pruned by 5-second time window.','Detection label baseline >=80px, below 0-50px camera title bar.'],
                'video_config':config['video']}
        (folder/'media_run.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
