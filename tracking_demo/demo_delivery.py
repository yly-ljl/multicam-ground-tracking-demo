"""New-output playback/cache/image entry points; heavy imports are deliberately lazy."""
import csv
from datetime import datetime
import json,os,sys,time
from pathlib import Path


def fresh_output(root,argument):
    if not argument:raise ValueError('--output is required: choose a NEW directory under outputs/')
    path=(root/argument).resolve();outputs=(root/'outputs').resolve()
    if not path.is_relative_to(outputs):raise ValueError('Demo outputs must stay under project outputs/')
    for name in ('E001_c1_baseline','E002_multicam','E003_improvement','E004_identity','E005_temporal_assignment','PHASE5_review'):
        if path.is_relative_to(outputs/name):raise ValueError('Choose a new output outside preserved experiment/review directories')
    if path.exists():raise FileExistsError(f'Output already exists: {path}; choose a new directory')
    path.mkdir(parents=True);return path

#准备对象-逐时刻处理-保存结果
def sequence(root,config,output,views=None,model=None):
    import cv2
    import numpy as np
    from .geometry import GroundCamera
    from .tracking import GroundTracker
    from .fusion import fuse_observations
    from .multicam import write_csv
    from .showcase import ShowcaseRenderer
    if model is not None:import torch
    output.mkdir(parents=True,exist_ok=True);begin=time.perf_counter();started=datetime.now().astimezone().isoformat()

    #创建四类对象
    cameras={c:GroundCamera(root/config['data_root'],config['calibrations'][c],config['roi_m']) for c in config['cameras']}
    tracker=GroundTracker(config['tracker']);assert config['tracker'].get('association_method','greedy')=='greedy'
    renderer=ShowcaseRenderer(cameras,config['roi_m'],config['overview_bounds'],config['zoom_bounds'])
    writer=cv2.VideoWriter(str(output/'prediction.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),2.,(1920,1080))

    if not writer.isOpened():raise RuntimeError('Video writer did not open')
    setup_seconds=time.perf_counter()-begin;all_obs=[];groups_out=[];members=[];tracks=[];ticks=[];image_times=[];tick_times=[]
    frame_ids=list(range(config['frame_start'],config['frame_stop'],config['frame_step']))
    try:
        #两层循环处理
        for number,frame in enumerate(frame_ids):
            tick_start=time.perf_counter();images={};observations=[]
            for name in config['cameras']:
                read_begin=time.perf_counter()

                #读取图像
                im=cv2.imread(str(root/config['data_root']/f'Image_subsets/{name}/{frame:08d}.png'))

                read_ms=(time.perf_counter()-read_begin)*1000
                if im is None or im.shape[:2]!=(1080,1920):raise ValueError(f'Invalid image {name}/{frame}')
                images[name]=im;prediction_ms=0.;projection_ms=0.;sync_before_ms=0.;speed={}
                if model is None:
                    local=[dict(r) for r in views[name] if r['frame']==frame]
                else:
                    sync_begin=time.perf_counter();torch.cuda.synchronize();sync_before_ms=(time.perf_counter()-sync_begin)*1000
                    predict_begin=time.perf_counter()

                    #模型接收图像，输出检测结果
                    result=model.predict(im,verbose=False,**config['detector'])[0]

                    torch.cuda.synchronize()
                    prediction_ms=(time.perf_counter()-predict_begin)*1000

                    speed=result.speed;project_begin=time.perf_counter();local=[]
                    device=str(next(model.model.parameters()).device)
                    if not device.startswith('cuda:'):raise RuntimeError(device)
                    for index,(box,score) in enumerate(zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy())):
                        x1,y1,x2,y2=[float(v) for v in box]

                        #取检测框底边中心作为脚点近似
                        foot=((x1+x2)/2,y2)
                        #根据相机标定，把脚点投到地面，得到米制位置 xy。
                        xy,reason=cameras[name].to_ground(foot)
                        #每个检测结果最后整理成一条观测

                        local.append({'frame':frame,'time_s':frame/10,'camera':name,'obs_id':f'{name}_{frame:08d}_{index:03d}','class_id':0,'confidence':float(score),
                                      'x1':x1,'y1':y1,'x2':x2,'y2':y2,'device':device,'foot_u':foot[0],'foot_v':foot[1],
                                      'x_m':float(xy[0]),'y_m':float(xy[1]),'z_m':0.,'reason':reason,'track_id':''})
                    projection_ms=(time.perf_counter()-project_begin)*1000
                observations.extend(local)
                image_times.append({'frame':frame,'camera':name,'read_ms':read_ms,'sync_before_ms':sync_before_ms,'synchronized_predict_wall_ms':prediction_ms,
                                    'cpu_projection_ms':projection_ms,'sdk_preprocess_ms':speed.get('preprocess',0.),'sdk_inference_ms':speed.get('inference',0.),'sdk_postprocess_ms':speed.get('postprocess',0.)})
            fusion_begin=time.perf_counter()

            #筛选投影被接受的观测，并按相机分类
            by_camera={
                c:[r for r in observations
                   if r['camera']==c and r['reason']=='accepted']
                for c in config['cameras']
            }

            #进行同一时刻的跨相机融合
            groups=fuse_observations(
                by_camera,config['cameras'],
                config['fusion_distance_m'],frame,'B3'
            )
            #将每个融合组的位置交给跟踪器，states为当前轨迹状态，events为时间步更新记录
            states,events=tracker.update(frame/10,[{'obs_id':g['fusion_id'],'x_m':g['x_m'],'y_m':g['y_m']} for g in groups])

            mapping={
                s['obs_id']:s['track_id']
                for s in states
                if s['state']=='observed'
            }
            #检查当前每个融合组都有对应的观测更新轨迹
            assert len(mapping)==len(groups)

            #沿着组成员，把轨迹ID对应回各相机观测。
            member_ids={}
            for g in groups:
                """
                  obs_id	某相机、某时刻的一条检测观测
                  fusion_id	某时刻的一个融合组
                  track_id	跨时刻维护的一条轨迹
                """
                tid=mapping[g['fusion_id']]
                groups_out.append({'frame':frame,'time_s':frame/10,'fusion_id':g['fusion_id'],'x_m':g['x_m'],'y_m':g['y_m'],'z_m':0.,'member_count':len(g['members']),
                                   'member_cameras':';'.join(m['camera'] for m in g['members']),'member_obs_ids':';'.join(m['obs_id'] for m in g['members']),
                                   'max_pair_distance_m':g['max_pair_distance_m'],'track_id':tid})
                for m in g['members']:
                    assert m['obs_id'] not in member_ids
                    member_ids[m['obs_id']]=tid #回答了某台相机的这条原始观测，最终属于哪条轨迹
                    members.append({'frame':frame,'time_s':frame/10,'camera':m['camera'],'obs_id':m['obs_id'],'fusion_id':g['fusion_id'],'raw_x_m':m['x_m'],'raw_y_m':m['y_m'],'track_id':tid})

            #把查到的轨迹ID写回原始观测记录。这样，三台相机中属于这个组的检测框，都能显示ID42
            for r in observations:
                r['track_id']=member_ids.get(r['obs_id'],'')

            for s in states:
                s.update(frame=frame,variant='B3',in_roi=cameras['C1'].in_roi([s['x_m'],s['y_m']]))
            fusion_ms=(time.perf_counter()-fusion_begin)*1000
            draw_begin=time.perf_counter()

            #绘制并保存本时刻结果，继续下一时刻
            renderer.update(frame/10,states)
            canvas=renderer.draw(images,frame,observations,states)

            draw_ms=(time.perf_counter()-draw_begin)*1000
            encode_begin=time.perf_counter()

            #把当前画面追加到视频，每处理一个时刻，就向视频写入一张展示画面
            writer.write(canvas)

            encode_ms=(time.perf_counter()-encode_begin)*1000
            if frame in config.get('screenshots',[]):
                cv2.imwrite(str(output/f'prediction_{frame:08d}.jpg'),
                canvas
                )
            all_obs.extend(observations);tracks.extend(states);ticks.extend(events)
            tick_times.append({'frame':frame,'time_s':frame/10,'fusion_tracking_ms':fusion_ms,'drawing_ms':draw_ms,'encoding_write_ms':encode_ms,'total_tick_ms':(time.perf_counter()-tick_start)*1000})
            if (number+1)%40==0:print(f'{output.name}: {number+1}/{len(frame_ids)} triplets',flush=True)
    finally:
        close_begin=time.perf_counter()
        writer.release() #完成视频写入并释放相关资源。
        close_seconds=time.perf_counter()-close_begin
    elapsed=time.perf_counter()-begin
    for filename,rows in [('observations.csv',all_obs),('fused_observations.csv',groups_out),('fusion_members.csv',members),('tracks.csv',tracks),('ticks.csv',ticks),('image_timings.csv',image_times),('tick_timings.csv',tick_times)]:write_csv(output/filename,rows)
    if model is not None:
        for c in config['cameras']:
            folder=output/'views'/c;folder.mkdir(parents=True,exist_ok=True);write_csv(folder/'observations.csv',[r for r in all_obs if r['camera']==c])
    report={'status':'completed','started_at':started,'finished_at':datetime.now().astimezone().isoformat(),'pid':os.getpid(),'command_argv':[sys.executable,*sys.argv],
            'mode':'images' if model is not None else 'cached_observations','ticks':len(frame_ids),'new_image_predict_calls':len(image_times) if model is not None else 0,
            'source_frames':[frame_ids[0],frame_ids[-1],config['frame_step']],'elapsed_seconds_after_setup_entry':elapsed,'setup_seconds':setup_seconds,'writer_close_seconds':close_seconds,
            'timing_scope':'Includes per-pass tracker/renderer/writer setup, reads, synchronized predict if applicable, projection, fusion/tracking, drawing/encoding, video close; excludes CSV writing.',
            'images_per_second_processing':len(image_times)/elapsed,'three_camera_ticks_per_second_processing':len(frame_ids)/elapsed,'playback_fps':2.,'video_duration_seconds':len(frame_ids)/2,
            'stage_seconds':{**{k:sum(r[k] for r in image_times)/1000 for k in ('read_ms','sync_before_ms','synchronized_predict_wall_ms','cpu_projection_ms')},
                             **{k:sum(r[k] for r in tick_times)/1000 for k in ('fusion_tracking_ms','drawing_ms','encoding_write_ms')}},
            'overview_clipped_states':renderer.overview_clipped_states,'final_overview_bounds':renderer.bounds,'overview_x_y_scale_equal':True,'zoom_x_y_scale_equal':True,
            'all_observed_and_predicted_states_displayed':True,'GT_used':False,'config':config}
    (output/'run.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report


def replay(root,config,output):
    from .multicam import read_observations
    start=time.perf_counter();views={c:read_observations(root/path) for c,path in config['observation_sources'].items()};load_seconds=time.perf_counter()-start
    report=sequence(root,config,output,views=views);report['observation_cache_load_seconds']=load_seconds
    (output/'run.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report


def cached_benchmark(root,config,source):
    from .multicam import read_observations
    from .fusion import fuse_observations
    from .tracking import GroundTracker
    start=time.perf_counter();views={c:read_observations(source/'views'/c/'observations.csv') for c in config['cameras']};load=time.perf_counter()-start;results=[]
    for repeat in range(3):
        tracker=GroundTracker(config['tracker']);begin=time.perf_counter();count=0
        for frame in range(config['frame_start'],config['frame_stop'],config['frame_step']):
            groups=fuse_observations({c:[r for r in views[c] if r['frame']==frame and r['reason']=='accepted'] for c in config['cameras']},config['cameras'],config['fusion_distance_m'],frame,'B3')
            states,_=tracker.update(frame/10,[{'obs_id':g['fusion_id'],'x_m':g['x_m'],'y_m':g['y_m']} for g in groups]);count+=1
        seconds=time.perf_counter()-begin;results.append({'repeat':repeat+1,'ticks':count,'seconds':seconds,'cached_triplet_ticks_per_second':count/seconds})
    return {'CSV_load_seconds':load,'scope':'CPU accepted-observation fusion + greedy tracking only; excludes images/inference/render/encoding; fresh tracker per repeat.','results':results}


def images(root,config,output,entry_start):
    os.environ['YOLO_CONFIG_DIR']=str(root/'.cache/ultralytics')
    os.environ['MPLCONFIGDIR']=str(root/'.cache/matplotlib')
    import_start=time.perf_counter()
    import cv2,numpy,torch
    from ultralytics import YOLO,settings
    from .showcase import ShowcaseRenderer
    from .tracking import GroundTracker
    from .fusion import fuse_observations
    import_seconds=time.perf_counter()-import_start
    settings.update({'sync':False})
    load_start=time.perf_counter()

    model=YOLO(str(root/config['weights']))

    model_seconds=time.perf_counter()-load_start;model_id=id(model)
    cuda_start=time.perf_counter();torch.cuda.synchronize();cuda_seconds=time.perf_counter()-cuda_start

    passes=config.get('passes',1);results=[];predictor_ids=[]

    for number in range(passes):
        #调用sequence返回报告
        report=sequence(root,config,output/f'pass_{number+1}',model=model)

        assert id(model)==model_id;predictor_ids.append(id(model.predictor));results.append(report)
    assert len(set(predictor_ids))==1
    first_total=import_seconds+model_seconds+cuda_seconds+results[0]['elapsed_seconds_after_setup_entry']
    report={'status':'completed','pid':os.getpid(),'command_argv':[sys.executable,*sys.argv],'heavy_import_seconds':import_seconds,'model_load_seconds':model_seconds,
            'CUDA_initialization_sync_seconds':cuda_seconds,'entry_module_setup_seconds_before_heavy_import':import_start-entry_start,
            'first_startup_plus_first_pass_seconds':first_total,'first_startup_images_per_second':results[0]['new_image_predict_calls']/first_total,
            'first_startup_three_camera_ticks_per_second':results[0]['ticks']/first_total,'model_objects_created':1,'model_object_id':model_id,'predictor_object_ids_per_pass':predictor_ids,
            'same_model_and_predictor_reused':True,'passes':results,'total_source_image_predict_calls':sum(r['new_image_predict_calls'] for r in results),
            'device':str(next(model.model.parameters()).device),'dtype':str(next(model.model.parameters()).dtype),'detector_config':config['detector'],
            'scope':'One process/model; first startup includes heavy imports+load+CUDA setup+first pass. Two later passes reuse the same loaded model/predictor. No OS page-cache flush; not cold-disk or production realtime certification.',
            'sdk_warmup_note':'Ultralytics may internally warm up using synthetic tensors on its first predict; image call counts count real source-image model.predict invocations.',
            'cached_CPU_fusion_tracking':cached_benchmark(root,config,output/'pass_1')}
    (output/'performance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('heavy_import_seconds','model_load_seconds','CUDA_initialization_sync_seconds','first_startup_plus_first_pass_seconds','total_source_image_predict_calls','same_model_and_predictor_reused')},indent=2));return report


def play(root,config,inspect=False):
    path=(root/config['video_artifact']).resolve()
    if not path.is_file():raise FileNotFoundError(path)
    if inspect:
        import cv2
        cap=cv2.VideoCapture(str(path));ok,frame=cap.read();meta={'video':str(path),'opened':bool(ok),'fps':cap.get(cv2.CAP_PROP_FPS),'frames':cap.get(cv2.CAP_PROP_FRAME_COUNT),'width':cap.get(cv2.CAP_PROP_FRAME_WIDTH),'height':cap.get(cv2.CAP_PROP_FRAME_HEIGHT)};cap.release()
        if not ok:raise RuntimeError('Video not readable')
        print(json.dumps(meta,indent=2));return meta
    os.startfile(str(path))


def execute(root,config,mode,output_argument,inspect=False,entry_start=None):
    if mode=='play':return play(root,config,inspect)
    output=fresh_output(root,output_argument)
    (output/'config_used.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    if mode=='replay':return replay(root,config,output)
    if mode=='images':return images(root,config,output,entry_start or time.perf_counter())
    raise ValueError(mode)
