"""Image-only formal prediction. This module never reads annotations or GT IDs."""
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
import torch

from .geometry import GroundCamera
from .render import Renderer
from .tracking import GroundTracker


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def predict(root, config, config_path):
    os.environ['YOLO_CONFIG_DIR'] = str(root / '.cache/ultralytics')
    os.environ['MPLCONFIGDIR'] = str(root / '.cache/matplotlib')
    from ultralytics import YOLO, settings
    settings.update({'sync': False})
    output = root / config['output']
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / 'source'
    snapshot.mkdir(exist_ok=True)
    sources = [root / 'run_demo.py', config_path, root / 'requirements.txt', root / 'tests/test_core.py']
    sources += sorted((root / 'tracking_demo').glob('*.py'))
    for path in sources:
        destination = snapshot / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    (output / 'config_used.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    started_at = datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    camera = GroundCamera(root / config['data_root'], config['camera_calibration'], config['roi_m'])
    tracker = GroundTracker(config['tracker'])
    renderer = Renderer(camera, config['video'])
    model_start = time.perf_counter()
    model = YOLO(str(root / config['weights']))
    model_load_seconds = time.perf_counter() - model_start
    writer = cv2.VideoWriter(str(output / 'C1_ground_tracking.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                             config['video']['fps'], (config['video']['width'], config['video']['height']))
    if not writer.isOpened():
        raise RuntimeError('MP4 video writer did not open')
    all_detections, all_observations, all_states, tick_events, timings = [], [], [], [], []
    frame_ids = list(range(config['frame_start'], config['frame_stop'], config['frame_step']))
    try:
        for number, frame in enumerate(frame_ids, 1):
            frame_start = time.perf_counter()
            time_s = frame / config['frame_number_hz']
            image_path = root / config['data_root'] / f'Image_subsets/{config["camera"]}/{frame:08d}.png'
            image = cv2.imread(str(image_path))
            if image is None or image.shape[:2] != (1080, 1920):
                raise ValueError(f'Expected calibration-size image: {image_path}')
            read_end = time.perf_counter()
            torch.cuda.synchronize()
            predict_start = time.perf_counter()
            result = model.predict(image, verbose=False, **config['detector'])[0]
            torch.cuda.synchronize()
            predict_end = time.perf_counter()
            device = str(next(model.model.parameters()).device)
            if not device.startswith('cuda:'):
                raise RuntimeError(f'Expected GPU inference, got {device}')
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            observations = []
            for index, (box, confidence) in enumerate(zip(boxes, scores)):
                x1, y1, x2, y2 = [float(value) for value in box]
                obs_id = f'{config["camera"]}_{frame:08d}_{index:03d}'
                detection = {'frame': frame, 'time_s': time_s, 'camera': config['camera'], 'obs_id': obs_id,
                             'class_id': 0, 'confidence': float(confidence),
                             'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'device': device}
                all_detections.append(detection)
                foot = ((x1+x2)/2, y2)
                xy, reason = camera.to_ground(foot)
                observations.append({**detection, 'foot_u': foot[0], 'foot_v': foot[1],
                                     'x_m': float(xy[0]), 'y_m': float(xy[1]), 'z_m': 0.,
                                     'reason': reason, 'track_id': ''})
            accepted = [row for row in observations if row['reason'] == 'accepted']
            states, events = tracker.update(time_s, accepted)
            mapping = {row['obs_id']: row['track_id'] for row in states if row['state'] == 'observed'}
            for row in observations:
                row['track_id'] = mapping.get(row['obs_id'], '')
            for row in states:
                row['frame'] = frame
                row['camera'] = config['camera']
                row['in_roi'] = camera.in_roi((row['x_m'], row['y_m']))
            ids = [row['track_id'] for row in states]
            if len(ids) != len(set(ids)) or len(mapping) != len(accepted):
                raise RuntimeError('Same-frame ID/observation mapping is not one-to-one')
            geometry_end = time.perf_counter()
            canvas = renderer.draw(image, frame, time_s, observations, states)
            writer.write(canvas)
            if frame in config['screenshots']:
                cv2.imwrite(str(output / f'C1_BEV_{frame:08d}.jpg'), canvas)
            all_observations.extend(observations)
            all_states.extend(states)
            tick_events.extend(events)
            timings.append({'frame': frame, 'time_s': time_s, 'detections': len(boxes),
                            'accepted': len(accepted), 'observed_tracks': len(mapping),
                            'predicted_tracks': len(states)-len(mapping), 'device': device,
                            'read_ms': (read_end-frame_start)*1000,
                            'predict_wall_ms': (predict_end-predict_start)*1000,
                            'preprocess_ms': result.speed['preprocess'], 'inference_ms': result.speed['inference'],
                            'postprocess_ms': result.speed['postprocess'],
                            'geometry_tracking_ms': (geometry_end-predict_end)*1000,
                            'render_encode_ms': (time.perf_counter()-geometry_end)*1000,
                            'total_frame_ms': (time.perf_counter()-frame_start)*1000})
            if number % 10 == 0:
                print(f'processed {number}/{len(frame_ids)} frame={frame:08d} t={time_s:.1f}s obs={len(accepted)} tracks={len(states)}', flush=True)
    finally:
        writer.release()
    processing_seconds = time.perf_counter() - started
    write_csv(output / 'detections.csv', all_detections)
    write_csv(output / 'observations.csv', all_observations)
    write_csv(output / 'tracks.csv', all_states)
    write_csv(output / 'ticks.csv', tick_events)
    write_csv(output / 'frame_timings.csv', timings)
    track_summaries = []
    for track_id in sorted({row['track_id'] for row in all_states}):
        rows = [row for row in all_states if row['track_id'] == track_id]
        speeds = [float(np.hypot(row['vx_mps'], row['vy_mps'])) for row in rows if row['velocity_status'] == 'estimated']
        track_summaries.append({'track_id': track_id, 'first_time_s': rows[0]['time_s'],
                                'last_time_s': rows[-1]['time_s'], 'observed_states': sum(r['state']=='observed' for r in rows),
                                'predicted_states': sum(r['state']=='predicted' for r in rows),
                                'max_estimated_speed_mps': max(speeds) if speeds else ''})
    write_csv(output / 'track_summary.csv', track_summaries)
    capture = cv2.VideoCapture(str(output / 'C1_ground_tracking.mp4'))
    video_info = {'fps': capture.get(cv2.CAP_PROP_FPS), 'frames': int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                  'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    capture.release()
    if video_info['frames'] != len(frame_ids) or video_info['fps'] != 2.:
        raise RuntimeError(f'Unexpected exported video timeline: {video_info}')
    summary = {'status': 'completed', 'started_at': started_at,
               'finished_at': datetime.now().astimezone().isoformat(), 'pid': os.getpid(),
               'command_argv': [sys.executable, *sys.argv], 'executable': sys.executable,
               'code_version': {'git_commit': None, 'reason': 'No Git repository; actual source/config snapshot saved.', 'source_snapshot': str(snapshot.relative_to(root))},
               'config': config, 'calibration': camera.describe(), 'model_load_seconds': model_load_seconds,
               'processing_seconds_including_model_load_and_video': processing_seconds,
               'frames': len(frame_ids), 'first_frame': frame_ids[0], 'last_frame': frame_ids[-1],
               'source_time_first_s': 0., 'source_time_last_s': frame_ids[-1]/10., 'tick_seconds': tracker.dt,
               'detections': len(all_detections), 'accepted_observations': sum(r['reason']=='accepted' for r in all_observations),
               'rejections': {reason: sum(r['reason']==reason for r in all_observations) for reason in sorted({r['reason'] for r in all_observations if r['reason']!='accepted'})},
               'track_ids_created': tracker.tracker.total_object_count,
               'ids_observed_at_least_twice': sum(r['observed_states']>=2 for r in track_summaries),
               'observed_track_states': sum(r['state']=='observed' for r in all_states),
               'pure_prediction_states': sum(r['state']=='predicted' for r in all_states),
               'video': video_info, 'device': device, 'dtype': str(next(model.model.parameters()).dtype),
               'gpu': torch.cuda.get_device_name(0), 'peak_torch_allocated_bytes': torch.cuda.max_memory_allocated(0),
               'versions': {name: importlib.metadata.version(name) for name in ['torch','torchvision','numpy','opencv-python','ultralytics','norfair','filterpy']},
               'prediction_inputs': 'Only images, local weights, calibration and fixed config/ROI. No annotation files or GT identity assignment.',
               'timing_scope': 'Per-frame internal inference vs full processing vs fixed 2fps playback are distinct. First predict includes setup/warmup.'}
    (output / 'run.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('status','frames','processing_seconds_including_model_load_and_video','accepted_observations','track_ids_created','pure_prediction_states','video')}, indent=2))
    return summary

