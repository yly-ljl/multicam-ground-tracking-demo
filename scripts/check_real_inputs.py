"""Read up to three initial annotated C1 images and run GPU person detection only."""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/setup_check'
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.cache/ultralytics')
os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache/matplotlib')
os.environ['TORCH_HOME'] = str(ROOT / '.cache/torch')

import cv2
import torch
from ultralytics import YOLO, settings


def read_calibration(path):
    values = {}
    for node in ET.parse(path).getroot():
        data = node.find('data')
        if data is not None:
            values[node.tag] = {
                'rows': int(node.findtext('rows')), 'cols': int(node.findtext('cols')),
                'values': [float(x) for x in data.text.split()],
            }
        else:
            values[node.tag] = (node.text or '').strip()
    return {'path': str(path.relative_to(ROOT)), 'values': values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/Wildtrack_dataset')
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    settings.update({'sync': False})
    image_dir = data_root / 'Image_subsets/C1'
    annotation_dir = data_root / 'annotations_positions'
    image_ids = {p.stem for p in image_dir.glob('*.png')}
    annotation_ids = {p.stem for p in annotation_dir.glob('*.json')}
    aligned_ids = sorted(image_ids & annotation_ids)
    selected = aligned_ids[:3]
    assert selected and selected == ['00000000', '00000005', '00000010'][:len(selected)], selected
    calibration = {
        'intrinsic': read_calibration(data_root / 'calibrations/intrinsic_zero/intr_CVLab1.xml'),
        'extrinsic': read_calibration(data_root / 'calibrations/extrinsic/extr_CVLab1.xml'),
        'scope': 'Read parameters only; no projection or coordinate-convention validation yet.',
    }
    data_check = {
        'data_root': str(data_root),
        'directory_scope': 'Counts describe only this local input directory, not proof of full dataset download.',
        'c1_image_count': len(image_ids), 'annotation_count': len(annotation_ids),
        'aligned_count': len(aligned_ids), 'images_without_gt': sorted(image_ids - annotation_ids),
        'selected_development_frames': selected, 'calibration': calibration,
        'time_rule': 'frame_number / 10 seconds; steps of 5 correspond to 0.5 seconds.',
        'gt_identity_semantics': 'Not established by this setup check.', 'annotations': [],
    }
    for frame in selected:
        annotations = json.loads((annotation_dir / f'{frame}.json').read_text(encoding='utf-8'))
        data_check['annotations'].append({
            'frame': frame, 'count': len(annotations),
            'first_record_structure': annotations[0] if annotations else None,
        })
    (OUT / 'data_check.json').write_text(json.dumps(data_check, indent=2), encoding='utf-8')

    model_path = ROOT / 'weights/yolo11n.pt'
    assert model_path.stat().st_size == 5613764
    load_start = time.perf_counter()
    model = YOLO(str(model_path))
    load_seconds = time.perf_counter() - load_start
    rows, frames = [], []
    for frame in selected:
        image_path = image_dir / f'{frame}.png'
        image = cv2.imread(str(image_path))
        assert image is not None, image_path
        torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = model.predict(image, device=0, classes=[0], imgsz=640,
                                   conf=.25, iou=.7, half=False, verbose=False)[0]
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
        boxes = prediction.boxes.xyxy.cpu().numpy()
        confidences = prediction.boxes.conf.cpu().numpy()
        classes = prediction.boxes.cls.cpu().numpy().astype(int)
        assert all(class_id == 0 for class_id in classes)
        model_device = str(next(model.model.parameters()).device)
        assert model_device.startswith('cuda:'), model_device
        annotated = image.copy()
        for box, confidence, class_id in zip(boxes, confidences, classes):
            x1, y1, x2, y2 = [float(v) for v in box]
            rows.append({'frame': frame, 'time_seconds': int(frame) / 10,
                         'camera': 'C1', 'class_id': int(class_id), 'class_name': 'person',
                         'confidence': float(confidence),
                         'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                         'device': model_device})
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
            cv2.putText(annotated, f'person {confidence:.2f}', (int(x1), max(18, int(y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 0), 1, cv2.LINE_AA)
        output_image = OUT / f'C1_{frame}_persons.jpg'
        assert cv2.imwrite(str(output_image), annotated)
        frames.append({'frame': frame, 'time_seconds': int(frame) / 10,
                       'input': str(image_path.relative_to(ROOT)),
                       'original_hw': list(image.shape[:2]), 'person_count': len(boxes),
                       'model_device': model_device, 'result_device': str(prediction.boxes.data.device),
                       'predict_wall_seconds': wall_seconds, 'ultralytics_speed_ms': prediction.speed,
                       'output_image': str(output_image.relative_to(ROOT))})
    columns = ['frame', 'time_seconds', 'camera', 'class_id', 'class_name', 'confidence',
               'x1', 'y1', 'x2', 'y2', 'device']
    with (OUT / 'person_detections.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        'status': 'passed', 'recorded_at': datetime.now().astimezone().isoformat(),
        'weights': str(model_path.relative_to(ROOT)), 'model_load_seconds': load_seconds,
        'settings': {'classes': [0], 'imgsz': 640, 'conf': .25, 'iou': .7, 'half': False},
        'frames': frames, 'detections': rows,
        'peak_torch_allocated_bytes': torch.cuda.max_memory_allocated(0),
        'timing_note': 'First predict includes model setup and warmup; wall time excludes input reading and drawing. Library speed_ms is per-image internal timing.',
        'scope': 'Only person detection and data readability. No ground projection, tracking, quality metric, or real-time claim. GT does not enter model inference.',
    }
    (OUT / 'inference_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'passed', 'frames': frames}, indent=2))


if __name__ == '__main__':
    main()
