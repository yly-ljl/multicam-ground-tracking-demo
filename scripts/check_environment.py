"""One-time dependency smoke check; synthetic XY observations are not a demo."""

import importlib.metadata
import json
import os
from pathlib import Path
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/setup_check'
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.cache/ultralytics')
os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache/matplotlib')
os.environ['TORCH_HOME'] = str(ROOT / '.cache/torch')

import cv2
import numpy as np
import torch
import torchvision
from norfair import Detection, Tracker
from ultralytics import settings


def main():
    settings.update({'sync': False})
    versions = dict(sorted(
        (d.metadata['Name'], d.version)
        for d in importlib.metadata.distributions()
    ))
    expected = {
        'torch': '2.4.1+cu124', 'torchvision': '0.19.1+cu124',
        'numpy': '1.26.4', 'opencv-python': '4.11.0.86',
        'ultralytics': '8.3.253', 'norfair': '2.3.0',
        'scipy': '1.13.1', 'matplotlib': '3.9.4', 'filterpy': '1.4.5',
        'ultralytics-thop': '2.0.18', 'setuptools': '80.9.0', 'wheel': '0.45.1',
    }
    for name, version in expected.items():
        assert importlib.metadata.version(name) == version, (name, version)
    assert Path(sys.prefix).resolve() == (ROOT / '.venv').resolve()
    assert torch.cuda.is_available(), 'Project environment cannot use CUDA'
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    device_tensor = torch.from_numpy(array).to('cuda:0')
    result_array = (device_tensor * 2 + 1).cpu().numpy()
    np.testing.assert_array_equal(result_array, array * 2 + 1)
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.], [20., 20., 30., 30.]])
    scores = torch.tensor([.9, .8, .7])
    nms = {}
    for device in ('cpu', 'cuda:0'):
        kept = torchvision.ops.nms(boxes.to(device), scores.to(device), .5).cpu().tolist()
        assert kept == [0, 2], (device, kept)
        nms[device] = kept
    tracker = Tracker(distance_function='euclidean', distance_threshold=1.0,
                      initialization_delay=0, hit_counter_max=5)
    observations = []
    for step in range(3):
        xy = np.array([[step * .1, step * .05]], dtype=np.float32)
        tracks = tracker.update(detections=[Detection(points=xy)])
        assert len(tracks) == 1
        observations.append({'step': step, 'input_xy': xy.tolist(),
                             'id': tracks[0].id, 'estimate_xy': tracks[0].estimate.tolist()})
    assert len({item['id'] for item in observations}) == 1
    report = {
        'status': 'passed', 'recorded_at': datetime.now().astimezone().isoformat(),
        'executable': sys.executable, 'python': sys.version,
        'versions': versions, 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory,
        'numpy_cuda_roundtrip': result_array.tolist(), 'nms_indices': nms,
        'norfair_synthetic_xy_updates': observations, 'opencv_version': cv2.__version__,
        'scope': 'Dependency smoke test only; no real ground tracking or accuracy evaluation.',
    }
    (OUT / 'environment_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = [
        '# Actual installed versions from the phase-1 environment smoke check.',
        '# Python 3.12.6 / Windows x64. Bootstrap setuptools/wheel before --no-build-isolation.',
        '--index-url https://pypi.org/simple',
    ]
    for name, version in versions.items():
        if name.lower() in ('torch', 'torchvision'):
            wheel_version = version.replace('+', '%2B')
            lines.append(f'{name} @ https://download.pytorch.org/whl/cu124/{name}-{wheel_version}-cp312-cp312-win_amd64.whl')
        else:
            lines.append(f'{name}=={version}')
    (ROOT / 'requirements.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
