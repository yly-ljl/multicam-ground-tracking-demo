"""Core risks only: metric geometry, physical time/lifecycle, IDs and gated matching."""
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from tracking_demo.geometry import GroundCamera, grid_to_world
from tracking_demo.tracking import GroundTracker
from tracking_demo.analysis import gated_pairs

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'configs/c1_baseline.json').read_text(encoding='utf-8'))


def obs(x, y=0., name='a'):
    return {'x_m': x, 'y_m': y, 'obs_id': name}


class CoreTests(unittest.TestCase):
    def test_gate_inside_assignment(self):
        # Unconstrained minimum cost followed by rejection would leave only one pair.
        pairs = gated_pairs([[0., .9], [.1, 0.]], [[0., 0.], [1., 0.]], 1.)
        self.assertEqual(len(pairs), 2)
        self.assertEqual({(a, b) for a, b, _ in pairs}, {(0, 0), (1, 1)})

    def test_metric_projection_and_opencv(self):
        camera = GroundCamera(ROOT / CONFIG['data_root'])
        np.testing.assert_allclose(camera.t * 100, camera.t_cm)
        np.testing.assert_allclose(grid_to_world(0), [-3, -9])
        np.testing.assert_allclose(grid_to_world(480), [-3, -8.975])
        points = np.array([[0., 0.], [3., 5.], [1., 12.]])
        uv_cv = cv2.projectPoints(np.c_[points, np.zeros(3)], camera.rvec, camera.t,
                                 camera.K, np.zeros(5))[0].reshape(-1, 2)
        for xy, uv in zip(points, uv_cv):
            np.testing.assert_allclose(camera.to_image(xy), uv, atol=1e-8)
            back, _ = camera.to_ground(uv)
            np.testing.assert_allclose(back, xy, atol=1e-9)
        self.assertFalse(camera.in_roi([9., 0.]))
        self.assertFalse(camera.in_roi([0., 27.]))
        outside, reason = camera.to_ground(camera.to_image([10., 10.]))
        self.assertEqual(reason, 'outside_roi')
        np.testing.assert_allclose(outside, [10., 10.])

    def test_half_second_velocity_and_one_second_gap(self):
        tracker = GroundTracker(CONFIG['tracker'])
        for tick in range(24):
            states, _ = tracker.update(tick * .5, [obs(tick * .5)])
        self.assertAlmostEqual(states[0]['vx_mps'], 1., delta=.02)
        previous = states[0]
        # Missing t=12.0: the next call at 12.5 must advance two physical predictions.
        states, events = tracker.update(12.5, [])
        self.assertEqual([event['time_s'] for event in events], [12., 12.5])
        self.assertTrue(events[0]['inserted_missing_tick'])
        self.assertAlmostEqual(states[0]['x_m'] - previous['x_m'], previous['vx_mps'], places=7)
        self.assertEqual(states[0]['since_observation_s'], 1.)
        self.assertEqual(states[0]['state'], 'predicted')
        same_id = states[0]['track_id']
        states, _ = tracker.update(13., [obs(13., name='return')])
        self.assertEqual(states[0]['track_id'], same_id)

    def test_empty_lifecycle_and_unique_ids(self):
        tracker = GroundTracker(CONFIG['tracker'])
        for tick in range(8):
            states, _ = tracker.update(tick * .5, [obs(0., name='a'), obs(4., name='b')])
            ids = [row['track_id'] for row in states]
            self.assertEqual(len(set(ids)), 2)
            self.assertEqual({row['obs_id'] for row in states}, {'a', 'b'})
        for tick in range(8, 12):
            states, _ = tracker.update(tick * .5, [])
            self.assertEqual(len(states), 2)
        states, _ = tracker.update(6., [])
        self.assertEqual(states, [])


if __name__ == '__main__':
    unittest.main()
