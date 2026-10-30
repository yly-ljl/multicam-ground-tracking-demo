import unittest
from tracking_demo.fusion import fuse_observations
from tracking_demo.tracking import GroundTracker
from tracking_demo.render import MultiRenderer
import json
from pathlib import Path


def point(camera, key, x, y=0):
    return {'camera':camera, 'obs_id':key, 'x_m':x, 'y_m':y}


class FusionTests(unittest.TestCase):
    def test_history_uses_physical_time_without_drawing(self):
        renderer=MultiRenderer({},[-3,9,-9,27],5)
        state={'track_id':1,'x_m':0,'y_m':0}
        renderer.update(0,[state])
        renderer.update(10,[state])
        self.assertEqual(len(renderer.history[1]),1)
        for tick in range(21,41): renderer.update(tick/2,[state])
        self.assertEqual(renderer.history[1][0][0],15)
        self.assertEqual(renderer.history[1][-1][0],20)
        renderer.update(30,[])
        self.assertEqual(len(renderer.history[1]),0)

    def check_invariants(self, groups, total):
        ids = [m['obs_id'] for g in groups for m in g['members']]
        self.assertEqual(len(ids), total)
        self.assertEqual(len(set(ids)), total)
        for group in groups:
            cameras = [m['camera'] for m in group['members']]
            self.assertEqual(len(cameras), len(set(cameras)))
            self.assertLessEqual(group['max_pair_distance_m'], .75)

    def test_same_camera_exclusion_and_single_use(self):
        rows={'C1':[point('C1','a',0), point('C1','b',.1)], 'C2':[point('C2','c',.05)]}
        groups=fuse_observations(rows,['C1','C2'],.75,0)
        self.check_invariants(groups,3)
        self.assertEqual(sorted(len(g['members']) for g in groups),[1,2])

    def test_complete_link_blocks_transitive_chain_and_gate_boundary(self):
        rows={'C1':[point('C1','a',0)],'C2':[point('C2','b',.75)],'C3':[point('C3','c',1.0)]}
        groups=fuse_observations(rows,['C1','C2','C3'],.75,0)
        self.check_invariants(groups,3)
        self.assertEqual(sorted(len(g['members']) for g in groups),[1,2])
        self.assertEqual([m['obs_id'] for m in groups[0]['members']],['a','b'])

    def test_ambiguous_one_to_one_and_arithmetic_mean(self):
        rows={'C1':[point('C1','a',0),point('C1','b',1)],
              'C2':[point('C2','c',.6),point('C2','d',1.4)]}
        groups=fuse_observations(rows,['C1','C2'],.75,0)
        self.check_invariants(groups,4)
        self.assertEqual([len(g['members']) for g in groups],[2,2])
        self.assertAlmostEqual(groups[0]['x_m'],.3)
        self.assertAlmostEqual(groups[1]['x_m'],1.2)

    def test_empty_and_missing_views(self):
        self.assertEqual(fuse_observations({},['C1','C2','C3'],.75,0),[])
        groups=fuse_observations({'C2':[point('C2','a',2)]},['C1','C2','C3'],.75,0)
        self.check_invariants(groups,1)
        self.assertEqual(groups[0]['x_m'],2)

    def test_missing_view_does_not_stop_global_update(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'configs/c1_baseline.json').read_text())
        tracker=GroundTracker(config['tracker'])
        previous=None
        for frame,views in [(0,{'C1':[point('C1','a',0)],'C2':[point('C2','b',.1)]}),
                            (5,{'C1':[point('C1','c',.2)]})]:
            groups=fuse_observations(views,['C1','C2'],.75,frame)
            states,_=tracker.update(frame/10,[{'obs_id':g['fusion_id'],'x_m':g['x_m'],'y_m':g['y_m']} for g in groups])
            self.assertEqual(states[0]['state'],'observed')
            if previous is not None: self.assertEqual(states[0]['track_id'],previous)
            previous=states[0]['track_id']
