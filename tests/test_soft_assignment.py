import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from norfair import Tracker
from tracking_demo.association import soft_global_pairs,SoftAssignmentTracker,GatedGlobalTracker
from tracking_demo.tracking import GroundTracker


def objective(d,rows,cols,tau=1.5):return sum(d[i,j]/tau for i,j in zip(rows,cols))+len(d)-len(rows)


class SoftAssignmentTests(unittest.TestCase):
    def test_empty_and_forbidden_including_negative(self):
        for shape in ((0,0),(0,3),(4,0)):self.assertEqual(soft_global_pairs(np.empty(shape),1.5),([],[]))
        self.assertEqual(soft_global_pairs([[-.1,np.nan],[np.inf,1.5]],1.5),([],[]))

    def test_strict_boundary_and_isolated_valid_edge(self):
        for d in (0.,1.499999):self.assertEqual(soft_global_pairs([[d]],1.5),([0],[0]))
        self.assertEqual(soft_global_pairs([[1.5]],1.5),([],[]))

    def test_known_255_two_pairs_preferred(self):
        d=np.array([[.5830567479,.7252393961],[.8809894919,1.5987279415]])
        rows,cols=soft_global_pairs(d,1.5);self.assertEqual((rows,cols),([0,1],[1,0]))
        self.assertAlmostEqual(objective(d,rows,cols),1.606228888/1.5)

    def test_known_840_nearest_and_unmatched_preferred(self):
        d=np.array([[.1228487566,1.0145442486],[1.3018926382,1.9825963974]])
        rows,cols=soft_global_pairs(d,1.5);self.assertEqual((rows,cols),([0],[0]))
        self.assertAlmostEqual(objective(d,rows,cols),1+.1228487566/1.5)

    def test_rectangular_and_input_preserved(self):
        d=np.array([[.1,np.inf],[.8,-1],[np.nan,.2]]);before=d.copy();rows,cols=soft_global_pairs(d,1.5)
        self.assertEqual((rows,cols),([0,2],[0,1]));np.testing.assert_array_equal(d,before)
        self.assertEqual(len(set(rows)),len(rows));self.assertEqual(len(set(cols)),len(cols))

    def test_equal_cost_does_not_require_cardinality_tiebreak(self):
        d=np.array([[0,.75],[.75,2.]])
        rows,cols=soft_global_pairs(d,1.5)
        self.assertIn(len(rows),(1,2));self.assertAlmostEqual(objective(d,rows,cols),1.)
        self.assertTrue(all(0<=d[i,j]<1.5 for i,j in zip(rows,cols)))

    def test_only_matching_override_and_explicit_method_selection(self):
        self.assertEqual({k for k,v in SoftAssignmentTracker.__dict__.items() if callable(v)},{'match_dets_and_objs'})
        c=json.loads((Path(__file__).resolve().parents[1]/'configs/c1_baseline.json').read_text())['tracker']
        self.assertIs(type(GroundTracker(c).tracker),Tracker)
        self.assertIs(type(GroundTracker({**c,'association_method':'greedy'}).tracker),Tracker)
        self.assertIs(type(GroundTracker({**c,'association_method':'gated_global'}).tracker),GatedGlobalTracker)
        self.assertIs(type(GroundTracker({**c,'association_method':'soft_global'}).tracker),SoftAssignmentTracker)

    def test_uninherited_observation_still_starts_new_track(self):
        c=json.loads((Path(__file__).resolve().parents[1]/'configs/c1_baseline.json').read_text())['tracker']
        tracker=GroundTracker({**c,'association_method':'soft_global'})
        tracker.update(0,[{'obs_id':'old_a','x_m':0.,'y_m':0.},{'obs_id':'old_b','x_m':1.2,'y_m':0.}])
        states,_=tracker.update(.5,[{'obs_id':'new_a','x_m':.1,'y_m':0.},{'obs_id':'new_b','x_m':-1.2,'y_m':0.}])
        observed={s['obs_id']:s['track_id'] for s in states if s['state']=='observed'}
        self.assertEqual(observed,{'new_a':1,'new_b':3})
        old=next(o for o in tracker.tracker.tracked_objects if o.id==2)
        self.assertEqual(old.last_detection.data['obs_id'],'old_b');self.assertEqual(old.hit_counter,0)
        tracker.update(1.,[]);self.assertNotIn(2,[s['track_id'] for s in tracker.update(1.5,[])[0]])

    def test_explicit_spatial_source_does_not_follow_first_track(self):
        from tracking_demo.identity_evaluation import collect_spatial_tables
        root=Path(__file__).resolve().parents[1]
        config={'output':'outputs/unused_test','segments':{'s':{'track_sources':{'B3_SOFT':'outputs/old/track/tracks.csv'},'spatial_source':'outputs/new/comparison.csv'}}}
        row={'variant':'B3_SOFT','prediction_count':'1','gt_count':'2','matched':'1'}
        with patch('tracking_demo.identity_evaluation.read_csv',return_value=[row]) as read,patch('tracking_demo.identity_evaluation.save_csv'):
            rows=collect_spatial_tables(root,config)
        read.assert_called_once_with(root/'outputs/new/comparison.csv');self.assertEqual(rows[0]['spatial_recall'],.5)
