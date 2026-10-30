import json
from pathlib import Path
import unittest
import numpy as np
from norfair import Tracker
from tracking_demo.association import gated_global_pairs,GatedGlobalTracker
from tracking_demo.tracking import GroundTracker


class AssociationTests(unittest.TestCase):
    def test_empty_and_no_valid_edges(self):
        for shape in ((0,0),(0,3),(4,0)):
            self.assertEqual(gated_global_pairs(np.empty(shape),1.5),([],[]))
        self.assertEqual(gated_global_pairs([[np.nan,np.inf],[1.5,2]],1.5),([],[]))

    def test_competition_maximizes_cardinality_before_distance(self):
        self.assertEqual(gated_global_pairs([[.9,1.1],[1.,3.]],1.5),([0,1],[1,0]))

    def test_same_cardinality_minimizes_distance(self):
        self.assertEqual(gated_global_pairs([[.1,.2],[.2,.9]],1.5),([0,1],[1,0]))

    def test_rectangular_unmatched_and_uniqueness(self):
        rows,cols=gated_global_pairs([[.2,5],[.3,4],[9,.4]],1.5)
        self.assertEqual((rows,cols),([0,2],[0,1]))
        self.assertEqual(len(set(rows)),len(rows));self.assertEqual(len(set(cols)),len(cols))

    def test_forbidden_edges_are_in_assignment_not_postfiltered(self):
        # The invalid full matching must not steal the .1m valid pair.
        self.assertEqual(gated_global_pairs([[.1,1.6],[1.4,100]],1.5),([0],[0]))

    def test_strict_boundary_and_input_not_mutated(self):
        a=np.array([[1.5,np.inf],[np.nan,1.499999]])
        before=a.copy();self.assertEqual(gated_global_pairs(a,1.5),([1],[1]))
        np.testing.assert_array_equal(a,before)

    def test_only_matching_is_overridden(self):
        methods={k for k,v in GatedGlobalTracker.__dict__.items() if callable(v)}
        self.assertEqual(methods,{'match_dets_and_objs'})

    def test_default_explicit_greedy_and_noncompeting_lifecycle(self):
        c=json.loads((Path(__file__).resolve().parents[1]/'configs/c1_baseline.json').read_text())['tracker']
        default=GroundTracker(c);greedy=GroundTracker({**c,'association_method':'greedy'});global_=GroundTracker({**c,'association_method':'gated_global'})
        self.assertIs(type(default.tracker),Tracker);self.assertIs(type(greedy.tracker),Tracker)
        self.assertIs(type(global_.tracker),GatedGlobalTracker)
        for time,obs in [(0,[{'obs_id':'a','x_m':0,'y_m':0}]),(.5,[{'obs_id':'b','x_m':.5,'y_m':0}]),(1.5,[]),(2.,[])]:
            a=default.update(time,obs);self.assertEqual(a,greedy.update(time,obs));self.assertEqual(a,global_.update(time,obs))
        with self.assertRaises(ValueError):GroundTracker({**c,'association_method':'unknown'})
