"""Hand-counted IDTP/IDFP/IDFN checks, independent of real data scores."""
import math
from pathlib import Path
import unittest
import motmetrics as mm
import numpy as np
from tracking_demo.identity_evaluation import accumulate,distance_matrix,metrics_for,load_gt,in_roi


def points(*pairs):return [{'id':i,'x_m':x,'y_m':0.} for i,x in pairs]


class IdentityTests(unittest.TestCase):
    def counts(self,frames,tp,fp,fn):
        result=metrics_for(accumulate(frames))
        self.assertEqual((result['idtp'],result['idfp'],result['idfn']),(tp,fp,fn))
        if 2*tp+fp+fn:self.assertAlmostEqual(result['idf1'],2*tp/(2*tp+fp+fn))
        return result

    def test_perfect_and_global_id_rename(self):
        g=points((1,0),(2,5))
        for ids in ((10,20),(104,903)):
            self.counts([(f,g,points((ids[0],0),(ids[1],5))) for f in (0,5,10)],6,0,0)

    def test_midway_id_split_and_two_people_swap(self):
        r=self.counts([(f,points((1,0)),points((10 if f<10 else 20,0))) for f in (0,5,10,15)],2,2,2)
        self.assertEqual(r['num_switches'],1)
        r=self.counts([(0,points((1,0),(2,5)),points((10,0),(20,5))),
                       (5,points((1,0),(2,5)),points((10,5),(20,0)))],2,2,2)
        self.assertEqual(r['num_switches'],2)

    def test_missed_and_extra_predictions(self):
        self.counts([(0,points((1,0)),points((10,0),(99,5))),(5,points((1,0)),[])],1,1,1)

    def test_empty_gt_prediction_and_wholly_empty_frame(self):
        self.counts([(0,[],points((10,0)))],0,1,0)
        self.counts([(0,points((1,0)),[])],0,0,1)
        r=self.counts([(0,[],[])],0,0,0);self.assertEqual(r['num_frames'],1)

    def test_inclusive_one_meter_uses_euclidean_not_squared(self):
        d=distance_matrix(points((1,0)),points((10,.5),(20,1.),(30,1.000001)))
        np.testing.assert_allclose(d[0,:2],[.5,1.]);self.assertTrue(np.isnan(d[0,2]))
        self.counts([(0,points((1,0)),points((10,1.)))],1,0,0)

    def test_empty_tick_and_segment_isolation(self):
        frames=[(0,points((1,0)),points((10,0))),(5,points((1,0)),[]),(10,points((1,0)),points((10,0)))]
        r=self.counts(frames,2,0,1);self.assertEqual(r['num_fragmentations'],1);self.assertEqual(r['num_frames'],3)
        # Independent segments may reuse an ID for another person without inheritance.
        self.counts([(800,points((2,5)),points((10,5)))],1,0,0)
        with self.assertRaises(ValueError):accumulate([frames[0],frames[0]])

    def test_full_relations_differ_from_spatial_winners(self):
        frames=[(0,points((1,0)),points((10,0),(20,.5))),
                (5,points((1,0)),points((10,.6),(20,.1)))]
        self.counts(frames,2,2,0)
        wrong=mm.MOTAccumulator(auto_id=False,max_switch_time=np.inf)
        wrong.update([1],[10,20],[[0,np.nan]],frameid=0)
        wrong.update([1],[10,20],[[np.nan,.1]],frameid=5)
        r=metrics_for(wrong);self.assertEqual((r['idtp'],r['idfp'],r['idfn']),(1,3,1))

    def test_missing_annotation_is_error_and_roi_half_open(self):
        missing=Path(__file__).resolve().parent/'fixtures'/'missing_annotation_data'
        with self.assertRaises(FileNotFoundError):load_gt(missing,0,[-3,9,-9,27])
        self.assertTrue(in_roi([-3,-9],[-3,9,-9,27]));self.assertFalse(in_roi([9,0],[-3,9,-9,27]))
        with self.assertRaises(ValueError):accumulate([(0,points((1,0),(1,1)),[])])
