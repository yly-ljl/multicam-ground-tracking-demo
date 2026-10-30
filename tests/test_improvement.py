import unittest
from pathlib import Path
import numpy as np
from tracking_demo.geometry import GroundCamera
from tracking_demo.improvement import ground_jacobian,weight_group

class WeightTests(unittest.TestCase):
 def test_analytic_jacobian_matches_finite_difference(self):
  root=Path(__file__).resolve().parents[1]/'data/Wildtrack_dataset'
  for name in ('CVLab1','CVLab2','CVLab3'):
   cam=GroundCamera(root,name)
   for xy in ([0,0],[5,15],[0,24]):
    uv=cam.to_image(xy);eps=1e-3
    numerical=np.column_stack([(cam.to_ground(uv+np.eye(2)[k]*eps)[0]-cam.to_ground(uv-np.eye(2)[k]*eps)[0])/(2*eps) for k in range(2)])
    np.testing.assert_allclose(ground_jacobian(cam,uv),numerical,rtol=1e-5,atol=1e-8)
 def test_lower_sensitivity_has_more_weight_and_singleton_is_unchanged(self):
  class Camera:
   def __init__(self,scale):self.H_inv=np.diag([scale,scale,1.])
  cams={'C1':Camera(1),'C2':Camera(2)}
  members=[{'camera':c,'foot_u':0,'foot_v':0,'x_m':x,'y_m':0} for c,x in [('C1',0),('C2',.5)]]
  group={'members':members,'x_m':.25,'y_m':0}
  weights,_=weight_group(group,cams,True)
  np.testing.assert_allclose(weights,[.8,.2]);self.assertAlmostEqual(group['x_m'],.1)
  self.assertEqual(group['members'],members)
  single={'members':members[:1],'x_m':0.,'y_m':0.}
  self.assertEqual(weight_group(single,cams,True)[0],[1.]);self.assertEqual(single['x_m'],0.)
 def test_disabled_keeps_arithmetic_point_and_equal_weights(self):
  class Camera:H_inv=np.eye(3)
  group={'members':[{'camera':'C1','foot_u':0,'foot_v':0,'x_m':0,'y_m':0},{'camera':'C2','foot_u':1,'foot_v':0,'x_m':.5,'y_m':0}], 'x_m':.25,'y_m':0}
  self.assertEqual(weight_group(group,{'C1':Camera(),'C2':Camera()},False)[0],[.5,.5])
  self.assertEqual(group['x_m'],.25)
