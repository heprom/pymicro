import unittest
import numpy as np
from pymicro.crystal.microstructure import Orientation, Grain
from pymicro.crystal.lattice import Lattice, HklPlane, HklDirection, SlipSystem
from pymicro.xray.xray_utils import lambda_keV_to_nm

class OrientationTests(unittest.TestCase):

  def setUp(self):
    print 'testing the Orientation class'

  def test_Orientation(self):
    o = Orientation.from_euler([45, 45, 0])
    self.assertAlmostEqual(o.phi1(), 45.)
    self.assertAlmostEqual(o.Phi(), 45.)
  
  def test_SchimdFactor(self):
    o = Orientation.from_euler([0., 0., 0.])
    ss = SlipSystem(HklPlane(1, 1, 1), HklDirection(0, 1, -1))
    self.assertAlmostEqual(o.schmid_factor(ss), 0.4082, 4)

  def test_MaxSchimdFactor(self):
    o = Orientation.from_euler([0., 0., 0.])
    oct_ss = SlipSystem.get_slip_systems(plane_type='111')
    self.assertAlmostEqual(max(o.compute_all_schmid_factors(oct_ss, verbose=True)), 0.4082, 4)

  def test_misorientation_angle(self):
    o1 = Orientation.from_euler((0., 0., 0.))
    o2 = Orientation.from_euler((60., 0., 0.))
    self.assertAlmostEqual(180/np.pi*o1.misorientation_angle(o2, crystal_structure='none'), 60)
    self.assertAlmostEqual(180/np.pi*o1.misorientation_angle(o2, crystal_structure='cubic'), 30)

  def test_misorientation_axis(self):
    o1 = Orientation.copper()
    o2 = Orientation.s3()
    self.assertAlmostEqual(180/np.pi*o1.misorientation_angle(o2, crystal_structure='none'), 19.38, 2) # check value of 19.576
    
  def test_Bragg_condition(self):
    al = Lattice.from_symbol('Al')
    p = HklPlane(0, 0, 2, lattice = al)
    lambda_keV = 42
    lambda_nm = lambda_keV_to_nm(lambda_keV)
    rod = [0.1449, -0.0281, 0.0616]
    o = Orientation.from_rodrigues(rod)
    (w1, w2) = o.dct_omega_angles(p, lambda_keV, verbose=False)
    omega = w1 + 180
    alpha = o.compute_XG_angle(p, omega, verbose=True)
    theta_bragg = p.bragg_angle(lambda_keV)
    self.assertAlmostEqual(alpha, 180/np.pi*(np.pi/2 - theta_bragg))

  def test_dct_omega_angles(self):
    lambda_keV = 30
    a = 0.3306 # lattice parameter in nm
    Ti_bcc = Lattice.cubic(a)
    p = HklPlane(0, 1, 1, lattice = Ti_bcc)
    o = Orientation.from_euler((103.517, 42.911, 266.452))
    (w1, w2) = o.dct_omega_angles(p, lambda_keV, verbose=False)
    self.assertAlmostEqual(w1, -151.665, 2)
    self.assertAlmostEqual(w2, 16.709, 2)
  
  def test_topotomo_tilts(self):
    al = Lattice.from_symbol('Al')
    p = HklPlane(0, 0, 2, lattice = al)
    rod = [0.1449, -0.0281, 0.0616]
    o = Orientation.from_rodrigues(rod)
    (ut, lt) = o.topotomo_tilts(p, verbose=True)
    self.assertAlmostEqual(180/np.pi*ut, 2.236, 3)
    self.assertAlmostEqual(180/np.pi*lt, -16.615, 3)    

if __name__ == '__main__':
  unittest.main()
