import unittest
import numpy as np
from pymicro.crystal.microstructure import Orientation, Grain
from pymicro.crystal.lattice import Lattice, HklPlane, HklDirection, SlipSystem

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
    oct_ss = SlipSystem.get_octaedral_slip_systems()
    self.assertAlmostEqual(max(o.compute_all_schmid_factors(oct_ss, verbose=True)), 0.4082, 4)

class OrientationTests(unittest.TestCase):

  def setUp(self):
    print 'testing the Grain class'

  def test_Orientation(self):
    lambda_keV = 30
    a = 0.3306 # lattice parameter in nm
    Ti_bcc = Lattice.cubic(a)
    p = HklPlane(0, 1, 1, lattice = Ti_bcc)
    g = Grain(4, Orientation.from_euler((103.517, 42.911, 266.452)))
    (w1, w2) = g.dct_omega_angles(p, lambda_keV, verbose=False)
    self.assertAlmostEqual(w1, -151.66)
    self.assertAlmostEqual(w2, 16.71)

if __name__ == '__main__':
  unittest.main()
