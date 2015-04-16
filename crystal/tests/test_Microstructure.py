import unittest
import numpy as np
from pymicro.crystal.microstructure import Orientation
from pymicro.crystal.lattice import HklPlane, HklDirection, SlipSystem

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

if __name__ == '__main__':
  unittest.main()
