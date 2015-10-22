import unittest
import numpy as np
from pymicro.crystal.lattice import Lattice, HklPlane, SlipSystem

class LatticeTests(unittest.TestCase):

  def setUp(self):
    print 'testing the Lattice class'

  def test_cubic(self):
    a = np.array([[ 0.5,  0. ,  0. ],
                  [ 0. ,  0.5,  0. ],
                  [ 0. ,  0. ,  0.5]])
    l = Lattice.cubic(0.5)
    for i in range(0, 3):
      for j in range(0, 3):
        self.assertEqual(l.matrix[i][j], a[i][j])

  def test_volume(self):
    l = Lattice.cubic(0.5)
    self.assertAlmostEqual(l.volume(), 0.125)
  
  def test_from_symbol(self):
    al = Lattice.from_symbol('Al')
    for i in range(0, 3):
      self.assertAlmostEqual(al._lengths[i], 0.40495)
      self.assertEqual(al._angles[i], 90.0)

class HklPlaneTests(unittest.TestCase):

  def setUp(self):
    print 'testing the HklPlane class'

  def test_HklPlane(self):
    p = HklPlane(1,1,1)
    n = p.normal()
    self.assertEqual(np.linalg.norm(n), 1)
  
  def test_bragg_angle(self):
    l = Lattice.cubic(0.287) # FCC iron
    hkl = HklPlane(2, 0, 0, l) # 200 reflection at 8 keV is at 32.7 deg
    self.assertAlmostEqual(hkl.bragg_angle(8), 0.5704164) 

class SlipSystemTests(unittest.TestCase):

  def setUp(self):
    print 'testing the SlipSystem class'

  def test_get_slip_system(self):
    ss = SlipSystem.get_slip_systems('111')
    self.assertEqual(len(ss), 12)
    for s in ss:
      n = s.get_slip_plane().normal()
      l = s.get_slip_direction().direction()
      self.assertEqual(np.dot(n, l), 0.)
    ss = SlipSystem.get_slip_systems('112')
    self.assertEqual(len(ss), 12)
    for s in ss:
      n = s.get_slip_plane().normal()
      l = s.get_slip_direction().direction()
      self.assertEqual(np.dot(n, l), 0.)
    
if __name__ == '__main__':
  unittest.main()
