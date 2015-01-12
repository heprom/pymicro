import unittest
import numpy as np
from crystal.lattice import Lattice, HklPlane

class LatticeTests(unittest.TestCase):

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
    
if __name__ == '__main__':
  unittest.main()
