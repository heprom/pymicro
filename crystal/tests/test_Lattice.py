import unittest
import numpy as np
from crystal.lattice import Lattice, HklPlane

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
  
  def test_HklPlane(self):
    p = HklPlane(1,1,1)
    n = p.normal
    self.assertEqual(np.linalg.norm(n), 1)
    
if __name__ == '__main__':
  unittest.main()
