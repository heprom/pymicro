import unittest
import numpy as np
from pymicro.crystal.lattice import Lattice

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
    
if __name__ == '__main__':
  unittest.main()
