import unittest
import numpy as np
from pymicro.crystal.lattice import Lattice, HklDirection, HklPlane, SlipSystem
from pymicro.crystal.microstructure import Orientation
from pymicro.xray.laue import select_lambda

class LaueTests(unittest.TestCase):

  def setUp(self):
    print 'testing the Lattice class'

  def test_angle_zone(self):
    '''Verify the angle between X and a particular zone axis expressed 
    in (X, Y, Z), given a crystal orientation.'''
    ni = Lattice.from_symbol('Ni')
    # euler angles in degrees
    phi1 = 89.4
    phi = 92.0
    phi2 = 86.8
    orientation = Orientation.from_euler([phi1, phi, phi2])
    Bt = orientation.orientation_matrix().transpose()
    # zone axis
    uvw = HklDirection(1, 0, 5, ni)
    ZA = Bt.dot(uvw.direction())
    if ZA[0] < 0:
      ZA *= -1 # make sur the ZA vector is going forward
    psi0 = np.arccos(np.dot(ZA, np.array([1., 0., 0.])))
    self.assertAlmostEqual(psi0*180/np.pi, 9.2922, 3)

  def test_select_lambda(self):
    '''Verify the wavelength diffracted by a given hkl plane.'''
    ni = Lattice.from_symbol('Ni')
    orientation = Orientation.cube()
    hkl = HklPlane(-1, -1, -1, ni)
    (the_lambda, theta) = select_lambda(hkl, orientation)
    self.assertAlmostEqual(the_lambda, 5.277, 3)
    self.assertAlmostEqual(theta*180/np.pi, 35.264, 3)

if __name__ == '__main__':
  unittest.main()
