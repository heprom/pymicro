import unittest
import numpy as np
from pymicro.crystal.lattice import Lattice, HklDirection, HklPlane, SlipSystem
from pymicro.crystal.microstructure import Orientation
from pymicro.xray.laue import select_lambda, index


class LaueTests(unittest.TestCase):
    def setUp(self):
        """testing the laue module:"""
        self.ni = Lattice.from_symbol('Ni')

    def test_angle_zone(self):
        """Verify the angle between X and a particular zone axis expressed
        in (X, Y, Z), given a crystal orientation."""
        # euler angles in degrees
        phi1 = 89.4
        phi = 92.0
        phi2 = 86.8
        orientation = Orientation.from_euler([phi1, phi, phi2])
        Bt = orientation.orientation_matrix().transpose()
        # zone axis
        uvw = HklDirection(1, 0, 5, self.ni)
        ZA = Bt.dot(uvw.direction())
        if ZA[0] < 0:
            ZA *= -1  # make sur the ZA vector is going forward
        psi0 = np.arccos(np.dot(ZA, np.array([1., 0., 0.])))
        self.assertAlmostEqual(psi0 * 180 / np.pi, 9.2922, 3)

    def test_select_lambda(self):
        """Verify the wavelength diffracted by a given hkl plane."""
        orientation = Orientation.cube()
        hkl = HklPlane(-1, -1, -1, self.ni)
        (the_lambda, theta) = select_lambda(hkl, orientation)
        self.assertAlmostEqual(the_lambda, 5.277, 3)
        self.assertAlmostEqual(theta * 180 / np.pi, 35.264, 3)

    def test_indexation(self):
        """Verify indexing solution from a known Laue pattern."""
        euler_angles = (191.9, 69.9, 138.9)  # degrees, /!\ not in fz
        orientation = Orientation.from_euler(euler_angles)
        # list of plane normals, obtained from the detector image
        hkl_normals = np.array([[0.11066932863248755, 0.8110118739480003, 0.5744667440465002],
                                [0.10259261224575777, 0.36808036454584847, -0.9241166599236196],
                                [0.12497400210731163, 0.38160000643453934, 0.9158400154428944],
                                [0.21941448008210823, 0.5527234994434788, -0.8039614537359691],
                                [0.10188581412204267, -0.17110594738052967, -0.9799704259066699],
                                [0.10832511255237177, -0.19018912890874434, 0.975752922227471],
                                [0.13621754927492466, -0.8942526135605741, 0.4263297343719016],
                                [0.04704092862601945, -0.45245473334950004, -0.8905458243704446]])
        miller_indices = [(3, -5, 0),
                          (5, 4, -2),
                          (2, -5, -1),
                          (3, -4, -5),
                          (2, -2, 3),
                          (-3, 4, -3),
                          (3, -4, 3),
                          (3, -2, 3),
                          (-5, 5, -1),
                          (5, -5, 1)]
        hkl_planes = []
        for indices in miller_indices:
            (h, k, l) = indices
            hkl_planes.append(HklPlane(h, k, l, self.ni))
        solutions = index(hkl_normals, hkl_planes, tol_angle=0.5, tol_disorientation=3.0)
        self.assertEqual(len(solutions), 1)
        final_orientation = Orientation(solutions[0])
        angle, ax1, ax2 = final_orientation.disorientation(orientation, crystal_structure='cubic')
        self.assertLess(angle * 180 / np.pi, 1.0)


if __name__ == '__main__':
    unittest.main()
