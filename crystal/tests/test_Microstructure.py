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
        self.assertAlmostEqual(180 / np.pi * o1.disorientation(o2, crystal_structure='none')[0], 60)
        self.assertAlmostEqual(180 / np.pi * o1.disorientation(o2, crystal_structure='cubic')[0], 30)

    def test_misorientation_axis(self):
        o1 = Orientation.copper()
        o2 = Orientation.s3()
        (angle, axis, axis_xyz) = o1.disorientation(o2, crystal_structure='none')
        self.assertAlmostEqual(180 / np.pi * angle, 19.38, 2)  # check value of 19.576
        val = np.array([-0.71518544, -0.60383062, -0.35199199])
        for i in range(3):
            self.assertAlmostEqual(axis[i], val[i], 6)

    def test_Bragg_condition(self):
        al = Lattice.from_symbol('Al')
        p = HklPlane(0, 0, 2, lattice=al)
        lambda_keV = 42
        lambda_nm = lambda_keV_to_nm(lambda_keV)
        rod = [0.1449, -0.0281, 0.0616]
        o = Orientation.from_rodrigues(rod)
        (w1, w2) = o.dct_omega_angles(p, lambda_keV, verbose=False)
        # test the two solution of the rotating crystal
        for omega in (w1, w2):
            alpha = o.compute_XG_angle(p, omega, verbose=True)
            theta_bragg = p.bragg_angle(lambda_keV)
            self.assertAlmostEqual(alpha, 180 / np.pi * (np.pi / 2 - theta_bragg))

    def test_dct_omega_angles(self):
        lambda_keV = 30
        lambda_nm = 1.2398 / lambda_keV
        a = 0.3306  # lattice parameter in nm
        Ti_bcc = Lattice.cubic(a)
        (h, k, l) = (0, 1, 1)
        hkl = HklPlane(h, k, l, lattice=Ti_bcc)
        o = Orientation.from_euler((103.517, 42.911, 266.452))
        theta = hkl.bragg_angle(lambda_keV, verbose=False)

        gt = o.orientation_matrix()  # our B (here called gt) corresponds to g^{-1} in Poulsen 2004
        A = h * gt[0, 0] + k * gt[1, 0] + l * gt[2, 0]
        B = -h * gt[0, 1] - k * gt[1, 1] - l * gt[2, 1]
        C = -2 * a * np.sin(theta) ** 2 / lambda_nm  # the minus sign comes from the main equation
        Delta = 4 * (A ** 2 + B ** 2 - C ** 2)
        self.assertEqual(Delta > 0, True)
        # print 'A=',A
        # print 'B=',B
        # print 'C=',C
        # print 'Delta=',Delta
        t1 = (B - 0.5 * np.sqrt(Delta)) / (A + C)
        t2 = (B + 0.5 * np.sqrt(Delta)) / (A + C)
        # print 'verifying Acos(w)+Bsin(w)=C:'
        for t in (t1, t2):
            x = A * (1 - t ** 2) / (1 + t ** 2) + B * 2 * t / (1 + t ** 2)
            self.assertAlmostEqual(x, C, 2)
        # print 'verifying (A+C)*t**2-2*B*t+(C-A)=0'
        for t in (t1, t2):
            self.assertAlmostEqual((A + C) * t ** 2 - 2 * B * t + (C - A), 0.0, 2)
        (w1, w2) = o.dct_omega_angles(hkl, lambda_keV, verbose=False)
        self.assertAlmostEqual(w1, 196.709, 2)
        self.assertAlmostEqual(w2, 28.334, 2)

    def test_topotomo_tilts(self):
        al = Lattice.from_symbol('Al')
        p = HklPlane(0, 0, 2, lattice=al)
        rod = [0.1449, -0.0281, 0.0616]
        o = Orientation.from_rodrigues(rod)
        (ut, lt) = o.topotomo_tilts(p, verbose=True)
        self.assertAlmostEqual(180 / np.pi * ut, 2.236, 3)
        self.assertAlmostEqual(180 / np.pi * lt, -16.615, 3)

    def test_IPF_color(self):
        o1 = Orientation.cube()  # 001 // Z
        o2 = Orientation.from_euler([35.264, 45., 0.])  # 011 // Z
        o3 = Orientation.from_euler([0., 54.736, 45.])  # 111 // Z
        orientations = [o1, o2, o3]
        targets = [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])]
        for case in range(2):
            o = orientations[case]
            print(o)
            target = targets[case]
            col = o.get_ipf_colour()
            print(col)
            for i in range(3):
                self.assertAlmostEqual(col[i], target[i])


if __name__ == '__main__':
    unittest.main()
