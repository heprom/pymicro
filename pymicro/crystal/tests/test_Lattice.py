import unittest
import numpy as np
from math import pi, cos, sin, acos, atan
from pymicro.crystal.lattice import Lattice, Symmetry, HklObject, HklDirection, HklPlane, SlipSystem


class LatticeTests(unittest.TestCase):
    def setUp(self):
        print('testing the Lattice class')

    def test_equality(self):
        l1 = Lattice.cubic(0.5)
        a = np.array([[0.5, 0., 0.],
                      [0., 0.5, 0.],
                      [0., 0., 0.5]])
        l2 = Lattice(a, symmetry=Symmetry.cubic)
        self.assertEqual(l1, l2)

    def test_cubic(self):
        a = np.array([[0.5, 0., 0.],
                      [0., 0.5, 0.],
                      [0., 0., 0.5]])
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
            self.assertAlmostEqual(al._lengths[i], 0.40495, 4)
            self.assertEqual(al._angles[i], 90.0)

    def test_reciprocal_lattice(self):
        Mg2Si = Lattice.from_parameters(1.534, 0.405, 0.683, 90., 106., 90., x_aligned_with_a=False)
        [astar, bstar, cstar] = Mg2Si.reciprocal_lattice()
        self.assertAlmostEqual(astar[0], 0.678, 3)
        self.assertAlmostEqual(astar[1], 0., 3)
        self.assertAlmostEqual(astar[2], 0., 3)
        self.assertAlmostEqual(bstar[0], 0., 3)
        self.assertAlmostEqual(bstar[1], 2.469, 3)
        self.assertAlmostEqual(bstar[2], 0., 3)
        self.assertAlmostEqual(cstar[0], 0.420, 3)
        self.assertAlmostEqual(cstar[1], 0., 3)
        self.assertAlmostEqual(cstar[2], 1.464, 3)


class HklDirectionTests(unittest.TestCase):
    def setUp(self):
        print('testing the HklDirection class')

    def test_angle_between_directions(self):
        d111 = HklDirection(1, 1, 1)
        d110 = HklDirection(1, 1, 0)
        d100 = HklDirection(1, 0, 0)
        dm111 = HklDirection(-1, 1, 1)
        self.assertAlmostEqual(d100.angle_with_direction(d110) * 180 / np.pi, 45.0)
        self.assertAlmostEqual(d111.angle_with_direction(d110) * 180 / np.pi, 35.26, 2)
        self.assertAlmostEqual(d111.angle_with_direction(dm111) * 180 / np.pi, 70.528, 2)

    def test_tetragonal_direction(self):
        bct = Lattice.body_centered_tetragonal(0.28, 0.40)
        d111 = HklDirection(1, 1, 1, bct)
        d110 = HklDirection(1, 1, 0, bct)
        self.assertAlmostEqual(d111.direction()[0], 0.49746834, 5)
        self.assertAlmostEqual(d111.direction()[1], 0.49746834, 5)
        self.assertAlmostEqual(d111.direction()[2], 0.71066905, 5)
        self.assertAlmostEqual(d110.direction()[0], 0.707106781, 5)
        self.assertAlmostEqual(d110.direction()[1], 0.707106781, 5)
        self.assertAlmostEqual(d110.direction()[2], 0.0, 5)

    def test_tetragonal_direction2(self):
        ZrO2 = Lattice.tetragonal(0.364, 0.527)
        d = HklDirection(1, 1, 1, ZrO2)
        target = np.array([1., 1., 1.448])
        target /= np.linalg.norm(target)
        self.assertAlmostEqual(d.direction()[0], target[0], 4)
        self.assertAlmostEqual(d.direction()[1], target[1], 4)
        self.assertAlmostEqual(d.direction()[2], target[2], 4)

    def test_angle_with_directions(self):
        (a, b, c) = (1.022, 0.596, 0.481)
        olivine = Lattice.orthorhombic(a, b, c)
        (h1, k1, l1) = (1., 1., 1.)
        (h2, k2, l2) = (3., 3., 2.)
        d1 = HklDirection(h1, k1, l1, olivine)
        d2 = HklDirection(h2, k2, l2, olivine)
        # compare with formula in orthorhombic lattice, angle must be 6.589 degrees
        angle = np.arccos(((h1 * h2 * a ** 2) + (k1 * k2 * b ** 2) + (l1 * l2 * c ** 2)) /
                          (np.sqrt(a ** 2 * h1 ** 2 + b ** 2 * k1 ** 2 + c ** 2 * l1 ** 2) *
                           np.sqrt(a ** 2 * h2 ** 2 + b ** 2 * k2 ** 2 + c ** 2 * l2 ** 2)))
        self.assertAlmostEqual(d1.angle_with_direction(d2), angle)

    def test_skip_higher_order(self):
        uvw = HklDirection(3, 3, 1)
        hkl_planes = uvw.find_planes_in_zone(max_miller=3)
        self.assertEqual(len(hkl_planes), 18)
        hkl_planes2 = HklObject.skip_higher_order(hkl_planes)
        self.assertEqual(len(hkl_planes2), 7)

    def test_4indices_representation(self):
        u, v, w = HklDirection.four_to_three_indices(2, -1, -1, 0)
        self.assertEqual(u, 1)
        self.assertEqual(v, 0)
        self.assertEqual(w, 0)
        U, V, T, W = HklDirection.three_to_four_indices(2, 1, 0)
        self.assertEqual(U, 1)
        self.assertEqual(V, 0)
        self.assertEqual(T, -1)
        self.assertEqual(W, 0)


class HklPlaneTests(unittest.TestCase):
    def setUp(self):
        print('testing the HklPlane class')
        self.hex = Lattice.hexagonal(0.2931, 0.4694)  # nm

    def test_equality(self):
        p1 = HklPlane(1, 1, 1)
        p2 = HklPlane(1, 1, 1)
        p3 = HklPlane(-1, 1, 1)
        self.assertEqual(p1, p2)
        self.assertTrue(p1 == p2)
        self.assertTrue(p1 != p3)

    def test_HklPlane(self):
        p = HklPlane(1, 1, 1)
        n = p.normal()
        self.assertEqual(np.linalg.norm(n), 1)

    def test_get_family(self):
        self.assertEqual(len(HklPlane.get_family('001', crystal_structure=Symmetry.cubic)), 3)
        self.assertEqual(len(HklPlane.get_family('001', crystal_structure=Symmetry.cubic, include_friedel_pairs=True)), 6)
        self.assertEqual(len(HklPlane.get_family('111', crystal_structure=Symmetry.cubic)), 4)
        self.assertEqual(len(HklPlane.get_family('111', crystal_structure=Symmetry.cubic, include_friedel_pairs=True)), 8)
        self.assertEqual(len(HklPlane.get_family('011', crystal_structure=Symmetry.cubic)), 6)
        self.assertEqual(len(HklPlane.get_family('011', crystal_structure=Symmetry.cubic, include_friedel_pairs=True)), 12)
        self.assertEqual(len(HklPlane.get_family('112', crystal_structure=Symmetry.cubic)), 12)
        self.assertEqual(len(HklPlane.get_family('112', crystal_structure=Symmetry.cubic, include_friedel_pairs=True)), 24)
        self.assertEqual(len(HklPlane.get_family('123', crystal_structure=Symmetry.cubic)), 24)
        self.assertEqual(len(HklPlane.get_family('123', crystal_structure=Symmetry.cubic, include_friedel_pairs=True)), 48)
        self.assertEqual(len(HklPlane.get_family('001', crystal_structure=Symmetry.tetragonal)), 1)
        self.assertEqual(len(HklPlane.get_family('001', crystal_structure=Symmetry.tetragonal, include_friedel_pairs=True)), 2)
        self.assertEqual(len(HklPlane.get_family('010', crystal_structure=Symmetry.tetragonal)), 2)
        self.assertEqual(len(HklPlane.get_family('010', crystal_structure=Symmetry.tetragonal, include_friedel_pairs=True)), 4)
        self.assertEqual(len(HklPlane.get_family('100', crystal_structure=Symmetry.tetragonal)), 2)
        self.assertEqual(len(HklPlane.get_family('100', crystal_structure=Symmetry.tetragonal, include_friedel_pairs=True)), 4)
        self.assertEqual(len(HklPlane.get_family([1, 0, 2], crystal_structure=Symmetry.tetragonal, include_friedel_pairs=True)), 8)
        self.assertEqual(len(HklPlane.get_family([-1, 0, 2], crystal_structure=Symmetry.tetragonal, include_friedel_pairs=True)), 8)
        self.assertEqual(len(HklPlane.get_family([0, 1, 2], crystal_structure=Symmetry.tetragonal, include_friedel_pairs=True)), 8)
        self.assertEqual(len(HklPlane.get_family([0, -1, 2], crystal_structure=Symmetry.tetragonal, include_friedel_pairs=True)), 8)
        self.assertEqual(len(HklPlane.get_family('001', crystal_structure=Symmetry.hexagonal)), 1)
        self.assertEqual(len(HklPlane.get_family('001', crystal_structure=Symmetry.hexagonal, include_friedel_pairs=True)), 2)
        self.assertEqual(len(HklPlane.get_family('100', crystal_structure=Symmetry.hexagonal)), 3)
        self.assertEqual(len(HklPlane.get_family('100', crystal_structure=Symmetry.hexagonal, include_friedel_pairs=True)), 6)
        self.assertEqual(len(HklPlane.get_family((1, 0, -1, 0), crystal_structure=Symmetry.hexagonal)), 3)
        self.assertEqual(len(HklPlane.get_family((1, 0, -1, 0), crystal_structure=Symmetry.hexagonal, include_friedel_pairs=True)), 6)
        self.assertEqual(len(HklPlane.get_family('102', crystal_structure=Symmetry.hexagonal)), 6)
        self.assertEqual(len(HklPlane.get_family('102', crystal_structure=Symmetry.hexagonal, include_friedel_pairs=True)), 12)

    def test_multiplicity(self):
        """Int Tables of Crystallography Vol. 1 p 32."""
        self.assertEqual(HklPlane(1, 0, 0).multiplicity(symmetry=Symmetry.cubic), 6)
        for h in range(1, 4):
            self.assertEqual(HklPlane(h, 0, 0).multiplicity(symmetry=Symmetry.tetragonal), 4)
            self.assertEqual(HklPlane(0, h, 0).multiplicity(symmetry=Symmetry.tetragonal), 4)
            self.assertEqual(HklPlane(h, h, 0).multiplicity(symmetry=Symmetry.tetragonal), 4)
            self.assertEqual(HklPlane(-h, h, 0).multiplicity(symmetry=Symmetry.tetragonal), 4)
            self.assertEqual(HklPlane(h, h, 1).multiplicity(symmetry=Symmetry.tetragonal), 8)
            self.assertEqual(HklPlane(-h, h, 1).multiplicity(symmetry=Symmetry.tetragonal), 8)
        self.assertEqual(HklPlane(0, 0, 1).multiplicity(symmetry=Symmetry.tetragonal), 2)
        self.assertEqual(HklPlane(1, 0, 2).multiplicity(symmetry=Symmetry.tetragonal), 8)
        self.assertEqual(HklPlane(-1, 0, 2).multiplicity(symmetry=Symmetry.tetragonal), 8)
        self.assertEqual(HklPlane(0, 1, 2).multiplicity(symmetry=Symmetry.tetragonal), 8)
        self.assertEqual(HklPlane(0, -1, 2).multiplicity(symmetry=Symmetry.tetragonal), 8)
        self.assertEqual(HklPlane(1, 2, 0).multiplicity(symmetry=Symmetry.tetragonal), 8)
        self.assertEqual(HklPlane(-1, 2, 0).multiplicity(symmetry=Symmetry.tetragonal), 8)
        self.assertEqual(HklPlane(1, 2, 3).multiplicity(symmetry=Symmetry.tetragonal), 16)

    def test_HklPlane_normal(self):
        ZrO2 = Lattice.tetragonal(3.64, 5.27)
        p = HklPlane(1, 1, 1, ZrO2)
        n = p.normal()
        self.assertAlmostEqual(n[0], 0.635, 3)
        self.assertAlmostEqual(n[1], 0.635, 3)
        self.assertAlmostEqual(n[2], 0.439, 3)

    def test_110_normal_monoclinic(self):
        """Testing (110) plane normal in monoclinic crystal structure.
        This test comes from
        http://www.mse.mtu.edu/~drjohn/my3200/stereo/sg5.html
        corrected for a few errors in the html page.
        In this test, the lattice is defined with the c-axis aligned with the Z direction of the Cartesian frame.
        """
        Mg2Si = Lattice.from_parameters(1.534, 0.405, 0.683, 90., 106., 90., x_aligned_with_a=False)
        a = Mg2Si.matrix[0]
        b = Mg2Si.matrix[1]
        c = Mg2Si.matrix[2]
        self.assertAlmostEqual(a[0], 1.475, 3)
        self.assertAlmostEqual(a[1], 0., 3)
        self.assertAlmostEqual(a[2], -0.423, 3)
        self.assertAlmostEqual(b[0], 0., 3)
        self.assertAlmostEqual(b[1], 0.405, 3)
        self.assertAlmostEqual(b[2], 0., 3)
        self.assertAlmostEqual(c[0], 0., 3)
        self.assertAlmostEqual(c[1], 0., 3)
        self.assertAlmostEqual(c[2], 0.683, 3)
        p = HklPlane(1, 1, 1, Mg2Si)
        Gc = p.scattering_vector()
        self.assertAlmostEqual(Gc[0], 1.098, 3)
        self.assertAlmostEqual(Gc[1], 2.469, 3)
        self.assertAlmostEqual(Gc[2], 1.464, 3)
        self.assertAlmostEqual(p.interplanar_spacing(), 0.325, 3)
        Ghkl = np.dot(Mg2Si.matrix, Gc)
        self.assertEqual(Ghkl[0], 1.)  # h
        self.assertEqual(Ghkl[1], 1.)  # k
        self.assertEqual(Ghkl[2], 1.)  # l

    def test_scattering_vector(self):
        Fe_fcc = Lattice.face_centered_cubic(0.287)  # FCC iron
        hkl = HklPlane(2, 0, 0, Fe_fcc)
        Gc = hkl.scattering_vector()
        self.assertAlmostEqual(np.linalg.norm(Gc), 1 / hkl.interplanar_spacing())
        Al_fcc = Lattice.face_centered_cubic(0.405)
        hkl = HklPlane(0, 0, 2, lattice=Al_fcc)
        Gc = hkl.scattering_vector()
        self.assertAlmostEqual(np.linalg.norm(Gc), 1 / hkl.interplanar_spacing())

    def test_scattering_vector_th(self):
        """ compute the scattering vector using the formal definition and compare it with the components obtained 
        using the reciprocal lattice. 
        The formulae are available in the Laue Atlas p61, one typo in Eq. 6.1 was corrected. """
        (a, b, c) = self.hex._lengths
        (alpha, beta, gamma) = np.radians(self.hex._angles)
        delta = pi / 2 - gamma
        chi = gamma - atan((cos(alpha) - cos(gamma) * cos(beta)) / (cos(beta) * cos(delta)))
        epsilon = pi / 2 - acos((cos(alpha) + cos(beta)) / (cos(chi) + cos(gamma - chi)))
        psi = acos(sin(epsilon) * cos(delta + chi))
        for (hp, kp, lp) in [(1, 1, 1), [1, 2, 0]]:
            # compute the h, k, l in the Cartesian coordinate system
            h = hp / a
            k = (a / hp - b / kp * cos(gamma)) / (a / hp * b / kp * cos(delta))
            l = (lp / c - hp / a * cos(beta) - kp / b * cos(psi)) / cos(epsilon)
            Gc = HklPlane(hp, kp, lp, self.hex).scattering_vector()
            self.assertAlmostEqual(Gc[0], h, 7)
            self.assertAlmostEqual(Gc[1], k, 7)
            self.assertAlmostEqual(Gc[2], l, 7)

    def test_bragg_angle(self):
        l = Lattice.cubic(0.287)  # FCC iron
        hkl = HklPlane(2, 0, 0, l)  # 200 reflection at 8 keV is at 32.7 deg
        self.assertAlmostEqual(hkl.bragg_angle(8), 0.5704164)


class SlipSystemTests(unittest.TestCase):
    def setUp(self):
        print('testing the SlipSystem class')

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
