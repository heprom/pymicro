import unittest
import os
import numpy as np
from pymicro.crystal.microstructure import Orientation, Grain, Microstructure
from pymicro.crystal.lattice import Symmetry, Lattice, HklPlane, HklDirection, SlipSystem
from pymicro.xray.xray_utils import lambda_keV_to_nm
from config import PYMICRO_EXAMPLES_DATA_DIR

class MicrostructureTests(unittest.TestCase):

    def setUp(self):
        print('testing the Microstructure class')
        self.test_eulers = [(45., 45, 0.), (10., 20, 30.), (191.9, 69.9, 138.9)]

    def test_add_grains(self):
        micro = Microstructure(name='test', autodelete=True)
        self.assertEqual(micro.get_number_of_grains(), 0)
        micro.add_grains(self.test_eulers)
        self.assertEqual(micro.get_number_of_grains(), 3)
        del micro

    def test_base(self):
        micro = Microstructure(name='test', autodelete=True)
        micro.add_grains(self.test_eulers)
        self.assertTrue(micro.get_sample_name() == 'test')
        self.assertTrue(os.path.exists(micro.h5_file))
        self.assertTrue(os.path.exists(micro.xdmf_file))
        h5_file = micro.h5_file
        xdmf_file = micro.xdmf_file
        del micro
        self.assertTrue(not os.path.exists(h5_file))
        self.assertTrue(not os.path.exists(xdmf_file))

    def test_from_dct(self):
        # read a microstructure from a DCT index.mat file
        m = Microstructure.from_dct(data_dir=PYMICRO_EXAMPLES_DATA_DIR,
                                    grain_file='t5_dct_cen_index.mat',
                                    use_dct_path=False)
        m.autodelete = True
        self.assertEqual(m.grains.nrows, 146)

    def test_from_file(self):
        # read a test microstructure already created
        m = Microstructure(filename=os.path.join(PYMICRO_EXAMPLES_DATA_DIR,
                                                 't5_dct_slice_data.h5'))
        self.assertEqual(m.grains.nrows, 21)
        self.assertEqual(m.get_voxel_size(), 0.0014)
        self.assertEqual(type(m.get_grain_map(as_numpy=True)), np.ndarray)
        self.assertEqual(type(m.get_mask(as_numpy=True)), np.ndarray)
        self.assertTrue(True)
        del m

    def test_from_copy(self):
        # test computing of grain geometrical quantities and copy of an
        # existing Microstructure dataset
        # create new microstructure copy of an already existing file
        filename = os.path.join(PYMICRO_EXAMPLES_DATA_DIR,
                                't5_dct_slice_data.h5')
        new_file = os.path.join(PYMICRO_EXAMPLES_DATA_DIR, 'tmp_slice_dct')
        m = Microstructure.copy_sample(filename, new_file, autodelete=True,
                                       get_object=True)
        h5_file = m.h5_path
        xdmf_file = m.xdmf_path
        self.assertTrue(os.path.exists(h5_file))
        self.assertTrue(os.path.exists(xdmf_file))
        m.recompute_grain_bounding_boxes()
        m.recompute_grain_centers()
        # m.recompute_grain_volumes()
        m_ref = Microstructure(filename=filename)
        for i in range(m_ref.grains.nrows):
            print(' n°1 :',m.grains[i])
            print(' n°2 :',m_ref.grains[i])
            self.assertEqual(m.grains[i], m_ref.grains[i])
        volume = np.sum(m.get_mask(as_numpy=True))
        self.assertEqual(volume, 194025)
        del m
        self.assertTrue(not os.path.exists(h5_file))
        self.assertTrue(not os.path.exists(xdmf_file))
        del m_ref

    def test_crop(self):
        # read a test microstructure
        m = Microstructure(os.path.join(PYMICRO_EXAMPLES_DATA_DIR, 'n27-id1_data.h5'))
        # crop the microstructure
        m1 = m.crop(x_start=20, x_end=40, y_start=10, y_end=40, z_start=15, z_end=55, autodelete=True)
        h5_file = m1.h5_file
        xdmf_file = m1.xdmf_file
        dims = (20, 30, 40)
        for i in range(3):
            self.assertEqual(m1.get_grain_map().shape[i], dims[i])
        self.assertEqual(m1.get_number_of_grains(), 16)
        gids_crop = [1, 2, 3, 4, 6, 7, 8, 13, 14, 15, 17, 20, 21, 22, 26, 27]
        for gid in m1.get_grain_ids():
            self.assertTrue(gid in gids_crop)
        self.assertEqual(np.sum(m1.get_grain_map(as_numpy=True) == 14), 396)
        del m1
        # verify that the mirostructure files have been deleted
        self.assertTrue(not os.path.exists(h5_file))
        self.assertTrue(not os.path.exists(xdmf_file))
        del m

    def test_grain_geometry(self):
        m = Microstructure(name='test', autodelete=True)
        grain_map = np.ones((8, 8, 8), dtype=np.uint8)
        m.set_grain_map(grain_map, voxel_size=1.0)
        grain = m.grains.row
        grain['idnumber'] = 1
        grain['orientation'] = Orientation.from_euler([10, 20, 30]).rod
        grain.append()
        m.grains.flush()
        m.recompute_grain_bounding_boxes()
        m.recompute_grain_centers()
        m.recompute_grain_volumes()
        bb1 = m.compute_grain_bounding_box(gid=1)
        self.assertEqual(bb1, ((0, 8), (0, 8), (0, 8)))
        c1 = m.compute_grain_center(gid=1).tolist()
        self.assertEqual(c1, [0., 0., 0.])
        self.assertEqual(m.compute_grain_volume(gid=1), 512)

    def test_renumber_grains(self):
        # read and copy a microstructure
        m1_path = os.path.join(PYMICRO_EXAMPLES_DATA_DIR, 'm1_data.h5')
        copy_path = os.path.join(PYMICRO_EXAMPLES_DATA_DIR, 'm1_copy_data.h5')
        m1 = Microstructure.copy_sample(m1_path, copy_path, autodelete=True,
                                        get_object=True)
        self.assertTrue(8 not in m1.get_grain_ids())
        m1.renumber_grains()
        self.assertTrue(8 in m1.get_grain_ids())
        m1.renumber_grains(sort_by_size=True)
        self.assertEqual(m1.get_grain_ids()[0], 18)
        del m1

    def test_merge_microstructures(self):
        m1 = Microstructure(os.path.join(PYMICRO_EXAMPLES_DATA_DIR, 'm1_data.h5'))
        m2 = Microstructure(os.path.join(PYMICRO_EXAMPLES_DATA_DIR, 'm2_data.h5'))
        # merge the two microstructures
        m1m2 = Microstructure.merge_microstructures([m1, m2], overlap=16)
        m1m2.autodelete = True
        h5_file = m1m2.h5_file
        xdmf_file = m1m2.xdmf_file
        dims = (64, 64, 64)
        for i in range(3):
            self.assertEqual(m1m2.get_grain_map().shape[i], dims[i])
        self.assertEqual(m1m2.get_number_of_grains(), 27)
        del m1m2
        self.assertTrue(not os.path.exists(h5_file))
        self.assertTrue(not os.path.exists(xdmf_file))
        del m1, m2

    def test_from_neper(self):
        # read a microstructure generated by neper
        m = Microstructure.from_neper(os.path.join(PYMICRO_EXAMPLES_DATA_DIR, 'n100-id1.tesr'))
        m.autodelete = True
        self.assertEqual(m.grains.nrows, 100)
        # verify all orienations
        euler_neper = np.genfromtxt(os.path.join(PYMICRO_EXAMPLES_DATA_DIR, 'n100-id1.ori-plain'))
        for i in range(100):
            euler_pymicro = m.get_grain(i + 1).orientation.euler
            for j in range(3):
                self.assertAlmostEqual(euler_pymicro[j],
                                       euler_neper[i][j] % 360, 4)
        self.assertEqual(m.__contains__('grain_map'), True)
        self.assertAlmostEqual(m.get_voxel_size(), 0.018, 2)
        dims = (54, 65, 75)
        for i in range(3):
            self.assertEqual(m.get_grain_map(as_numpy=True).shape[i], dims[i])
        del m

    def test_find_neighbors(self):
        # read a test microstructure already created
        m = Microstructure(filename=os.path.join(PYMICRO_EXAMPLES_DATA_DIR,
                                                 't5_dct_slice_data.h5'))
        neighbors = m.find_neighbors(grain_id=5, distance=3)
        self.assertEqual(len(neighbors), 9)
        for gid in [0, 1, 3, 14, 17, 18, 25, 51, 115]:
            self.assertTrue(gid in neighbors)
        del m


class OrientationTests(unittest.TestCase):

    def setUp(self):
        print('testing the Orientation class')
        self.test_eulers = [(45., 45, 0.), (10., 20, 30.), (191.9, 69.9, 138.9)]

    def test_Orientation(self):
        o = Orientation.from_euler([45, 45, 0])
        self.assertAlmostEqual(o.phi1(), 45.)
        self.assertAlmostEqual(o.Phi(), 45.)

    def test_RodriguesConversion(self):
        rod = [0.1449, -0.0281, 0.0616]
        g = Orientation.Rodrigues2OrientationMatrix(rod)
        calc_rod = Orientation.OrientationMatrix2Rodrigues(g)
        for i in range(3):
            self.assertAlmostEquals(calc_rod[i], rod[i])

    def test_OrientationMatrix2Euler(self):
        for test_euler in self.test_eulers:
            o = Orientation.from_euler(test_euler)
            g = o.orientation_matrix()
            calc_euler = Orientation.OrientationMatrix2Euler(g)
            calc_euler2 = Orientation.OrientationMatrix2EulerSF(g)
            for i in range(3):
                self.assertAlmostEquals(calc_euler[i], test_euler[i])
                #self.assertAlmostEquals(calc_euler2[i], test_euler[i])

    def test_SchimdFactor(self):
        o = Orientation.from_euler([0., 0., 0.])
        ss = SlipSystem(HklPlane(1, 1, 1), HklDirection(0, 1, -1))
        self.assertAlmostEqual(o.schmid_factor(ss), 0.4082, 4)

    def test_MaxSchimdFactor(self):
        o = Orientation.from_euler([0., 0., 0.])
        oct_ss = SlipSystem.get_slip_systems(slip_type='111')
        self.assertAlmostEqual(max(o.compute_all_schmid_factors(oct_ss, verbose=True)), 0.4082, 4)

    def test_misorientation_matrix(self):
        for test_euler in self.test_eulers:
            o = Orientation.from_euler(test_euler)
            g = o.orientation_matrix()
            delta = np.dot(g, g.T)
            self.assertEqual(Orientation.misorientation_angle_from_delta(delta), 0.0)

    def test_misorientation_angle(self):
        o1 = Orientation.from_euler((0., 0., 0.))
        o2 = Orientation.from_euler((60., 0., 0.))
        self.assertAlmostEqual(180 / np.pi * o1.disorientation(o2, crystal_structure=Symmetry.triclinic)[0], 60)
        self.assertAlmostEqual(180 / np.pi * o1.disorientation(o2, crystal_structure=Symmetry.cubic)[0], 30)

    def test_misorientation_axis(self):
        o1 = Orientation.copper()
        o2 = Orientation.s3()
        (angle, axis, axis_xyz) = o1.disorientation(o2, crystal_structure=Symmetry.triclinic)
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

    def test_solve_trig_equation(self):
        x1, x2 = Orientation.solve_trig_equation(2, -6, 3)
        self.assertAlmostEqual(x1, 3.9575 * 180 / np.pi, 2)
        self.assertAlmostEqual(x2, 6.1107 * 180 / np.pi, 2)
        x1, x2 = Orientation.solve_trig_equation(5, 4, 6)
        self.assertAlmostEqual(x1, 0.3180 * 180 / np.pi, 2)
        self.assertAlmostEqual(x2, 1.0314 * 180 / np.pi, 2)

    def test_dct_omega_angles(self):
        # test with a BCC Titanium lattice
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
        t1 = (B - 0.5 * np.sqrt(Delta)) / (A + C)
        t2 = (B + 0.5 * np.sqrt(Delta)) / (A + C)
        # verifying A cos(w) + B sin(w) = C:'
        for t in (t1, t2):
            x = A * (1 - t ** 2) / (1 + t ** 2) + B * 2 * t / (1 + t ** 2)
            self.assertAlmostEqual(x, C, 2)
        # verifying (A + C) * t**2 - 2 * B * t + (C - A) = 0'
        for t in (t1, t2):
            self.assertAlmostEqual((A + C) * t ** 2 - 2 * B * t + (C - A), 0.0, 2)
        (w1, w2) = o.dct_omega_angles(hkl, lambda_keV, verbose=False)
        self.assertAlmostEqual(w1, 196.709, 2)
        self.assertAlmostEqual(w2, 28.334, 2)
        # test with an FCC Aluminium-Lithium lattice
        a = 0.40495  # lattice parameter in nm
        Al_fcc = Lattice.face_centered_cubic(a)
        hkl = HklPlane(-1, 1, 1, Al_fcc)
        o = Orientation.from_rodrigues([0.0499, -0.3048, 0.1040])
        w1, w2 = o.dct_omega_angles(hkl, 40, verbose=False)
        self.assertAlmostEqual(w1, 109.2, 1)
        self.assertAlmostEqual(w2, 296.9, 1)


    def test_topotomo_tilts(self):
        # tests cases from ma2285 experiment on id11, omega offset = -90
        T = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        al = Lattice.from_symbol('Al')
        p = HklPlane(0, 0, 2, lattice=al)
        rod = [0.1449, -0.0281, 0.0616]
        o = Orientation.from_rodrigues(rod)
        (ut, lt) = o.topotomo_tilts(p, T)
        self.assertAlmostEqual(180 / np.pi * ut, 2.236, 3)
        self.assertAlmostEqual(180 / np.pi * lt, 16.615, 3)
        # use test case from AlLi_sam8_dct_cen_
        p = HklPlane(2, 0, 2, lattice=al)
        rod = [0.0499, -0.3048, 0.1040]
        o = Orientation.from_rodrigues(rod)
        (ut, lt) = o.topotomo_tilts(p, T)
        self.assertAlmostEqual(180 / np.pi * ut, -11.04, 2)
        self.assertAlmostEqual(180 / np.pi * lt, -0.53, 2)
        # test case from ma3921
        T = Orientation.compute_instrument_transformation_matrix(-1.2, 0.7, 90)
        Ti7Al = Lattice.hexagonal(0.2931, 0.4694)  # nm
        (h, k, l) = HklPlane.four_to_three_indices(-1, 2, -1, 0)
        p = HklPlane(h, k, l, Ti7Al)
        o = Orientation.from_rodrigues([0.7531, 0.3537, 0.0621])
        (ut, lt) = o.topotomo_tilts(p, T)
        self.assertAlmostEqual(180 / np.pi * ut, 11.275, 2)
        self.assertAlmostEqual(180 / np.pi * lt, -4.437, 2)


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

    def test_in_fundamental_zone(self):
        rod = [0.1449, -0.0281, 0.0616]
        o = Orientation.from_rodrigues(rod)
        self.assertTrue(o.inFZ(symmetry=Symmetry.cubic))
        o = Orientation.from_euler([191.9, 69.9, 138.9])
        self.assertFalse(o.inFZ(symmetry=Symmetry.cubic))

    def test_move_to_fundamental_zone(self):
        o = Orientation.from_euler([191.9, 69.9, 138.9])
        # move rotation to cubic FZ
        o_fz = o.move_to_FZ(symmetry=Symmetry.cubic, verbose=False)
        # double check new orientation in is the FZ
        self.assertTrue(o_fz.inFZ(symmetry=Symmetry.cubic))
        # verify new Euler angle values
        val = np.array([303.402, 44.955, 60.896])
        for i in range(3):
            self.assertAlmostEqual(o_fz.euler[i], val[i], 3)

    def test_small_disorientation(self):
        o_ref = Orientation(np.array([[-0.03454188, 0.05599919, -0.99783313],
                                      [-0.01223192, -0.99837784, -0.05560633],
                                      [-0.99932839, 0.01028467, 0.03517083]]))
        o_12 = Orientation(np.array([[-0.03807341, -0.06932796, -0.99686712],
                                     [-0.0234124, -0.99725469, 0.07024911],
                                     [-0.99900064, 0.02601367, 0.03634576]]))
        (angle, axis, axis_xyz) = o_ref.disorientation(o_12, crystal_structure=Symmetry.cubic)
        self.assertAlmostEqual(angle * 180 / np.pi, 7.24, 2)
        o_ref_fz = o_ref.move_to_FZ(symmetry=Symmetry.cubic, verbose=False)
        o_12_fz = o_12.move_to_FZ(symmetry=Symmetry.cubic, verbose=False)
        delta = np.dot(o_ref_fz.orientation_matrix(), o_12_fz.orientation_matrix().T)
        mis_angle = Orientation.misorientation_angle_from_delta(delta)
        self.assertAlmostEqual(mis_angle * 180 / np.pi, 7.24, 2)

if __name__ == '__main__':
    unittest.main()
