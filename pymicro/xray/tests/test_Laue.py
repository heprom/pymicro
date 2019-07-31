import unittest
import numpy as np
from pymicro.crystal.lattice import Lattice, HklDirection, HklPlane, Symmetry
from pymicro.crystal.microstructure import Orientation
from pymicro.xray.laue import select_lambda, diffracted_vector, gnomonic_projection_point, gnomonic_projection, index
from pymicro.xray.detectors import RegArrayDetector2d


class LaueTests(unittest.TestCase):
    def setUp(self):
        """testing the laue module:"""
        self.ni = Lattice.from_symbol('Ni')
        self.al = Lattice.face_centered_cubic(0.40495)
        self.g4 = Orientation.from_rodrigues([0.0499199, -0.30475322, 0.10396082])
        self.spots = np.array([[ 76, 211], [ 77, 281], [ 86, 435], [ 90, 563], [112, 128], [151, 459], [151, 639],
                               [161, 543], [170, 325], [176, 248], [189,  70], [190, 375], [213, 670], [250, 167],
                               [294,  54], [310, 153], [323, 262], [358, 444], [360, 507], [369, 163], [378, 535],
                               [384,  86], [402, 555], [442, 139], [444, 224], [452, 565], [476, 292], [496,  88],
                               [501, 547], [514, 166], [522, 525], [531, 433], [536, 494], [559, 264], [581,  57],
                               [625, 168], [663, 607], [679,  69], [686, 363], [694, 240], [703, 315], [728, 437],
                               [728, 518], [743, 609], [756, 128], [786, 413], [789, 271], [790, 534], [791, 205],
                               [818, 123]])

    def test_angle_zone(self):
        """Verify the angle between X and a particular zone axis expressed
        in (X, Y, Z), given a crystal orientation."""
        # euler angles in degrees
        phi1 = 89.4
        phi = 92.0
        phi2 = 86.8
        orientation = Orientation.from_euler([phi1, phi, phi2])
        gt = orientation.orientation_matrix().transpose()
        # zone axis
        uvw = HklDirection(1, 0, 5, self.ni)
        ZA = gt.dot(uvw.direction())
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

    def test_select_lambda(self):
        """Verify that the rotating crystal conditions correspond to the selected wave length diffracted 
        after rotating the crystal in both positions."""
        hkl_dif = HklPlane(2, 0, 2, self.al)
        lambda_keV = 40.0
        w1, w2 = self.g4.dct_omega_angles(hkl_dif, lambda_keV, verbose=False)
        for omega in [w1, w2]:
            omegar = omega * np.pi / 180
            R = np.array([[np.cos(omegar), -np.sin(omegar), 0], [np.sin(omegar), np.cos(omegar), 0], [0, 0, 1]])
            o_rot = Orientation(np.dot(self.g4.orientation_matrix(), R.T))
            self.assertAlmostEqual(select_lambda(hkl_dif, o_rot, verbose=False)[0], lambda_keV, 6)

    def test_gnomonic_projection_point(self):
        """Verify that the gnomonic projection of two diffracted points on a detector give access to the angle 
        between the lattice plane normals."""
        olivine = Lattice.orthorhombic(1.022, 0.596, 0.481)  # nm Barret & Massalski convention
        orientation = Orientation.cube()
        p1 = HklPlane(2, 0, -3, olivine)
        p2 = HklPlane(3, -1, -3, olivine)
        detector = RegArrayDetector2d(size=(512, 512))
        detector.pixel_size = 0.200  # mm, 0.1 mm with factor 2 binning
        detector.ucen = 235
        detector.vcen = 297
        detector.ref_pos = np.array([131., 0., 0.]) + \
                           (detector.size[0] / 2 - detector.ucen) * detector.u_dir * detector.pixel_size + \
                           (detector.size[1] / 2 - detector.vcen) * detector.v_dir * detector.pixel_size  # mm

        angle = 180 / np.pi * np.arccos(np.dot(p1.normal(), p2.normal()))
        # test the gnomonic projection for normal and not normal X-ray incidence
        for ksi in [0.0, 1.0]:  # deg
            Xu = np.array([np.cos(ksi * np.pi / 180), 0., np.sin(ksi * np.pi / 180)])
            OC = detector.project_along_direction(Xu)  # C is the intersection of the direct beam with the detector
            K1 = diffracted_vector(p1, orientation, Xu=Xu)
            K2 = diffracted_vector(p2, orientation, Xu=Xu)
            R1 = detector.project_along_direction(K1, origin=[0., 0., 0.])
            R2 = detector.project_along_direction(K2, origin=[0., 0., 0.])
            OP1 = gnomonic_projection_point(R1, OC=OC)[0]
            OP2 = gnomonic_projection_point(R2, OC=OC)[0]
            hkl_normal1 = OP1 / np.linalg.norm(OP1)
            hkl_normal2 = (OP2 / np.linalg.norm(OP2))
            # the projection must give the normal to the diffracting plane
            for i in range(3):
                self.assertAlmostEqual(hkl_normal1[i], p1.normal()[i], 6)
                self.assertAlmostEqual(hkl_normal2[i], p2.normal()[i], 6)
            angle_gp = 180 / np.pi * np.arccos(np.dot(hkl_normal1, hkl_normal2))
            self.assertAlmostEqual(angle, angle_gp, 6)

    def test_gnomonic_projection(self):
        """Testing the gnomonic_projection function on a complete image."""
        # incident beam
        ksi = 0.4  # deg
        Xu = np.array([np.cos(ksi * np.pi / 180), 0., np.sin(ksi * np.pi / 180)])

        # create our detector
        detector = RegArrayDetector2d(size=(919, 728))
        detector.pixel_size = 0.254  # mm binning 2x2
        detector.ucen = 445
        detector.vcen = 380
        detector.ref_pos = np.array([127.8, 0., 0.]) + \
                           (detector.size[0] // 2 - detector.ucen) * detector.u_dir * detector.pixel_size + \
                           (detector.size[1] // 2 - detector.vcen) * detector.v_dir * detector.pixel_size  # mm
        OC = detector.project_along_direction(Xu)  # C is the intersection of the direct beam with the detector
        # create test image
        pattern = np.zeros(detector.size, dtype=np.uint8)
        for i in range(self.spots.shape[0]):
            # light corresponding pixels
            pattern[self.spots[i, 0], self.spots[i, 1]] = 255
        detector.data = pattern
        gnom = gnomonic_projection(detector, pixel_size=4, OC=OC)
        import os
        test_dir = os.path.dirname(os.path.realpath(__file__))
        ref_gnom_data = np.load(os.path.join(test_dir, 'ref_gnom_data.npy'))
        self.assertTrue(np.array_equal(gnom.data, ref_gnom_data))

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
        final_orientation = Orientation(solutions[0])
        angle, ax1, ax2 = final_orientation.disorientation(orientation, crystal_structure=Symmetry.cubic)
        self.assertLess(angle * 180 / np.pi, 1.0)


if __name__ == '__main__':
    unittest.main()
