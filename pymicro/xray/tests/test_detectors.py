import unittest
import numpy as np
from pymicro.xray.detectors import RegArrayDetector2d

class DetectorsTests(unittest.TestCase):

    def setUp(self):
        """testing the detectors module:"""
        self.detector = RegArrayDetector2d(size=(1024, 512))
        self.detector.pixel_size = 0.1  # mm
        self.detector.ref_pos = np.array([100., 0., 0.])  # position in the laboratory frame of the middle of the detector

    def test_project_along_direction(self):
        """Verify the project_along_direction method."""
        R = self.detector.project_along_direction(direction=(1., 0., 0.), origin=(0., 0., 0.))
        (u, v) = self.detector.lab_to_pixel(R)[0]
        self.assertEqual(u, 512)
        self.assertEqual(v, 256)
        # project in the top left corner of the detector
        v = np.array([0.1, 0.04, 0.02])
        v /= np.linalg.norm(v)
        R = self.detector.project_along_direction(direction=v, origin=(0., 0., 0.))
        (u, v) = self.detector.lab_to_pixel(R)[0]
        self.assertEqual(int(u), 112)
        self.assertEqual(int(v), 56)
        RR = self.detector.pixel_to_lab(u, v)[0]
        self.assertListEqual(RR.tolist(), R.tolist())
        size_mm_1 = self.detector.get_size_mm()

        # use a 2x2 binning
        self.detector.pixel_size = 0.2  # mm
        self.detector.size = (512, 256)
        size_mm_2 = self.detector.get_size_mm()
        self.assertEqual(size_mm_1[0], size_mm_2[0])
        self.assertEqual(size_mm_1[1], size_mm_2[1])
        (u, v) = self.detector.lab_to_pixel(R)[0]
        self.assertEqual(int(u), 56)
        self.assertEqual(int(v), 28)
        RR = self.detector.pixel_to_lab(u, v)[0]
        self.assertListEqual(RR.tolist(), R.tolist())

    def test_detector_tilt(self):
        """Verify the tilted coordinate frame calculation.
        
        The local frame is calculated by composing 3 rotations and the definition of the detector-to-pixel conversion.
        We have verified that:
        u[0] == np.cos(delta) * np.sin(omega)
        u[1] == -np.cos(kappa) * np.cos(omega) + np.sin(kappa) * np.sin(delta) * np.sin(omega)
        u[2] == -np.sin(kappa) * np.cos(omega) - np.cos(kappa) * np.sin(delta) * np.sin(omega)
        v[0] == -np.sin(delta)
        v[1] == np.sin(kappa) * np.cos(delta)
        v[2] == -np.cos(kappa) * np.cos(delta)
        w[0] == np.cos(omega) * np.cos(delta)
        w[1] == np.cos(omega) * np.sin(delta) * np.sin(kappa) + np.sin(omega) * np.cos(kappa)
        w[2] == np.sin(omega) * np.sin(kappa) - np.cos(omega) * np.sin(delta) * np.cos(kappa)
        
        so the test chack that the matrix composition gives the final result.
        """
        for tilt in [1, 5, 10, 15]:  # degrees
            # Detector tilt kappa/X ; delta/Y ; omega/Z
            det_tilt = RegArrayDetector2d(size=(487, 619), tilts=(tilt, tilt, tilt))
            kappa, delta, omega = np.radians([tilt, tilt, tilt])
            # compute u, v, w using trigonometry
            u1 = np.cos(delta) * np.sin(omega)
            u2 = -np.cos(kappa) * np.cos(omega) + np.sin(kappa) * np.sin(delta) * np.sin(omega)
            u3 = -np.sin(kappa) * np.cos(omega) - np.cos(kappa) * np.sin(delta) * np.sin(omega)
            v1 = -np.sin(delta)
            v2 = np.sin(kappa) * np.cos(delta)
            v3 = -np.cos(kappa) * np.cos(delta)
            w1 = np.cos(omega) * np.cos(delta)
            w2 = np.cos(omega) * np.sin(delta) * np.sin(kappa) + np.sin(omega) * np.cos(kappa)
            w3 = np.sin(omega) * np.sin(kappa) - np.cos(omega) * np.sin(delta) * np.cos(kappa)
            self.assertAlmostEqual(u1, det_tilt.u_dir[0], 7)
            self.assertAlmostEqual(u2, det_tilt.u_dir[1], 7)
            self.assertAlmostEqual(u3, det_tilt.u_dir[2], 7)
            self.assertAlmostEqual(v1, det_tilt.v_dir[0], 7)
            self.assertAlmostEqual(v2, det_tilt.v_dir[1], 7)
            self.assertAlmostEqual(v3, det_tilt.v_dir[2], 7)
            self.assertAlmostEqual(w1, det_tilt.w_dir[0], 7)
            self.assertAlmostEqual(w2, det_tilt.w_dir[1], 7)
            self.assertAlmostEqual(w3, det_tilt.w_dir[2], 7)
