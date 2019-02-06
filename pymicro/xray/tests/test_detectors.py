import unittest
import numpy as np
from pymicro.xray.detectors import RegArrayDetector2d

class DetectorsTests(unittest.TestCase):

    def setUp(self):
        """testing the detectors module:"""
        self.detector = RegArrayDetector2d(size=(1024, 512), u_dir=[0, -1, 0], v_dir=[0, 0, -1])
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
        RR = self.detector.pixel_to_lab(u, v)
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
        RR = self.detector.pixel_to_lab(u, v)
        self.assertListEqual(RR.tolist(), R.tolist())

    def test_detector_tilt(self):
        """Verify the tilted coordinate frame """
        for tilt in [1, 5, 10, 15]:
            # Detector tilt alpha/X ; beta/Y ; gamma/Z
            alpha = np.radians(tilt)  # degree to rad, rotate around X axis
            beta =  np.radians(tilt)  # degree to rad, rotate around Y axis
            gamma = np.radians(tilt)  # degree to rad, rotate around Z axis


            u1 = np.sin(gamma) * np.cos(beta)
            u2 = - np.cos(gamma) * np.cos(alpha) + np.sin(gamma) * np.sin(beta) * np.sin(alpha)
            u3 = - np.cos(gamma) * np.sin(alpha) - np.sin(gamma) * np.sin(beta) * np.cos(alpha)

            v1 = - np.sin(beta)
            v2 = np.cos(beta) * np.sin(alpha)
            v3 = - np.cos(beta) * np.cos(alpha)

            det_tilt = RegArrayDetector2d(size=(487, 619),
                        u_dir=[u1, u2, u3], v_dir=[v1, v2, v3])

            # compute w using trigonometry
            w1 = np.cos(gamma) * np.cos(beta)
            w2 = np.cos(gamma) * np.sin(beta) * np.sin(alpha) + np.sin(gamma) * np.cos(alpha)
            w3 = np.sin(gamma) * np.sin(alpha) - np.cos(gamma) * np.sin(beta) * np.cos(alpha)

            self.assertAlmostEqual(w1, det_tilt.w_dir[0], 7)
            self.assertAlmostEqual(w2, det_tilt.w_dir[1], 7)
            self.assertAlmostEqual(w3, det_tilt.w_dir[2], 7)