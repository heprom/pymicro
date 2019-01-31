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
