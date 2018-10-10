import unittest
import numpy as np
from pymicro.xray.detectors import RegArrayDetector2d

class DetectorsTests(unittest.TestCase):

    def setUp(self):
        """testing the detectors module:"""
        self.detector = RegArrayDetector2d(size=(1024, 512), u_dir=[0, -1, 0], v_dir=[0, 0, -1])
        self.detector.ref_pos = np.array([100., 0., 0.])  # position in the laboratory frame of the middle of the detector

    def test_project_along_direction(self):
        """Verify the project_along_direction method."""
        R = self.detector.project_along_direction(direction=(1., 0., 0.), origin=(0., 0., 0.))
        (u, v) = self.detector.lab_to_pixel(R)[0]
        self.assertEqual(u, 512)
        self.assertEqual(v, 256)
