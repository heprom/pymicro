import unittest
import numpy as np
from pymicro.xray.experiment import Experiment
from pymicro.xray.detectors import RegArrayDetector2d

class ExperimentTests(unittest.TestCase):

    def setUp(self):
        """testing the experiment module:"""
        self.experiment = Experiment()

    def test_add_detector(self):
        detector = RegArrayDetector2d(size=(512, 512))
        self.experiment.add_detector(detector)
        self.assertEqual(self.experiment.get_number_of_detectors(), 1)