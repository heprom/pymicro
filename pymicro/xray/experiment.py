import json
from pymicro.xray.detectors import Detector2d

class Sample:

    def __init__(self, name):
        self.name = name

class Experiment:
    """Class to represent an actual or virtual X-ray experiment.
    
    A cartesion coordinate system (X, Y, Z) is associated with the experiment. By default X is the direction of X-rays 
    and the sampel is placed at the origin (0, 0, 0).
    """

    def __init__(self):
        self.sample = Sample()
        self.detectors = []

    def add_detector(self, detector):
        assert detector.isinstance(Detector2d)
        self.detectors.append(detector)

    def get_number_of_detectors(self):
        return len(self.detectors)