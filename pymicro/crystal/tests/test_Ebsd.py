import unittest
import numpy as np
from pymicro.crystal.ebsd import OimScan, OimPhase
from pymicro.crystal.microstructure import Orientation


class EbsdTests(unittest.TestCase):

    def setUp(self):
        print('testing the OimScan class')
        # create a test EBSD scan
        self.scan = OimScan((10, 10))
        phase = OimPhase(1)
        self.scan.phase_list.append(phase)
        # populate the euler field with 4 random grains
        for i in range(2):
            for j in range(2):
                o = Orientation.random()
                phi1, Phi, phi2 = np.radians(o.euler)
                self.scan.euler[i * 5:(i + 1) * 5, j * 5:(j + 1) * 5, 0] = phi1
                self.scan.euler[i * 5:(i + 1) * 5, j * 5:(j + 1) * 5, 1] = Phi
                self.scan.euler[i * 5:(i + 1) * 5, j * 5:(j + 1) * 5, 2] = phi2
        self.scan.ci = 1.0  # mark all pixels as good
        self.scan.euler.shape

    def test_segment_grains(self):
        grain_ids = self.scan.segment_grains()
        n = len(np.unique(grain_ids))
        self.assertEqual(n, 4)
