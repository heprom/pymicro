import unittest
import numpy as np
from pymicro.xray.fitting import lin_reg

class FittingTests(unittest.TestCase):

    def setUp(self):
        """testing the fitting module:"""

    def test_lin_reg(self):
        """Verify the linear regression, example from wikipedia."""
        xi = np.array([1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83])
        yi = np.array([52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46])
        alpha, beta, r = lin_reg(xi, yi)
        self.assertAlmostEqual(beta, 61.2721865, 7)
        self.assertAlmostEqual(alpha, -39.0619559, 7)
        self.assertAlmostEqual(r,  0.99458379, 7)
