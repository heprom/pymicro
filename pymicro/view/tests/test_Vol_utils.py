import unittest
import numpy as np
from pymicro.view.vol_utils import min_max_cumsum, auto_min_max

class VolUtilsTests(unittest.TestCase):

    def setUp(self):
        print('testing the vol_utils module')
        self.mu, self.sigma = 0, 0.2  # mean and standard deviation
        np.random.seed(42)
        self.data = np.random.normal(self.mu, self.sigma, 3000)

    def test_min_max_cumsum(self):
        count, bins = np.histogram(self.data, 30, range=(-3 * self.sigma, 3 * self.sigma), density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        mini, maxi = min_max_cumsum(count, 0.005, verbose=True)
        self.assertEqual(mini, 2)
        self.assertEqual(maxi, 26)

    def test_auto_min_max(self):
        from scipy import misc
        image = misc.ascent().astype(np.float32)
        mini, maxi = auto_min_max(image, cut=0.001, nb_bins=256, verbose=False)
        self.assertAlmostEqual(mini, 2.9882, 3)
        self.assertAlmostEqual(maxi, 244.0429, 3)
