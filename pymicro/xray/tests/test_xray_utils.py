import unittest
from pymicro.xray.xray_utils import f_atom

class XrayUtilsTests(unittest.TestCase):

    def test_f_atom(self):
        """Verify the calculation of the atom form factor."""
        for Z in range(1, 30):
            # error is less than 3%
            self.assertLess(abs(f_atom(0., Z) - Z) / Z, 0.03)