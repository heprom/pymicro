import unittest
import numpy as np
from pymicro.crystal.mechanical import *

class MechanicalTests(unittest.TestCase):
    def setUp(self):
        print('testing the mechanical module')
        # Random stiffness tensor
        self.C = np.random.rand(3, 3, 3, 3)
