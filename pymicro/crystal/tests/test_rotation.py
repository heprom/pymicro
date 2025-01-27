import unittest
import numpy as np
import math
from pymicro.crystal.rotation import *


class RotationTests(unittest.TestCase):

    def setUp(self):
        print('testing the rotation module')
        euler_deg = [343.580, 128.653, 290.986]
        self.euler = np.radians(euler_deg)
        self.om = eu2om(self.euler)
        '''
        n = 11
        self.eulers = []
        phi1 = np.linspace(0.+0.00001, np.pi-0.00001, n, endpoint=True)
        Phi = phi1.copy()
        phi2 = phi1.copy()
        for i in range(len(phi1)):
            for j in range(len(Phi)):
                for k in range(len(phi2)):
                    self.eulers.append([phi1[i], Phi[j], phi2[k]])
        self.conversions = [[None, eu2om, eu2ax, eu2ro, eu2qu],
                            [om2eu, None, om2ax, om2ro, om2qu],
                            [ax2eu, ax2om, None, ax2ro, ax2qu],
                            [ro2eu, ro2om, ro2ax, None, ro2qu],
                            [qu2eu, qu2om, qu2ax, qu2ro, None]]
        '''
        self.conversions = [[ None, eu2om, eu2ro, eu2qu],
                            [om2eu,  None, om2ro, om2qu],
                            [ro2eu, ro2om,  None, ro2qu],
                            [qu2eu, qu2om, qu2ro,  None]]

    def dist(self, a, b):
        # manhattan distance
        return np.max(np.abs(a - b))

    def euler_dist(self, a, b):
        return self.dist(eu2qu(a), eu2qu(b))

    def om_dist(self, a, b):
        return self.dist(a.flatten(), b.flatten())

    def test_eu2om(self):
        print('output euler', om2eu(eu2om(self.euler)))
        print(self.euler)
        self.assertAlmostEqual(self.euler_dist(self.euler, om2eu(eu2om(self.euler))), 0.)

    def test_eu2qu(self):
        # a simple test with known values
        euler_1 = np.radians([79.6679, 137.1016, 209.3178])
        euler_2 = np.radians([80.2498, 136.5144, 210.0200])
        self.assertTrue(np.allclose(eu2qu(euler_1), [0.29767613, -0.39592397, 0.84233317, -0.21238637]))
        self.assertTrue(np.allclose(eu2qu(euler_2), [0.30394669, -0.39423896, 0.84104062, -0.21176103]))

    def test_eu_conversions(self):
        # test each conversion from euler angles, forth and back
        for i in range(len(self.conversions)):
            if i == 0:
                continue
            euler = self.conversions[i][0](self.conversions[0][i](self.euler))
            self.assertAlmostEqual(self.euler_dist(self.euler, euler), 0.)

    def test_om_conversions(self):
        # test each conversion from orientation matrice, forth and back
        for i in range(len(self.conversions)):
            if i == 1:
                continue
            om = self.conversions[i][1](self.conversions[1][i](self.om))
            self.assertAlmostEqual(self.om_dist(self.om, om), 0.)
