import unittest
import numpy as np
from math import cos, sin
from pymicro.crystal.microstructure import Orientation


class QuaternionTests(unittest.TestCase):

    def setUp(self):
        print('testing the Quaternion class')
        self.euler_deg = [343.580, 128.653, 290.986]

    def test_Euler2Quaternion(self):
        euler_rad = np.radians(self.euler_deg)
        # compute quaternion using the passive convention
        q = Orientation.Euler2Quaternion(self.euler_deg, P=-1)

        euler = 0.5 * euler_rad
        c1 = cos(euler[0])
        s1 = sin(euler[0])
        c2 = cos(euler[1])
        s2 = sin(euler[1])
        c3 = cos(euler[2])
        s3 = sin(euler[2])
        q0 = c1 * c2 * c3 - s1 * c2 * s3
        q1 = c1 * s2 * c3 + s1 * s2 * s3
        q2 = -c1 * s2 * s3 + s1 * s2 * c3
        q3 = c1 * c2 * s3 + s1 * c2 * c3
        self.assertAlmostEqual(q.q0, q0)
        self.assertAlmostEqual(q.q1, q1)
        self.assertAlmostEqual(q.q2, q2)
        self.assertAlmostEqual(q.q3, q3)