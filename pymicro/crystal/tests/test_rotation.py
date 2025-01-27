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


    def test_eu2om_single_vectorized(self):
        """Test eu2om with a single Euler angle set (shape (3,))."""
        # Example single euler set (radians)
        euler_single = np.array([0.1, 0.2, 0.3])
        om_single = eu2om(euler_single)  # should be (3,3)

        # Check shape
        self.assertEqual(om_single.shape, (3, 3),
                        msg="eu2om(single) should return shape (3,3).")

        # Optionally, check that converting back to Euler matches original (within tolerance)
        # We can re-use om2eu from the module to check round-trip error:
        euler_back = om2eu(om_single)
        self.assertAlmostEqual(self.euler_dist(euler_single, euler_back), 0.,
                            msg="Round-trip conversion with single euler failed.")


    def test_eu2om_array_vectorized(self):
        """Test eu2om with multiple Euler angle sets (shape (n, 3))."""
        np.random.seed(0)  # for reproducible tests
        n = 100
        eulers = np.random.rand(n, 3) * np.pi  # 10 random rows, each is [phi1, Phi, phi2]
        om_array = eu2om(eulers)  # should be (10, 3, 3)

        # Check shape
        self.assertEqual(om_array.shape, (n, 3, 3),
                        msg="eu2om(array) should return shape (n,3,3).")

        # Check consistency: row-by-row, it should match single-euler usage
        for i in range(n):
            om_single = eu2om(eulers[i])  # shape (3,3)
            self.assertTrue(np.allclose(om_array[i], om_single),
                            msg=f"Row {i} of eu2om(array) differs from eu2om(single).")

        # (Optional) Round-trip check for one or more rows
        euler_back_0 = om2eu(om_array[0])
        self.assertAlmostEqual(self.euler_dist(eulers[0], euler_back_0), 0.,
                                msg="Round-trip conversion with euler[0] failed.")


    def test_eu2qu_single_vectorized(self):
        """Test eu2qu with a single Euler angle set (shape (3,))."""
        # Example single Euler set (radians)
        euler_single = np.array([0.1, 0.2, 0.3])
        qu_single = eu2qu(euler_single)  # should be shape (4,)

        # Check shape
        self.assertEqual(qu_single.shape, (4,),
                        msg="eu2qu(single) should return shape (4,).")

        # Scalar part should be non-negative
        self.assertGreaterEqual(qu_single[0], 0.,
                                msg="Scalar part of quaternion should be >= 0.")
        
        # Check quaternion norm is 1
        self.assertAlmostEqual(np.linalg.norm(qu_single), 1.0,
                             msg="Quaternion should have unit norm.")


    def test_eu2qu_array_vectorized(self):
        """Test eu2qu with multiple Euler angle sets (shape (n, 3))."""
        np.random.seed(0)  # for reproducible tests
        n = 100
        eulers = np.random.rand(n, 3) * np.pi  # 10 random rows
        qu_array = eu2qu(eulers)  # should be (10, 4)

        # Check shape
        self.assertEqual(qu_array.shape, (n, 4),
                        msg="eu2qu(array) should return shape (n,4).")

        # Check scalar part non-negative
        self.assertTrue(np.all(qu_array[:, 0] >= 0),
                        msg="Scalar part of each quaternion should be >= 0.")
        
        # Check all quaternions have unit norm
        norms = np.linalg.norm(qu_array, axis=1)
        self.assertTrue(np.allclose(norms, 1.0),
                        msg="All quaternions should have unit norm.")

        # Compare row-by-row: eu2qu(eulers[i]) vs qu_array[i]
        for i in range(n):
            qu_single = eu2qu(eulers[i])  # shape (4,)
            self.assertTrue(np.allclose(qu_array[i], qu_single),
                            msg=f"Row {i} of eu2qu(array) differs from eu2qu(single).")