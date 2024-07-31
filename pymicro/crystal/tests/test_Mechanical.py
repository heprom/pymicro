import unittest
import numpy as np
from pymicro.crystal.mechanical import *
from pymicro.crystal.rotation import *

class MechanicalTests(unittest.TestCase):

    def setUp(self):
        print('testing the mechanical module')
        # Random stiffness tensor
        self.euler_deg = [90,180,180]
        self.Cmatrix = getC_tial("tetra")

    def test_elasticityMatrixRotation(self):
        print('testing the elasticity matrix rotation functions')
        euler_rad = np.radians(self.euler_deg)
        rotated_Cmatrix=rotateC_euler(self.Cmatrix,euler_rad)
        self.assertAlmostEqual(self.Cmatrix[0,0], rotated_Cmatrix[0,0])
        self.assertAlmostEqual(self.Cmatrix[1,1], rotated_Cmatrix[1,1])
        self.assertAlmostEqual(self.Cmatrix[2,2], rotated_Cmatrix[2,2])
        self.assertAlmostEqual(self.Cmatrix[3,3], rotated_Cmatrix[3,3])
        self.assertAlmostEqual(self.Cmatrix[4,4], rotated_Cmatrix[4,4])
        self.assertAlmostEqual(self.Cmatrix[5,5], rotated_Cmatrix[5,5])


test1=MechanicalTests()
test1.setUp()
test1.test_elasticityMatrixRotation()