"""The mechanical module provide functions to work with quantity such as tensors 
and mechanics of material related calculations.
"""
import numpy as np
from pymicro.crystal.rotation import *


## Takes two pairs of ijkl indices and returns 1 (equivalent) or 0 (not equivalent) according to both major and minor symmetries
def symm_check(i1, j1, k1, l1, i2, j2, k2, l2):
    if i1 == i2 and j1 == j2 and k1 == k2 and l1 == l2:
        return 1
    elif i1 == j2 and j1 == i2 and k1 == k2 and l1 == l2:
        return 1
    elif i1 == j2 and j1 == i2 and k1 == l2 and l1 == k2:
        return 1
    elif i1 == i2 and j1 == j2 and k1 == l2 and l1 == k2:
        return 1
    elif i1 == k2 and j1 == l2 and k1 == i2 and l1 == j2:
        return 1
    else:
        return 0


## Transfer a pair of indices i,j (inside a 4th order tensor for example) into Voigt notation (components from 1 to 6, here 0 to 5)
def index_transf(i, j):
    if i == 0 and j == 0:
        return 0
    elif i == 1 and j == 1:
        return 1
    elif i == 2 and j == 2:
        return 2
    elif (i == 1 and j == 2) or (i == 2 and j == 1):
        return 3
    elif (i == 0 and j == 2) or (i == 2 and j == 0):
        return 4
    elif (i == 0 and j == 1) or (i == 1 and j == 0):
        return 5


## Converts an elasticty matrix (rank 2 tensor) to an elasticity tensor (rank 4 tensor)
def sec2fourth(Cmatrix,dtype=np.float64):
    dummy = np.zeros(shape=[3, 3, 3, 3],dtype=dtype)
    dummy[0, 0, 0, 0] = Cmatrix[0, 0]
    dummy[1, 1, 1, 1] = Cmatrix[1, 1]
    dummy[2, 2, 2, 2] = Cmatrix[2, 2]
    dummy[0, 0, 1, 1] = Cmatrix[0, 1]
    dummy[0, 0, 2, 2] = Cmatrix[0, 2]
    dummy[1, 1, 2, 2] = Cmatrix[1, 2]
    dummy[1, 2, 1, 2] = Cmatrix[3, 3]
    dummy[0, 2, 0, 2] = Cmatrix[4, 4]
    dummy[0, 1, 0, 1] = Cmatrix[5, 5]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            for o in range(3):
                                for p in range(3):
                                    if symm_check(i, j, k, l, m, n, o, p) == 1:
                                        dummy[i, j, k, l] = max(
                                            dummy[i, j, k, l], dummy[m, n, o, p]
                                        )
    return dummy


## Converts an elasticity tensor (rank 4 tensor) to an elasticty matrix (rank 2 tensor)
def fourth2sec(Ctensor,dtype=np.float64):
    Cmatrix = np.zeros(shape=[6, 6], dtype=dtype)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    a = index_transf(i, j)
                    b = index_transf(k, l)
                    Cmatrix[a, b] = Ctensor[i, j, k, l]
    return Cmatrix


## Rotates an elasticty matrix (rank 2 tensor) for given Euler angles
def rotateC_euler(Cmatrix0, euler):

    Ctensor0 = sec2fourth(Cmatrix0)

    g = eu2om(euler)

    Gtensor = np.einsum(
            "mi,nj,ok,pl->ijklmnop",
            g,
            g,
            g,
            g,
    )

    rotC=np.einsum("ijklmnop,mnop->ijkl", Gtensor, Ctensor0)

    rotCvoigt = fourth2sec(rotC)
    return rotCvoigt


## Assigns to a variable the reference elasticity matrix for a gamma-TiAl single-crystal and given symmetry (cubic or tetragonal)
def getC_tial(symmetry, euler=[0,0,0]):
    if symmetry == "tetra":
        C11 = 183000
        C12 = 74000
        C22 = C11
        C33 = 178000
        C13 = C12
        C23 = C12
        C44 = 105000
        C55 = C44
        C66 = 78000

    elif symmetry == "cubic":
        C11 = 183000
        C12 = 74000
        C22 = C11
        C33 = 183000
        C13 = C12
        C23 = C12
        C44 = 105000
        C55 = C44
        C66 = 105000

    Cunrot = np.eye(6)
    Cunrot = np.array(
        [
            [C11, C12, C13, 0, 0, 0],
            [C12, C22, C23, 0, 0, 0],
            [C13, C23, C33, 0, 0, 0],
            [0, 0, 0, C44, 0, 0],
            [0, 0, 0, 0, C55, 0],
            [0, 0, 0, 0, 0, C66],
        ],
        dtype="float64",
    )

    Crot = rotateC_euler(Cunrot, euler)

    return Crot