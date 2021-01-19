"""The lattice module define the class to handle 3D crystal lattices (the 14 Bravais lattices).
"""
import os
from pymicro.external import CifFile_module as CifFile
import enum
import functools
import numpy as np
from numpy import pi, dot, transpose, radians
from matplotlib import pyplot as plt


class Crystal:
    '''
    The Crystal class to create any particular crystal structure.

    A crystal instance is composed by:

     * one of the 14 Bravais lattice
     * a point basis (or motif)
    '''

    def __init__(self, lattice, basis=None, basis_labels=None, basis_sizes=None, basis_colors=None):
        '''
        Create a Crystal instance with the given lattice and basis.

        This create a new instance of a Crystal object. The given lattice
        is assigned to the crystal. If the basis is not specified, it will
        be one atom at (0., 0., 0.).

        :param lattice: the :py:class:`~pymicro.crystal.lattice.Lattice` instance of the crystal.
        :param list basis: A list of tuples containing the position of the atoms in the motif.
        :param list basis_labels: A list of strings containing the description of the atoms in the motif.
        :param list basis_labels: A list of float between 0. and 1. (default 0.1) to sale the atoms in the motif.
        :param list basis_colors: A list of vtk colors of the atoms in the motif.
        '''
        self._lattice = lattice
        if basis == None:
            # default to one atom at (0, 0, 0)
            self._basis = [(0., 0., 0.)]
            self._labels = ['?']
            self._sizes = [0.1]
            self._colors = [(0., 0., 1.)]
        else:
            self._basis = basis
            self._labels = basis_labels
            self._sizes = basis_sizes
            self._colors = basis_colors


class CrystallinePhase:

    def __init__(self, phase_id=1):
        """Create a new crystalline phase.

        The `phase_id` attribute is used to identify the phase in data sets
        where it can be referred to in phase_map for instance."""
        self.phase_id = phase_id
        self.name = ''
        self.description = ''
        self.formula = ''
        # symmetry should be an instance of Symmetry
        self.symmetry = None
        # lattice constants: a, b, c, alpha, beta, gamma (degrees)
        self.lattice_parameters = [0, 0, 0, 0, 0, 0]
        # a list of C_IJ values
        self.elastic_constants = []


class Symmetry(enum.Enum):
    """
    Class to describe crystal symmetry defined by its Laue class symbol.
    """
    cubic = 'm3m'
    hexagonal = '6/mmm'
    orthorhombic = 'mmm'
    tetragonal = '4/mmm'
    trigonal = 'bar3m'
    monoclinic = '2/m'
    triclinic = 'bar1'

    @staticmethod
    def from_string(s):
        if s == 'cubic':
            return Symmetry.cubic
        elif s == 'hexagonal':
            return Symmetry.hexagonal
        elif s == 'orthorhombic':
            return Symmetry.orthorhombic
        elif s == 'tetragonal':
            return Symmetry.tetragonal
        elif s == 'trigonal':
            return Symmetry.trigonal
        elif s == 'monoclinic':
            return Symmetry.monoclinic
        elif s == 'triclinic':
            return Symmetry.triclinic
        else:
            return None

    def to_string(self):
        if self is Symmetry.cubic:
            return 'cubic'
        elif self is Symmetry.hexagonal:
            return 'hexagonal'
        elif self is Symmetry.orthorhombic:
            return 'orthorhombic'
        elif self is Symmetry.tetragonal:
            return 'tetragonal'
        elif self is Symmetry.trigonal:
            return 'trigonal'
        elif self is Symmetry.monoclinic:
            return 'monoclinic'
        elif self is Symmetry.triclinic:
            return 'triclinic'
        else:
            return None

    def symmetry_operators(self, use_miller_bravais=False):
        """Define the equivalent crystal symmetries.

        Those come from Randle & Engler, 2000. For instance in the cubic
        crystal struture, for instance there are 24 equivalent cube orientations.

        :returns array: A numpy array of shape (n, 3, 3) where n is the \
        number of symmetries of the given crystal structure.
        """
        if self is Symmetry.cubic:
            sym = np.zeros((24, 3, 3), dtype=np.float)
            sym[0] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            sym[1] = np.array([[0., 0., -1.], [0., -1., 0.], [-1., 0., 0.]])
            sym[2] = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
            sym[3] = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
            sym[4] = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
            sym[5] = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
            sym[6] = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
            sym[7] = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
            sym[8] = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
            sym[9] = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
            sym[10] = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
            sym[11] = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
            sym[12] = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
            sym[13] = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
            sym[14] = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
            sym[15] = np.array([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]])
            sym[16] = np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])
            sym[17] = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
            sym[18] = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
            sym[19] = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
            sym[20] = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
            sym[21] = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
            sym[22] = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
            sym[23] = np.array([[-1., 0., 0.], [0., 0., -1.], [0., -1., 0.]])
        elif self is Symmetry.hexagonal:
            if use_miller_bravais:
              # using the Miller-Bravais representation here
              sym = np.zeros((12, 4, 4), dtype=np.int)
              sym[0] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
              sym[1] = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
              sym[2] = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
              sym[3] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
              sym[4] = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1]])
              sym[5] = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, -1]])
              sym[6] = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
              sym[7] = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
              sym[8] = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
              sym[9] = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
              sym[10] = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, -1]])
              sym[11] = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, -1]])
            else:
              sym = np.zeros((12, 3, 3), dtype=np.float)
              s60 = np.sin(60 * np.pi / 180)
              sym[0] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
              sym[1] = np.array([[0.5, s60, 0.], [-s60, 0.5, 0.], [0., 0., 1.]])
              sym[2] = np.array([[-0.5, s60, 0.], [-s60, -0.5, 0.], [0., 0., 1.]])
              sym[3] = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
              sym[4] = np.array([[-0.5, -s60, 0.], [s60, -0.5, 0.], [0., 0., 1.]])
              sym[5] = np.array([[0.5, -s60, 0.], [s60, 0.5, 0.], [0., 0., 1.]])
              sym[6] = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
              sym[7] = np.array([[0.5, s60, 0.], [s60, -0.5, 0.], [0., 0., -1.]])
              sym[8] = np.array([[-0.5, s60, 0.], [s60, 0.5, 0.], [0., 0., -1.]])
              sym[9] = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
              sym[10] = np.array([[-0.5, -s60, 0.], [-s60, 0.5, 0.], [0., 0., -1.]])
              sym[11] = np.array([[0.5, -s60, 0.], [-s60, -0.5, 0.], [0., 0., -1.]])
        elif self is Symmetry.orthorhombic:
            sym = np.zeros((4, 3, 3), dtype=np.float)
            sym[0] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            sym[1] = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
            sym[2] = np.array([[-1., 0., -1.], [0., 1., 0.], [0., 0., -1.]])
            sym[3] = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
        elif self is Symmetry.tetragonal:
            sym = np.zeros((8, 3, 3), dtype=np.float)
            sym[0] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            sym[1] = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
            sym[2] = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
            sym[3] = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
            sym[4] = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
            sym[5] = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
            sym[6] = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
            sym[7] = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
        elif self is Symmetry.triclinic:
            sym = np.zeros((1, 3, 3), dtype=np.float)
            sym[0] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        else:
            raise ValueError('warning, symmetry not supported: %s' % self)
        return sym

    def move_rotation_to_FZ(self, g, verbose=False):
        """
        Compute the rotation matrix in the Fundamental Zone of a given `Symmetry` instance.

        :param g: a 3x3 matrix representing the rotation 
        :param verbose: flag for verbose mode
        :return: a new 3x3 matrix for the rotation in the fundamental zone.
        """
        omegas = []  # list to store all the rotation angles
        syms = self.symmetry_operators()
        for sym in syms:
            # apply the symmetry operator
            om = np.dot(sym, g)
            if verbose:
                print(om)
                print(om.trace())
            # compute the Rodrigues vector of the corresponding orientation matrix
            # from pymicro.crystal.microstructure import Orientation
            # r = Orientation.OrientationMatrix2Rodrigues(om)
            # print(r)
            # and then the rotation angle
            # omega = 2 * np.arctan(np.linalg.norm(r)) * 180 / np.pi
            # todo: check if we can avoid computing the R vector
            cw = 0.5 * (om.trace() - 1)
            omega = np.arccos(cw)
            omegas.append(omega)
        index = np.argmin(omegas)
        if verbose:
            print(omegas)
            print('moving to FZ, index = %d' % index)
        return np.dot(syms[index], g)

    def stiffness_matrix(self, elastic_constants):
        """Build the stiffness matrix for this symmetry.

        :param list elastic_constants: the elastic constants (the number must
        correspond to the type of symmetry, eg 3 for cubic).
        :return ndarray: a numpy array of shape (6, 6) representing
        the stiffness matrix.
        """
        if self is Symmetry.cubic:
            if len(elastic_constants) is not 3:
                raise ValueError('Error: need 3 elastic constants for cubic '
                                 'symmetry, got %d' % len(elastic_constants))
            C11, C12, C44 = elastic_constants
            C = np.array([[C11, C12, C12,   0,   0,   0],
                          [C12, C11, C12,   0,   0,   0],
                          [C12, C12, C11,   0,   0,   0],
                          [  0,   0,   0, C44,   0,   0],
                          [  0,   0,   0,   0, C44,   0],
                          [  0,   0,   0,   0,   0, C44]])
            return C
        elif self is Symmetry.hexagonal:
            if len(elastic_constants) is not 5:
                raise ValueError('Error: need 5 elastic constants for hexagonal '
                                 'symmetry, got %d' % len(elastic_constants))
                C11, C12, C13, C33, C44 = elastic_constants
                C66 = (C11 - C12) / 2
                C = np.array([[C11, C12, C13,   0,   0,   0],
                              [C12, C11, C13,   0,   0,   0],
                              [C13, C13, C33,   0,   0,   0],
                              [  0,   0,   0, C44,   0,   0],
                              [  0,   0,   0,   0, C44,   0],
                              [  0,   0,   0,   0,   0, C66]])
                return C
        elif self is Symmetry.tetragonal:
            if len(elastic_constants) is not 6:
                raise ValueError('Error: need 6 elastic constants for tetragonal '
                                 'symmetry, got %d' % len(elastic_constants))
                C11, C12, C13, C33, C44, C66 = elastic_constants
                C = np.array([[C11, C12, C13,   0,   0,   0],
                              [C12, C11, C13,   0,   0,   0],
                              [C13, C13, C33,   0,   0,   0],
                              [  0,   0,   0, C44,   0,   0],
                              [  0,   0,   0,   0, C44,   0],
                              [  0,   0,   0,   0,   0, C66]])
                return C
        elif self is Symmetry.orthorhombic:
            if len(elastic_constants) is not 9:
                raise ValueError('Error: need 9 elastic constants for tetragonal '
                                 'symmetry, got %d' % len(elastic_constants))
                C11, C12, C13, C22, C23, C33, C44, C55, C66 = elastic_constants
                C = np.array([[C11, C12, C13,   0,   0,   0],
                              [C12, C22, C23,   0,   0,   0],
                              [C13, C23, C33,   0,   0,   0],
                              [  0,   0,   0, C44,   0,   0],
                              [  0,   0,   0,   0, C55,   0],
                              [  0,   0,   0,   0,   0, C66]])
                return C
        elif self is Symmetry.monoclinic:
            if len(elastic_constants) is not 13:
                raise ValueError('Error: need 13 elastic constants for monoclinic '
                                 'symmetry, got %d' % len(elastic_constants))
                C11, C12, C13, C16, C22, C23, C26, C33, C36, C44, C45, \
                C55, C66 = elastic_constants
                C = np.array([[C11, C12, C13,   0,   0, C16],
                              [C12, C22, C23,   0,   0, C26],
                              [C13, C23, C33,   0,   0, C36],
                              [  0,   0,   0, C44, C45,   0],
                              [  0,   0,   0, C45, C55, C56],
                              [C16, C26, C36,   0,   0, C66]])
                return C
        elif self is Symmetry.triclinic:
            if len(elastic_constants) is not 21:
                raise ValueError('Error: need 21 elastic constants for triclinic '
                                 'symmetry, got %d' % len(elastic_constants))
                C11, C12, C13, C14, C15, C16, C22, C23, C24, C25, C26, C33, \
                C34, C35, C36, C44, C45, C46, C55, C56, C66 = elastic_constants
                C = np.array([[C11, C12, C13, C14, C15, C16],
                              [C12, C22, C23, C24, C25, C26],
                              [C13, C23, C33, C34, C35, C36],
                              [C14, C24, C34, C44, C45, C46],
                              [C15, C25, C35, C45, C55, C56],
                              [C16, C26, C36, C46, C56, C66]])
                return C
        else:
            raise ValueError('warning, symmetry not supported: %s' % self)


class Lattice:
    '''
    The Lattice class to create one of the 14 Bravais lattices.

    This particular class has been partly inspired from the pymatgen
    project at https://github.com/materialsproject/pymatgen

    Any of the 7 lattice systems (each corresponding to one point group)
    can be easily created and manipulated.

    The lattice centering can be specified to form any of the 14 Bravais
    lattices:

     * Primitive (P): lattice points on the cell corners only (default);
     * Body (I): one additional lattice point at the center of the cell;
     * Face (F): one additional lattice point at the center of each of
       the faces of the cell;
     * Base (A, B or C): one additional lattice point at the center of
       each of one pair of the cell faces.

    ::

      a = 0.352 # FCC Nickel
      l = Lattice.face_centered_cubic(a)
      print(l.volume())

    Addditionnally the point-basis can be controlled to address non
    Bravais lattice cells. It is set to a single atoms at (0, 0, 0) by
    default so that each cell is a Bravais lattice but may be changed to
    something more complex to achieve HCP structure or Diamond structure
    for instance.
    '''

    def __init__(self, matrix, centering='P', symmetry=None):
        '''Create a crystal lattice (unit cell).

        Create a lattice from a 3x3 matrix.
        Each row in the matrix represents one lattice vector.
        '''
        m = np.array(matrix, dtype=np.float64).reshape((3, 3))
        lengths = np.sqrt(np.sum(m ** 2, axis=1))
        angles = np.zeros(3)
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            angles[i] = dot(m[j], m[k]) / (lengths[j] * lengths[k])
        angles = np.arccos(angles) * 180. / pi
        self._angles = angles
        self._lengths = lengths
        self._matrix = m
        self._centering = centering
        self._symmetry = symmetry

    def __eq__(self, other):
        """Override the default Equals behavior.

        The equality of two Lattice objects is based on the equality of their angles, lengths, and centering.
        """
        if not isinstance(other, self.__class__):
            return False
        for i in range(3):
            if self._angles[i] != other._angles[i]:
                return False
            elif self._lengths[i] != other._lengths[i]:
                return False
        if self._centering != other._centering:
            return False
        if self._symmetry != other._symmetry:
            return False
        return True

    def __repr__(self):
        f = lambda x: "%0.1f" % x
        out = ["Lattice", " abc : " + " ".join(map(f, self._lengths)),
               " angles : " + " ".join(map(f, self._angles)),
               " volume : %0.4f" % self.volume(),
               " A : " + " ".join(map(f, self._matrix[0])),
               " B : " + " ".join(map(f, self._matrix[1])),
               " C : " + " ".join(map(f, self._matrix[2]))]
        return "\n".join(out)

    def reciprocal_lattice(self):
        '''Compute the reciprocal lattice.

        The reciprocal lattice defines a crystal in terms of vectors that
        are normal to a plane and whose lengths are the inverse of the
        interplanar spacing. This method computes the three reciprocal
        lattice vectors defined by:

        .. math::

         * a.a^* = 1
         * b.b^* = 1
         * c.c^* = 1
        '''
        [a, b, c] = self._matrix
        V = self.volume()
        astar = np.cross(b, c) / V
        bstar = np.cross(c, a) / V
        cstar = np.cross(a, b) / V
        return [astar, bstar, cstar]

    @property
    def matrix(self):
        """Returns a copy of matrix representing the Lattice."""
        return np.copy(self._matrix)

    def get_symmetry(self):
        """Returns the type of `Symmetry` of the Lattice."""
        return self._symmetry

    @staticmethod
    def symmetry(crystal_structure=Symmetry.cubic, use_miller_bravais=False):
        """Define the equivalent crystal symmetries.

        Those come from Randle & Engler, 2000. For instance in the cubic
        crystal struture, for instance there are 24 equivalent cube orientations.

        :param crystal_structure: an instance of the `Symmetry` class describing the crystal symmetry.
        :raise ValueError: if the given symmetry is not supported.
        :returns array: A numpy array of shape (n, 3, 3) where n is the \
        number of symmetries of the given crystal structure.
        """
        return crystal_structure.symmetry_operators(use_miller_bravais=use_miller_bravais)

    def get_lattice_parameters(self):
        """This function create a list of the independent lattice parameters depending on the symmetry.

        :return: a list of the lattice parameters.
        """
        sym = self.get_symmetry()
        (a, b, c) = self._lengths
        (alpha, beta, gamma) = self._angles
        # craft a list of the lattice parameters
        if sym is Symmetry.cubic:
            parameters = [a]
        elif sym in [Symmetry.hexagonal, Symmetry.trigonal, Symmetry.tetragonal]:
            parameters = [a, c]
        elif sym is Symmetry.orthorhombic:
            parameters = [a, b, c]
        elif sym is Symmetry.monoclinic:
            parameters = [a, b, c, alpha]
        else:
            parameters = [a, b, c, alpha, beta, gamma]
        return parameters

    def guess_symmetry(self):
        """Guess the lattice symmetry from the geometry."""
        (a, b, c) = self._lengths
        (alpha, beta, gamma) = self._angles
        return Lattice.guess_symmetry_from_parameters(a, b, c, alpha, beta, gamma)

    @staticmethod
    def guess_symmetry_from_parameters(a, b, c, alpha, beta, gamma):
        """Guess the lattice symmetry from the geometrical parameters."""
        if alpha == 90. and beta == 90. and gamma == 90:
            if a == b and a == c:
                return Symmetry.cubic
            elif a == b and a != c:
                return Symmetry.tetragonal
            else:
                return Symmetry.orthorhombic
        elif alpha == 90. and beta == 90. and gamma == 120 and a == b and a != c:
            return Symmetry.hexagonal
        elif a == b and a == c and alpha == beta and alpha == gamma:
            return Symmetry.trigonal
        elif a != b and a != c and beta == gamma and alpha != beta:
            return Symmetry.monoclinic
        else:
            return Symmetry.triclinic

    @staticmethod
    def from_cif(file_path):
        """
        Create a crystal Lattice using information contained in a given CIF
        file (Crystallographic Information Framework, a standard for
        information interchange in crystallography).

        Reference: S. R. Hall, F. H. Allen and I. D. Brown,
        The crystallographic information file (CIF): a new standard archive file for crystallography,
        Acta Crystallographica Section A, 47(6):655-685 (1991)
        doi = 10.1107/S010876739101067X

        .. note::

           Lattice constants are given in Angstrom in CIF files and so
           converted to nanometer.

        :param str file_path: The path to the CIF file representing the crystal structure.
        :returns: A `Lattice` instance corresponding to the given CIF file.
        """
        cf = CifFile.ReadCif(file_path)
        # crystal = eval('cf[\'%s\']' % symbol)
        crystal = cf.first_block()
        a = 0.1 * float(crystal['_cell_length_a'])
        b = 0.1 * float(crystal['_cell_length_b'])
        c = 0.1 * float(crystal['_cell_length_c'])
        alpha = float(crystal['_cell_angle_alpha'])
        beta = float(crystal['_cell_angle_beta'])
        gamma = float(crystal['_cell_angle_gamma'])
        try:
            symmetry = Symmetry.from_string(crystal['_symmetry_cell_setting'])
        except KeyError:
            symmetry = Lattice.guess_symmetry_from_parameters(a, b, c, alpha, beta, gamma)
        return Lattice.from_parameters(a, b, c, alpha, beta, gamma, symmetry=symmetry)

    @staticmethod
    def from_symbol(symbol):
        '''
        Create a crystal Lattice using information contained in a unit cell.

        *Parameters*

        **symbol**: The chemical symbol of the crystal (eg 'Al')

        *Returns*

        A `Lattice` instance corresponding to the given element.
        '''
        path = os.path.dirname(__file__)
        return Lattice.from_cif(os.path.join(path, 'cif', '%s.cif' % symbol))

    @staticmethod
    def cubic(a):
        '''
        Create a cubic Lattice unit cell.

        *Parameters*

        **a**: first lattice length parameter (a = b = c here)

        *Returns*

        A `Lattice` instance corresponding to a primitice cubic lattice.
        '''
        return Lattice([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]], symmetry=Symmetry.cubic)

    @staticmethod
    def body_centered_cubic(a):
        '''
        Create a body centered cubic Lattice unit cell.

        *Parameters*

        **a**: first lattice length parameter (a = b = c here)

        *Returns*

        A `Lattice` instance corresponding to a body centered cubic
        lattice.
        '''
        return Lattice.from_parameters(a, a, a, 90, 90, 90, centering='I', symmetry=Symmetry.cubic)

    @staticmethod
    def face_centered_cubic(a):
        '''
        Create a face centered cubic Lattice unit cell.

        *Parameters*

        **a**: first lattice length parameter (a = b = c here)

        *Returns*

        A `Lattice` instance corresponding to a face centered cubic
        lattice.
        '''
        return Lattice.from_parameters(a, a, a, 90, 90, 90, centering='F', symmetry=Symmetry.cubic)

    @staticmethod
    def tetragonal(a, c):
        '''
        Create a tetragonal Lattice unit cell.

        *Parameters*

        **a**: first lattice length parameter

        **c**: third lattice length parameter (b = a here)

        *Returns*

        A `Lattice` instance corresponding to a primitive tetragonal
        lattice.
        '''
        return Lattice.from_parameters(a, a, c, 90, 90, 90, symmetry=Symmetry.tetragonal)

    @staticmethod
    def body_centered_tetragonal(a, c):
        '''
        Create a body centered tetragonal Lattice unit cell.

        *Parameters*

        **a**: first lattice length parameter

        **c**: third lattice length parameter (b = a here)

        *Returns*

        A `Lattice` instance corresponding to a body centered tetragonal
        lattice.
        '''
        return Lattice.from_parameters(a, a, c, 90, 90, 90, centering='I', symmetry=Symmetry.tetragonal)

    @staticmethod
    def orthorhombic(a, b, c):
        '''
        Create a tetragonal Lattice unit cell with 3 different length
        parameters a, b and c.
        '''
        return Lattice.from_parameters(a, b, c, 90, 90, 90, symmetry=Symmetry.orthorhombic)

    @staticmethod
    def base_centered_orthorhombic(a, b, c):
        '''
        Create a based centered orthorombic Lattice unit cell.

        *Parameters*

        **a**: first lattice length parameter

        **b**: second lattice length parameter

        **c**: third lattice length parameter

        *Returns*

        A `Lattice` instance corresponding to a based centered orthorombic
        lattice.
        '''
        return Lattice.from_parameters(a, b, c, 90, 90, 90, centering='C', symmetry=Symmetry.orthorhombic)

    @staticmethod
    def body_centered_orthorhombic(a, b, c):
        '''
        Create a body centered orthorombic Lattice unit cell.

        *Parameters*

        **a**: first lattice length parameter

        **b**: second lattice length parameter

        **c**: third lattice length parameter

        *Returns*

        A `Lattice` instance corresponding to a body centered orthorombic
        lattice.
        '''
        return Lattice.from_parameters(a, b, c, 90, 90, 90, centering='I', symmetry=Symmetry.orthorhombic)

    @staticmethod
    def face_centered_orthorhombic(a, b, c):
        '''
        Create a face centered orthorombic Lattice unit cell.

        *Parameters*

        **a**: first lattice length parameter

        **b**: second lattice length parameter

        **c**: third lattice length parameter

        *Returns*

        A `Lattice` instance corresponding to a face centered orthorombic
        lattice.
        '''
        return Lattice.from_parameters(a, b, c, 90, 90, 90, centering='F', symmetry=Symmetry.orthorhombic)

    @staticmethod
    def hexagonal(a, c):
        '''
        Create a hexagonal Lattice unit cell with length parameters a and c.
        '''
        return Lattice.from_parameters(a, a, c, 90, 90, 120, symmetry=Symmetry.hexagonal)

    @staticmethod
    def rhombohedral(a, alpha):
        '''
        Create a rhombohedral Lattice unit cell with one length
        parameter a and the angle alpha.
        '''
        return Lattice.from_parameters(a, a, a, alpha, alpha, alpha, symmetry=Symmetry.trigonal)

    @staticmethod
    def monoclinic(a, b, c, alpha):
        '''
        Create a monoclinic Lattice unit cell with 3 different length
        parameters a, b and c. The cell angle is given by alpha.
        The lattice centering id primitive ie. 'P'
        '''
        return Lattice.from_parameters(a, b, c, alpha, 90, 90, symmetry=Symmetry.monoclinic)

    @staticmethod
    def base_centered_monoclinic(a, b, c, alpha):
        '''
        Create a based centered monoclinic Lattice unit cell.

        *Parameters*

        **a**: first lattice length parameter

        **b**: second lattice length parameter

        **c**: third lattice length parameter

        **alpha**: first lattice angle parameter

        *Returns*

        A `Lattice` instance corresponding to a based centered monoclinic
        lattice.
        '''
        return Lattice.from_parameters(a, b, c, alpha, 90, 90, centering='C', symmetry=Symmetry.monoclinic)

    @staticmethod
    def triclinic(a, b, c, alpha, beta, gamma):
        '''
        Create a triclinic Lattice unit cell with 3 different length
        parameters a, b, c and three different cell angles alpha, beta
        and gamma.

        ..note::

           This method is here for the sake of completeness since one can
           create the triclinic cell directly using the `from_parameters`
           method.
        '''
        return Lattice.from_parameters(a, b, c, alpha, beta, gamma, symmetry=Symmetry.triclinic)

    @staticmethod
    def from_symmetry(symmetry, parameters):
        """Create a new lattice based on a type of symmetry and a list of lattice parameters.

        The type of symmetry should be an instance of `Symmetry` and the list of parameters should contain the
        appropriate number: 1 for cubic, 2 for hexagonal, tetragonal or trigonal, 3 for orthorhombic, 4 for monoclinic
        or 6 for triclinic.

        :param symmetry: an instance of `Symmetry`.
        :param list parameters: a list of the lattice parameters.
        :return: the newly created `Lattice` instance.
        """
        if symmetry is Symmetry.cubic:
            if len(parameters) != 1:
                raise(ValueError('The number of parameters for %s symmetry should be 1, got %d' % (symmetry, len(parameters))))
            return Lattice.cubic(parameters[0])
        elif symmetry in [Symmetry.hexagonal, Symmetry.trigonal]:
            if len(parameters) != 2:
                raise(ValueError('The number of parameters for %s symmetry should be 2, got %d' % (symmetry, len(parameters))))
            return Lattice.hexagonal(parameters[0], parameters[1])
        elif symmetry is Symmetry.orthorhombic:
            if len(parameters) != 3:
                raise(ValueError('The number of parameters for %s symmetry should be 3, got %d' % (symmetry, len(parameters))))
            return Lattice.orthorhombic(parameters[0], parameters[1], parameters[2])
        elif symmetry is Symmetry.tetragonal:
            if len(parameters) != 2:
                raise(ValueError('The number of parameters for %s symmetry should be 2, got %d' % (symmetry, len(parameters))))
            return Lattice.tetragonal(parameters[0], parameters[1])
        elif symmetry is Symmetry.monoclinic:
            if len(parameters) != 4:
                raise(ValueError('The number of parameters for %s symmetry should be 4, got %d' % (symmetry, len(parameters))))
            return Lattice.monoclinic(parameters[0], parameters[1], parameters[2], parameters[3])
        else:
            if len(parameters) != 6:
                raise(ValueError('The number of parameters for triclinic symmetry should be 6, got %d' % len(parameters)))
            return Lattice.triclinic(*parameters)

    @staticmethod
    def from_parameters(a, b, c, alpha, beta, gamma, x_aligned_with_a=True, centering='P', symmetry=Symmetry.triclinic):
        """
        Create a Lattice using unit cell lengths and angles (in degrees).
        The lattice centering can also be specified (among 'P', 'I', 'F',
        'A', 'B' or 'C').

        :param float a: first lattice length parameter.
        :param float b: second lattice length parameter.
        :param float c: third lattice length parameter.
        :param float alpha: first lattice angle parameter.
        :param float beta: second lattice angle parameter.
        :param float gamma: third lattice angle parameter.
        :param bool x_aligned_with_a: flag to control the convention used to define the Cartesian frame.
        :param str centering: lattice centering ('P' by default) passed to the `Lattice` class.
        :param symmetry: a `Symmetry` instance to be passed to the lattice.
        :return: A `Lattice` instance with the specified lattice parameters and centering.
        """
        alpha_r = radians(alpha)
        beta_r = radians(beta)
        gamma_r = radians(gamma)
        if x_aligned_with_a:  # first lattice vector (a) is aligned with X
            vector_a = a * np.array([1, 0, 0])
            vector_b = b * np.array([np.cos(gamma_r), np.sin(gamma_r), 0])
            c1 = c * np.cos(beta_r)
            c2 = c * (np.cos(alpha_r) - np.cos(gamma_r) * np.cos(beta_r)) / np.sin(gamma_r)
            vector_c = np.array([c1, c2, np.sqrt(c ** 2 - c1 ** 2 - c2 ** 2)])
        else:  # third lattice vector (c) is aligned with Z
            cos_gamma_star = (np.cos(alpha_r) * np.cos(beta_r) - np.cos(gamma_r)) / (np.sin(alpha_r) * np.sin(beta_r))
            sin_gamma_star = np.sqrt(1 - cos_gamma_star ** 2)
            vector_a = [a * np.sin(beta_r), 0.0, a * np.cos(beta_r)]
            vector_b = [-b * np.sin(alpha_r) * cos_gamma_star, b * np.sin(alpha_r) * sin_gamma_star, b * np.cos(alpha_r)]
            vector_c = [0.0, 0.0, float(c)]
        return Lattice([vector_a, vector_b, vector_c], centering=centering, symmetry=symmetry)

    def volume(self):
        """Compute the volume of the unit cell."""
        m = self._matrix
        return abs(np.dot(np.cross(m[0], m[1]), m[2]))

    def get_hkl_family(self, hkl):
        """Get a list of the hkl planes composing the given family for
        this crystal lattice.

        *Parameters*

        **hkl**: miller indices of the requested family

        *Returns*

        A list of the hkl planes in the given family.
        """
        planes = HklPlane.get_family(hkl, lattice=self, crystal_structure=self._symmetry)
        return planes


class SlipSystem:
    '''A class to represent a crystallographic slip system.

    A slip system is composed of a slip plane (most widely spaced planes
    in the crystal) and a slip direction (highest linear density of atoms
    in the crystal).
    '''

    def __init__(self, plane, direction):
        '''Create a new slip system object with the given slip plane and
        slip direction.
        '''
        self._plane = plane
        self._direction = direction

    def __repr__(self):
        out = '(%d%d%d)' % self._plane.miller_indices()
        out += '[%d%d%d]' % self._direction.miller_indices()
        return out

    def get_slip_plane(self):
        return self._plane

    def get_slip_direction(self):
        return self._direction

    @staticmethod
    def from_indices(plane_indices, direction_indices, lattice=None):
        """create a slip system from the indices of the plane and the direction.

        This method create a `SlipSystem` instance by associating a slip plane
        and a slip direction both given by their Miller indices.

        :param tuple plane_indices: the miller indices for the slip plane.
        :param tuple direction_indices: the miller indices for the slip direction.
        :param Lattice lattice: the crystal lattice.
        :return: the new `SlipSystem` instance.
        """
        if len(plane_indices) == 4:
            plane_indices = HklPlane.four_to_three_indices(plane_indices)
        plane = HklPlane(*plane_indices, lattice)
        if len(plane_indices) == 4:
            direction_indices = HklDirection.four_to_three_indices(direction_indices)
        direction = HklDirection(*direction_indices, lattice)
        return SlipSystem(plane, direction)

    @staticmethod
    def get_slip_systems(slip_type='oct', lattice=None):
        """Get all slip systems for a given hkl plane family.

        A string is used to describe the slip system type:
         * cube or 001, for [110] slip in (001) planes
         * oct or 111 for [110] slip in (111) planes
         * 112 for [111] slip in (112) planes
         * basal for [11-20] slip in (0001) planes (hexagonal)
         * prism for [11-20] slip in (1-100) planes (hexagonal)

        :param str slip_type: a string describing the slip system type.
        :return list: a list of :py:class:`~pymicro.crystal.lattice.SlipSystem`.
        """
        slip_systems = []
        if slip_type in ['cube', '001']:
            slip_systems.append(SlipSystem.from_indices((0, 0, 1), (-1, 1, 0), lattice))  # E5
            slip_systems.append(SlipSystem.from_indices((0, 0, 1), (1, 1, 0), lattice))  # E6
            slip_systems.append(SlipSystem.from_indices((1, 0, 0), (0, 1, 1), lattice))  # F1
            slip_systems.append(SlipSystem.from_indices((1, 0, 0), (0, -1, 1), lattice))  # F2
            slip_systems.append(SlipSystem.from_indices((0, 1, 0), (-1, 0, 1), lattice))  # G4
            slip_systems.append(SlipSystem.from_indices((0, 1, 0), (1, 0, 1), lattice))  # G3
        elif slip_type in ['oct', '111']:
            slip_systems.append(SlipSystem.from_indices((1, 1, 1), (-1, 0, 1), lattice))  # Bd
            slip_systems.append(SlipSystem.from_indices((1, 1, 1), (0, -1, 1), lattice))  # Ba
            slip_systems.append(SlipSystem.from_indices((1, 1, 1), (-1, 1, 0), lattice))  # Bc
            slip_systems.append(SlipSystem.from_indices((1, -1, 1), (-1, 0, 1), lattice))  # Db
            slip_systems.append(SlipSystem.from_indices((1, -1, 1), (0, 1, 1), lattice))  # Dc
            slip_systems.append(SlipSystem.from_indices((1, -1, 1), (1, 1, 0), lattice))  # Da
            slip_systems.append(SlipSystem.from_indices((-1, 1, 1), (0, -1, 1), lattice))  # Ab
            slip_systems.append(SlipSystem.from_indices((-1, 1, 1), (1, 1, 0), lattice))  # Ad
            slip_systems.append(SlipSystem.from_indices((-1, 1, 1), (1, 0, 1), lattice))  # Ac
            slip_systems.append(SlipSystem.from_indices((1, 1, -1), (-1, 1, 0), lattice))  # Cb
            slip_systems.append(SlipSystem.from_indices((1, 1, -1), (1, 0, 1), lattice))  # Ca
            slip_systems.append(SlipSystem.from_indices((1, 1, -1), (0, 1, 1), lattice))  # Cd
        elif slip_type == '112':
            slip_systems.append(SlipSystem.from_indices((1, 1, 2), (1, 1, -1), lattice))
            slip_systems.append(SlipSystem.from_indices((-1, 1, 2), (1, -1, 1), lattice))
            slip_systems.append(SlipSystem.from_indices((1, -1, 2), (-1, 1, 1), lattice))
            slip_systems.append(SlipSystem.from_indices((1, 1, -2), (1, 1, 1), lattice))
            slip_systems.append(SlipSystem.from_indices((1, 2, 1), (1, -1, 1), lattice))
            slip_systems.append(SlipSystem.from_indices((-1, 2, 1), (1, 1, -1), lattice))
            slip_systems.append(SlipSystem.from_indices((1, -2, 1), (1, 1, 1), lattice))
            slip_systems.append(SlipSystem.from_indices((1, 2, -1), (-1, 1, 1), lattice))
            slip_systems.append(SlipSystem.from_indices((2, 1, 1), (-1, 1, 1), lattice))
            slip_systems.append(SlipSystem.from_indices((-2, 1, 1), (1, 1, 1), lattice))
            slip_systems.append(SlipSystem.from_indices((2, -1, 1), (1, 1, -1), lattice))
            slip_systems.append(SlipSystem.from_indices((2, 1, -1), (1, -1, 1), lattice))
        elif slip_type == 'basal':
            p_basal = HklPlane(0, 0, 1, lattice)  # basal plane
            # basal slip systems
            bss1 = SlipSystem(p_basal, HklDirection(*HklDirection.four_to_three_indices(2, -1, -1, 0), lattice))
            bss2 = SlipSystem(p_basal, HklDirection(*HklDirection.four_to_three_indices(-1, 2, -1, 0), lattice))
            bss3 = SlipSystem(p_basal, HklDirection(*HklDirection.four_to_three_indices(-1, -1, 2, 0), lattice))
            slip_systems = [bss1, bss2, bss3]
        elif slip_type == 'prism':
            p_prism1 = HklPlane(0, 1, 0, lattice)
            p_prism2 = HklPlane(-1, 0, 0, lattice)
            p_prism3 = HklPlane(-1, 1, 0, lattice)
            # prismatic slip systems
            pss1 = SlipSystem(p_prism1, HklDirection(*HklDirection.four_to_three_indices(2, -1, -1, 0), lattice))
            pss2 = SlipSystem(p_prism2, HklDirection(*HklDirection.four_to_three_indices(-1, 2, -1, 0), lattice))
            pss3 = SlipSystem(p_prism3, HklDirection(*HklDirection.four_to_three_indices(-1, -1, 2, 0), lattice))
            slip_systems = [pss1, pss2, pss3]
        else:
            print('unsupported slip system type: %s, try one of (cube, oct, 112, basal, prism)' % slip_type)
        return slip_systems


class HklObject:
    def __init__(self, h, k, l, lattice=None):
        '''Create a new hkl object with the given Miller indices and
           crystal lattice.
        '''
        if lattice == None:
            lattice = Lattice.cubic(1.0)
        self._lattice = lattice
        self._h = h
        self._k = k
        self._l = l

    @property
    def lattice(self):
        return self._lattice

    def set_lattice(self, lattice):
        """Assign a new `Lattice` to this instance.

        :param lattice: the new crystal lattice.
        """
        self._lattice = lattice

    @property
    def h(self):
        return self._h

    @property
    def k(self):
        return self._k

    @property
    def l(self):
        return self._l

    def miller_indices(self):
        '''
        Returns an immutable tuple of the plane Miller indices.
        '''
        return (self._h, self._k, self._l)

    @staticmethod
    def skip_higher_order(hkl_list, keep_friedel_pair=False, verbose=False):
        """Create a copy of a list of some hkl object retaining only the first order.

        :param list hkl_list: The list of `HklObject`.
        :param bool keep_friedel_pair: flag to keep order -1 in the list. 
        :param bool verbose: activate verbose mode.
        :returns list: A new list of :py:class:`~pymicro.crystal.lattice.HklObject` without any multiple reflection.
        """
        # create array with all the miller indices
        hkl_array = np.empty((len(hkl_list), 3), dtype=int)
        for i in range(len(hkl_list)):
            hkl = hkl_list[i]
            hkl_array[i] = np.array(hkl.miller_indices())
        # first start by ordering the HklObjects by ascending miller indices sum
        hkl_sum = np.sum(np.abs(hkl_array), axis=1)
        hkl_sum_sort = np.argsort(hkl_sum)
        first_order_list = [hkl_list[hkl_sum_sort[0]]]
        for i in range(1, len(hkl_sum_sort)):
            hkl_next = hkl_sum_sort[i]
            hkl = hkl_list[hkl_next]
            (h, k, l) = hkl.miller_indices()
            if verbose:
                print('trying hkl object (%d, %d, %d)' % (h, k, l))
            lower = False
            # check if a hkl object already exist in the list
            for uvw in first_order_list:
                # try to assess the multiple from the sum of miller indices
                (u, v, w) = uvw.miller_indices()
                if verbose:
                    print('looking at: (%d, %d, %d)' % (u, v, w))
                n = hkl_sum[hkl_next] / np.sum(np.abs(np.array((u, v, w))), axis=0)
                for order in [-n, n]:
                    if keep_friedel_pair and order == -1:
                        if verbose:
                            print('keeping Friedel pair reflexion: (%d, %d, %d) with n=%d' % (u, v, w, order))
                        continue
                    if (u * order == h) and (v * order == k) and (w * order) == l:
                        if verbose:
                            print('lower order reflexion was found: (%d, %d, %d) with n=%d' % (u, v, w, order))
                        lower = True
                        break
            # if no lower order reflexion was found, add the hkl object to the list
            if not lower:
                if verbose:
                    print('adding hkl object (%d, %d, %d) to the list' % (h, k, l))
                first_order_list.append(hkl)
        return first_order_list


class HklDirection(HklObject):
    def __repr__(self):
        f = lambda x: "%0.3f" % x
        out = ['HKL Direction',
               ' Miller indices:',
               ' h : ' + str(self._h),
               ' k : ' + str(self._k),
               ' l : ' + str(self._l),
               ' crystal lattice : ' + str(self._lattice)]
        return '\n'.join(out)

    def direction(self):
        '''Returns a normalized vector, expressed in the cartesian
        coordinate system, corresponding to this crystallographic direction.
        '''
        (h, k, l) = self.miller_indices()
        M = self._lattice.matrix.T  # the columns of M are the a, b, c vector in the cartesian coordinate system
        l_vect = M.dot(np.array([h, k, l]))
        return l_vect / np.linalg.norm(l_vect)

    def angle_with_direction(self, hkl):
        '''Computes the angle between this crystallographic direction and
        the given direction (in radian).'''
        return np.arccos(np.dot(self.direction(), hkl.direction()))

    @staticmethod
    def angle_between_directions(hkl1, hkl2, lattice=None):
        '''Computes the angle between two crystallographic directions (in radian).

        :param tuple hkl1: The triplet of the miller indices of the first direction.
        :param tuple hkl2: The triplet of the miller indices of the second direction.
        :param Lattice lattice: The crystal lattice, will default to cubic if not specified.

        :returns float: The angle in radian.
        '''
        d1 = HklDirection(*hkl1, lattice=lattice)
        d2 = HklDirection(*hkl2, lattice=lattice)
        return d1.angle_with_direction(d2)

    @staticmethod
    def three_to_four_indices(u, v, w):
        """Convert from Miller indices to Miller-Bravais indices. this is used for hexagonal crystal lattice."""
        return (2 * u - v) / 3., (2 * v - u) / 3., -(u + v) / 3., w

    @staticmethod
    def four_to_three_indices(U, V, T, W):
        """Convert from Miller-Bravais indices to Miller indices. this is used for hexagonal crystal lattice."""
        import fractions
        u, v, w = U - T, V - T, W
        gcd = functools.reduce(fractions.gcd, (u, v, w))
        return u / gcd, v / gcd, w / gcd

    @staticmethod
    def angle_between_4indices_directions(hkil1, hkil2, ac):
        """Computes the angle between two crystallographic directions in a hexagonal lattice.

        The solution was derived by F. Frank in:
        On Miller - Bravais indices and four dimensional vectors. Acta Cryst. 18, 862-866 (1965)

        :param tuple hkil1: The quartet of the indices of the first direction.
        :param tuple hkil2: The quartet of the indices of the second direction.
        :param tuple ac: the lattice parameters of the hexagonal structure in the form (a, c).
        :returns float: The angle in radian.
        """
        h1, k1, i1, l1 = hkil1
        h2, k2, i2, l2 = hkil2
        a, c = ac
        lambda_square = 2. / 3 * (c / a) ** 2
        value = (h1 * h2 + k1 * k2 + i1 * i2 + lambda_square * l1 * l2) / \
                (np.sqrt(h1 ** 2 + k1 ** 2 + i1 ** 2 + lambda_square * l1 ** 2) *
                 np.sqrt(h2 ** 2 + k2 ** 2 + i2 ** 2 + lambda_square * l2 ** 2))
        return np.arccos(value)

    def find_planes_in_zone(self, max_miller=5):
        '''
        This method finds the hkl planes in zone with the crystallographic
        direction. If (u,v,w) denotes the zone axis, this means finding all
        hkl planes which verify :math:`h.u + k.v + l.w = 0`.

        :param max_miller: The maximum miller index to limt the search`
        :returns list: A list of :py:class:`~pymicro.crystal.lattice.HklPlane` objects \
        describing all the planes in zone with the direction.
        '''
        (u, v, w) = self.miller_indices()
        indices = range(-max_miller, max_miller + 1)
        hklplanes_in_zone = []
        for h in indices:
            for k in indices:
                for l in indices:
                    if h == k == l == 0:  # skip (0, 0, 0)
                        continue
                    if np.dot(np.array([h, k, l]), np.array([u, v, w])) == 0:
                        hklplanes_in_zone.append(HklPlane(h, k, l, self._lattice))
        return hklplanes_in_zone


class HklPlane(HklObject):
    '''
    This class define crystallographic planes using Miller indices.

    A plane can be create by speficying its Miller indices and the
    crystal lattice (default is cubic with lattice parameter of 1.0).

    ::

      a = 0.405 # FCC Aluminium
      l = Lattice.cubic(a)
      p = HklPlane(1, 1, 1, lattice=l)
      print(p)
      print(p.scattering_vector())
      print(p.interplanar_spacing())

    .. note::

      Miller indices are defined in terms of the inverse of the intercept
      of the plane on the three crystal axes a, b, and c.
    '''

    def __eq__(self, other):
        """Override the default Equals behavior.

        The equality of two HklObjects is based on the equality of their miller indices.
        """
        if isinstance(other, self.__class__):
            return self._h == other._h and self._k == other._k and \
                   self._l == other._l and self._lattice == other._lattice
        return False

    def __ne__(self, other):
        """Define a non-equality test"""
        return not self.__eq__(other)

    def normal(self):
        '''Returns the unit vector normal to the plane.

        We use of the repiprocal lattice to compute the normal to the plane
        and return a normalised vector.
        '''
        n = self.scattering_vector()
        return n / np.linalg.norm(n)

    def scattering_vector(self):
        '''Calculate the scattering vector of this `HklPlane`.

        The scattering vector (or reciprocal lattice vector) is normal to
        this `HklPlane` and its length is equal to the inverse of the
        interplanar spacing. In the cartesian coordinate system of the
        crystal, it is given by:

        ..math

          G_c = h.a^* + k.b^* + l.c^*

        :returns: a numpy vector expressed in the cartesian coordinate system of the crystal.
        '''
        [astar, bstar, cstar] = self._lattice.reciprocal_lattice()
        (h, k, l) = self.miller_indices()
        # express (h, k, l) in the cartesian crystal CS
        Gc = h * astar + k * bstar + l * cstar
        return Gc

    def __repr__(self):
        f = lambda x: "%0.3f" % x
        out = ['HKL Plane',
               ' Miller indices:',
               ' h : ' + str(self._h),
               ' k : ' + str(self._k),
               ' l : ' + str(self._l),
               ' plane normal : ' + str(self.normal()),
               ' crystal lattice : ' + str(self._lattice)]
        return '\n'.join(out)

    def friedel_pair(self):
        """Create the Friedel pair of the HklPlane."""
        (h, k, l) = self.miller_indices()
        pair = HklPlane(-h, -k, -l, self._lattice)
        return pair

    def interplanar_spacing(self):
        '''
        Compute the interplanar spacing.
        For cubic lattice, it is:

        .. math::

           d = a / \sqrt{h^2 + k^2 + l^2}

        The general formula comes from 'Introduction to Crystallography'
        p. 68 by Donald E. Sands.
        '''
        (a, b, c) = self._lattice._lengths
        (h, k, l) = self.miller_indices()
        (alpha, beta, gamma) = radians(self._lattice._angles)
        # d = a / np.sqrt(h**2 + k**2 + l**2) # for cubic structure only
        d = self._lattice.volume() / np.sqrt(h ** 2 * b ** 2 * c ** 2 * np.sin(alpha) ** 2 + \
                                             k ** 2 * a ** 2 * c ** 2 * np.sin(
                                                 beta) ** 2 + l ** 2 * a ** 2 * b ** 2 * np.sin(gamma) ** 2 + \
                                             2 * h * l * a * b ** 2 * c * (
                                                 np.cos(alpha) * np.cos(gamma) - np.cos(beta)) + \
                                             2 * h * k * a * b * c ** 2 * (
                                                 np.cos(alpha) * np.cos(beta) - np.cos(gamma)) + \
                                             2 * k * l * a ** 2 * b * c * (
                                                 np.cos(beta) * np.cos(gamma) - np.cos(alpha)))
        return d

    def bragg_angle(self, lambda_keV, verbose=False):
        '''Compute the Bragg angle for this `HklPlane` at the given energy.

        .. note::

          For this calculation to work properly, the lattice spacing needs
          to be in nm units.
        '''
        d = self.interplanar_spacing()
        lambda_nm = 1.2398 / lambda_keV
        theta = np.arcsin(lambda_nm / (2 * d))
        if verbose:
            theta_deg = 180 * theta / np.pi
            (h, k, l) = self.miller_indices()
            print('\nBragg angle for %d%d%d at %.1f keV is %.1f deg\n' % (h, k, l, lambda_keV, theta_deg))
        return theta

    @staticmethod
    def four_to_three_indices(U, V, T, W):
        """Convert four to three index representation of a slip plane (used for hexagonal crystal lattice)."""
        #return (6 * h / 5. - 3 * k / 5., 3 * h / 5. + 6 * k / 5., l)
        return U, V, W

    @staticmethod
    def three_to_four_indices(u, v, w):
        """Convert three to four index representation of a slip plane (used for hexagonal crystal lattice)."""
        return u, v, -(u + v), w

    def is_in_list(self, hkl_planes, friedel_pair=False):
        """Check if the hkl plane is in the given list.

        By default this relies on the built in in test from the list type which in turn calls in the __eq__ method.
        This means it will return True if a plane with the exact same miller indices (and same lattice) is in the list.
        Turning on the friedel_pair flag will allow to test also the Friedel pair (-h, -k, -l) and return True if it is
        in the list.
        For instance (0,0,1) and (0,0,-1) are in general considered as the same lattice plane.
        """
        if not friedel_pair:
            return self in hkl_planes
        else:
            return self in hkl_planes or self.friedel_pair() in hkl_planes

    @staticmethod
    def is_same_family(hkl1, hkl2, crystal_structure=Symmetry.cubic):
        """Static method to test if both lattice planes belongs to the same family.
        
        A family {hkl} is composed by all planes that are equivalent to (hkl) using the symmetry of the lattice. 
        The lattice assoiated with `hkl2`is not taken into account here.
        """
        return hkl1.is_in_list(HklPlane.get_family(hkl2.miller_indices(), lattice=hkl1._lattice,
                                                   crystal_structure=crystal_structure))

    @staticmethod
    def get_family(hkl, lattice=None, include_friedel_pairs=False, crystal_structure=Symmetry.cubic):
        """Static method to obtain a list of the different crystallographic
        planes in a particular family.

        :param str hkl: a sequence of 3 (4 for hexagonal) numbers corresponding to the miller indices.
        :param Lattice lattice: The reference crystal lattice (default None).
        :param bool include_friedel_pairs: Flag to include the Friedel pairs in the list (False by default).
        :param str crystal_structure: A string descibing the crystal structure (cubic by default). 
        :raise ValueError: if the given string does not correspond to a supported family.
        :returns list: a list of the :py:class:`~pymicro.crystal.lattice.HklPlane` in the given hkl family.

        .. note::

          The method account for the lattice symmetry to create a list of equivalent lattice plane from the point 
          of view of the point group symmetry. A flag can be used to include or not the Friedel pairs. If not, the 
          family is contstructed using the miller indices limited the number of minus signs. For instance  (1,0,0) 
          will be in the list and not (-1,0,0).
        """
        if not (len(hkl) == 3 or (len(hkl) == 4 and crystal_structure == Symmetry.hexagonal)):
            raise ValueError('warning, family not supported: {}'.format(hkl))
        # handle hexagonal case
        if len(hkl) == 4:
            h = int(hkl[0])
            k = int(hkl[1])
            i = int(hkl[2])
            l = int(hkl[3])
            (h, k, l) = HklPlane.four_to_three_indices(h, k, i, l)  # useless as it just drops i
        else:  # 3 indices
            h = int(hkl[0])
            k = int(hkl[1])
            l = int(hkl[2])
            if crystal_structure == Symmetry.hexagonal:
                i = -(h + k)
        family = []
        # construct lattice plane family from the symmetry operators
        if crystal_structure == Symmetry.hexagonal:
          syms = Lattice.symmetry(crystal_structure, use_miller_bravais=True)  # use 4x4 symmetry operators
        else:
          syms = Lattice.symmetry(crystal_structure)
        for sym in syms:
            if crystal_structure == Symmetry.hexagonal:
                n_sym = np.dot(sym, np.array([h, k, i, l]))
                n_sym = HklPlane.four_to_three_indices(*n_sym)
            else:  # 3 indices
                n_sym = np.dot(sym, np.array([h, k, l]))
            hkl_sym = HklPlane(*n_sym, lattice=lattice)
            if not hkl_sym.is_in_list(family, friedel_pair=True):
                family.append(hkl_sym)
            if include_friedel_pairs:
                hkl_sym = HklPlane(-n_sym[0], -n_sym[1], -n_sym[2], lattice=lattice)
                if not hkl_sym.is_in_list(family, friedel_pair=False):
                    family.append(hkl_sym)
        if not include_friedel_pairs:
            # for each hkl plane chose between (h, k, l) and (-h, -k, -l) to have the less minus signs
            for i in range(len(family)):
                hkl = family[i]
                (h, k, l) = hkl.miller_indices()
                if np.where(np.array([h, k, l]) < 0)[0].size > 0 and np.where(np.array([h, k, l]) <= 0)[0].size >= 2:
                    family[i] = hkl.friedel_pair()
                    #print('replacing plane (%d%d%d) by its pair: (%d%d%d)' % (h, k, l, -h, -k, -l))


        '''
        if not len(hkl) == 3:
            raise ValueError('warning, family not supported: %s' % hkl)
        family = []
        hkl_list = list(hkl)
        hkl_list.sort()  # miller indices are now sorted by increasing order
        h = int(hkl[0])
        k = int(hkl[1])
        l = int(hkl[2])
        if hkl == '001' or hkl == '010' or hkl == '100':
            family.append(HklPlane(1, 0, 0, lattice))
            family.append(HklPlane(0, 1, 0, lattice))
            family.append(HklPlane(0, 0, 1, lattice))
        elif hkl in ['011', '101', '110']:
            family.append(HklPlane(1, 1, 0, lattice))
            family.append(HklPlane(-1, 1, 0, lattice))
            family.append(HklPlane(1, 0, 1, lattice))
            family.append(HklPlane(-1, 0, 1, lattice))
            family.append(HklPlane(0, 1, 1, lattice))
            family.append(HklPlane(0, -1, 1, lattice))
        elif hkl == '111':
            family.append(HklPlane(1, 1, 1, lattice))
            family.append(HklPlane(-1, 1, 1, lattice))
            family.append(HklPlane(1, -1, 1, lattice))
            family.append(HklPlane(1, 1, -1, lattice))
        elif hkl in ['112', '121', '211']:
            family.append(HklPlane(1, 1, 2, lattice))
            family.append(HklPlane(-1, 1, 2, lattice))
            family.append(HklPlane(1, -1, 2, lattice))
            family.append(HklPlane(1, 1, -2, lattice))
            family.append(HklPlane(1, 2, 1, lattice))
            family.append(HklPlane(-1, 2, 1, lattice))
            family.append(HklPlane(1, -2, 1, lattice))
            family.append(HklPlane(1, 2, -1, lattice))
            family.append(HklPlane(2, 1, 1, lattice))
            family.append(HklPlane(-2, 1, 1, lattice))
            family.append(HklPlane(2, -1, 1, lattice))
            family.append(HklPlane(2, 1, -1, lattice))
        elif hkl in ['113', '131', '311']:
            family.append(HklPlane(1, 1, 3, lattice))
            family.append(HklPlane(-1, 1, 3, lattice))
            family.append(HklPlane(1, -1, 3, lattice))
            family.append(HklPlane(1, 1, -3, lattice))
            family.append(HklPlane(1, 3, 1, lattice))
            family.append(HklPlane(-1, 3, 1, lattice))
            family.append(HklPlane(1, -3, 1, lattice))
            family.append(HklPlane(1, 3, -1, lattice))
            family.append(HklPlane(3, 1, 1, lattice))
            family.append(HklPlane(-3, 1, 1, lattice))
            family.append(HklPlane(3, -1, 1, lattice))
            family.append(HklPlane(3, 1, -1, lattice))
        elif hkl in ['331', '313', '133']:
            family.append(HklPlane(3, 3, 1, lattice))
            family.append(HklPlane(-3, 3, 1, lattice))
            family.append(HklPlane(3, -3, 1, lattice))
            family.append(HklPlane(3, 3, -1, lattice))
            family.append(HklPlane(3, 1, 3, lattice))
            family.append(HklPlane(-3, 1, 3, lattice))
            family.append(HklPlane(3, -1, 3, lattice))
            family.append(HklPlane(3, 1, -3, lattice))
            family.append(HklPlane(1, 3, 3, lattice))
            family.append(HklPlane(-1, 3, 3, lattice))
            family.append(HklPlane(1, -3, 3, lattice))
            family.append(HklPlane(1, 3, -3, lattice))
        elif hkl in ['662', '626', '266']:
            family.append(HklPlane(6, 6, 2, lattice))
            family.append(HklPlane(-6, 6, 2, lattice))
            family.append(HklPlane(6, -6, 2, lattice))
            family.append(HklPlane(6, 6, -2, lattice))
            family.append(HklPlane(6, 2, 6, lattice))
            family.append(HklPlane(-6, 2, 6, lattice))
            family.append(HklPlane(6, -2, 6, lattice))
            family.append(HklPlane(6, 2, -6, lattice))
            family.append(HklPlane(2, 6, 6, lattice))
            family.append(HklPlane(-2, 6, 6, lattice))
            family.append(HklPlane(2, -6, 6, lattice))
            family.append(HklPlane(2, 6, -6, lattice))
        elif hkl == '002':
            family.append(HklPlane(2, 0, 0, lattice))
            family.append(HklPlane(0, 2, 0, lattice))
            family.append(HklPlane(0, 0, 2, lattice))
        elif hkl == '022':
            family.append(HklPlane(2, 2, 0, lattice))
            family.append(HklPlane(-2, 2, 0, lattice))
            family.append(HklPlane(2, 0, 2, lattice))
            family.append(HklPlane(-2, 0, 2, lattice))
            family.append(HklPlane(0, 2, 2, lattice))
            family.append(HklPlane(0, -2, 2, lattice))
        elif hkl == '123':
            family.append(HklPlane(1, 2, 3, lattice))
            family.append(HklPlane(-1, 2, 3, lattice))
            family.append(HklPlane(1, -2, 3, lattice))
            family.append(HklPlane(1, 2, -3, lattice))
            family.append(HklPlane(3, 1, 2, lattice))
            family.append(HklPlane(-3, 1, 2, lattice))
            family.append(HklPlane(3, -1, 2, lattice))
            family.append(HklPlane(3, 1, -2, lattice))
            family.append(HklPlane(2, 3, 1, lattice))
            family.append(HklPlane(-2, 3, 1, lattice))
            family.append(HklPlane(2, -3, 1, lattice))
            family.append(HklPlane(2, 3, -1, lattice))
            family.append(HklPlane(1, 3, 2, lattice))
            family.append(HklPlane(-1, 3, 2, lattice))
            family.append(HklPlane(1, -3, 2, lattice))
            family.append(HklPlane(1, 3, -2, lattice))
            family.append(HklPlane(2, 1, 3, lattice))
            family.append(HklPlane(-2, 1, 3, lattice))
            family.append(HklPlane(2, -1, 3, lattice))
            family.append(HklPlane(2, 1, -3, lattice))
            family.append(HklPlane(3, 2, 1, lattice))
            family.append(HklPlane(-3, 2, 1, lattice))
            family.append(HklPlane(3, -2, 1, lattice))
            family.append(HklPlane(3, 2, -1, lattice))
        elif len(np.unique(list(hkl))) == 1 and not '0' in hkl:
            # hhh planes -> 4 planes
            family.append(HklPlane(h, h, h, lattice))
            family.append(HklPlane(-h, h, h, lattice))
            family.append(HklPlane(h, -h, h, lattice))
            family.append(HklPlane(h, h, -h, lattice))
        elif len(np.unique(hkl_list)) == 2 and hkl_list[0] == hkl_list[1] and not '0' in hkl:
            # 2 different ints, hhl type -> 12 planes
            family.append(HklPlane(h, h, l, lattice))
            family.append(HklPlane(-h, h, l, lattice))
            family.append(HklPlane(h, -h, l, lattice))
            family.append(HklPlane(h, h, -l, lattice))
            family.append(HklPlane(h, l, h, lattice))
            family.append(HklPlane(-h, l, h, lattice))
            family.append(HklPlane(h, -l, h, lattice))
            family.append(HklPlane(h, l, -h, lattice))
            family.append(HklPlane(l, h, h, lattice))
            family.append(HklPlane(-l, h, h, lattice))
            family.append(HklPlane(l, -h, h, lattice))
            family.append(HklPlane(l, h, -h, lattice))
        elif len(np.unique(hkl_list)) == 2 and hkl_list[1] == hkl_list[2] and not '0' in hkl:
            # 2 different ints, hkk type -> 12 planes
            family.append(HklPlane(k, k, h, lattice))
            family.append(HklPlane(-k, k, h, lattice))
            family.append(HklPlane(k, -k, h, lattice))
            family.append(HklPlane(k, k, -h, lattice))
            family.append(HklPlane(k, h, k, lattice))
            family.append(HklPlane(-k, h, k, lattice))
            family.append(HklPlane(k, -h, k, lattice))
            family.append(HklPlane(k, h, -k, lattice))
            family.append(HklPlane(h, k, k, lattice))
            family.append(HklPlane(-h, k, k, lattice))
            family.append(HklPlane(h, -k, k, lattice))
            family.append(HklPlane(h, k, -k, lattice))
        elif len(np.unique(list(hkl))) == 3 and not '0' in hkl:
            # 3 different ints, all nonzeros -> 24 planes
            family.append(HklPlane(h, k, l, lattice))
            family.append(HklPlane(-h, k, l, lattice))
            family.append(HklPlane(h, -k, l, lattice))
            family.append(HklPlane(h, k, -l, lattice))
            family.append(HklPlane(l, h, k, lattice))
            family.append(HklPlane(-l, h, k, lattice))
            family.append(HklPlane(l, -h, k, lattice))
            family.append(HklPlane(l, h, -k, lattice))
            family.append(HklPlane(k, l, h, lattice))
            family.append(HklPlane(-k, l, h, lattice))
            family.append(HklPlane(k, -l, h, lattice))
            family.append(HklPlane(k, l, -h, lattice))
            family.append(HklPlane(h, l, k, lattice))
            family.append(HklPlane(-h, l, k, lattice))
            family.append(HklPlane(h, -l, k, lattice))
            family.append(HklPlane(h, l, -k, lattice))
            family.append(HklPlane(k, h, l, lattice))
            family.append(HklPlane(-k, h, l, lattice))
            family.append(HklPlane(k, -h, l, lattice))
            family.append(HklPlane(k, h, -l, lattice))
            family.append(HklPlane(l, k, h, lattice))
            family.append(HklPlane(-l, k, h, lattice))
            family.append(HklPlane(l, -k, h, lattice))
            family.append(HklPlane(l, k, -h, lattice))
        else:
            raise ValueError('warning, family not supported: %s' % hkl)
        '''
        return family

    def multiplicity(self, symmetry=Symmetry.cubic):
        """compute the general multiplicity for this `HklPlane` and the given `Symmetry`.

        :param Symmetry symmetry: The crystal symmetry to take into account.
        :return: the number of equivalent planes in the family.
        """
        return len(HklPlane.get_family(self.miller_indices(), include_friedel_pairs=True, crystal_structure=symmetry))

    def slip_trace(self, orientation, n_int=np.array([0, 0, 1]), view_up=np.array([0, 1, 0]), trace_size=100, verbose=False):
        """
        Compute the intersection of the lattice plane with a particular plane defined by its normal.

        :param orientation: The crystal orientation.
        :param n_int: normal to the plane of intersection (laboratory local frame).
        :param view_up: vector to place upwards on the plot.
        :param int trace_size: size of the trace.
        :param verbose: activate verbose mode.
        :return: a numpy array with the coordinates of the two points defining the trace.
        """
        gt = orientation.orientation_matrix().transpose()
        n_rot = gt.dot(self.normal())
        trace_xyz = np.cross(n_rot, n_int)
        trace_xyz /= np.linalg.norm(trace_xyz)
        # now we have the trace vector expressed in the XYZ coordinate system
        # we need to change the coordinate system to the intersection plane
        # (then only the first two component will be non zero)
        P = np.zeros((3, 3), dtype=np.float)
        Zp = n_int
        Yp = view_up / np.linalg.norm(view_up)
        Xp = np.cross(Yp, Zp)
        for k in range(3):
            P[k, 0] = Xp[k]
            P[k, 1] = Yp[k]
            P[k, 2] = Zp[k]
        trace = trace_size * P.transpose().dot(trace_xyz)  # X'=P^-1.X
        if verbose:
            print('n_rot = %s' % n_rot)
            print('trace in XYZ', trace_xyz)
            print(P)
            print('trace in (XpYpZp):', trace)
        return trace

    @staticmethod
    def plot_slip_traces(orientation, hkl='111', n_int=np.array([0, 0, 1]), \
                         view_up=np.array([0, 1, 0]), verbose=False, title=True, legend=True, \
                         trans=False, str_plane=None):
        """
        A method to plot the slip planes intersection with a particular plane
        (known as slip traces if the plane correspond to the surface).
        A few parameters can be used to control the plot looking.
        Thank to Jia Li for starting this code.

        :param orientation: The crystal orientation.
        :param hkl: the slip plane family (eg. 111 or 110)
        :param n_int: normal to the plane of intersection.
        :param view_up: vector to place upwards on the plot.
        :param verbose: activate verbose mode.
        :param title: display a title above the plot.
        :param legend: display the legend.
        :param trans: use a transparent background for the figure (useful to overlay the figure on top of another image).
        :param str_plane: particular string to use to represent the plane in the image name.
        """
        plt.figure()
        hkl_planes = HklPlane.get_family(hkl)
        colors = 'rgykcmbw'
        for i, hkl_plane in enumerate(hkl_planes):
            trace = hkl_plane.slip_trace(orientation, n_int=n_int, view_up=view_up, trace_size=1, verbose=verbose)
            x = [-trace[0] / 2, trace[0] / 2]
            y = [-trace[1] / 2, trace[1] / 2]
            plt.plot(x, y, colors[i % len(hkl_planes)], label='%d%d%d' % hkl_plane.miller_indices(), linewidth=2)
        plt.axis('equal')
        t = np.linspace(0., 2 * np.pi, 100)
        plt.plot(0.5 * np.cos(t), 0.5 * np.sin(t), 'k')
        plt.axis([-0.51, 0.51, -0.51, 0.51])
        plt.axis('off')
        if not str_plane: str_plane = '(%.1f, %.1f, %.1f)' % (n_int[0], n_int[1], n_int[2])
        if title:
            plt.title('{%s} family traces on plane %s' % (hkl, str_plane))
        if legend: plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
        plt.savefig('slip_traces_%s_%s.png' % (hkl, str_plane), transparent=trans, format='png')

    @staticmethod
    def plot_XY_slip_traces(orientation, hkl='111', title=True,
                            legend=True, trans=False, verbose=False):
        """Helper method to plot the slip traces on the XY plane."""
        HklPlane.plot_slip_traces(orientation, hkl=hkl, n_int=np.array([0, 0, 1]),
                                  view_up=np.array([0, 1, 0]), title=title, legend=legend,
                                  trans=trans, verbose=verbose, str_plane='XY')

    @staticmethod
    def plot_YZ_slip_traces(orientation, hkl='111', title=True,
                            legend=True, trans=False, verbose=False):
        """Helper method to plot the slip traces on the YZ plane."""
        HklPlane.plot_slip_traces(orientation, hkl=hkl, n_int=np.array([1, 0, 0]),
                                  view_up=np.array([0, 0, 1]), title=title, legend=legend,
                                  trans=trans, verbose=verbose, str_plane='YZ')

    @staticmethod
    def plot_XZ_slip_traces(orientation, hkl='111', title=True,
                            legend=True, trans=False, verbose=False):
        """Helper method to plot the slip traces on the XZ plane."""
        HklPlane.plot_slip_traces(orientation, hkl=hkl, n_int=np.array([0, -1, 0]),
                                  view_up=np.array([0, 0, 1]), title=title, legend=legend,
                                  trans=trans, verbose=verbose, str_plane='XZ')

    @staticmethod
    def indices_from_two_directions(uvw1, uvw2):
        """
        Two crystallographic directions :math:`uvw_1` and :math:`uvw_2`define a unique set of hkl planes. 
        This does not depends on the crystal symmetry.
        
        .. math::

           h = v_1 . w_2 - w_1 . v_2 \\\\
           k = w_1 . u_2 - u_1 . w_2 \\\\
           l = u_1 . v_2 - v_1 . u_2

        :param uvw1: The first instance of the `HklDirection` class.
        :param uvw2: The second instance of the `HklDirection` class.
        :return h, k, l: the miller indices of the `HklPlane` defined by the two directions.
        """
        (u1, v1, w1) = uvw1.miller_indices()
        (u2, v2, w2) = uvw2.miller_indices()
        h = v1 * w2 - w1 * v2
        k = w1 * u2 - u1 * w2
        l = u1 * v2 - v1 * u2
        return h, k, l
