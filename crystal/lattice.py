'''The lattice module define the class to handle 
   3D crystal lattices (the 14 Bravais lattices).
'''
import os
from pymicro.crystal.cif import CifFile
import itertools
import numpy as np
from numpy import pi, dot, transpose, radians
from matplotlib import pyplot as plt

class Lattice:
  '''
  Crystal lattice class.

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
  Bravais lattice cells. It is set to a single atoms at (0,0,0) by 
  default so that each cell is a Bravais lattice but may be changed to 
  something more complex to achieve HCP structure or Diamond structure 
  for instance.
  '''
  
  def __init__(self, matrix, centering='P'):
    '''Create a crystal lattice (unit cel).
    
    Create a lattice from a 3x3 matrix.
    Each row in the matrix represents one lattice vector.
    '''
    m = np.array(matrix, dtype=np.float64).reshape((3, 3))
    lengths = np.sqrt(np.sum(m ** 2, axis=1))
    angles = np.zeros(3)
    for i in xrange(3):
      j = (i + 1) % 3
      k =  (i + 2) % 3
      angles[i] = dot(m[j], m[k]) / (lengths[j] * lengths[k])
    angles = np.arccos(angles) * 180. / pi
    self._angles = angles
    self._lengths = lengths
    self._matrix = m
    self._centering = centering
    self._basis = [(0.,0.,0.)]
  
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
    
    This computes the three reciprocal lattice vectors defined by
    
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
    '''Returns a copy of matrix representing the Lattice.'''
    return np.copy(self._matrix)

  @staticmethod
  def symmetry(crystal_structure='cubic'):
    ''' define the 24 equivalent cube orientations.
    
    :params str crystal_structure: a string describing the crystal structure.
    :raise ValueError: if the given crystal structure is not cubic or none.

    Those come from Randle & Engler, 2000. For cubic, they correspond to:

     * 1 for pure cube
     * 9 rot90 arount <001> axes
     * 6 rot180 around <110> axes
     * 8 rot 120 around <111> axes)
    '''
    if crystal_structure == 'cubic':
      cubes = np.zeros((24, 3, 3), dtype= np.float)
      cubes[0] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
      cubes[1] = np.array([[ 0.,  0., -1.], [0., -1.,  0.], [-1.,  0.,  0.]])
      cubes[2] = np.array([[ 0.,  0., -1.], [0.,  1.,  0.], [1.,  0.,  0.]])
      cubes[3] = np.array([[-1.,  0.,  0.], [0.,  1.,  0.], [0.,  0., -1.]])
      cubes[4] = np.array([[ 0.,  0.,  1.], [0.,  1.,  0.], [-1.,  0.,  0.]])
      cubes[5] = np.array([[ 1.,  0.,  0.], [0.,  0., -1.], [0.,  1.,  0.]])
      cubes[6] = np.array([[ 1.,  0.,  0.], [0., -1.,  0.], [0.,  0., -1.]])
      cubes[7] = np.array([[ 1.,  0.,  0.], [0.,  0.,  1.], [0., -1.,  0.]])
      cubes[8] = np.array([[ 0., -1.,  0.], [1.,  0.,  0.], [0.,  0.,  1.]])
      cubes[9] = np.array([[-1.,  0.,  0.], [0., -1.,  0.], [0.,  0.,  1.]])
      cubes[10] = np.array([[ 0.,  1.,  0.], [-1.,  0.,  0.], [0.,  0.,  1.]])
      cubes[11] = np.array([[ 0.,  0.,  1.], [1.,  0.,  0.], [0.,  1.,  0.]])
      cubes[12] = np.array([[ 0.,  1.,  0.], [0.,  0.,  1.], [1.,  0.,  0.]])
      cubes[13] = np.array([[ 0.,  0., -1.], [-1.,  0.,  0.], [0.,  1.,  0.]])
      cubes[14] = np.array([[ 0., -1.,  0.], [0.,  0.,  1.], [-1.,  0.,  0.]])
      cubes[15] = np.array([[ 0.,  1.,  0.], [0.,  0., -1.], [-1.,  0.,  0.]])
      cubes[16] = np.array([[ 0.,  0., -1.], [1.,  0.,  0.], [0., -1.,  0.]])
      cubes[17] = np.array([[ 0.,  0.,  1.], [-1.,  0.,  0.], [0., -1.,  0.]])
      cubes[18] = np.array([[ 0., -1.,  0.], [0.,  0., -1.], [1.,  0.,  0.]])
      cubes[19] = np.array([[ 0.,  1.,  0.], [1.,  0.,  0.], [0.,  0., -1.]])
      cubes[20] = np.array([[-1.,  0.,  0.], [0.,  0.,  1.], [0.,  1.,  0.]])
      cubes[21] = np.array([[ 0.,  0.,  1.], [0., -1.,  0.], [1.,  0.,  0.]])
      cubes[22] = np.array([[ 0., -1.,  0.], [-1.,  0.,  0.], [0.,  0., -1.]])
      cubes[23] = np.array([[-1.,  0.,  0.], [0.,  0., -1.], [0., -1.,  0.]])
      return cubes
    elif crystal_structure == 'none':
      equiv = np.zeros((1, 3, 3), dtype= np.float)
      equiv[0] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
      return equiv
    else:
      raise ValueError('warning, crystal structure not supported: %s' % crystal_structure)

  @staticmethod
  def from_cif(file_path):
    '''
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

    *Parameters*
    
    **file_path**: The path to the CIF file representing the crystal structure.
    
    *Returns*
    
    A `Lattice` instance corresponding to the given CIF file.
    '''
    cf = CifFile.ReadCif(file_path)
    #crystal = eval('cf[\'%s\']' % symbol)
    crystal = cf.first_block()
    a = 0.1*float(crystal['_cell_length_a'])
    b = 0.1*float(crystal['_cell_length_b'])
    c = 0.1*float(crystal['_cell_length_c'])
    alpha = float(crystal['_cell_angle_alpha'])
    beta = float(crystal['_cell_angle_beta'])
    gamma = float(crystal['_cell_angle_gamma'])
    return Lattice.from_parameters(a, b, c, alpha, beta, gamma)

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
    return Lattice([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]])

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
    return Lattice.from_parameters(a, a, a, 90, 90, 90, 'I')

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
    return Lattice.from_parameters(a, a, a, 90, 90, 90, 'F')

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
    return Lattice.from_parameters(a, a, c, 90, 90, 90)

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
    return Lattice.from_parameters(a, a, c, 90, 90, 90, 'I')

  @staticmethod
  def orthorombic(a, b, c):
    '''
    Create a tetragonal Lattice unit cell with 3 different length 
    parameters a, b and c.
    '''
    return Lattice.from_parameters(a, b, c, 90, 90, 90)

  @staticmethod
  def base_centered_orthorombic(a, b, c):
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
    return Lattice.from_parameters(a, b, c, 90, 90, 90, 'C')

  @staticmethod
  def body_centered_orthorombic(a, b, c):
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
    return Lattice.from_parameters(a, b, c, 90, 90, 90, 'I')

  @staticmethod
  def face_centered_orthorombic(a, b, c):
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
    return Lattice.from_parameters(a, b, c, 90, 90, 90, 'F')

  @staticmethod    
  def hexagonal(a, c):
    '''
    Create a hexagonal Lattice unit cell with length parameters a and c.
    '''
    return Lattice.from_parameters(a, a, c, 90, 90, 120)

  @staticmethod
  def rhombohedral(a, alpha):
    '''
    Create a rhombohedral Lattice unit cell with one length 
    parameter a and the angle alpha.
    '''
    return Lattice.from_parameters(a, a, a, alpha, alpha, alpha)

  @staticmethod
  def monoclinic(a, b, c, alpha):
    '''
    Create a monoclinic Lattice unit cell with 3 different length 
    parameters a, b and c. The cell angle is given by alpha.
    The lattice centering id primitive ie. 'P'
    '''
    return Lattice.from_parameters(a, b, c, alpha, 90, 90)

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
    return Lattice.from_parameters(a, b, c, alpha, 90, 90, 'C')
      
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
    return Lattice.from_parameters(a, b, c, alpha, beta, gamma)
      
  @staticmethod
  def from_parameters(a, b, c, alpha, beta, gamma, centering='P'):
    '''
    Create a Lattice using unit cell lengths and angles (in degrees).
    The lattice centering can also be specified (among 'P', 'I', 'F', 
    'A', 'B' or 'C').

    *Parameters*
    
    **a**: first lattice length parameter
    
    **b**: second lattice length parameter
    
    **c**: third lattice length parameter
    
    **alpha**: first lattice angle parameter
    
    **beta**: second lattice angle parameter
    
    **gamma**: third lattice angle parameter
    
    **centering**: lattice centering ('P' by default)
    
    *Returns*
    
    A `Lattice` instance with the specified lattice parameters.
    '''
    alpha_r = radians(alpha)
    beta_r = radians(beta)
    gamma_r = radians(gamma)    
    val = (np.cos(alpha_r) * np.cos(beta_r) - np.cos(gamma_r)) \
      / (np.sin(alpha_r) * np.sin(beta_r))
    #Sometimes rounding errors result in values slightly > 1.
    val = val if abs(val) <= 1 else val / abs(val)
    gamma_star = np.arccos(val)
    vector_a = [a * np.sin(beta_r), 0.0, a * np.cos(beta_r)]
    vector_b = [-b * np.sin(alpha_r) * np.cos(gamma_star), b * np.sin(alpha_r) * np.sin(gamma_star), b * np.cos(alpha_r)]
    vector_c = [0.0, 0.0, float(c)]
    return Lattice([vector_a, vector_b, vector_c], centering)

  def volume(self):
    '''Compute the volume of the unit cell.'''
    m = self._matrix
    return abs(np.dot(np.cross(m[0], m[1]), m[2]))

  def get_hkl_family(self, hkl):
    '''Get a list of the hkl planes composing the given family for 
    this crystal lattice.

    *Parameters*
    
    **hkl**: miller indices of the requested family

    *Returns*
    
    A list of the hkl planes in the given family.
    '''
    planes = HklPlane.get_family(hkl)
    for p in planes:
      p._lattice = self
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
  def get_slip_systems(plane_type='111'):
    '''A static method to get all slip systems for a given hkl plane family.
    
    :params str plane_type: a string of the 3 miller indices of the crystallographic plane family.
    :returns list: a list of :py:class:`~pymicro.crystal.lattice.SlipSystem`. 
    
    .. warning::
    
      only working for 111 and 112 planes...
    '''
    slip_systems = []
    if plane_type == '111':
      slip_systems.append(SlipSystem(HklPlane(1, 1, 1), HklDirection(-1, 0, 1))) # Bd
      slip_systems.append(SlipSystem(HklPlane(1, 1, 1), HklDirection(0, -1, 1))) # Ba
      slip_systems.append(SlipSystem(HklPlane(1, 1, 1), HklDirection(-1, 1, 0))) # Bc
      slip_systems.append(SlipSystem(HklPlane(1, -1, 1), HklDirection(-1, 0, 1))) # Db
      slip_systems.append(SlipSystem(HklPlane(1, -1, 1), HklDirection(0, 1, 1))) # Dc
      slip_systems.append(SlipSystem(HklPlane(1, -1, 1), HklDirection(1, 1, 0))) # Da
      slip_systems.append(SlipSystem(HklPlane(-1, 1, 1), HklDirection(0, -1, 1))) # Ab
      slip_systems.append(SlipSystem(HklPlane(-1, 1, 1), HklDirection(1, 1, 0))) # Ad
      slip_systems.append(SlipSystem(HklPlane(-1, 1, 1), HklDirection(1, 0, 1))) # Ac
      slip_systems.append(SlipSystem(HklPlane(1, 1, -1), HklDirection(-1, 1, 0))) # Cb
      slip_systems.append(SlipSystem(HklPlane(1, 1, -1), HklDirection(1, 0, 1))) # Ca
      slip_systems.append(SlipSystem(HklPlane(1, 1, -1), HklDirection(0, 1, 1))) # Cd
    elif plane_type == '112':
      slip_systems.append(SlipSystem(HklPlane(1, 1, 2), HklDirection(1, 1, -1)))
      slip_systems.append(SlipSystem(HklPlane(-1, 1, 2), HklDirection(1, -1, 1)))
      slip_systems.append(SlipSystem(HklPlane(1, -1, 2), HklDirection(-1, 1, 1)))
      slip_systems.append(SlipSystem(HklPlane(1, 1, -2), HklDirection(1, 1, 1)))
      slip_systems.append(SlipSystem(HklPlane(1, 2, 1), HklDirection(1, -1, 1)))
      slip_systems.append(SlipSystem(HklPlane(-1, 2, 1), HklDirection(1, 1, -1)))
      slip_systems.append(SlipSystem(HklPlane(1, -2, 1), HklDirection(1, 1, 1)))
      slip_systems.append(SlipSystem(HklPlane(1, 2, -1), HklDirection(-1, 1, 1)))
      slip_systems.append(SlipSystem(HklPlane(2, 1, 1), HklDirection(-1, 1, 1)))
      slip_systems.append(SlipSystem(HklPlane(-2, 1, 1), HklDirection(1, 1, 1)))
      slip_systems.append(SlipSystem(HklPlane(2, -1, 1), HklDirection(1, 1, -1)))
      slip_systems.append(SlipSystem(HklPlane(2, 1, -1), HklDirection(1, -1, 1)))
    else:
      print 'warning only 111 or 112 slip planes supported for the moment!'
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

  def miller_indices(self):
    '''
    Returns an immutable tuple of the plane Miller indices.
    '''
    return (self._h, self._k, self._l)

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
    '''Returns a normalized vector corresponding to this crystallographic direction.'''
    (h, k, l) = self.miller_indices()
    l_vect = np.array([h, k, l])
    return l_vect/np.linalg.norm(l_vect)

  def angle_with_direction(self, hkl):
    '''Computes the angle between this crystallographic direction and 
    the given direction (in radians).'''
    return np.arccos(np.dot(self.direction(), hkl.direction()))

  @staticmethod
  def angle_between_directions((h1, k1, l1), (h2, k2, l2)):
    '''Computes the angle between two crystallographic directions (in degrees).'''
    d1 = HklDirection(h1, k1, l1)
    d2 = HklDirection(h2, k2, l2)
    return d1.angle_with_direction(d2)*180/np.pi
  
  def find_planes_in_zone(self, max_miller=5):
    '''
    This method finds the hkl planes in zone with the crystallographic 
    direction. If (u,v,w) denotes the zone axis, this means finding all 
    hkl planes which verify :math:`h.u + k.v + l.w = 0`.
    
    :param max_miller: The maximum miller index to limt the search`
    :returns list: A list of :py:class:`~pymicro.crystal.lattice.HklPlane` 
    objects describing all the planes in zone with the direction.
    '''
    indices = range(-max_miller, max_miller+1)
    hklplanes_in_zone = []
    for h in indices:
      for k in indices:
        for l in indices:
          if h == k == l == 0: # skip (0, 0, 0)
            continue
          if np.dot(np.array([h, k, l]), self.direction()) == 0:
            hklplanes_in_zone.append(HklPlane(h, k, l, self._lattice))
    return(hklplanes_in_zone)
    
class HklPlane(HklObject):
  '''
  This class define crystallographic planes using Miller indices.
  
  A plane can be create by speficying its Miller indices and the 
  crystal lattice (default is cubic with lattice parameter of 1.0)
  ::

    a = 0.405 # FCC Aluminium
    l = Lattice.cubic(a)
    p = HklPlane(1, 1, 1, lattice=l)
    print p
    print p.interplanar_spacing()

  .. warning::

     The calculations have not yet been thourouhly tested appart 
     from cubic crystal lattice.
  '''

  def normal(self, verbose=False):
    '''Returns the unit vector normal to the plane.

    .. warning::

       Right now the we do not make use of the repiprocal lattice to 
       compute the plane... this should be corrected in the future.

       Proof hkl plane hexagonal calculations
    '''
    (alpha, beta, gamma) = self._lattice._angles
    (h, k, l) = self.miller_indices()
    isHexagonal = np.linalg.norm(np.array([alpha, beta, gamma]) - np.array([90.0, 90.0, 120.0])) < 0.001
    (h, k, l) = self.miller_indices()
    n = np.zeros(3)
    a = self._lattice._matrix[0]
    b = self._lattice._matrix[1]
    c = self._lattice._matrix[2]
    if verbose:
      print 'computing hkl plane normal with indices',h,k,l
      print 'lattice angles:', (alpha, beta, gamma)
      print 'lattice vectors:', a, b, c
    n = h*a + k*b + l*c
    if isHexagonal:
      n_hex = np.array([2*n[0] + n[1], n[0] + 2*n[1], n[2]])
      print 'HCP plane', self.miller_indices(), 'is normal to', n_hex
      n = n_hex
    return n/np.linalg.norm(n)
  
  def scattering_vector(self):
    [astar, bstar, cstar] = self._lattice.reciprocal_lattice()
    (h, k, l) = self.miller_indices()
    # express (h,k,l) in the cartesian crystal CS
    Gc = h*astar + k*bstar + l*cstar
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
    #d = a / np.sqrt(h**2 + k**2 + l**2) # for cubic structure
    d = self._lattice.volume() / np.sqrt(h**2*b**2*c**2*np.sin(alpha)**2 + \
      k**2*a**2*c**2*np.sin(beta)**2 + l**2*a**2*b**2*np.sin(gamma)**2 + \
      2*h*l*a*b**2*c*(np.cos(alpha)*np.cos(gamma) - np.cos(beta)) + \
      2*h*k*a*b*c**2*(np.cos(alpha)*np.cos(beta) - np.cos(gamma)) + \
      2*k*l*a**2*b*c*(np.cos(beta)*np.cos(gamma) - np.cos(alpha)))
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
      print '\nBragg angle for %d%d%d at %.1f keV is %.1f deg\n' % (h, k, l, lambda_keV, theta_deg)
    return theta
    
  @staticmethod
  def four_to_three_index(h, k, i, l):
    '''Convert four to three index direction (used for hexagonal crystal lattice).'''
    return (6*h/5. - 3*k/5., 3*h/5. + 6*k/5., l)

  @staticmethod
  def three_to_four_index(u, v, w):
    '''Convert three to four index direction (used for hexagonal crystal lattice).'''
    return ((2*u - v)/3., (2*v - u)/3., -(u + v)/3., w)
    
  @staticmethod
  def get_family(hkl):
    '''Static method to obtain a list of the different crystallographic 
    planes in a particular family.
    
    :params str hkl: a string of 3 numbers corresponding to the miller indices.
    :raise ValueError: if the given string does not correspond to a supported family.
    :returns list: a list of the :py:class:`~pymicro.crystal.lattice.HklPlane` in the given hkl family.

    .. note::

      We could build any family with 3 integers automatically:
      * 1 int nonzero -> 3 planes, eg (001)
      * 2 equal ints nonzero -> 6 planes, eg (011)
      * 3 equal ints nonzero -> 4 planes, eg (111)
      * 2 different ints, all nonzeros -> 12 planes, eg (112) 
      * 3 different ints, all nonzeros -> 24 planes, eg (123)
    '''
    if not len(hkl) == 3:
      raise ValueError('warning, family not supported: %s' % hkl)
    family = []
    l = list(hkl)
    l.sort() # miller indices are now sorted by increasing order
    if hkl == '001' or hkl == '010' or hkl == '100':
      family.append(HklPlane(1, 0, 0))
      family.append(HklPlane(0, 1, 0))
      family.append(HklPlane(0, 0, 1))
    elif hkl in ['011', '101', '110']:
      family.append(HklPlane(1, 1, 0))
      family.append(HklPlane(-1, 1, 0))
      family.append(HklPlane(1, 0, 1))
      family.append(HklPlane(-1, 0, 1))
      family.append(HklPlane(0, 1, 1))
      family.append(HklPlane(0, -1, 1))
    elif hkl == '111':
      family.append(HklPlane(1, 1, 1))
      family.append(HklPlane(-1, 1, 1))
      family.append(HklPlane(1, -1, 1))
      family.append(HklPlane(1, 1, -1))
    elif hkl == '112' or hkl == '211':
      family.append(HklPlane(1, 1, 2))
      family.append(HklPlane(-1, 1, 2))
      family.append(HklPlane(1, -1, 2))
      family.append(HklPlane(1, 1, -2))
      family.append(HklPlane(1, 2, 1))
      family.append(HklPlane(-1, 2, 1))
      family.append(HklPlane(1, -2, 1))
      family.append(HklPlane(1, 2, -1))
      family.append(HklPlane(2, 1, 1))
      family.append(HklPlane(-2, 1, 1))
      family.append(HklPlane(2, -1, 1))
      family.append(HklPlane(2, 1, -1))
    elif hkl == '002':
      family.append(HklPlane(2, 0, 0))
      family.append(HklPlane(0, 2, 0))
      family.append(HklPlane(0, 0, 2))
    elif hkl == '022':
      family.append(HklPlane(2, 2, 0))
      family.append(HklPlane(-2, 2, 0))
      family.append(HklPlane(2, 0, 2))
      family.append(HklPlane(-2, 0, 2))
      family.append(HklPlane(0, 2, 2))
      family.append(HklPlane(0, -2, 2))
    elif hkl == '123':
      family.append(HklPlane(1, 2, 3))
      family.append(HklPlane(-1, 2, 3))
      family.append(HklPlane(1, -2, 3))
      family.append(HklPlane(1, 2, -3))
      family.append(HklPlane(3, 1, 2))
      family.append(HklPlane(-3, 1, 2))
      family.append(HklPlane(3, -1, 2))
      family.append(HklPlane(3, 1, -2))
      family.append(HklPlane(2, 3, 1))
      family.append(HklPlane(-2, 3, 1))
      family.append(HklPlane(2, -3, 1))
      family.append(HklPlane(2, 3, -1))
      family.append(HklPlane(1, 3, 2))
      family.append(HklPlane(-1, 3, 2))
      family.append(HklPlane(1, -3, 2))
      family.append(HklPlane(1, 3, -2))
      family.append(HklPlane(2, 1, 3))
      family.append(HklPlane(-2, 1, 3))
      family.append(HklPlane(2, -1, 3))
      family.append(HklPlane(2, 1, -3))
      family.append(HklPlane(3, 2, 1))
      family.append(HklPlane(-3, 2, 1))
      family.append(HklPlane(3, -2, 1))
      family.append(HklPlane(3, 2, -1))
    else:
      raise ValueError('warning, family not supported: %s' % hkl)
    return family
  
  @staticmethod
  def plot_slip_traces(orientation, hkl='111', n_int=np.array([0, 0, 1]), \
    view_up=np.array([0, 1, 0]), verbose=False, title=True, legend=True, \
    trans=False, str_plane=None):
    '''
    A method to plot the slip planes intersection with a particular plane 
    (known as slip traces if the plane correspond to the surface).
    Thank to Jia Li for starting this code.
    
    * orientation: The crystal orientation.
    * hkl: the slip plane family (eg. 111 or 110)
    * n_int: normal to the plane of intersection.
    * view_up: vector to place upwards on the plot
    * verbose: activate verbose mode.
    
    A few additional parameters can be used to control the plot looking.

    * title: display a title above the plot
    * legend: display the legend
    * trans: use a transparent background for the figure (useful to 
      overlay the figure on top of another image).
    * str_plane: particuler string to use to represent the plane in the image name.
    '''
    n_int /= np.linalg.norm(n_int)
    view_up /= np.linalg.norm(view_up)
    Bt = orientation.orientation_matrix().transpose()
    hklplanes = HklPlane.get_family(hkl)
    plt.figure(figsize=(7, 5))
    colors = 'rgykcmbw'
    for i, p in enumerate(hklplanes):
      n_rot = Bt.dot(p.normal())
      trace_xyz = np.cross(n_rot, n_int)
      trace_xyz /= np.linalg.norm(trace_xyz)
      # now we have the trace vector expressed in the XYZ coordinate system
      # we need to change the coordinate system to the intersection plane
      # (then only the first two component will be non zero)
      P = np.zeros((3,3), dtype=np.float)
      Zp = n_int
      Yp = view_up / np.linalg.norm(view_up)
      Xp = np.cross(Yp, Zp)
      for k in range(3):
        P[k, 0] = Xp[k]
        P[k, 1] = Yp[k]
        P[k, 2] = Zp[k]
      trace = P.transpose().dot(trace_xyz) # X'=P^-1.X
      if verbose:
        print 'trace in XYZ',trace_xyz
        print P
        print 'trace in (XpYpZp):',trace
      x = [-trace[0]/2, trace[0]/2]
      y = [-trace[1]/2, trace[1]/2]
      plt.plot(x, y, colors[i % len(hklplanes)], label='%d%d%d' % (p._h, p._k, p._l), linewidth=2)
    plt.axis('equal')
    t = np.linspace(0., 2*np.pi, 100)
    plt.plot(0.5*np.cos(t), 0.5*np.sin(t), 'k')
    plt.axis([-0.51,0.51,-0.51,0.51])
    plt.axis('off')
    if not str_plane: str_plane = '(%.1f, %.1f, %.1f)' % (n_int[0], n_int[1], n_int[2])
    if title:
      plt.title('{%s} family traces on plane %s' % (hkl, str_plane))
    if legend: plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
    plt.savefig('slip_traces_%s_%s.png' % (hkl, str_plane), transparent=trans, format='png')

  @staticmethod
  def plot_XY_slip_traces(orientation, hkl='111', title=True, \
    legend=True, trans=False, verbose=False):
    ''' Helper method to plot the slip traces on the XY plane.'''
    HklPlane.plot_slip_traces(orientation, hkl=hkl, n_int = np.array([0, 0, 1]), \
      view_up = np.array([0, 1, 0]), title=title, legend=legend, \
      trans=trans, verbose=verbose, str_plane='XY')

  @staticmethod
  def plot_YZ_slip_traces(orientation, hkl='111', title=True, \
    legend=True, trans=False, verbose=False):
    ''' Helper method to plot the slip traces on the YZ plane.'''
    HklPlane.plot_slip_traces(orientation, hkl=hkl, n_int = np.array([1, 0, 0]), \
      view_up = np.array([0, 0, 1]), title=title, legend=legend, \
      trans=trans, verbose=verbose, str_plane='YZ')

  @staticmethod
  def plot_XZ_slip_traces(orientation, hkl='111', title=True, \
    legend=True, trans=False, verbose=False):
    ''' Helper method to plot the slip traces on the XZ plane.'''
    HklPlane.plot_slip_traces(orientation, hkl=hkl, n_int = np.array([0, -1, 0]), \
      view_up = np.array([0, 0, 1]), title=title, legend=legend, \
      trans=trans, verbose=verbose, str_plane='XZ')
    
