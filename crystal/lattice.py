'''The microstructure module define the class to handle 
   3D crystal lattices (the 7 Bravais lattices).
   This particular class has been largely inspired from the pymatgen 
   project at https://github.com/materialsproject/pymatgen
'''
import itertools
import numpy as np
from numpy import pi, dot, transpose, radians

class Lattice:
  '''
  Crystal lattice class.
  It supports only cubic and hexagonal at the moment.
  '''
  
  def __init__(self, matrix):
    '''
     Create a lattice from any sequence of 9 numbers.
     Note that the sequence is assumed to be read one row at a time. 
     Each row represents one lattice vector.
    '''
    m = np.array(matrix, dtype=np.float64).reshape((3, 3))
    lengths = np.sqrt(np.sum(m ** 2, axis=1))
    angles = np.zeros(3)
    for i in xrange(3):
      j = (i + 1) % 3
      k = (i + 2) % 3
      angles[i] = dot(m[j], m[k]) / (lengths[j] * lengths[k])
    angles = np.arccos(angles) * 180. / pi
    self._angles = angles
    self._lengths = lengths
    self._matrix = m
  
  def __repr__(self):
    f = lambda x: "%0.1f" % x
    out = ["Lattice", " abc : " + " ".join(map(f, self._lengths)),
           " angles : " + " ".join(map(f, self._angles)),
           " volume : %0.4f" % self.volume(),
           " A : " + " ".join(map(f, self._matrix[0])),
           " B : " + " ".join(map(f, self._matrix[1])),
           " C : " + " ".join(map(f, self._matrix[2]))]
    return "\n".join(out)
  
  @property
  def matrix(self):
    '''Returns a copy of matrix representing the Lattice.'''
    return np.copy(self._matrix)

  @staticmethod
  def cubic(a):
    '''
    Create a cubic Lattice unit cell with length parameter a.
    '''
    return Lattice([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]])

  @staticmethod
  def tetragonal(a, c):
    '''
    Create a tetragonal Lattice unit cell with 2 different length 
    parameters a and c.
    '''
    return Lattice.from_parameters(a, a, c, 90, 90, 90)

  @staticmethod
  def orthorombic(a, b, c):
    '''
    Create a tetragonal Lattice unit cell with 3 different length 
    parameters a, b and c.
    '''
    return Lattice.from_parameters(a, b, c, 90, 90, 90)

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
    '''
    return Lattice.from_parameters(a, b, c, alpha, 90, 90)
      
  @staticmethod
  def triclinic(a, b, c, alpha, beta, gamma):
    '''
    Create a triclinic Lattice unit cell with 3 different length 
    parameters a, b, c and three different cell angles alpha, beta 
    and gamma.
    This method is here for the sake of completeness since one can 
    create the triclinic cell directly using the from_parameters method.
    '''
    return Lattice.from_parameters(a, b, c, alpha, beta, gamma)
      
  @staticmethod
  def from_parameters(a, b, c, alpha, beta, gamma):
    '''
    Create a Lattice using unit cell lengths and angles (in degrees).
    Returns: A Lattice with the specified lattice parameters.
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
    return Lattice([vector_a, vector_b, vector_c])    

  def volume(self):
    '''
    Volume of the unit cell.
    '''
    m = self._matrix
    return abs(np.dot(np.cross(m[0], m[1]), m[2]))

class HklPlane:
  '''
  This class define crystallographic planes using Miller indices.
  A cubic crystal lattice is assumed for the moment.
  FIXME right now the we do not make use of the repiprocal lattice to 
  compute the plane... this should be corrected in the future.
  '''
  def __init__(self, h, k, l, lattice=Lattice.cubic(1.0)):
    self._lattice = lattice
    self._h = h
    self._k = k
    self._l = l

  def normal(self):
    '''Returns the unit vector normal to the plane.'''
    (h, k, l) = self.miller_indices()
    n = np.array([h, k, l])
    return n/np.linalg.norm(n)
  
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

  def miller_indices(self):
    '''Returns an immutable tuple of the plane Miller indices.'''
    return (self._h, self._k, self._l)

  def interplanar_spacing(self):
    '''
    Compute the interplanar spacing.
    The formula comes from 'Introduction to Crystallography' p. 68
    by Donald E. Sands.
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

  @staticmethod
  def get_family(hkl):
    '''Helper static method to obtain a list of the different
    slip plane in a particular family.'''
    family = []
    if hkl == '110':
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
    else:
      print 'warning, family not supported:', hkl
    return family
      
if __name__ == '__main__':
  a = 0.405 # Al FCC
  l = Lattice([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]])
  p = HklPlane(1, 1, 1, lattice=l)
  print p.__repr__
  print p.interplanar_spacing()
