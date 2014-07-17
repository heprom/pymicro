'''The microstructure module define the class to handle 
   3D crystal lattices (the 7 Bravais lattices).
   This particular class has been largely inspired from the pymatgen 
   project at https://github.com/materialsproject/pymatgen
'''
import itertools
import numpy as np
from numpy import pi, dot, transpose, radians
#from pymicro.crystal.microstructure import Orientation
from matplotlib import pyplot as plt

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
    '''Returns the unit vector normal to the plane.
    FIXME do not handle non straight lattices like hexagonal
    '''
    #(a, b, c) = self._lattice._lengths
    #(h, k, l) = self.miller_indices()
    n = np.zeros(3)
    for i in range(3):
      if self.miller_indices()[i] != 0:
        n[i] = self._lattice._lengths[i]/float(self.miller_indices()[i])
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

if __name__ == '__main__':
  a = 0.405 # Al FCC
  l = Lattice([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]])
  p = HklPlane(1, 1, 1, lattice=l)
  print p.__repr__
  print p.interplanar_spacing()
