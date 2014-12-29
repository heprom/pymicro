'''The lattice module define the class to handle 
   3D crystal lattices (the 14 Bravais lattices).
'''
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
      k = (i + 2) % 3
      angles[i] = dot(m[j], m[k]) / (lengths[j] * lengths[k])
    angles = np.arccos(angles) * 180. / pi
    self._angles = angles
    self._lengths = lengths
    self._matrix = m
    self._centering = centering
  
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

  @staticmethod
  def get_slip_systems(plane_type='111'):
    if plane_type != '111':
      print 'warning only 111 slip supported for the moment!'
    planes = HklPlane.get_family(plane_type)
    slip_systems = []
    for plane in planes:
      (h, k, l) = plane.miller_indices()
      for i in range(3): # 3 slip directions by plane
        dir_type = np.array([h, k, l])
        dir_type[i] = 0
        direction = np.cross(np.array([h, k, l]), dir_type)
        (u, v, w) = direction
        slip_dir = HklDirection(u, v, w)
        slip_systems.append(SlipSystem(plane, slip_dir))
    return slip_systems
      
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
    (h, k, l) = self.miller_indices()
    l_vect = np.array([h, k, l])
    return l_vect/np.linalg.norm(l_vect)

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

  .. note::

     The calculations have not been thourouhly tested appart 
     from cubic crystal lattice.

  .. warning::

     Right now the we do not make use of the repiprocal lattice to 
     compute the plane... this should be corrected in the future.
  '''

  def normal(self, verbose=False):
    '''Returns the unit vector normal to the plane.

    .. warning::

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
    
    .. warning::
    
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
    '''Helper static method to obtain a list of the different
    slip plane in a particular family.'''
    family = []
    if hkl == '100':
      family.append(HklPlane(1, 0, 0))
      family.append(HklPlane(0, 1, 0))
      family.append(HklPlane(0, 0, 1))
    elif hkl == '110':
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
    elif hkl == '200':
      family.append(HklPlane(2, 0, 0))
      family.append(HklPlane(0, 2, 0))
      family.append(HklPlane(0, 0, 2))
    elif hkl == '220':
      family.append(HklPlane(2, 2, 0))
      family.append(HklPlane(-2, 2, 0))
      family.append(HklPlane(2, 0, 2))
      family.append(HklPlane(-2, 0, 2))
      family.append(HklPlane(0, 2, 2))
      family.append(HklPlane(0, -2, 2))
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
    
