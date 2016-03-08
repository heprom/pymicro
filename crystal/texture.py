#!/usr/bin/env python
import numpy as np
from pymicro.crystal.lattice import Lattice, SlipSystem
from pymicro.crystal.microstructure import Orientation, Grain, Microstructure
from matplotlib import pyplot as plt, colors, cm

class PoleFigure:
  '''A class to handle pole figures.
  
  A pole figure is a popular tool to plot multiple crystal orientations, 
  either in the sample coordinate system (direct pole figure) or 
  alternatively plotting a particular direction in the crystal 
  coordinate system (inverse pole figure).
  '''

  def __init__(self, microstructure=None, lattice=None, axis = 'Z', hkl='111', 
    proj='stereo', verbose=False):
    '''
    Create an empty PoleFigure object associated with an empty Microstructure.
    
    :param microstructure: the :py:class:`~pymicro.crystal.microstructure.Microstructure` containing the collection of orientations to plot (None by default).
    :param lattice: the crystal :py:class:`~pymicro.crystal.lattice.Lattice`.
    :param str axis: the pole figure axis ('Z' by default), vertical axis in the direct pole figure and direction plotted on the inverse pole figure.

    .. warning::

       Any crystal structure is now supported (you have to set the proper 
       crystal lattice) but it has only really be tested for cubic.

    :param str hkl: slip plane family ('111' by default)    
    :param str proj: projection type, can be either 'stereo' (default) or 'flat'
    :param bool verbose: verbose mode (False by default)
    '''
    self.proj = proj
    self.axis = axis
    self.map_field = None
    if microstructure:
      self.microstructure = microstructure
    else:
      self.microstructure = Microstructure()
    if lattice:
      self.lattice = lattice
    else:
      self.lattice = Lattice.cubic(1.0)
    self.set_hkl_poles(hkl)
    self.verbose = verbose
    self.mksize = 12
    self.color_by_grain_id = False
    self.pflegend = False
    self.x = np.array([1,0,0])
    self.y = np.array([0,1,0])
    self.z = np.array([0,0,1])
    
    # list all crystal directions
    self.c001s = np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=np.float)
    self.c011s = np.array([[0,1,1],[1,0,1],[1,1,0],[0,-1,1],[-1,0,1],[-1,1,0]], dtype=np.float) / np.sqrt(2)
    self.c111s = np.array([[1,1,1],[-1,-1,1],[1,-1,1],[-1,1,1]], dtype=np.float) / np.sqrt(3)

  def set_hkl_poles(self, hkl):
    '''Set the pole list to plot.

    :params str hkl: slip plane family ('111' by default)
    '''
    self.family = hkl # keep a record of this
    planes = self.lattice.get_hkl_family(self.family)
    poles = []
    for p in planes:
      poles.append(p.normal())
    self.poles = poles
  
  def set_map_field(self, field_name, field=None, field_min_level=None, field_max_level=None, lut='hot'):
    '''Set the PoleFigure to color poles with the given field.
    
    This method activates a mode where each symbol in the pole figure 
    is color coded with respect to a field, which can be either the 
    grain id, or a given field given in form of a list. If the grain 
    volume or strain. For the grain id, the color is set according the 
    each grain id in the :py:class:`~pymicro.crystal.microstructure.Microstructure` 
    and the :py:meth:`~pymicro.crystal.microstructure.rand_cmap` function. 
    For a given field, the color is set from the lookup table and 
    according to the value in the given list. The list must contain a 
    record for each grain. Minimum and maximum value to map the field 
    values and the colors can be specify, if not they are directly taken 
    as the min() and max() of the field. 
    
    :param str field_name: The field name, could be 'grain_id', or any other name describing the field.
    :param list field: A list containing a record for each grain.
    :param float field_min_level: The minimum value to use for this field.
    :param float field_max_level: The maximum value to use for this field.
    :param str lut: A string describing the colormap to use (among matplotlib ones available).
    :raise ValueError: If the given field does not contain enough values.
    '''
    self.map_field = field_name
    self.lut = lut
    if field_name == 'grain_id':
      self.field = [g.id for g in self.microstructure.grains]
    else:
      if len(field) < len(self.microstructure.grains):
        raise ValueError('The field must contain a record for each grain in the microstructure')
      self.field = field
      if not field_min_level:
        self.field_min_level = field.min()
      else:
        self.field_min_level = field_min_level
      if not field_max_level:
        self.field_max_level = field.max()
      else:
        self.field_max_level = field_max_level
      
  def plot_pole_figures(self, plot_sst=True, display=True, save_as='pdf'):
    '''Plot and save a picture with both direct and inverse pole figures.
    
    :param bool plot_sst: controls wether to plot the full inverse pole \
    figure or only the standard stereographic triangle (True by default).
    :param bool display: display the plot if True, else save a picture \
    of the pole figures (True by default)
    :param str save_as: File format used to save the image such as pdf \
    or png ('pdf' by default)
    
    ::
    
      micro = Microstructure(name = 'AlLi_sam8')
      micro.grains.append(Grain(11, Orientation.from_euler(np.array([262.364, 16.836, 104.691]))))
      Al_fcc = Lattice.face_centered_cubic(0.405) # not really necessary since default lattice is cubic
      pf = PoleFigure(microstructure=micro, proj='stereo', lattice=Al_fcc, hkl='111')
      pf.mksize = 12
      pf.set_map_field('grain_id')
      pf.pflegend = True # this works well for a few grains
      pf.plot_pole_figures()
    
    .. figure:: _static/AlLi_sam8_pole_figure.png
        :width: 750 px
        :height: 375 px
        :alt: AlLi_sam8_pole_figure
        :align: center

        A 111 pole figure plotted for a single crystal orientation.
    '''
    fig = plt.figure(figsize=(10,5))
    # direct PF
    ax1 = fig.add_subplot(121, aspect='equal')
    self.plot_pf(ax = ax1, mk='o', col='k', ann=False)
    # inverse PF
    ax2 = fig.add_subplot(122, aspect='equal')
    if plot_sst:
      self.plot_sst(ax = ax2)
    else:
      self.plot_ipf(ax = ax2)
    if display:
      plt.show()
    else:
      plt.savefig('%s_pole_figure.%s' % (self.microstructure.name, save_as) , format=save_as)

  def plot_crystal_dir(self, c_dir, mk='o', col='k', ax=None, ann=False, lab=''):
    '''Function to plot a crystal direction on a pole figure.
    
    :param c_dir: A vector describing the crystal direction.    
    :param mk: marker used to plot the pole (disc by default).
    :param col: color used to plot the pole (black by default).
    :param ax: a reference to a pyplot ax to draw the pole.
    :param bool ann: Annotate the pole with the coordinates of the vector if True (False by default).    
    :param str lab: Label to use in the legend of the plot ('' by default).
    :raise ValueError: if the projection type is not supported
    '''
    if c_dir[2] < 0: c_dir *= -1 # make unit vector have z>0
    if self.proj == 'flat':
      cp = c_dir
    elif self.proj == 'stereo':
      c = c_dir + self.z
      c /= c[2] # SP'/SP = r/z with r=1
      cp = c
      #cp = np.cross(c, self.z)
    else:
      raise ValueError('Error, unsupported projection type', proj)
    ax.plot(cp[0], cp[1], linewidth=0, markerfacecolor=col, marker=mk, \
      markeredgecolor=col, markersize=self.mksize, label=lab)
    # Next 3 lines are necessary in case c_dir[2]=0, as for Euler angles [45, 45, 0]
    if c_dir[2] < 0.000001:
      ax.plot(-cp[0], -cp[1], linewidth=0, markerfacecolor=col, marker=mk, \
        markersize=self.mksize, label=lab)
    if ann:
      ax.annotate(c_dir.view(), (cp[0], cp[1]-0.1), xycoords='data',
      fontsize=8, horizontalalignment='center', verticalalignment='center')

  def plot_line_between_crystal_dir(self, c1, c2, ax=None, steps=11, col='k'):
    '''Plot a curve between two crystal directions.

    The curve is actually composed of several straight lines segments to 
    draw from direction 1 to direction 2.

    :param c1: vector describing crystal direction 1
    :param c2: vector describing crystal direction 2
    :param ax: a reference to a pyplot ax to draw the line
    :param int: steps: number of straight lines composing the curve (11 by default)
    :param col: line color (black by default)
    '''
    path = np.zeros((steps, 2), dtype=float)
    for j, i in enumerate(np.linspace(0., 1., steps)):
      ci = i*c1 + (1-i)*c2
      ci /= np.linalg.norm(ci)
      if self.proj == 'stereo':
        ci += self.z
        ci /= ci[2]
      path[j, 0] = ci[0]
      path[j, 1] = ci[1]
    ax.plot(path[:, 0], path[:, 1], color=col, markersize=self.mksize, linewidth=2)

  def plot_pf_background(self, ax, labels=True):
    '''Function to plot the background of the pole figure.
    
    :param ax: a reference to a pyplot ax to draw the backgroud.
    :params bool labels: add lables to axes (True by default).
    '''
    an = np.linspace(0,2*np.pi,100)
    plt.hold('on')
    ax.plot(np.cos(an), np.sin(an), 'k-')
    ax.plot([-1,1], [0,0], 'k-')
    ax.plot([0,0], [-1,1], 'k-')
    axe_labels = ['X', 'Y', 'Z']
    if self.axis == 'Z':
      h = 0; v = 1; u = 2
    elif self.axis == 'Y':
      h = 0; v = 2; u = 1
    else:
      h = 1; v = 2; u = 0
    ax.annotate(axe_labels[h], (1.01, 0.0), xycoords='data',
      fontsize=16, horizontalalignment='left', verticalalignment='center')
    ax.annotate(axe_labels[v], (0.0, 1.01), xycoords='data',
      fontsize=16, horizontalalignment='center', verticalalignment='bottom')

  def plot_pf_dir(self, c_dir, ax=None, mk='o', col='k', ann=False, lab=''):
    '''Plot a crystal direction in a direct pole figure.'''
    if self.axis == 'Z':
      h = 0; v = 1; u = 2
    elif self.axis == 'Y':
      h = 0; v = 2; u = 1
    else:
      h = 1; v = 2; u = 0
	# the direction to plot is given by c_dir[h,v,u]
    if self.verbose: print 'corrected for pf axis:',c_dir[[h,v,u]]
    self.plot_crystal_dir(c_dir[[h,v,u]], mk=mk, col=col, ax=ax, ann=ann, lab=lab)
	  
  def plot_pf(self, ax=None, mk='o', col='k', ann=False):
    '''Create the direct pole figure. 
    
    :param ax: a reference to a pyplot ax to draw the poles.
    :param mk: marker used to plot the poles (disc by default).
    :param col: symbol color (black by default)
    :param bool ann: Annotate the pole with the coordinates of the vector if True (False by default).    
    '''
    self.plot_pf_background(ax)
    for grain in self.microstructure.grains:
      B = grain.orientation_matrix()
      Bt = B.transpose()
      for i, c in enumerate(self.poles):
        label = ''
        c_rot = Bt.dot(c)
        if self.verbose: print 'plotting ',c,' in sample CS (corrected for pf axis):',c_rot
        if self.map_field:
          if self.map_field == 'grain_id':
            col = Microstructure.rand_cmap().colors[grain.id]
            if self.pflegend and i == 0:
              # only add grain legend for its first pole
              label = 'grain ' + str(grain.id)
          else: # use the field value for this grain
            color = int(255*(self.field[grain.id] - self.field_min_level) / float(self.field_max_level - self.field_min_level))
            col_cmap = cm.get_cmap(self.lut, 256)
            col = col_cmap(np.arange(256))[color] # directly access the color
        self.plot_pf_dir(c_rot, mk=mk, col=col, ax=ax, ann=ann, lab=label)
    ax.axis([-1.1,1.1,-1.1,1.1])
    if self.pflegend and self.map_field == 'grain_id':
      ax.legend(bbox_to_anchor=(0.05, 1), loc=1, numpoints=1, \
        prop={'size':10})
    ax.axis('off')
    ax.set_title('{%s} direct %s projection' % (self.family, self.proj))

  def create_pf_contour(self, ax=None, ang_step=10):
    '''Compute the distribution of orientation and plot it using contouring.
    
    This plot the distribution of orientation in the microstructure 
    associated with this PoleFigure instance, as a continuous 
    distribution using angular bining with the specified step.
    the distribution is constructed at runtime by discretizing the 
    angular space and counting the number of poles in each bin.
    Then the plot_pf_contour method is called to actually plot the data.
    
    :param ax: a reference to a pyplot ax to draw the contours.
    :param int ang_step: angular step in degrees to use for constructing the orientation distribution data (10 degrees by default)
    '''
    # discretise the angular space (azimuth and altitude)
    ang_step *= np.pi / 180 # change to radians
    n_phi = 1 + 2*np.pi/ang_step
    n_psi = 1 + 0.5*np.pi/ang_step
    phis = np.linspace(0, 2*np.pi, n_phi)
    psis = np.linspace(0, np.pi/2, n_psi)
    xv, yv = np.meshgrid(phis, psis)
    values = np.zeros((n_psi, n_phi), dtype=int)
    for grain in self.microstructure.grains:
      B = grain.orientation_matrix()
      Bt = B.transpose()
      for c in self.poles:
        c_rot = Bt.dot(c)
        # handle poles pointing down
        if c_rot[2] < 0: c_rot *= -1 # make unit vector have z>0
        if c_rot[1] >=0:
          phi = np.arccos(c_rot[0]/np.sqrt(c_rot[0]**2 + c_rot[1]**2))
        else:
          phi = 2*np.pi - np.arccos(c_rot[0]/np.sqrt(c_rot[0]**2 + c_rot[1]**2))
        psi = np.arccos(c_rot[2]) # since c_rot is normed
        i_phi = int((phi + 0.5*ang_step) / ang_step) % n_phi
        j_psi = int((psi + 0.5*ang_step)/ ang_step) % n_psi
        values[j_psi, i_phi] += 1
    if self.proj == 'stereo': # double check which one is flat/stereo
      x = (2*yv/np.pi)*np.cos(xv)
      y = (2*yv/np.pi)*np.sin(xv)
    else:
      x = np.sin(yv)*np.cos(xv)
      y = np.sin(yv)*np.sin(xv)
    # close the pole figure by duplicating azimuth=0
    values[:,-1] = values[:,0]
    self.plot_pf_contour(ax, x, y, values)
    
  def plot_pf_contour(self, ax, x, y, values):
    '''Plot the direct pole figure using contours. '''
    self.plot_pf_background(ax)
    ax.contourf(x, y, values)
    #ax.plot(x, y, 'ko')
    ax.axis([-1.1,1.1,-1.1,1.1])
    ax.axis('off')
    ax.set_title('{%s} direct %s projection' % (self.family, self.proj))

  @staticmethod
  def sst_symmetry_cubic(z_rot):
    '''Transform a given vector according to the cubic symmetry.
    
    This function transform a vector so that it lies in the unit SST triangle.
    
    :param z_rot: vector to transform.
    :return: the transformed vector.
    '''
    if z_rot[0] < 0: z_rot[0] = -z_rot[0]
    if z_rot[1] < 0: z_rot[1] = -z_rot[1]
    if z_rot[2] < 0: z_rot[2] = -z_rot[2]

    if (z_rot[2] > z_rot[1]):
      z_rot[1], z_rot[2] = z_rot[2], z_rot[1]
    
    if (z_rot[1] > z_rot[0]):
      z_rot[0], z_rot[1] = z_rot[1], z_rot[0]
      
    if (z_rot[2] > z_rot[1]):
      z_rot[1], z_rot[2] = z_rot[2], z_rot[1]
      
    return np.array([z_rot[1], z_rot[2], z_rot[0]])
    
  def get_color_from_field(self, grain):
    if self.map_field:
      if self.map_field == 'grain_id':
        col = Microstructure.rand_cmap().colors[grain.id]
      else: # use the field value for this grain
        color = int(255*(self.field[grain.id] - self.field_min_level) / float(self.field_max_level - self.field_min_level))
        col_cmap = cm.get_cmap(self.lut, 256)
        col = col_cmap(np.arange(256))[color] # directly access the color
      return col
    else:
      return (0, 0, 0)

  def plot_sst(self, ax=None, mk='s', col='k', ann=False):
    ''' Create the inverse pole figure in the unit standard triangle. 

    :param ax: a reference to a pyplot ax to draw the poles.
    :param mk: marker used to plot the poles (square by default).
    :param col: symbol color (black by default)
    :param bool ann: Annotate the pole with the coordinates of the vector if True (False by default).    
    '''
    c001 = np.array([0,0,1])
    c101 = np.array([1,0,1])
    c111 = np.array([1,1,1])
    self.plot_line_between_crystal_dir(c001, c101, ax=ax)
    self.plot_line_between_crystal_dir(c001, c111, ax=ax)
    self.plot_line_between_crystal_dir(c101, c111, ax=ax)
    # now plot the sample axis
    for grain in self.microstructure.grains:
      B = grain.orientation_matrix()
      # compute axis and apply SST symmetry
      if self.axis == 'Z':
        axis = self.z
      elif self.axis == 'Y':
        axis = self.y
      else:
        axis = self.x
      axis_rot = self.sst_symmetry_cubic(B.dot(axis))
      label = ''
      if self.map_field == 'grain_id':
        label = 'grain ' + str(grain.id)
      self.plot_crystal_dir(axis_rot, mk=mk, col=self.get_color_from_field(grain), ax=ax, ann=ann, lab=label)
      if self.verbose: print 'plotting ',self.axis,' in crystal CS:',axis_rot
    ax.axis('off')
    ax.axis([-0.05,0.45,-0.05,0.40])
    ax.set_title('%s-axis SST inverse %s projection' % (self.axis, self.proj))
    
  def plot_ipf(self, ax=None, mk='s', col='k', ann=False):
    ''' Create the inverse pole figure for direction Z. 
    
    :param ax: a reference to a pyplot ax to draw the poles.
    :param mk: marker used to plot the poles (square by default).
    :param col: symbol color (black by default)
    :param bool ann: Annotate the pole with the coordinates of the vector if True (False by default).    
    '''
    self.plot_pf_background(ax)
    for c in self.c111s:
      for i in range(3):
        d = c.copy(); d[i] = 0
        e = np.zeros_like(c); e[i] = c[i]
        self.plot_line_between_crystal_dir(c, d, ax=ax)
        self.plot_line_between_crystal_dir(c, e, ax=ax)
    markers = ['s', 'o', '^']
    for i, dirs in enumerate([self.c001s, self.c011s, self.c111s]):
      [self.plot_crystal_dir(c, mk=markers[i], col='k', ax=ax, ann=False) for c in dirs]
      # also plot the negative direction of those lying in the plane z==0
      for c in dirs:
        if np.dot(c,self.z) == 0.0:
          self.plot_crystal_dir(-c, mk=markers[i], col='k', ax=ax, ann=False)
    # now plot the sample axis
    for grain in self.microstructure.grains:
      B = grain.orientation_matrix()
      if self.axis == 'Z':
        axis = self.z
      elif self.axis == 'Y':
        axis = self.y
      else:
        axis = self.x
      axis_rot = B.dot(axis)
      if self.map_field == 'grain_id':
        col = Microstructure.rand_cmap().colors[grain.id]
      self.plot_crystal_dir(axis_rot, mk=mk, col=col, ax=ax, ann=ann)
      if self.verbose: print 'plotting ',self.axis,' in crystal CS:',axis_rot
    ax.axis([-1.1,1.1,-1.1,1.1])
    ax.axis('off')
    ax.set_title('%s-axis inverse %s projection' % (self.axis, self.proj))

  @staticmethod
  def plot(orientation):
    '''Plot a pole figure (both direct and inverse) for a single orientation.

    :param orientation: the crystalline :py:class:`~pymicro.crystal.microstructure.Microstructure` to plot.
    '''
    micro = Microstructure()
    micro.grains.append(Grain(1, orientation))
    pf = PoleFigure(microstructure=micro)
    pf.plot_pole_figures(display=True)

  @staticmethod
  def plot_euler(phi1, Phi, phi2):
    '''Directly plot a pole figure for a single orientation given its 
    three Euler angles.

    ::
    
      PoleFigure.plot_euler(10, 20, 30)

    :param float phi1: first Euler angle (in degree).
    :param float Phi: second Euler angle (in degree).
    :param float phi2: third Euler angle (in degree).
    '''
    PoleFigure.plot(Orientation.from_euler(np.array([phi1, Phi, phi2])))
    
class TaylorModel:
  '''A class to carry out texture evolution with the Taylor model.
  
  Briefly explain the full constrained Taylor model [ref 1938].  
  '''

  def __init__(self, microstructure):
    self.micro = microstructure # Microstructure instance
    self.slip_systems = SlipSystem.get_slip_systems('111')
    self.nact = 5 # number of active slip systems in one grain to accomodate the plastic strain
    self.dt = 1.e-3
    self.max_time = 0.001 # sec
    self.time = 0.0
    self.L = np.array([[-0.5, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 1.0]]) # velocity gradient
  
  def compute_step(self, g, check=True):
    Wc = np.zeros((3, 3), dtype=np.float)
    # compute Schmid factors
    SF = []
    for s in self.slip_systems:
      SF.append(g.schmid_factor(s))
    ss_rank = np.zeros(self.nact, dtype=int)
    # rank the slip systems by SF
    for i in range(self.nact):
      ss_rank[i] = np.argmax(SF)
      print('index of ss % d is %d' % (i, ss_rank[i]))
      SF[ss_rank[i]] = 0.0
    # now we need to solve: L = gam1*m1 + gam2*m2+ ...
    iu = np.triu_indices(3) # indices of the upper part of a 3x3 matrix
    L = self.L[iu][:5] # form a vector with the velocity gradient components
    M = np.zeros((5, self.nact), dtype=np.float)
    for i in range(len(ss_rank)):
      s = self.slip_systems[ss_rank[i]]
      m = g.orientation.slip_system_orientation_tensor(s)
      #m = g.orientation.slip_system_orientation_strain_tensor(s)
      M[0, i] += m[0, 0]
      M[1, i] += m[0, 1]
      M[2, i] += m[0, 2]
      M[3, i] += m[1, 1]
      M[4, i] += m[1, 2]
      #M[5, i] += m[2, 2]
    dgammas = np.linalg.lstsq(M, L, rcond=1.e-3)[0]
    '''
    U, s, V = np.linalg.svd(M) # solve by SVD
    print 'U:\n'
    print U
    print 's:\n'
    print s
    print 'V:\n'
    print V
    pinv_svd = np.dot(np.dot(V.T, np.linalg.inv(np.diag(s))), U.T)
    dgammas_svd = np.dot(pinv_svd, L) # solving Ax=b computing x = A^-1*b
    print 'dgammas (SVD) =', dgammas_svd
    '''
    print 'dgammas (LST) =', dgammas
    if check:
      # check consistency
      Lcheck = np.zeros((3, 3), dtype=np.float)
      for i in range(len(ss_rank)):
        s = self.slip_systems[ss_rank[i]]
        ms = g.orientation.slip_system_orientation_tensor(s)
        #ms = g.orientation.slip_system_orientation_strain_tensor(s)
        Lcheck += dgammas[i] * ms
      print 'check:',np.sum(Lcheck - self.L),'\n', Lcheck
      if abs(np.sum(Lcheck - self.L)) > 1e-1:
        raise ValueError('Problem with solving for plastic slip, trying to increase the number of active slip systems')
    # compute the plastic spin
    for i in range(len(ss_rank)):
      s = self.slip_systems[ss_rank[i]]
      qs = g.orientation.slip_system_orientation_rotation_tensor(s)
      Wc += dgammas[i] * qs
    print 'plastic spin:\n', Wc
    return Wc, dgammas
