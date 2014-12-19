#!/usr/bin/env python
import numpy as np
from pymicro.crystal.microstructure import Orientation, Grain, Microstructure
from matplotlib import pyplot as plt, colors, cm

class PoleFigure:
  '''A class to handle pole figures.
  
  A pole figure is a popular tool to plot multiple crystal orientations, 
  either in the sample coordinate system (direct pole figure) or 
  alternatively plotting a particular direction in the crystal 
  coordinate system (inverse pole figure).
  '''

  def __init__(self, microstructure=None, structure='cubic', proj='stereo'):
    '''
    Create an empty PoleFigure object associated with an empty Microstructure.
    
    *Parameters*

    **microstructure**: the microstructure containing the collection of orientations to plot (None by default)

    **structure**: the crystal structure

    .. warning::

       Right now only the cubic crystal structure is supported.

    **proj**: projection type, can be either 'stereo' (default) or 'flat'
    
    **mksize**: marker size as displayed on the plots (12 pts by default)
    
    **verbose**: verbose mode (False by default)
    
    **color_by_grain_id**: color the spot on the pole figure with grain color (False by default)
    
    **pflegend**: show the legend (only if color_by_grain_id is active, False by default)
    '''
    self.structure = structure
    self.proj = proj
    if microstructure:
      self.microstructure = microstructure
    else:
      self.microstructure = Microstructure()
    self.mksize = 12
    self.verbose = False
    self.color_by_grain_id = False
    self.pflegend = False
    self.x = np.array([1,0,0])
    self.y = np.array([0,1,0])
    self.z = np.array([0,0,1])
    # list all crystal directions
    if self.structure == 'cubic':
      self.c001s = np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=np.float)
      self.c011s = np.array([[0,1,1],[1,0,1],[1,1,0],[0,-1,1],[-1,0,1],[-1,1,0]], dtype=np.float) / np.sqrt(2)
      self.c111s = np.array([[1,1,1],[-1,-1,1],[1,-1,1],[-1,1,1]], dtype=np.float) / np.sqrt(3)
      #self.c111s = np.array([[-1,-1,-1],[1,1,-1],[-1,1,-1],[1,-1,-1]], dtype=np.float) / np.sqrt(3)
    #elif self.structure == 'hexagonal':
    else:
      raise TypeError('unsupported crystal structure', structure)

  def plot_pole_figures(self, display=True):
    '''Plot and save a picture with both direct and inverse pole figures.
    
    ::
    
      micro = Microstructure(name = 'AlLi_sam8')
      micro.grains.append(Grain(11, Orientation.from_euler(np.array([262.364, 16.836, 104.691]))))
      pf = PoleFigure(structure='cubic', proj='stereo', microstructure=micro)
      pf.color_by_grain_id = True
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
    self.plot_ipf(ax = ax2)
    if display:
      plt.show()
    else:
      plt.savefig(self.microstructure.name + '_pole_figure.pdf', format='pdf')

  def plot_crystal_dir(self, c_dir, mk='o', col='k', ax=None, ann=False, lab=''):
    '''Helper function to plot a crystal direction.'''
    if c_dir[2] < 0: c_dir *= -1 # make unit vector have z>0
    if self.proj == 'flat':
      cp = c_dir
    elif self.proj == 'stereo':
      c = c_dir + self.z
      c /= c[2] # SP'/SP = r/z with r=1
      cp = c
      #cp = np.cross(c, self.z)
    else:
      raise TypeError('Error, unsupported projection type', proj)
    ax.plot(cp[0], cp[1], linewidth=0, markerfacecolor=col, marker=mk, \
      markersize=self.mksize, label=lab)
    if ann:
      ax.annotate(c_dir.view(), (cp[0], cp[1]-0.1), xycoords='data',
        fontsize=8, horizontalalignment='center', verticalalignment='center')

  def plot_line_between_crystal_dir(self, c1, c2, ax=None, steps=11, col='g'):
    '''Helper function to plot a curve between two crystal directions.'''
    path = np.zeros((steps,2), dtype=float)
    for j,i in enumerate(np.linspace(0,1,steps)):
      ci = i*c1 + (1-i)*c2
      ci /= np.linalg.norm(ci)
      if self.proj == 'stereo':
        ci += self.z
        ci /= ci[2]
      path[j,0] = ci[0]
      path[j,1] = ci[1]
    ax.plot(path[:,0], path[:,1], color=col, markersize=self.mksize)

  def plot_pf_background(self, ax):
    '''Helper function to plot the background of the pole figure. '''
    an = np.linspace(0,2*np.pi,100)
    plt.hold('on')
    ax.plot(np.cos(an), np.sin(an), 'k-')
    ax.plot([-1,1], [0,0], 'k-')
    ax.plot([0,0], [-1,1], 'k-')

  def plot_pf(self, ax=None, mk='o', col='k', ann=False):
    '''Create the direct pole figure. '''
    self.plot_pf_background(ax)
    ax.annotate('x', (1.01, 0.0), xycoords='data',
      fontsize=16, horizontalalignment='left', verticalalignment='center')
    ax.annotate('y', (0.0, 1.01), xycoords='data',
      fontsize=16, horizontalalignment='center', verticalalignment='bottom')
    for grain in self.microstructure.grains:
      B = grain.orientation_matrix()
      Bt = B.transpose()
      for c in self.c111s:
        label = ''
        c_rot = Bt.dot(c)
        if self.verbose: print 'plotting ',c,' in sample CS:',c_rot
        if self.color_by_grain_id:
          col = Microstructure.rand_cmap().colors[grain.id]
          if self.pflegend and self.c111s.tolist().index(c.tolist()) == 0:
            # only add grain legend for the first crystal direction
            label = 'grain ' + str(grain.id)
        self.plot_crystal_dir(c_rot, mk=mk, col=col, ax=ax, ann=ann, lab=label)
    ax.axis([-1.1,1.1,-1.1,1.1])
    if self.pflegend and self.color_by_grain_id:
      ax.legend(bbox_to_anchor=(0.05, 1), loc=1, numpoints=1, \
        prop={'size':10})
    ax.axis('off')
    ax.set_title('direct %s projection' % self.proj)

  def sst_symmetry(self, z_rot):
    '''Perform symmetry according to the unit SST triangle.'''
    if z_rot[0] < 0: z_rot[0] *= -1.
    if z_rot[1] < 0: z_rot[1] *= -1.
    if z_rot[1] > z_rot[0]:
      z_rot[1], z_rot[0] = z_rot[0], z_rot[1]
    return z_rot
    
  def plot_sst(self, ax=None, mk='s', col='r', ann=False):
    ''' Create the inverse pole figure in the unit standard triangle. 
    '''
    c001 = np.array([0,0,1])
    c101 = np.array([1,0,1])
    c111 = np.array([1,1,1])
    self.plot_line_between_crystal_dir(c001, c101, ax=ax)
    self.plot_line_between_crystal_dir(c001, c111, ax=ax)
    self.plot_line_between_crystal_dir(c101, c111, ax=ax)
    # now plot the sample z-axis
    for grain in self.microstructure.grains:
      B = grain.orientation_matrix()
      # compute z axis and apply SST symmetry
      z_rot = self.sst_symmetry(B.dot(self.z))
      print z_rot
      self.plot_crystal_dir(z_rot, mk=mk, col=col, ax=ax, ann=ann)
      if self.verbose: print 'plotting ',self.z,' in sample CS:',z_rot
    ax.axis('off')
    ax.set_title('inverse %s projection, unit standard triangle' % self.proj)
    
  def plot_ipf(self, ax=None, mk='s', col='r', ann=False):
    ''' Create the inverse pole figure for direction Z. 
    
    Parameters:
    
    **ax**: a reference to a pyplot ax to draw the figure
    
    **mk**: marker type (square by default)
    
    **col**: marker color (red by default)
    
    **ann**: draw annotation near the crystal direction (False by default)
    '''
    self.plot_pf_background(ax)
    for c in self.c111s:
      for i in range(3):
        d = c.copy(); d[i] = 0
        e = np.zeros_like(c); e[i] = c[i]
        self.plot_line_between_crystal_dir(c, d, ax=ax)
        self.plot_line_between_crystal_dir(c, e, ax=ax)
    #print self.c001s + self.c011s
    markers = ['s', 'o', '^']
    for i, dirs in enumerate([self.c001s, self.c011s, self.c111s]):
      [self.plot_crystal_dir(c, mk=markers[i], col='k', ax=ax, ann=False) for c in dirs]
      # also plot the negative direction of those lying in the plane z==0
      for c in dirs:
        if np.dot(c,self.z) == 0.0:
          self.plot_crystal_dir(-c, mk=markers[i], col='k', ax=ax, ann=False)
    # now plot the sample z-axis
    for grain in self.microstructure.grains:
      B = grain.orientation_matrix()
      z_rot = B.dot(self.z) # HP 09 march 2014 changed from Bt to B
      self.plot_crystal_dir(z_rot, mk=mk, col=col, ax=ax, ann=ann)
      if self.verbose: print 'plotting ',self.z,' in sample CS:',z_rot
    ax.axis([-1.1,1.1,-1.1,1.1])
    ax.axis('off')
    ax.set_title('inverse %s projection' % self.proj)

  @staticmethod
  def plot(orientation):
    '''Plot a pole figure for a single orientation.

    A file empty.pdf will be written with both direct and inverse pole
    figures.
    
    Parameters:
    
    **orientation**: the crystalline `Orientation` to plot.
    '''
    micro = Microstructure()
    micro.grains.append(Grain(1, orientation))
    pf = PoleFigure(micro)
    pf.plot_pole_figures(display=True)

  @staticmethod
  def plot_euler(phi1, Phi, phi2):
    '''Directly plot a pole figure for a single orientation given its 
    three Euler angles.

    Parameters:
    
    **phi1**: first Euler angle.
    
    **Phi**: second Euler angle.
    
    **phi2**: third Euler angle.
    '''
    PoleFigure.plot(Orientation.from_euler(np.array([phi1, Phi, phi2])))
    
