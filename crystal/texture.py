#!/usr/bin/env python
import numpy as np
from microstructure import Orientation, Grain, Microstructure
from matplotlib import pyplot as plt, colors, cm

class PoleFigure:

  '''
  Create an empty PoleFigure object associated with an empty Microstructure.
  
  structure: crystal structure (only 'cubic' for now)
  proj: projection type, can be either 'stereo' (default) or 'flat'
  mksize: marker size as displayed on the plots
  verbose: verbose mode
  color_by_grain_id: color the spot on the pole figure with grain color
  pflegend: show the legend (only if color_by_grain_id is active)
  '''
  def __init__(self, structure='cubic', proj='stereo', \
               microstructure = Microstructure()):
    self.structure = structure
    self.proj = proj
    self.microstructure = microstructure
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

  '''Plot and save a picture with both direct and inverse pole figures...'''
  def plot_pole_figures(self):
    fig = plt.figure(figsize=(10,5))
    # direct PF
    ax1 = fig.add_subplot(121, aspect='equal')
    self.plot_pf(ax = ax1, mk='o', col='k', ann=False)
    # inverse PF
    ax2 = fig.add_subplot(122, aspect='equal')
    self.plot_ipf(ax = ax2)
    plt.savefig(self.microstructure.name + '_pole_figure.pdf',format='pdf')

  '''Helper function to plot a crystal direction.'''
  def plot_crystal_dir(self, c_dir, mk='o', col='k', ax=None, ann=False, lab=''):
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

  '''Helper function to plot a curve between two crystal directions.'''
  def plot_line_between_crystal_dir(self, c1, c2, ax=None, steps=11):
    path = np.zeros((steps,2), dtype=float)
    for j,i in enumerate(np.linspace(0,1,steps)):
      ci = i*c1 + (1-i)*c2
      ci /= np.linalg.norm(ci)
      if self.proj == 'stereo':
        ci += self.z
        ci /= ci[2]
      path[j,0] = ci[0]
      path[j,1] = ci[1]
    ax.plot(path[:,0], path[:,1], 'g-', markersize=self.mksize)

  '''Helper function to plot the background of the pole figure. '''
  def plot_pf_background(self, ax):
    an = np.linspace(0,2*np.pi,100)
    plt.hold('on')
    ax.plot(np.cos(an), np.sin(an), 'k-')
    ax.plot([-1,1], [0,0], 'k-')
    ax.plot([0,0], [-1,1], 'k-')

  ''' Create the direct pole figure. '''
  def plot_pf(self, ax=None, mk='o', col='k', ann=False):
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
      ax.legend(bbox_to_anchor=(0.1, 1), loc=1, numpoints=1)#borderaxespad=0.)
    ax.axis('off')
    ax.set_title('direct %s projection' % self.proj)

  ''' Create the inverse pole figure for direction Z. '''
  def plot_ipf(self, ax=None, mk='s', col='r', ann=False):
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

if __name__ == '__main__':
  '''
  The following is a counter example for OrientationMatrix2Euler.
  Indeed it gives g[3,3] = 1 so according to DCTdoc_Crystal_orientation_descriptions.pdf
  we should have Phi=0.0 and phi1=phi2=10.0 which is obviously not the case here.
  '''
  grain = Grain(1, Orientation.from_euler(np.array([10.0, 0.0, 0.0])))
  #grain = Grain(1, Orientation.from_euler(np.array([142.8, 32.0, 214.4])))
  euler = grain.orientation.euler # those are computed by OrientationMatrix2Euler
  grain2 = Grain(2, Orientation.from_euler(euler))
  print grain.orientation
  print grain2.orientation
  micro = Microstructure()
  micro.grains.append(grain)
  micro.grains.append(grain2)
  pf = PoleFigure(structure='cubic', proj='flat', microstructure=micro)
  print pf.c111s
  fig = plt.figure(figsize=(12,5))
  ax1 = fig.add_subplot(131, aspect='equal')
  pf.plot_pf(ax = ax1)
  ax2 = fig.add_subplot(132, aspect='equal')
  pf.proj = 'flat'
  pf.plot_ipf(ax = ax2)
  # change projection mode to stereo
  pf.proj = 'stereo'
  ax3 = fig.add_subplot(133, aspect='equal')
  pf.plot_ipf(ax = ax3)
  plt.subplots_adjust(top=1, bottom=0, left=0, right=1)
  plt.savefig('ipf_full.pdf',format='pdf')
  plt.show()
