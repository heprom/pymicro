#!/usr/bin/env python
import numpy as np
from microstructure import Grain
from matplotlib import pyplot as plt, colors, cm

class PoleFigure:

  def __init__(self, structure='cubic', proj='stereo'):
    self.structure = structure
    self.proj = proj
    self.x = np.array([1,0,0])
    self.y = np.array([0,1,0])
    self.z = np.array([0,0,1])
    # list all crystal directions
    if self.structure == 'cubic':
      self.c001s = np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=np.float)
      self.c011s = np.array([[0,1,1],[1,0,1],[1,1,0],[0,-1,1],[-1,0,1],[-1,1,0]], dtype=np.float) / np.sqrt(2)
      self.c111s = np.array([[1,1,1],[-1,-1,1],[1,-1,1],[-1,1,1]], dtype=np.float) / np.sqrt(3)
    #elif self.structure == 'hexagonal':
    else:
      raise TypeError('unsupported crystal structure', structure)

  '''Helper function to plot a crystal direction.'''
  def plot_crystal_dir(self, c_dir, mk='o', col='k', ax=None, ann=False):
    if self.proj == 'flat':
      cp = c_dir
    elif self.proj == 'stereo':
      c = c_dir + self.z
      c /= c[2] # SP'/SP = r/z with r=1
      cp = np.cross(c, self.z)
    else:
      raise TypeError('Error, unsupported projection type', proj)
    if c_dir[2] < 0: cp *= -1
    ax.plot(cp[0], cp[1], col, marker=mk, markersize=12)
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
    ax.plot(path[:,0], path[:,1], 'g-', markersize=6)

  def plot_ipf(self, ax=None):
    # create the inverse pole figure
    an = np.linspace(0,2*np.pi,100)
    plt.hold('on')
    ax.plot(np.cos(an), np.sin(an), 'k-')
    ax.plot([-1,1], [0,0], 'k-')
    ax.plot([0,0], [-1,1], 'k-')
    # inverse pole figure for direction Z
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

    ax.axis([-1.1,1.1,-1.1,1.1])
    ax.axis('off')
    ax.set_title(self.proj + ' projection')

if __name__ == '__main__':
  pf1 = PoleFigure(structure='cubic', proj='flat')
  pf2 = PoleFigure(structure='cubic', proj='stereo')
  fig = plt.figure(figsize=(10,5))
  ax1 = fig.add_subplot(121, aspect='equal')
  pf1.plot_ipf(ax=ax1)
  ax2 = fig.add_subplot(122, aspect='equal')
  pf2.plot_ipf(ax=ax2)
  plt.savefig('ipf_full.pdf',format='pdf')
  plt.show()
