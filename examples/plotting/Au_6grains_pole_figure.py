#!/usr/bin/env python
import os, numpy as np
from pymicro.crystal.microstructure import Microstructure, Grain, Orientation
from pymicro.crystal.texture import PoleFigure
from matplotlib import pyplot as plt, colors, cm

if __name__ == '__main__':
  '''
  Pole figure of a gold sample containing 6 grains with a strong <111> fiber texture.
  A Microstructure object is first created with the 6 grains of interest.
  The grain ids corerespond to the actual grain number (in an EBSD scan for instance).
  A PoleFigure object is then created using this microstructure and the pole figures
  (both direct and inverse) are drawn by calling the plot_pole_figures() method.
  '''
  micro = Microstructure(name = 'Au_6grains')
  micro.grains.append(Grain(1158, Orientation.from_euler(np.array([344.776,52.2589,53.9933]))))
  micro.grains.append(Grain(1349, Orientation.from_euler(np.array([344.899,125.961,217.330]))))
  micro.grains.append(Grain(1585, Orientation.from_euler(np.array([228.039,57.4791,143.171]))))
  micro.grains.append(Grain(1805, Orientation.from_euler(np.array([186.741,60.333,43.311]))))
  micro.grains.append(Grain(1833, Orientation.from_euler(np.array([151.709,55.0406,44.1051]))))
  micro.grains.append(Grain(2268, Orientation.from_euler(np.array([237.262,125.149,225.615]))))

  # create pole figure (both direct and inverse)
  pf = PoleFigure(hkl='111', axis='Z', proj='stereo', microstructure=micro)
  pf.color_by_grain_id = True
  pf.pflegend = True # this works well for a few grains
  pf.plot_pole_figures(plot_sst=True, display=False, save_as='png')

  image_name = os.path.splitext(__file__)[0] + '.png'
  print 'writting %s' % image_name

  from matplotlib import image
  image.thumbnail(image_name, 'thumb_' + image_name, 0.2)

