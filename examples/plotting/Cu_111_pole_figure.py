#!/usr/bin/env python
import os, numpy as np
from pymicro.crystal.microstructure import Microstructure, Grain, Orientation
from pymicro.crystal.texture import PoleFigure
from matplotlib import pyplot as plt, colors, cm

if __name__ == '__main__':
  '''
  111 Pole figure of a copper sample containing 10000 grains with a fibre 
  texture.
  '''
  eulers = Orientation.read_euler_txt('../data/Cu_111.dat')
  micro = Microstructure(name = 'Cu_111')
  for index in eulers:
    micro.grains.append(Grain(index, eulers[index]))

  # create pole figure (both direct and inverse)
  pf = PoleFigure(hkl='111', proj='stereo', microstructure=micro, verbose=False)
  pf.color_by_grain_id = False
  pf.mksize = 5
  pf.pflegend = False
  pf.plot_pole_figures(plot_sst=True, display=False, save_as='png')

  image_name = os.path.splitext(__file__)[0] + '.png'
  print 'writting %s' % image_name

  from matplotlib import image
  image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
