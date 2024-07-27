#!/usr/bin/env python
import os, numpy as np
from pymicro.crystal.microstructure import Microstructure
from pymicro.crystal.texture import PoleFigure

from pymicro import get_examples_data_dir
PYMICRO_EXAMPLES_DATA_DIR = get_examples_data_dir()

'''
200 Pole figure of a copper sample containing 10000 grains with a fibre
texture.
'''
euler_path = os.path.join(PYMICRO_EXAMPLES_DATA_DIR, 'Cu_111.dat')
euler_list = np.genfromtxt(euler_path, usecols=(0, 1, 2), max_rows=1000)
micro = Microstructure(name='Cu_200', autodelete=True)
micro.add_grains(euler_list)

# create pole figure (both direct and inverse)
pf = PoleFigure(hkl='200', proj='stereo', microstructure=micro)
pf.color_by_grain_id = False
pf.mksize = 5
pf.pflegend = False
pf.plot_pole_figures(plot_sst=True, display=False, save_as='png')
del pf
del micro

image_name = os.path.splitext(__file__)[0] + '.png'
print('writing %s' % image_name)

from matplotlib import image

image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
