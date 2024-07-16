import os
from pymicro.crystal.microstructure import Microstructure
from pymicro.crystal.lattice import HklPlane
#from pymicro.examples import PYMICRO_EXAMPLES_DATA_DIR
import pathlib as pl 

from matplotlib import pyplot as plt, image

'''
Plot an EBSD map for Titanium with IPF color and crystal lattices.
'''
PYMICRO_EXAMPLES_DATA_DIR = "../data" 
m = Microstructure(os.path.join(PYMICRO_EXAMPLES_DATA_DIR, 'Ti_ebsd_demo_data.h5'))
fig, ax = m.view_slice(color='ipf', axis=[1, 0, 0], show_lattices=True, display=False)
ax.set_title('EBSD map with [100] IPF coloring and showing crystal lattices')

image_name = os.path.splitext(__file__)[0] + '.png'
print('writing %s' % image_name)
plt.savefig(image_name, format='png')
del m

image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
