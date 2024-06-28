import os
from pymicro.crystal.microstructure import Microstructure
from pymicro.crystal.lattice import HklPlane

from pymicro import get_examples_data_dir
PYMICRO_EXAMPLES_DATA_DIR = get_examples_data_dir()

from matplotlib import pyplot as plt, image

'''
Plot an EBSD map for Titanium with IPF color and slip traces.
'''
m = Microstructure(os.path.join(PYMICRO_EXAMPLES_DATA_DIR, 'Ti_ebsd_demo_data.h5'))
p_basal = HklPlane(0, 0, 1, m.get_lattice())  # basal plane
fig, ax = m.view_slice(color='ipf', axis=[1, 0, 0], show_slip_traces=True, hkl_planes=[p_basal], display=False)
ax.set_title('EBSD map with [100] IPF coloring and basal slip traces')

image_name = os.path.splitext(__file__)[0] + '.png'
print('writing %s' % image_name)
plt.savefig(image_name, format='png')
del m

image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
