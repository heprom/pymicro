import os, numpy as np
from pymicro.crystal.texture import PoleFigure
from pymicro.crystal.microstructure import Microstructure

"""
Example of a pole figure of a random microstructure with 200 grains. The 
poles are colored according using IPF coloring and resized proportionally 
to each grain volume.
"""
# create a microstructure with a random texture and 200 grains
micro = Microstructure.random_texture(n=200)
micro.autodelete = True

# set random values for the grain volumes
np.random.seed(22)
for g in micro.grains:
    g['volume'] = 100 * np.random.random() ** 3
    g.update()
micro.grains.flush()

# first pole figure
pf = PoleFigure(microstructure=micro)
pf.resize_markers = True
pf.set_hkl_poles('001')
pf.axis = 'Z'
pf.set_map_field('ipf')
pf.plot_pole_figures(plot_sst=True, display=False, save_as='png')
del pf
del micro

image_name = os.path.splitext(__file__)[0] + '.png'
print('writing %s' % image_name)

from matplotlib import image

image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
