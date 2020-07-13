import os, numpy as np
from pymicro.crystal.texture import PoleFigure
from pymicro.crystal.microstructure import Microstructure, Grain, Orientation
from matplotlib import pyplot as plt, colors, colorbar, cm

'''
An inverse pole figure with symboled colored by the grain size.
'''
eulers = Orientation.read_orientations('../data/EBSD_20grains.txt', data_type='euler', usecols=[1, 2, 3])
grain_sizes = np.genfromtxt('../data/EBSD_20grains.txt', usecols=[9])
micro = Microstructure(name='test')
for i in range(20):
    micro.grains.append(Grain(i + 1, eulers[i + 1]))
    micro.get_grain(i + 1).volume = grain_sizes[i]

# build a custom pole figure
pf = PoleFigure(microstructure=micro, hkl='001')#, lattice=Ti7Al)
#pf.resize_markers = True
pf.mksize = 100
pf.set_map_field('strain', grain_sizes, field_min_level=0.0, field_max_level=1000., lut='jet')
fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9], aspect='equal')
pf.plot_sst(ax=ax1, mk='o')
ax1.set_title('%s-axis SST inverse %s projection' % (pf.axis, pf.proj))

# to add the color bar
ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
norm = colors.Normalize(vmin=0., vmax=1000.)
cb = colorbar.ColorbarBase(ax2, cmap=cm.jet, norm=norm, orientation='vertical')
cb.set_label('Grain size (pixels)')

image_name = os.path.splitext(__file__)[0] + '.png'
print('writting %s' % image_name)
plt.savefig(image_name, format='png')

from matplotlib import image

image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
