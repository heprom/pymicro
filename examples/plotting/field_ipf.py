import os, numpy as np
from pymicro.crystal.texture import PoleFigure
from pymicro.crystal.microstructure import Microstructure, Grain, Orientation
from matplotlib import pyplot as plt, colors, colorbar, cm

'''
An inverse pole figure with symbols colored by the grain size.
'''
euler_list = np.genfromtxt('../data/EBSD_20grains.txt', usecols=[1, 2, 3]).tolist()
grain_sizes = np.genfromtxt('../data/EBSD_20grains.txt', usecols=[9])
micro = Microstructure(name='test', autodelete=True)
micro.add_grains(euler_list)
micro.set_volumes(grain_sizes)

# build a custom pole figure
pf = PoleFigure(microstructure=micro, hkl='001')#, lattice=Ti7Al)
#pf.resize_markers = True
pf.mksize = 100
pf.set_map_field('grain_size', field_min_level=0.0, field_max_level=1000., lut='jet')
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
print('writing %s' % image_name)
plt.savefig(image_name, format='png')
del pf
del micro

from matplotlib import image

image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
