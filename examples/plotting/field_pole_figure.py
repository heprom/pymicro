from pymicro.crystal.microstructure import *
from pymicro.crystal.texture import *

from pymicro import get_examples_data_dir # import file directory path
PYMICRO_EXAMPLES_DATA_DIR = get_examples_data_dir() # get the file directory path

from matplotlib import pyplot as plt, colors, colorbar, cm
import pathlib as pl 

'''This example demonstrate how a field can be used to color each symbol on
the pole figure with the :py:meth:~`pymicro.crystal.texture.set_map_field`
method.
'''
#orientations = Orientation.read_euler_txt('../data/orientation_set.inp')
#for i in range(600):
#    micro.grains.append(Grain(i, orientations[i + 1]))
euler_list = np.genfromtxt(PYMICRO_EXAMPLES_DATA_DIR / 'orientation_set.inp').tolist()
micro = Microstructure(name='field', autodelete=True)
micro.add_grains(euler_list)

# load strain from dat files
strain_field = np.genfromtxt(PYMICRO_EXAMPLES_DATA_DIR / 'strain_avg_per_grain.dat')[19, ::2]

# build custom pole figures
pf = PoleFigure(microstructure=micro)
pf.mksize = 40
pf.set_map_field('strain', strain_field, field_min_level=0.015, field_max_level=0.025)
fig = plt.figure()
# direct PF
ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9], aspect='equal')
pf.plot_pf(ax=ax1)
plt.title('111 pole figure, cubic elasticity')

# to add the color bar
ax2 = fig.add_axes([0.8, 0.05, 0.05, 0.9])
norm = colors.Normalize(vmin=0.015, vmax=0.025)
cb = colorbar.ColorbarBase(ax2, cmap=cm.hot, norm=norm, orientation='vertical')
cb.set_label('Average strain (mm/mm)')

image_name = os.path.splitext(__file__)[0] + '.png'
print('writing %s' % image_name)
plt.savefig('%s' % image_name, format='png')
del pf
del micro

from matplotlib import image

image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
