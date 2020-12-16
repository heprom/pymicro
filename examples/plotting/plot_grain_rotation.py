import numpy as np
import os
from pymicro.crystal.microstructure import Microstructure, Orientation, Grain
from pymicro.crystal.texture import PoleFigure
from matplotlib import pyplot as plt

# read data from Z-set calculation (50% tension load)
data = np.genfromtxt('../data/R_1g.dat')
t, R11, R22, R33, R12, R23, R31, R21, R32, R13, _, _, _, _ = data.T
step = 1 # plot every step point
max_step = data.shape[0]

# create a microstructure with the initial grain orientation
micro = Microstructure(name='1g', autodelete=True)
g = Grain(50, Orientation.from_euler((12.293, 149.266, -167.068)))
micro.add_grains([(12.293, 149.266, -167.068)], grain_ids=[50])

ipf = PoleFigure(proj='stereo', microstructure=micro)
ipf.mksize = 100
ipf.set_map_field('grain_id')

fig = plt.figure(1, figsize=(6, 5)) # for IPF
ax1 = fig.add_subplot(111, aspect='equal')
print('** plotting the initial orientation (with label for legend) **')
ipf.plot_sst(ax = ax1, mk='.', col='k', ann=False)
ax1.set_title('grain rotation in tension')
axis = np.array([0, 0, 1])

grain = micro.get_grain(50)
cgid = Microstructure.rand_cmap().colors[grain.id] # color by grain id
g = grain.orientation_matrix()
axis_rot_sst_prev = np.array(ipf.sst_symmetry_cubic(g.dot(axis)))
print('** plotting ipf loading axis trajectory **')
for k in range(0, max_step, step):
  rot = np.array([[R11[k], R12[k], R13[k]], 
                  [R21[k], R22[k], R23[k]],
                  [R31[k], R32[k], R33[k]]])
  # apply R^t to the initial orientation given by g
  new_g = rot.transpose().dot(g)
  axis_rot_sst = ipf.sst_symmetry_cubic(new_g.dot(axis))
  ipf.plot_line_between_crystal_dir(axis_rot_sst_prev, axis_rot_sst, ax=ax1, col=cgid, steps=2)
  axis_rot_sst_prev = axis_rot_sst
del ipf
del micro  # automatically delete .h5 and .xdmf

plt.subplots_adjust(bottom=0.0, top=0.9, left=0.0, right=1.0)
image_name = os.path.splitext(__file__)[0] + '.png'
print('writting %s' % image_name)
plt.savefig(image_name, format='png')

from matplotlib import image

image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
