#!/usr/bin/env python
import os, numpy as np
from matplotlib import pyplot as plt
from pymicro.crystal.microstructure import Microstructure, Orientation
from pymicro.crystal.texture import PoleFigure

N = 500  # number of grains
micro = Microstructure.random_texture(N)
# look at misorientation between pair of grains
misorientations = []
for i in range(2, len(micro.grains)):
    o1 = micro.get_grain(i - 1).orientation
    o2 = micro.get_grain(i).orientation
    w = 180 / np.pi * o1.disorientation(o2)[0]
    misorientations.append(w)

# plt misorientations histogram
plt.hist(misorientations, bins=20, normed=True, cumulative=False)
plt.title('misorientation distribution, random texture %d grains' % N)
psis_dg = np.linspace(0, 63, 5 * 63 + 1)
misorientations_MacKenzie = []
for psidg in psis_dg:
    psi = np.pi * psidg / 180
    misorientations_MacKenzie.append(Orientation.misorientation_MacKenzie(psi))
plt.plot(psis_dg, misorientations_MacKenzie, 'k--', linewidth=2, label='MacKenzie (1958)')
plt.ylim(0, 0.05)
plt.legend(loc='upper left')
image_name = os.path.splitext(__file__)[0] + '.png'
print('writting %s' % image_name)
plt.savefig(image_name, format='png')

from matplotlib import image

image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
