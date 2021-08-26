import numpy as np
from matplotlib import pyplot as plt
from pymicro.xray.xray_utils import f_atom

# plot X-ray atomic structure factor for Al and Ni
q_values = np.arange(0., 2.01, 0.01)
for element, Z in [('Al', 13), ('Ni', 28)]:
    f = f_atom(q_values, Z)
    plt.plot(q_values, f, label=element)
plt.xlim(0, 2)
plt.xlabel(r'$\sin(\theta)/\lambda$')
plt.ylabel(r'Atomic structure factor $f$')
plt.legend()

import os

image_name = os.path.splitext(__file__)[0] + '.png'
print('writting %s' % image_name)
plt.savefig(image_name, format='png')

from matplotlib import image

image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
