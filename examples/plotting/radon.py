import os, numpy as np
from pymicro.file.file_utils import HST_read
from skimage.transform import radon
from matplotlib import pyplot as plt

'''
Example of use of the radon transform.
'''
data = HST_read('../data/mousse_250x250x250_uint8.raw', autoparse_filename=True, zrange=range(1))[:, :, 0]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
ax1.set_title('Original data')
ax1.imshow(data.T, cmap=plt.cm.Greys_r)

theta = np.linspace(0., 180., max(data.shape), endpoint=False)
sinogram = radon(data, theta=theta, circle=False)
ax2.set_title('Radon transform (Sinogram)')
ax2.set_xlabel('Projection angle (deg)')
ax2.set_ylabel('Projection position (pixels)')
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
            extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
fig.subplots_adjust(left=0.05, right=0.95)

image_name = os.path.splitext(__file__)[0] + '.png'
print('writting %s' % image_name)
plt.savefig(image_name, format='png')

from matplotlib import image

image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
