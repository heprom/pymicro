import os, numpy as np
from pymicro.xray.detectors import Xpad
from matplotlib import pyplot as plt, cm

xpad = Xpad()  # create an Xpad object
xpad.mask_flag = 0
# calib
xpad.factorIdoublePixel = 2.64
xpad.XcenDetector = 451.7 + 5 * 3
xpad.YcenDetector = 116.0
xpad.deltaOffset = 13.0
xpad.correction = 'none'

# load image
xpad.load_image('../data/scan_138.raw')
xpad.compute_TwoTh_Psi_arrays(0., 0.)

image_name = os.path.splitext(__file__)[0] + '.png'
print('writting %s' % image_name)
plt.imsave(image_name, np.log10(xpad.corr_data))

from matplotlib import image

image.thumbnail(image_name, 'thumb_' + image_name, scale=200. / 560)
