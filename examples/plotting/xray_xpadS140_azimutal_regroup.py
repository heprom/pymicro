import os, numpy as np
from pymicro.xray.detectors import Xpad
from matplotlib import pyplot as plt, cm

xpad = Xpad()  # create an Xpad object
xpad.mask_flag = 0
# calib
xpad.orientation = 'vertical'
xpad.calib = 85.62
xpad.factorIdoublePixel = 2.64
xpad.XcenDetector = 451.7 + 5 * 3
xpad.YcenDetector = 116.0
xpad.deltaOffset = 13.0
xpad.correction = 'none'

# load image
xpad.load_image('../data/scan_138.raw')
xpad.compute_TwoTh_Psi_arrays(diffracto_delta=5., diffracto_gamma=0.)

two_theta_values, intensity, counts = xpad.azimuthal_regroup(17.0, \
                                                                23.0, 1. / xpad.calib, psi_min=-3, psi_max=3,
                                                                write_txt=False, \
                                                                output_image=False)
plt.figure()
plt.plot(two_theta_values, intensity, 'ko-', label=r"data #%d" % 138)
plt.xlabel('2 theta (deg)')
plt.ylabel('Intensity')
image_name = os.path.splitext(__file__)[0] + '.png'
print('writting %s' % image_name)
plt.savefig(image_name, format='png')

from matplotlib import image

image.thumbnail(image_name, 'thumb_' + image_name, scale=0.2)
