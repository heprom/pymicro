import os, numpy as np
from pymicro.xray.detectors import Mar165
from pymicro.file.file_utils import HST_read
from matplotlib import pyplot as plt, cm

mar = Mar165()  # create a mar object
mar.ucen = 981  # position of the direct beam on the x axis
mar.vcen = 1060  # position of the direct beam on the y axis
mar.calib = 495. / 15.  # identified on (111) ring of CeO2 powder
mar.correction = 'bg'
mar.compute_TwoTh_Psi_arrays()

# load background for correction
mar.bg = HST_read('../data/c1_exptime_bgair_5.raw', data_type=np.uint16, dims=(2048, 2048, 1))[:, :, 0].astype(
    np.float32)
# load image
mar.load_image('../data/c1_exptime_air_5.raw')

image_name = os.path.splitext(__file__)[0] + '.png'
print('writting %s' % image_name)
plt.imsave(image_name, mar.corr_data, vmin=0, vmax=2000)

from matplotlib import image

image.thumbnail(image_name, 'thumb_' + image_name, scale=200. / 2048)
