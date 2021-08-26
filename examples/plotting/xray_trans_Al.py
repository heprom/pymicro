from pymicro.xray.xray_utils import plot_xray_trans

# plot X-ray transmission through 1 mm of Al
plot_xray_trans('Al', display=False)

import os

image_name = os.path.splitext(__file__)[0] + '.png'
print('writting %s' % image_name)

from matplotlib import image

image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
