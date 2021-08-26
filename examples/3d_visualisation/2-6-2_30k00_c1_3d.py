import os, vtk, numpy as np
from vtk.util import numpy_support
from vtk.util.colors import *
from pymicro.file.file_utils import HST_read, HST_write, HST_info
from pymicro.view.scene3d import Scene3D
from pymicro.view.vtk_utils import *

'''
Create a 3d scene showing the 3d corner crack highlighted with an
elevation filter. The sample edge is also shown and made semi-transparent
The axes are labeled (L,T,S) accordingly to the material directions.
'''
# create the 3D scene
base_name = os.path.splitext(__file__)[0]
s3d = Scene3D(display=False, ren_size=(1000, 800), name=base_name)

# reading volume
data_dir = '../data/'
scan_name = '2-6-2_30k00_c1_masked-labels'
scan_path = os.path.join(data_dir, scan_name)
volsize = np.array(HST_info(scan_path + '.raw.info'))
grid = read_image_data(scan_path + '.raw', volsize, header_size=0, data_type='uint8')

# create 3d actors
crack = elevationFilter(grid, 1, (40, volsize[2] - 25))
skin = contourFilter(grid, 2, opacity=0.5, discrete=True)
outline = data_outline(grid)

# add a color bar
lut = crack.GetMapper().GetLookupTable()
bar = color_bar('Elevation', lut, fmt='%.0f', width=0.5, height=0.075, num_labels=5, font_size=26)

# add actors to the scene
s3d.add(outline)
s3d.add(skin)
s3d.add(crack)
s3d.add(bar)

# setup LTS axes
axes = axes_actor(length=volsize[2], axisLabels=('L', 'T', 'S'), fontSize=30)
s3d.add(axes)

# setup camera and render
cam = setup_camera(size=(volsize))
cam.SetPosition(0 * volsize[0], 2.5 * volsize[1], 5 * volsize[2])
cam.Dolly(1.0)
cam.SetViewUp(0, 0, 1)
s3d.set_camera(cam)
s3d.render()

from matplotlib import image

image_name = base_name + '.png'
image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
