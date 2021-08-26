#!/usr/bin/env python
import os, vtk
from pymicro.view.scene3d import Scene3D
from pymicro.file.file_utils import HST_read
from pymicro.view.vtk_utils import show_grains, box_3d, axes_actor, setup_camera, show_grains

'''
Create a 3d scene showing the grain map of a polycrystal. Each grain
is colored with a random color.
'''

# create the 3D scene
base_name = os.path.splitext(__file__)[0]
s3d = Scene3D(display=False, ren_size=(800, 800), name=base_name)

data_dir = '../data'
scan_name = 'pure_Ti_216x216x141_uint16.raw'
scan_path = os.path.join(data_dir, scan_name)
data = HST_read(scan_path, autoparse_filename=True)
size = data.shape
print('done reading, volume size is ', size)

# add all the grains
grains = show_grains(data)
s3d.add(grains)

# add outline
outline = box_3d(size=size, line_color=(0., 0., 0.))
s3d.add(outline)

# add axes actor
axes = axes_actor(0.5 * size[0], fontSize=60)
s3d.add(axes);

cam = setup_camera(size=(size))
cam.SetPosition(2.0 * size[0], 0.0 * size[1], 2.0 * size[2])
cam.Dolly(0.75)
s3d.set_camera(cam)
s3d.render()

# thumbnail for the image gallery
from matplotlib import image

image_name = base_name + '.png'
image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
