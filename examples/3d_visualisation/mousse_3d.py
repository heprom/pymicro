import os, vtk
import numpy as np

from pymicro.view.scene3d import Scene3D
from pymicro.view.vtk_utils import *
from vtk.util.colors import white, grey, black, lamp_black

'''
Create a 3d scene with a tomographic view of a polymer foam.
The shape is displayed using a simple contour filter. Bounding box
and axes are also added to the scene.
'''

# Create the 3D scene
base_name = os.path.splitext(__file__)[0]
s3d = Scene3D(display=False, ren_size=(800, 800), name=base_name, background=black)

data_dir = '../data'
scan = 'mousse_250x250x250_uint8.raw'
im_file = os.path.join(data_dir, scan)
s_size = scan[:-4].split('_')[-2].split('x')
s_type = scan[:-4].split('_')[-1]
size = [int(s_size[0]), int(s_size[1]), int(s_size[2])]
data = read_image_data(im_file, size, data_type=s_type, verbose=True)

print('adding bounding box')
outline = data_outline(data)
outline.GetProperty().SetColor(white)
s3d.add(outline)

print('isolating the foam with vtkContourFilter')
foam = contourFilter(data, 80, color=grey, diffuseColor=white)
foam.GetProperty().SetSpecular(.4)
foam.GetProperty().SetSpecularPower(10)
s3d.add(foam)

print('adding XYZ axes')
axes = axes_actor(length=100, fontSize=60)
axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(grey)
s3d.add(axes);

print('setting up camera and rendering')
cam = setup_camera(size=size)
cam.SetClippingRange(1, 2000)
s3d.set_camera(cam)
s3d.render()

# thumbnail for the image gallery
from matplotlib import image

image_name = base_name + '.png'
image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
