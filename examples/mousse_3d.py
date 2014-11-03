from pymicro.view.vtk_utils import *
from vtk.util.colors import white, grey, black, lamp_black
import os, vtk
import numpy as np

display = False

data_dir = '.'
scan = 'mousse_250x250x250_uint8.raw'
im_file = os.path.join(data_dir, scan)
s_size = scan[:-4].split('_')[-2].split('x')
s_type = scan[:-4].split('_')[-1]
size = [int(s_size[0]), int(s_size[1]), int(s_size[2])]
data = read_image_data(im_file, size, data_type=s_type, verbose=True)

print 'creating renderer'
ren = vtk.vtkRenderer()
ren.SetBackground(0.0, 0.0, 0.0)

print 'adding bounding box'
outline = data_outline(data)
outline.GetProperty().SetColor(white)
ren.AddActor(outline)

print 'isolating the foam with vtkContourFilter'
foam = contourFilter(data, 80, color=grey, diffuseColor=white)
foam.GetProperty().SetSpecular(.4)
foam.GetProperty().SetSpecularPower(10)
ren.AddActor(foam)

print 'adding XYZ axes'
axes = axes_actor(length = 100, axisLabels = True)
axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(grey)
ren.AddViewProp(axes);

print 'setting up camera'
cam = setup_camera(size=size)
cam.SetClippingRange(1, 2000)
ren.SetActiveCamera(cam)

print '3d rendering'
render(ren, ren_size=(600, 600), display=False, save=True, name='%s_3d.png' % scan[:-4])
print 'done'
