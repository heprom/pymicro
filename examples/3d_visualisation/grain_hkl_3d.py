from pymicro.file.file_utils import HST_read
from pymicro.view.vtk_utils import *
from pymicro.crystal.lattice import HklPlane
from pymicro.crystal.microstructure import Orientation, Grain
from vtk.util.colors import white, grey, green, black, lamp_black
import os, vtk
from scipy import ndimage
import numpy as np

data_dir = '../data'
scan = 'grain1_112x112x121_uint8.raw'
im_file = os.path.join(data_dir, scan)

# create a python Grain object
orientation = Orientation.from_rodrigues(np.array([0.3889, -0.0885, 0.3268]))
grain = Grain(1, orientation)
grain_data = HST_read(im_file, header_size=0, autoparse_filename= True, verbose=True)
grain.position = ndimage.measurements.center_of_mass(grain_data, grain_data)
print 'grain position:',grain.position
grain.volume = ndimage.measurements.sum(grain_data) # label is 1.0 here
grain.add_vtk_mesh(grain_data, contour=False)

print 'adding bounding box'
grain_bbox = box_3d(size=np.shape(grain_data), line_color=white)
print 'adding grain with slip planes'
hklplanes = [HklPlane(1, 1, 1)]
grain_with_planes = grain_3d(grain, hklplanes, show_normal=False, \
  plane_opacity=1.0, show_orientation=True)
tr = vtk.vtkTransform()
tr.Translate(grain.position)
grain_with_planes.SetUserTransform(tr)

print 'adding a lattice to picture the grain orientation'
lat_size = 20
l = Lattice.cubic(lat_size)
cubic = lattice_3d_with_planes(l, hklplanes, crystal_orientation=grain.orientation, \
  show_normal=True, plane_opacity=1.0)
tra = cubic.GetUserTransform()
tra.PostMultiply()
tra.Translate(lat_size, lat_size, lat_size)

print 'adding axes'
axes = axes_actor(length = 100, axisLabels = True)
axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(grey)

print 'setting up camera'
cam = setup_camera(size=np.shape(grain_data))
cam.Dolly(0.9)

# Create renderer and add all the actors
ren = vtk.vtkRenderer()
ren.SetBackground(0.0, 0.0, 0.0)
ren.AddActor(grain_bbox)
ren.AddActor(grain_with_planes)
ren.AddActor(cubic)
ren.AddViewProp(axes);
ren.SetActiveCamera(cam)

image_name = os.path.splitext(__file__)[0] + '.png'
print 'writting %s' % image_name
render(ren, ren_size=(600, 700), save=True, display=False, name=image_name)

from matplotlib import image
image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
