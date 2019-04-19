import os, vtk
import numpy as np
from scipy import ndimage
from vtk.util.colors import white, grey, black
from pymicro.file.file_utils import HST_read
from pymicro.view.scene3d import Scene3D
from pymicro.view.vtk_utils import *
from pymicro.crystal.lattice import HklPlane
from pymicro.crystal.microstructure import Orientation, Grain

'''
Create a 3d scene showing a grain with a specific hkl plane inside.
A small crystal lattice is also displayed aside the grain to picture
its orientation.
'''
data_dir = '../data'
scan = 'grain1_112x112x121_uint8.raw'
im_file = os.path.join(data_dir, scan)

# Create the 3D scene
base_name = os.path.splitext(__file__)[0]
s3d = Scene3D(display=False, ren_size=(800, 800), name=base_name, background=black)

# create a python Grain object from the image data
orientation = Orientation.from_rodrigues(np.array([0.3889, -0.0885, 0.3268]))
grain = Grain(1, orientation)
grain_data = HST_read(im_file, header_size=0, autoparse_filename=True, verbose=True)
grain.position = ndimage.measurements.center_of_mass(grain_data, grain_data)
print('grain position: %s' % str(grain.position))
grain.volume = ndimage.measurements.sum(grain_data)  # label is 1.0 here
grain.add_vtk_mesh(grain_data, contour=False)

print('adding bounding box')
grain_bbox = box_3d(size=np.shape(grain_data), line_color=white)
print('adding grain with slip planes')

z_offsets = np.linspace(-50, 50, 6, endpoint=True)
print(z_offsets)
plane_origins = np.zeros((len(z_offsets), 3), dtype=float)
plane_origins[:, 2] = z_offsets

hkl_planes = [HklPlane(1, 1, 1)] * len(z_offsets)
grain_with_planes = grain_3d(grain, hkl_planes,  plane_origins=plane_origins, show_normal=False, \
                         plane_opacity=1.0, show_orientation=False)
tr = vtk.vtkTransform()
tr.Translate(grain.position)
grain_with_planes.SetUserTransform(tr)

print('adding axes')
axes = axes_actor(length=100, fontSize=60)
axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(grey)

print('setting up camera')
cam = setup_camera(size=np.shape(grain_data))
cam.Dolly(0.9)

# add all actors to the 3d scene and render
s3d.add(grain_bbox)
s3d.add(grain_with_planes)
s3d.add(axes);
s3d.set_camera(cam)
s3d.render()

# thumbnail for the image gallery
from matplotlib import image

image_name = base_name + '.png'
image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
