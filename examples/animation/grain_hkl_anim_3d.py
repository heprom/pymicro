import os, vtk
import numpy as np
from scipy import ndimage

from pymicro.file.file_utils import edf_read
from pymicro.view.vtk_utils import *
from pymicro.view.vtk_anim import vtkRotateActorAroundZAxis
from pymicro.crystal.lattice import HklPlane
from pymicro.crystal.microstructure import Orientation, Grain

data_dir = os.path.join('../rawdata')
scan = 'grain1_112x112x121_uint8.raw'
im_file = os.path.join(data_dir, scan)

print('create a python Grain object')
orientation = Orientation.from_rodrigues(np.array([0.3889, -0.0885, 0.3268]))
grain = Grain(1, orientation)
grain_data = edf_read(im_file, header_size=0, autoparse_filename= True, verbose=True)
grain.position = ndimage.measurements.center_of_mass(grain_data, grain_data)
grain.volume = ndimage.measurements.sum(grain_data) # label is 1.0 here
grain.add_vtk_mesh(grain_data, contour=False)
#grain.save_vtk_repr() # save the grain mesh in vtk format

print('adding bounding box')
grain_bbox = box_3d(size=np.shape(grain_data), line_color=white)

print('adding grain with slip planes')
p1 = HklPlane(1, 1, 1)
p2 = HklPlane(1, 1, -1)
hklplanes = [p1]
grain_with_planes = add_grain_to_3d_scene(grain, hklplanes, show_orientation=True)
tr = vtk.vtkTransform()
tr.Translate(grain.position)
grain_with_planes.SetUserTransform(tr)

print('creating 3d renderer')
ren = vtk.vtkRenderer()
ren.SetBackground(0.0, 0.0, 0.0)
ren.AddActor(grain_bbox)
ren.AddActor(grain_with_planes)
cam = setup_camera(size=np.shape(grain_data))
cam.Dolly(0.9)
ren.SetActiveCamera(cam)
ren_size = (600, 700)
name = scan[:-4] + '_anim_3d'
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(ren_size)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
renWin.Render()
iren.Initialize()

print('initialize the a vtkRotateActorAroundZAxis instance')
anim_actor = vtkRotateActorAroundZAxis(0)
anim_actor.actor = grain_with_planes
anim_actor.actor_position = grain.position
anim_actor.save_image = True
anim_actor.timer_incr = 10
anim_actor.time_anim_ends = 360
if anim_actor.save_image:
  print('images will be saved to folder %s' % name)
  if not os.path.exists(name):
    os.mkdir(name) # create a folder to store the images
  anim_actor.prefix = name
iren.AddObserver('TimerEvent', anim_actor.execute)
timerId = iren.CreateRepeatingTimer(100); # time in ms
iren.Start()
