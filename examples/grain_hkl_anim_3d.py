from pymicro.file.file_utils import edf_read
from pymicro.view.vtk_utils import *
from pymicro.view.vtk_anim import vtkRotateActorAroundZAxis
from pymicro.crystal.lattice import HklPlane
from pymicro.crystal.microstructure import Orientation, Grain
from vtk.util.colors import white, grey, green, black, lamp_black
import os, vtk
from scipy import ndimage
import numpy as np

data_dir = os.path.join(os.environ['RAWDATA'], '2014_feb_topotomo/raw_stacks')
#scan = 'grain.raw'
#scan = 'grain1_450x450x487_uint8.raw'
#scan = 'grain1_225x225x243_uint8.raw'
scan = 'grain1_112x112x121_uint8.raw'
im_file = os.path.join(data_dir, scan)

# create a python Grain object
#orientation = Orientation.from_euler(np.array([0., 0., 0.]))
orientation = Orientation.from_rodrigues(np.array([0.3889, -0.0885, 0.3268]))
grain = Grain(1, orientation)
grain_data = edf_read(im_file, header_size=0, autoparse_filename= True, verbose=True)
grain.position = ndimage.measurements.center_of_mass(grain_data, grain_data)
print 'grain position:',grain.position
grain.volume = ndimage.measurements.sum(grain_data) # label is 1.0 here
grain.add_vtk_mesh(grain_data, contour=False)
#grain.save_vtk_repr() # save the grain mesh in vtk format

print 'adding bounding box'
grain_bbox = box_3d(size=np.shape(grain_data), line_color=white)
print 'adding grain with slip planes'
p1 = HklPlane(1, 1, 1)
p2 = HklPlane(1, 1, -1)
#hklplanes = [p1, p2]
hklplanes = [p1]
grain_with_planes = add_grain_to_3d_scene(grain, hklplanes, show_orientation=True)
tr = vtk.vtkTransform()
tr.Translate(grain.position)
grain_with_planes.SetUserTransform(tr)

print 'setting up camera'
cam = setup_camera(size=np.shape(grain_data))
#cam.SetPosition(2*np.shape(grain_data)[0], -2*np.shape(grain_data)[1], 1.5*np.shape(grain_data)[2])
cam.Dolly(0.9)
# Create renderer
ren = vtk.vtkRenderer()
ren.SetBackground(0.0, 0.0, 0.0)
ren.AddActor(grain_bbox)
ren.AddActor(grain_with_planes)
ren.SetActiveCamera(cam)

#render(ren, ren_size=(600, 700), display=True, name=scan[:-4] + '_3d.png')
ren_size = (600, 700)
display = False
name = scan[:-4] + '_3d'
# Create a window for the renderer
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(ren_size)

# Start the initialization and rendering
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
renWin.Render()
iren.Initialize()
# Sign up to receive TimerEvent
anim_actor = vtkRotateActorAroundZAxis()
anim_actor.actor = grain_with_planes
anim_actor.actor_position = grain.position
anim_actor.display = display
if not display:
  if not os.path.exists(name):
    os.mkdir(name) # create a folder to store the images
  anim_actor.prefix = name
iren.AddObserver('TimerEvent', anim_actor.execute)
timerId = iren.CreateRepeatingTimer(100); # time in ms
iren.Start()
