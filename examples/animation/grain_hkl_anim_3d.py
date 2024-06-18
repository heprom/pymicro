from scipy import ndimage

from pymicro.file.file_utils import HST_read
from pymicro.view.vtk_utils import *
from pymicro.view.vtk_anim import vtkAnimationScene, vtkRotateActorAroundZAxis
from pymicro.crystal.lattice import HklPlane
from pymicro.crystal.microstructure import Orientation, Grain

from pymicro import get_examples_data_dir # import file directory path
PYMICRO_EXAMPLES_DATA_DIR = get_examples_data_dir() # get the file directory path

data_dir = PYMICRO_EXAMPLES_DATA_DIR
scan = 'grain1_112x112x121_uint8.raw'
im_file = os.path.join(data_dir, scan)

print('create a python Grain object')
orientation = Orientation.from_rodrigues(np.array([0.3889, -0.0885, 0.3268]))
grain = Grain(1, orientation)
grain_data = HST_read(im_file, autoparse_filename=True, verbose=True)
grain.position = ndimage.measurements.center_of_mass(grain_data, grain_data)
grain.volume = ndimage.measurements.sum(grain_data)  # label is 1.0 here
grain.add_vtk_mesh(grain_data, contour=False)
# grain.save_vtk_repr() # save the grain mesh in vtk format

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

print('creating the animation scene')
scene = vtkAnimationScene(ren, ren_size)
scene.save_image = True
scene.timer_incr = 10
scene.time_anim_ends = 360
scene.prefix = name
print('initialize the a vtkRotateActorAroundZAxis instance')
anim_actor = vtkRotateActorAroundZAxis(0)
anim_actor.set_actor(grain_with_planes)
anim_actor.time_anim_ends = 360
scene.add_animation(anim_actor)
scene.render()
