from pymicro.fe.FE import FE_Calc
from pymicro.view.scene3d import Scene3D
from pymicro.view.vtk_utils import axes_actor, setup_camera, rand_cmap, show_mesh
import os, vtk

base_name = os.path.splitext(__file__)[0]
calc = FE_Calc(wdir='../data', prefix='calcul')
calc.read_ut()
# compute the grain id field
gid_field = calc._mesh.compute_grain_id_field(grain_prefix='_ELSET')
calc.add_integ_field('grain_ids', gid_field)
# get the sig33 field at card 5
field_name = 'sig33'
field = calc.read_integ(5, field_name, verbose=False)
calc.add_integ_field(field_name, field)
vtk_mesh = calc.build_vtk()

# initialize a 3d scene
s3d = Scene3D(display=False, ren_size=(800, 800), name=base_name)

# create mapper and mesh actor
lut = rand_cmap(N=40, first_is_black=False, table_range=(0, 39))
mesh = show_mesh(vtk_mesh, map_scalars=True, lut=lut, show_edges=True, edge_color=(0.2, 0.2, 0.2), edge_line_width=0.5)
s3d.add(mesh)

# add axes actor
axes = axes_actor(50, fontSize=60)
s3d.add(axes)

# set up camera and render
cam = setup_camera(size=(100, 100, 100))
s3d.set_camera(cam)
s3d.render()

# thumbnail for the image gallery
from matplotlib import image

image_name = base_name + '.png'
image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
