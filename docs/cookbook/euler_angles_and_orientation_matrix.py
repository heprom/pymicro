import numpy as np
import vtk
from pymicro.view.vtk_utils import lattice_3d, unit_arrow_3d, axes_actor, text, setup_camera, \
    apply_orientation_to_actor, set_opacity
from pymicro.view.scene3d import Scene3D
from pymicro.crystal.microstructure import Orientation
from pymicro.crystal.lattice import Lattice, HklDirection

s3d = Scene3D(display=False, ren_size=(600, 600))
s3d.name = 'euler_angles_and_orientation_matrix'
euler_angles = np.array([142.8, 32.0, 214.4])
(phi1, Phi, phi2) = euler_angles
orientation = Orientation.from_euler(euler_angles)
g = orientation.orientation_matrix()

lab_frame = axes_actor(1, fontSize=50)
lab_frame.SetCylinderRadius(0.02)
s3d.add(lab_frame)

crystal_frame = axes_actor(0.6, fontSize=50, axisLabels=None)
crystal_frame.SetCylinderRadius(0.05)
collection = vtk.vtkPropCollection()
crystal_frame.GetActors(collection)
for i in range(collection.GetNumberOfItems()):
    collection.GetItemAsObject(i).GetProperty().SetColor(0.0, 0.0, 0.0)
apply_orientation_to_actor(crystal_frame, orientation)
s3d.add(crystal_frame)

a = 1.0
l = Lattice.face_centered_cubic(a)
fcc_lattice = lattice_3d(l, crystal_orientation=orientation)
set_opacity(fcc_lattice, 0.3)
s3d.add(fcc_lattice)

# arrow to show 111 lattice vector
Vc = np.array([a, a, a])
Vs = np.dot(g.T, Vc)
vector = unit_arrow_3d((0., 0., 0.), Vs, make_unit=False)
s3d.add(vector)

# add some text actors
euler_str = 'Crystal Euler angles = (%.1f, %.1f, %.1f)\n' \
            'Vc=[1, 1, 1]\n' \
            'Vs=[%.3f, %.3f, %.3f]' % (phi1, Phi, phi2, Vs[0], Vs[1], Vs[2])
euler_text = text(euler_str, coords=(0.5, 0.05))
s3d.get_renderer().AddActor2D(euler_text)
rotation_text = text('', coords=(0.5, 0.95))
s3d.get_renderer().AddActor2D(rotation_text)

# camera settings
cam = setup_camera(size=(1.3 * a, 1.3 * a, 1.3 * a))
cam.SetFocalPoint(a / 2, a / 2, 0.)
s3d.set_camera(cam)

s3d.render()
