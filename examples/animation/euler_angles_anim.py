from pymicro.view.vtk_utils import *
from pymicro.view.vtk_anim import *
from pymicro.view.scene3d import Scene3D
from pymicro.crystal.microstructure import Orientation
from pymicro.crystal.lattice import Lattice

s3d = Scene3D(display=True, ren_size=(600, 600))
euler_angles = np.array([142.8, 32.0, 214.4])
(phi1, Phi, phi2) = euler_angles
orientation = Orientation.from_euler(euler_angles)

scene = vtkAnimationScene(s3d.get_renderer(), s3d.renWin.GetSize())
scene.save_image = True
scene.timer_incr = 1
scene.timer_end = 179
scene.verbose = False
scene.prefix = 'euler_angles_anim'

lab_frame = axes_actor(1, fontSize=50)
lab_frame.SetCylinderRadius(0.02)
s3d.add(lab_frame)

crystal_frame = axes_actor(0.6, fontSize=50, axisLabels=None)
crystal_frame.SetCylinderRadius(0.04)
collection = vtk.vtkPropCollection()
crystal_frame.GetActors(collection)
for i in range(collection.GetNumberOfItems()):
    collection.GetItemAsObject(i).GetProperty().SetColor(0.0, 0.0, 0.0)
crystal_frame.SetVisibility(0)
s3d.add(crystal_frame)

a = 0.4045  # nm, value for Al
l = Lattice.cubic(a)
cubic_lattice = lattice_3d(l, crystal_orientation=orientation, tubeRadius=0.1 * a, sphereRadius=0.2 * a)
s3d.add(cubic_lattice)

# display the crystal frame progressively
crystal_frame_visibility = vtkSetVisibility(5, crystal_frame, gradually=True)
crystal_frame_visibility.time_anim_ends = 20
scene.add_animation(crystal_frame_visibility)

# apply Euler angles one by one with the Bunge convention (ZXZ)
crystal_frame_rotate_phi1 = vtkRotateActorAroundAxis(30, duration=40, axis=[0., 0., 1.], angle=phi1)
crystal_frame_rotate_phi1.set_actor(crystal_frame)
scene.add_animation(crystal_frame_rotate_phi1)

o_phi1 = Orientation.from_euler((phi1, 0., 0.))
x_prime = np.dot(o_phi1.orientation_matrix().T, [1., 0., 0.])
print('after phi1, X axis is {0}'.format(x_prime))
crystal_frame_rotate_Phi = vtkRotateActorAroundAxis(80, duration=40, axis=[1., 0., 0.], angle=Phi)
crystal_frame_rotate_Phi.set_actor(crystal_frame)
# fix the reference to the user_transform_matrix after phi1
m = vtk.vtkMatrix4x4()  # row major order, 16 elements matrix
m.DeepCopy(crystal_frame.GetUserTransform().GetMatrix())
for j in range(3):
    for i in range(3):
        m.SetElement(j, i, o_phi1.orientation_matrix()[i, j])
crystal_frame_rotate_Phi.user_transform_matrix = m
scene.add_animation(crystal_frame_rotate_Phi)

o_phi1_Phi = Orientation.from_euler((phi1, Phi, 0.))
z_prime = np.dot(o_phi1_Phi.orientation_matrix().T, [0., 0., 1.])
print('after phi1 and Phi, Z axis is {0}'.format(z_prime))
crystal_frame_rotate_phi2 = vtkRotateActorAroundAxis(130, duration=40, axis=[0., 0., 1.], angle=phi2)
crystal_frame_rotate_phi2.set_actor(crystal_frame)
# fix the reference to the user_transform_matrix after phi1 and Phi
m2 = vtk.vtkMatrix4x4()  # row major order, 16 elements matrix
m2.DeepCopy(crystal_frame.GetUserTransform().GetMatrix())
for j in range(3):
    for i in range(3):
        m2.SetElement(j, i, o_phi1_Phi.orientation_matrix()[i, j])
crystal_frame_rotate_phi2.user_transform_matrix = m2
scene.add_animation(crystal_frame_rotate_phi2)

# add some text actors
euler_str = 'The orientation matrix brings the lab frame\n' \
            'into coincidence with the crystal frame.\n' \
            'Crystal Euler angles = (%.1f, %.1f, %.1f)' % (phi1, Phi, phi2)
euler_text = text(euler_str, coords=(0.5, 0.05))
s3d.get_renderer().AddActor2D(euler_text)
rotation_text = text('', coords=(0.5, 0.95))
s3d.get_renderer().AddActor2D(rotation_text)


def update_rotation_text():
    if scene.timer_count < crystal_frame_rotate_phi1.time_anim_starts:
        return ''
    elif crystal_frame_rotate_phi1.time_anim_starts < scene.timer_count <= crystal_frame_rotate_phi1.time_anim_ends:
        the_phi1 = (scene.timer_count - crystal_frame_rotate_phi1.time_anim_starts) / \
                   float(crystal_frame_rotate_phi1.time_anim_ends - crystal_frame_rotate_phi1.time_anim_starts) * phi1
        print('t=%d, computed the_phi1 = %.1f' % (scene.timer_count, the_phi1))
        return 'Applying first rotation: phi1 = %.1f degrees' % the_phi1
    elif crystal_frame_rotate_Phi.time_anim_starts < scene.timer_count <= crystal_frame_rotate_Phi.time_anim_ends:
        the_Phi = (scene.timer_count - crystal_frame_rotate_Phi.time_anim_starts) / \
                  float(crystal_frame_rotate_Phi.time_anim_ends - crystal_frame_rotate_Phi.time_anim_starts) * Phi
        return 'Applying second rotation: Phi = %.1f degrees' % the_Phi
    elif crystal_frame_rotate_phi2.time_anim_starts < scene.timer_count <= crystal_frame_rotate_phi2.time_anim_ends:
        the_phi2 = (scene.timer_count - crystal_frame_rotate_phi2.time_anim_starts) / \
                   float(crystal_frame_rotate_phi2.time_anim_ends - crystal_frame_rotate_phi2.time_anim_starts) * phi2
        return 'Applying third rotation: phi2 = %.1f degrees' % the_phi2
    return None


update_text = vtkUpdateText(rotation_text, update_rotation_text, t=0, duration=scene.timer_end)
scene.add_animation(update_text)

# camera settings
cam = setup_camera(size=(1., 1., 1.))
cam.SetFocalPoint(a / 2, a / 2, a / 2)
s3d.set_camera(cam)

# crystal_frame.SetVisibility(1)
# scene.render_at(100)
scene.render()
