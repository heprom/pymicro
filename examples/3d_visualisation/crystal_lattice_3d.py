#!/usr/bin/env python
import vtk, os
from pymicro.crystal.lattice import Lattice
from pymicro.view.scene3d import Scene3D
from pymicro.view.vtk_utils import lattice_grid, lattice_3d, lattice_3d_with_planes, axes_actor, \
    apply_translation_to_actor

'''
Create a 3d scene with all the crystal lattice types.
Each lattice is created separatly and added to the scene
with a small offset so it is displayed nicely.
'''

# create the 3D scene
base_name = os.path.splitext(__file__)[0]
s3d = Scene3D(display=False, ren_size=(800, 400), name=base_name)

# create all the different unit lattice cells
a = 1.0
b = 1.5
c = 2.0
alpha = 66
beta = 66
gamma = 66

l = Lattice.cubic(a)
cubic = lattice_3d(l)
apply_translation_to_actor(cubic, (0.5, 0.5, 0.0))

l = Lattice.tetragonal(a, c)
tetragonal = lattice_3d(l)
apply_translation_to_actor(tetragonal, (2.0, 2.0, 0.0))

l = Lattice.orthorombic(a, b, c)
orthorombic = lattice_3d(l)
apply_translation_to_actor(orthorombic, (3.5, 3.5, 0.0))

l = Lattice.hexagonal(a, c)
hexagonal = lattice_3d(l)
apply_translation_to_actor(hexagonal, (5.0, 5.0, 0.0))

l = Lattice.rhombohedral(a, alpha)
rhombohedral = lattice_3d(l)
apply_translation_to_actor(rhombohedral, (6.5, 6.5, 0.0))

l = Lattice.monoclinic(a, b, c, alpha)
monoclinic = lattice_3d(l)
apply_translation_to_actor(monoclinic, (8.0, 8.0, 0.0))

l = Lattice.triclinic(a, b, c, alpha, beta, gamma)
triclinic = lattice_3d(l)
apply_translation_to_actor(triclinic, (9.5, 9.5, 0.0))

# add actors to the 3d scene
s3d.add(cubic)
s3d.add(tetragonal)
s3d.add(orthorombic)
s3d.add(hexagonal)
s3d.add(rhombohedral)
s3d.add(monoclinic)
s3d.add(triclinic)

# add axes actor
axes = axes_actor(length=0.5)
transform = vtk.vtkTransform()
transform.Translate(-0.5, -0.5, 0.0)
axes.SetUserTransform(transform)
s3d.add(axes)

# set up camera
cam = vtk.vtkCamera()
cam.SetViewUp(0, 0, 1)
cam.SetPosition(7, 4, 3)
cam.SetFocalPoint(5.5, 5.5, 0)
cam.SetClippingRange(-20, 20)
cam.Dolly(0.2)
s3d.set_camera(cam)
s3d.render()

# thumbnail for the image gallery
from matplotlib import image

image_name = base_name + '.png'
image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
