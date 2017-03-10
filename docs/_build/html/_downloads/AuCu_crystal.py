#!/usr/bin/env python
import vtk, os
from pymicro.crystal.lattice import Crystal
from pymicro.view.scene3d import Scene3D
from pymicro.view.vtk_utils import *
from vtk.util.colors import gold

'''
Create a 3d scene with a unit cells of primitive tetragonal crystal 
lattice. The basis associated consist of one Au atom at (0., 0., 0.) 
and one Cu atom at (0.5, 0.5, 0.5), so thet the unit cell contains 
2 atoms (tP2 Pearson symbol). 4 units cells are shown.
'''
copper = (1.000000, 0.780392, 0.494117)
a = 0.2867
c = 0.411
tl = Lattice.tetragonal(a, c)
tl._centering = 'P'
[A, B, C] = tl._matrix
origin = (0., 0., 0.)
AuCu = Crystal(tl, basis=[(0., 0., 0.), (0.5, 0.5, 0.5)], basis_labels=['Au', 'Cu'], basis_sizes=[0.5, 0.5],
               basis_colors=[gold, copper])
AuCu_actor = crystal_3d(AuCu, origin, m=2, n=2, p=1, hide_outside=True)

# create the 3D scene
base_name = os.path.splitext(__file__)[0]
s3d = Scene3D(display=False, ren_size=(800, 800), name=base_name)
s3d.add(AuCu_actor)

cam = setup_camera(size=A + B + C)
cam.SetViewUp(0, 0, 1)
cam.SetFocalPoint(A + B + C / 2)
cam.SetPosition(4.0, 1.5, 1.5)
cam.Dolly(2.0)
s3d.set_camera(cam)
s3d.render()

# thumbnail for the image gallery
from matplotlib import image

image_name = base_name + '.png'
image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
