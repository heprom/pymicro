#!/usr/bin/env python
import vtk, numpy, os
from pymicro.crystal.microstructure import Orientation, Grain
from pymicro.crystal.texture import PoleFigure
from pymicro.view.scene3d import Scene3D
from pymicro.view.vtk_utils import pole_figure_3d, axes_actor, setup_camera

'''
Create a 3d scene with a cubic crystal lattice at the center.
Hkl planes are added to the lattice and their normal displayed.
A sphere is added to show how a pole figure can be constructed.
'''

base_name = os.path.splitext(__file__)[0]
s3d = Scene3D(display=False, ren_size=(800, 800), name=base_name)

orientation = Orientation.from_euler(numpy.array([142.8, 32.0, 214.4]))
pf = PoleFigure(hkl='111')
pf.microstructure.grains.append(Grain(1, orientation))
pole_figure = pole_figure_3d(pf, radius=1.0, show_lattice=True)

# add all actors to the 3d scene
s3d.add(pole_figure)
axes = axes_actor(1.0, fontSize=60)
s3d.add(axes)

# set up camera
cam = setup_camera(size=(1, 1, 1))
cam.SetViewUp(0, 0, 1)
cam.SetPosition(0, -4, 0)
cam.SetFocalPoint(0, 0, 0)
s3d.set_camera(cam)
s3d.render()

# thumbnail for the image gallery
from matplotlib import image
image_name = base_name + '.png'
image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
