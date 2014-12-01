#!/usr/bin/env python
import vtk, numpy
from pymicro.crystal.lattice import Lattice, HklPlane
from pymicro.crystal.microstructure import Orientation
from pymicro.view.vtk_utils import *

if __name__ == '__main__':
  '''
  Create a 3d scene with a cubic crystal lattice.
  Hkl planes are added to the lattice and displayed.
  '''
  # Create the Renderer and RenderWindow
  ren = vtk.vtkRenderer()
  ren.SetBackground(1., 1., 1.)

  # create the unit lattice cell
  l = Lattice.cubic(1.0)
  
  # create the slip planes and the cubic lattice actor
  hklplanes = HklPlane.get_family('111')
  cubic  = lattice_3d_with_planes(l, hklplanes, crystal_orientation=None, \
    show_normal=True, plane_opacity=0.5)
  ren.AddActor(cubic)

  # add axes actor
  axes = axes_actor(0.5)
  ren.AddViewProp(axes)

  # set up camera and render
  cam = setup_camera(size=(1, 1, 1))
  cam.SetFocalPoint(0, 0, 0)
  cam.SetPosition(4, -1.5, 1.5) # change the position to something better
  cam.Dolly(1.1) # get a little closer
  ren.SetActiveCamera(cam)
  image_name = os.path.splitext(__file__)[0] + '.png'
  print 'writting %s' % image_name
  render(ren, save=True, display=False, ren_size=(800,800), name=image_name)
