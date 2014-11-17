from pymicro.view.vtk_utils import *
from pymicro.crystal.lattice import HklPlane
from pymicro.crystal.microstructure import Orientation
from math import sqrt
import numpy as np

if __name__ == '__main__':
  '''
  Create a 3d scene with a hexagonal crystal lattice.
  Hkl planes are added to the lattice and displayed.
  '''
  # Create the Renderer and RenderWindow
  ren = vtk.vtkRenderer()
  ren.SetBackground(white)

  # crystal orientation
  o = Orientation.from_euler((15.0, -45.0, 0.0))
  o = None

  # hexagonal lattice
  a = 1.0 #0.321 # nm
  c = 1.5 #0.521 #nm
  l = Lattice.hexagonal(a, c)
  #grid = lattice_grid(l)
  grid = hexagonal_lattice_grid(l)
  print np.array(HklPlane.four_to_three_index(1,0,-1,0))/0.6 # prismatic 1
  print np.array(HklPlane.four_to_three_index(0,1,-1,0))/0.6 # prismatic 2
  print np.array(HklPlane.four_to_three_index(0,1,-1,1))/0.6*3 # pyramidal 1
  print np.array(HklPlane.four_to_three_index(1,1,-2,2))/0.6*3 # pyramidal 2
  print np.array(HklPlane.four_to_three_index(0,0,0,1)) # basal
  p1 = HklPlane(2., 1, 0, lattice=l) # attach the plane to the hexagonal lattice
  p2 = HklPlane(-1, 2, 0, lattice=l)
  p3 = HklPlane(-3., 6., 5., lattice=l)
  p4 = HklPlane(3, 9, 10, lattice=l)
  p5 = HklPlane(0, 0, 1, lattice=l)
  #hklplanes = [p1, p3]
  #hklplanes = [p2, p4]
  hklplanes = [p3,p5]
  hexagon = vtk.vtkAssembly()
  Edges, Vertices = lattice_3d(grid, tubeRadius=0.025*a, sphereRadius=0.1*a)
  hexagon.AddPart(Edges)
  hexagon.AddPart(Vertices)
  hexagon.SetOrigin(a/2, -a*sqrt(3)/2., c/2)
  hexagon.AddPosition(-a/2, a*sqrt(3)/2., -c/2)

  # display all the hkl planes (with normal)
  for hklplane in hklplanes:
    origin = (a/2, -a*sqrt(3)/2., c/2)
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(hklplane.normal())
    print 'normal is', plane.GetNormal()
    hklplaneActor = add_plane_to_grid(plane, grid, origin)
    hklplaneActor.GetProperty().SetOpacity(0.5)
    hexagon.AddPart(hklplaneActor)
    # add an arrow to display the normal to the plane
    arrowActor = unit_arrow_3d(origin, a*np.array(plane.GetNormal()), make_unit=False)
    hexagon.AddPart(arrowActor)

  if o != None:
    apply_orientation_to_actor(hexagon, o)
  ren.AddActor(hexagon)

  # add axes actor
  axes = axes_actor(0.5)
  ren.AddViewProp(axes)

  # set up camera and render
  cam = setup_camera(size=(1, 1, 1))
  cam.SetFocalPoint(0, 0, 0)
  cam.SetPosition(0., -2, 1.0) # change the position to something better
  cam.ParallelProjectionOn()
  cam.Dolly(1.1) # get a little closer
  ren.SetActiveCamera(cam)
  render(ren, display=True, ren_size=(800,800), name='hexagonal_crystal_3d.png')
  print 'done'
