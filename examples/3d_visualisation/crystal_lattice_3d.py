#!/usr/bin/env python
import vtk
from pymicro.crystal.lattice import Lattice
from pymicro.view.vtk_utils import lattice_grid, lattice_3d, lattice_3d_with_planes, axes_actor, render

if __name__ == '__main__':
  '''
  Create a 3d scene with all the crystal lattice types.
  Each lattice is created separatly and added to the scene
  with a small offset so it is displayed nicely.
  '''

  # create all the different unit lattice cells
  a = 1.0
  b = 1.5
  c = 2.0
  alpha = 66
  beta = 66
  gamma = 66

  l = Lattice.cubic(a)
  cubic = lattice_3d(l)
  transform = vtk.vtkTransform()
  transform.Translate(0.5,0.5,0)
  cubic.SetUserTransform(transform)

  l = Lattice.tetragonal(a, c)
  tetragonal = lattice_3d(l)
  transform = vtk.vtkTransform()
  transform.Translate(2.0,2.0,0)
  tetragonal.SetUserTransform(transform)

  l = Lattice.orthorombic(a, b, c)
  orthorombic = lattice_3d(l)
  transform = vtk.vtkTransform()
  transform.Translate(3.5,3.5,0)
  orthorombic.SetUserTransform(transform)

  l = Lattice.hexagonal(a, c)
  hexagonal = lattice_3d(l)
  transform = vtk.vtkTransform()
  transform.Translate(5.0,5.0,0)
  hexagonal.SetUserTransform(transform)

  l = Lattice.rhombohedral(a, alpha)
  rhombohedral = lattice_3d(l)
  transform = vtk.vtkTransform()
  transform.Translate(6.5,6.5,0)
  rhombohedral.SetUserTransform(transform)

  l = Lattice.monoclinic(a, b, c, alpha)
  monoclinic = lattice_3d(l)
  transform = vtk.vtkTransform()
  transform.Translate(8.0,8.0,0)
  monoclinic.SetUserTransform(transform)

  l = Lattice.triclinic(a, b, c, alpha, beta, gamma)
  triclinic = lattice_3d(l)
  transform = vtk.vtkTransform()
  transform.Translate(9.5,9.5,0)
  triclinic.SetUserTransform(transform)

  # Create the Renderer and RenderWindow
  ren = vtk.vtkRenderer()
  ren.AddActor(cubic)
  ren.AddActor(tetragonal)
  ren.AddActor(orthorombic)
  ren.AddActor(hexagonal)
  ren.AddActor(rhombohedral)
  ren.AddActor(monoclinic)
  ren.AddActor(triclinic)

  # add axes actor
  axes = axes_actor(length = 0.5)
  transform = vtk.vtkTransform()
  transform.Translate(-0.5, -0.5, 0.0)
  axes.SetUserTransform(transform)
  ren.AddViewProp(axes)

  # Set the background color.
  ren.SetBackground(1.0, 1.0, 1.0)
  # set up camera
  cam = vtk.vtkCamera()
  cam.SetViewUp(0, 0, 1)
  cam.SetPosition(7, 4, 3)
  cam.SetFocalPoint(5.5, 5.5, 0)
  cam.SetClippingRange(-20,20)
  cam.Dolly(0.2)
  ren.SetActiveCamera(cam)
  import os
  image_name = os.path.splitext(__file__)[0] + '.png'
  print 'writting %s' % image_name
  render(ren, ren_size=(800, 400), save=True, display=False, name=image_name)

  from matplotlib import image
  image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
