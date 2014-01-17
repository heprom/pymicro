#!/usr/bin/env python
import vtk
from lattice import Lattice
from PyMicro.view.vtk_utils import lattice_grid, lattice_3d, axes_actor, render

def create_lattice_3d(lattice):
  grid = lattice_grid(lattice)
  Edges, Vertices = lattice_3d(grid)
  assembly = vtk.vtkAssembly()
  assembly.AddPart(Edges)
  assembly.AddPart(Vertices)
  return assembly

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
  cubic = create_lattice_3d(l)
  transform = vtk.vtkTransform()
  transform.Translate(0.5,0.5,0)
  cubic.SetUserTransform(transform)

  l = Lattice.tetragonal(a, c)
  tetragonal = create_lattice_3d(l)
  transform = vtk.vtkTransform()
  transform.Translate(2.0,2.0,0)
  tetragonal.SetUserTransform(transform)

  l = Lattice.orthorombic(a, b, c)
  orthorombic = create_lattice_3d(l)
  transform = vtk.vtkTransform()
  transform.Translate(3.5,3.5,0)
  orthorombic.SetUserTransform(transform)

  l = Lattice.hexagonal(a, c)
  hexagonal = create_lattice_3d(l)
  transform = vtk.vtkTransform()
  transform.Translate(5.0,5.0,0)
  hexagonal.SetUserTransform(transform)

  l = Lattice.rhombohedral(a, alpha)
  rhombohedral = create_lattice_3d(l)
  transform = vtk.vtkTransform()
  transform.Translate(6.5,6.5,0)
  rhombohedral.SetUserTransform(transform)

  l = Lattice.monoclinic(a, b, c, alpha)
  monoclinic = create_lattice_3d(l)
  transform = vtk.vtkTransform()
  transform.Translate(8.0,8.0,0)
  monoclinic.SetUserTransform(transform)

  l = Lattice.triclinic(a, b, c, alpha, beta, gamma)
  triclinic = create_lattice_3d(l)
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
  ren.SetBackground(white)
  # set up camera
  cam = vtk.vtkCamera()
  cam.SetViewUp(0, 0, 1)
  cam.SetPosition(7, 4, 3)
  cam.SetFocalPoint(5.5, 5.5, 0)
  cam.SetClippingRange(-20,20)
  cam.Dolly(0.2)
  ren.SetActiveCamera(cam)
  render(ren, ren_size=(800, 400), display=False, name='crystal_lattice_3d.png')
