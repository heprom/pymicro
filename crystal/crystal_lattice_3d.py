#!/usr/bin/env python
import vtk
from vtk.util.colors import *
from lattice import Lattice
from view.vtk_utils import lattice_3d

def create_lattice_3d(lattice):
  Grid = lattice_3d(lattice)

  # transform the points using the three euler angles
  transform = vtk.vtkTransform()
  transform.Identity()
  #transform.Translate(2,0,0)
  #transform.RotateZ(344.0)
  #transform.RotateX(125.0)
  #transform.RotateZ(217.0)
  transF = vtk.vtkTransformFilter()
  transF.SetInput(Grid)
  transF.SetTransform(transform)

  Edges = vtk.vtkExtractEdges()
  Edges.SetInput(transF.GetOutput())
  Tubes = vtk.vtkTubeFilter()
  Tubes.SetInputConnection(Edges.GetOutputPort())
  Tubes.SetRadius(.02)
  Tubes.SetNumberOfSides(6)
  Tubes.UseDefaultNormalOn()
  Tubes.SetDefaultNormal(.577, .577, .577)
  # Create the mapper and actor to display the cell edges.
  TubeMapper = vtk.vtkPolyDataMapper()
  TubeMapper.SetInputConnection(Tubes.GetOutputPort())
  Edges = vtk.vtkActor()
  Edges.SetMapper(TubeMapper)

  # Create a sphere to use as a glyph source for vtkGlyph3D.
  Sphere = vtk.vtkSphereSource()
  Sphere.SetRadius(0.1)
  Sphere.SetPhiResolution(20)
  Sphere.SetThetaResolution(20)
  Vertices = vtk.vtkGlyph3D()
  Vertices.SetInputConnection(transF.GetOutputPort())
  Vertices.SetSource(Sphere.GetOutput())
  # Create a mapper and actor to display the glyphs.
  SphereMapper = vtk.vtkPolyDataMapper()
  SphereMapper.SetInputConnection(Vertices.GetOutputPort())
  SphereMapper.ScalarVisibilityOff()
  Vertices = vtk.vtkActor()
  Vertices.SetMapper(SphereMapper)
  Vertices.GetProperty().SetDiffuseColor(blue)
  # finally, add the two actors to the renderer
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
  # Create the Renderer and RenderWindow
  ren = vtk.vtkRenderer()
  renWin = vtk.vtkRenderWindow()
  renWin.AddRenderer(ren)
  renWin.SetSize(800, 400)
  iren = vtk.vtkRenderWindowInteractor()
  iren.SetRenderWindow(renWin)

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

  ren.AddActor(cubic)
  ren.AddActor(tetragonal)
  ren.AddActor(orthorombic)
  ren.AddActor(hexagonal)
  ren.AddActor(rhombohedral)
  ren.AddActor(monoclinic)
  ren.AddActor(triclinic)

  # add axes actor
  axes = vtk.vtkAxesActor()
  axes.SetTotalLength(0.5,0.5,0.5)
  axes.SetXAxisLabelText('x')
  axes.SetYAxisLabelText('y')
  axes.SetZAxisLabelText('z')
  axes.SetShaftTypeToCylinder()
  axes.SetCylinderRadius(0.02)
  axprop = vtk.vtkTextProperty()
  axprop.SetColor(0, 0, 0)
  axprop.SetFontSize(1)
  axprop.SetFontFamilyToArial()
  axes.GetXAxisCaptionActor2D().SetCaptionTextProperty(axprop)
  axes.GetYAxisCaptionActor2D().SetCaptionTextProperty(axprop)
  axes.GetZAxisCaptionActor2D().SetCaptionTextProperty(axprop)
  transform = vtk.vtkTransform()
  transform.Translate(-0.5,-0.5,0)
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

  # capture the display and write a png image
  w2i = vtk.vtkWindowToImageFilter()
  writer = vtk.vtkPNGWriter()
  w2i.SetInput(renWin)
  w2i.Update()
  writer.SetInputConnection(w2i.GetOutputPort())
  writer.SetFileName('crystal_lattice_3d.png')
  renWin.Render()
  writer.Write()
  
  # display the scene using the RenderWindowInteractor
  iren.Initialize()
  renWin.Render()
  iren.Start()
