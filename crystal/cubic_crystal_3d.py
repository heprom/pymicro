#!/usr/bin/env python
import vtk, numpy
from lattice import Lattice, HklPlane
from microstructure import Orientation
from vtk.util.colors import *
from view.vtk_utils import *

def create_lattice_3d_with_planes(lattice, hklplanes):
  # get the unstructured grid corresponding to the crystal lattice
  grid = lattice_grid(lattice)

  # an assembly is used to gather all the actors together
  assembly = vtk.vtkAssembly()

  # add a vertical plane to show the slip plane intersection (could be any surface)
  vplane = vtk.vtkPlane()
  vplane.SetOrigin(0.5, 0.25, 0.5)
  vplane.SetNormal(0, 1, 0)

  # add all the hkl planes
  for hklplane in hklplanes:
    origin = (0.5, 0.5, 0.5) # see if we can compute this automatically
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(hklplane.normal)
    hklplaneActor = add_plane_to_grid(plane, grid, origin)
    assembly.AddPart(hklplaneActor)
    # get a reference to the vtkPolyData representing the hkl plane
    hklplanePolyData = hklplaneActor.GetMapper().GetInput()

    # cut the rotated hkl plane with the reference plane
    vplaneCut = vtk.vtkCutter()
    vplaneCut.SetInput(hklplanePolyData)
    vplaneCut.SetCutFunction(vplane)
    vplaneCut.Update() # this is a vtkPolyData
    vplaneCutMapper = vtk.vtkPolyDataMapper()
    vplaneCutMapper.SetInputConnection(vplaneCut.GetOutputPort())
    vplaneCutActor = vtk.vtkActor()
    vplaneCutActor.SetMapper(vplaneCutMapper)
    vplaneCutActor.GetProperty().SetColor(peacock)
    vplaneCutActor.GetProperty().SetLineWidth(5)
    assembly.AddPart(vplaneCutActor)  
    
    # add an arrow to display the normal to the plane
    arrowActor = unit_arrow_3d(origin, hklplane.normal)
    assembly.AddPart(arrowActor)  
  
  Edges, Vertices = lattice_3d(grid)
  # add the two actors to the renderer
  assembly.AddPart(Edges)
  assembly.AddPart(Vertices)

  # finally, apply crystal orientation to the lattice
  apply_orientation_to_actor(assembly, Orientation(344.0, 125.0, 217.0, type='euler'))
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

  l = Lattice.cubic(a)
  print l
  p1 = HklPlane(1, 1, 1)
  p2 = HklPlane(-1, 1, 1)
  p3 = HklPlane(1, -1, 1)
  p4 = HklPlane(1, 1, -1)
  hklplanes = [p1, p2, p3, p4]
  #hklplanes = [p1]
  print hklplanes
  cubic = create_lattice_3d_with_planes(l, hklplanes)
  transform = cubic.GetUserTransform()
  #transform.Translate(0.5,0.5,0)
  ren.AddActor(cubic)

  # add axes actor
  axes = axes_actor(0.5)
  #axesTransform = vtk.vtkTransform()
  #axesTransform.Translate(-0.5,-0.5,0)
  #axes.SetUserTransform(axesTransform)
  ren.AddViewProp(axes)

  # Set the background color.
  ren.SetBackground(white)

  # set up camera
  cam = vtk.vtkCamera()
  cam.SetViewUp(0, 0, 1)
  cam.SetPosition(4, -1.5, 1.5)
  cam.SetFocalPoint(0.5, 0.5, 0.5)
  #cam.SetClippingRange(-20,20)
  cam.Dolly(1.1)
  ren.SetActiveCamera(cam)

  # capture the display and write a png image
  w2i = vtk.vtkWindowToImageFilter()
  writer = vtk.vtkPNGWriter()
  w2i.SetInput(renWin)
  w2i.Update()
  writer.SetInputConnection(w2i.GetOutputPort())
  writer.SetFileName('cubic_crystal_3d.png')
  renWin.Render()
  writer.Write()
  
  # display the scene using the RenderWindowInteractor
  iren.Initialize()
  renWin.Render()
  iren.Start()
