#!/usr/bin/env python
import vtk, numpy
from pymicro.crystal.lattice import Lattice, HklPlane
from pymicro.crystal.microstructure import Orientation
from pymicro.view import vtk_utils
from vtk.util.colors import white, peacock, tomato, red, green, yellow, black, cyan, magenta

def create_pole_figure_3d(grain_orientation, show_arrows=True, show_slip_traces=False, verbose=False):
  '''
  Create a 3d scene with a cubic crystal lattice at the center.
  Hkl planes are added to the lattice and their normal displayed.
  A sphere is added to show how a pole figure can be constructed.
  Slip traces on a particular plane can also be shown.
  '''
  # Create the Renderer
  ren = vtk.vtkRenderer()
  m = grain_orientation.orientation_matrix()
  mt = m.transpose()

  # outbounding sphere radius
  r = 1.
    
  # create the cubic lattice cell
  a = r/2
  l = Lattice.cubic(a)
  grid = vtk_utils.lattice_grid(l)
  cubic_lattice = vtk.vtkAssembly()
  Edges, Vertices = vtk_utils.lattice_3d(grid, tubeRadius=0.01, sphereRadius=0.05)
  # get a list of hkl planes in the 110 family
  hklplanes = HklPlane.get_family('110')
  colors = [red, green, yellow, black, cyan, magenta] # correspond to pyplot 'rgykcmbw'
  
  # add a vertical plane to show the slip plane intersection
  # note that the plane is rotated towards the crystal CS
  # to make the intersection with the slip plane...
  vplane = vtk.vtkPlane()
  vplane.SetOrigin(a/2, a/2, a/2)
  #n_inter = numpy.array([0, 0, 1]) # to plot slip traces on XY
  n_inter = numpy.array([0, 1, 0]) # to plot slip traces on XZ
  vplane.SetNormal(m.dot(n_inter)) # here the plane is rotated to crystal CS

  for i, hklplane in enumerate(hklplanes):
    origin = (a/2, a/2, a/2)
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(hklplane.normal())
    #FIXME here an assembly is retruned and we cannot access the mapper anymore
    hklplaneActor = vtk_utils.add_plane_to_grid(plane, grid, origin)
    cubic_lattice.AddPart(hklplaneActor)
    # get a reference to the vtkPolyData representing the hkl plane
    hklplanePolyData = hklplaneActor.GetMapper().GetInput()
    if show_arrows:
      # add an arrow to display the normal to the plane
      arrowActor = vtk_utils.unit_arrow_3d(origin, hklplane.normal())
      cubic_lattice.AddPart(arrowActor)
    if verbose:
      # debug infos:
      print 'slip plane normal', hklplane.normal
      print 'rotated slip plane normal', mt.dot(hklplane.normal())
      print 'intersection plane', n_inter
      print 'intersection line has unit vector:', numpy.cross(mt.dot(hklplane.normal()), n_inter)
    if show_slip_traces:
      # cut the rotated plane with the vertical plane to display the trace
      slipTrace = vtk.vtkCutter()
      slipTrace.SetInput(hklplanePolyData)
      slipTrace.SetCutFunction(vplane)
      slipTrace.Update() # this is a vtkPolyData
      slipTraceMapper = vtk.vtkPolyDataMapper()
      slipTraceMapper.SetInputConnection(slipTrace.GetOutputPort())
      slipTraceActor = vtk.vtkActor()
      slipTraceActor.SetMapper(slipTraceMapper)
      slipTraceActor.GetProperty().SetColor(colors[i])
      slipTraceActor.GetProperty().SetLineWidth(5)
      cubic_lattice.AddPart(slipTraceActor)

  # add the two actors to the renderer
  cubic_lattice.AddPart(Edges)
  cubic_lattice.AddPart(Vertices)

  # place the center of the lattice at (0.0, 0.0, 0.0)
  cubic_lattice.SetOrigin(a/2, a/2, a/2)
  cubic_lattice.AddPosition(-a/2,-a/2,-a/2)
  cubic_lattice.RotateZ(grain_orientation.phi1())
  cubic_lattice.RotateX(grain_orientation.Phi())
  cubic_lattice.RotateZ(grain_orientation.phi2())

  # add an outbounding sphere
  sphereSource = vtk.vtkSphereSource()
  sphereSource.SetCenter(0.0, 0.0, 0.0)
  sphereSource.SetRadius(r)
  sphereSource.SetPhiResolution(40)
  sphereSource.SetThetaResolution(40)
  sphereMapper = vtk.vtkPolyDataMapper()
  sphereMapper.SetInputConnection(sphereSource.GetOutputPort())
  sphereMapper.ScalarVisibilityOff()
  sphereActor = vtk.vtkActor()
  sphereActor.SetMapper(sphereMapper)
  sphereActor.GetProperty().SetOpacity(0.1)
  
  # add actors and set the background color.
  ren.AddActor(cubic_lattice)
  ren.AddActor(sphereActor)
  axes = vtk_utils.axes_actor(r)
  ren.AddViewProp(axes)
  ren.SetBackground(white)

  # set up camera
  cam = vtk.vtkCamera()
  #cam.SetViewUp(0, 1, 0)
  cam.SetViewUp(0, 0, 1)
  #cam.SetPosition(0, 0, 4*r)
  cam.SetPosition(0, -4*r, 0)
  cam.SetFocalPoint(0, 0, 0)
  ren.SetActiveCamera(cam)
  
  image_name = os.path.splitext(__file__)[0] + '.png'
  print 'writting %s' % image_name
  vtk_utils.render(ren, ren_size=(800, 800), save=True, display=False, name=image_name)

if __name__ == '__main__':
  orientation = Orientation.from_euler(numpy.array([142.8, 32.0, 214.4]))
  create_pole_figure_3d(orientation, show_arrows=True, show_slip_traces=True, verbose=False)
