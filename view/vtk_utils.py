import os
import sys
import vtk
import numpy
from vtk.util.colors import black, white, grey, blue, orange
from vtk.util import numpy_support

# see if some of the stuff needs to be moved to the Microstructure module
from pymicro.crystal.lattice import Lattice, HklPlane

def to_vtk_type(type):
  '''Function to get the VTK data type given a numpy data type.

  *Parameters*
  
  **type**: the numpy data type like 'uint8', 'uint16'...

  *Returns*

  A VTK data type.
  '''
  if type == 'uint8':
    return vtk.VTK_UNSIGNED_CHAR
  elif type == 'uint16':
    return vtk.VTK_UNSIGNED_SHORT
  elif type == 'uint32':
    return vtk.VTK_UNSIGNED_INT
  elif type == 'float':
    return vtk.VTK_FLOAT
  elif type == 'float64':
    return vtk.VTK_DOUBLE
  
def rand_cmap(N=256, first_is_black = False, table_range=(0,255)):
  '''Create a VTK look up table with random colors.
  
     The first color can be enforced to black and usually figure out 
     the image background. The random seed is fixed to 13 in order 
     to consistently produce the same colormap. '''
  numpy.random.seed(13)
  rand_colors = numpy.random.rand(N,3)
  if first_is_black:
    rand_colors[0] = [0., 0., 0.] # enforce black background
  lut = vtk.vtkLookupTable()
  lut.SetNumberOfTableValues(N)
  lut.Build()
  for i in range(N):
    lut.SetTableValue(i,rand_colors[i][0],rand_colors[i][1],rand_colors[i][2],1.0)
  lut.SetRange(table_range)
  return lut

def pv_rand_cmap(N=256, first_is_black = False):
  '''Write out the random color map in paraview xml format. '''
  numpy.random.seed(13)
  rand_colors = numpy.random.rand(N,3)
  if first_is_black:
    rand_colors[0] = [0., 0., 0.] # enforce black background
  print '<ColorMap name="random" space="RGB">'
  for i in range(N):
    print '<Point x="%d" o="1" r="%8.6f" g="%8.6f" b="%8.6f"/>' % (i, rand_colors[i][0], rand_colors[i][1], rand_colors[i][2])
  print '</ColorMap>'

def gray_cmap(table_range=(0,255)):
  '''create a black and white colormap.

  *Parameters*
  
  **table_range**: 2 values tuple (default: (0,255))
  start and end values for the table range.

  *Returns*

  A vtkLookupTable from black to white.
  '''
  lut = vtk.vtkLookupTable()
  lut.SetSaturationRange(0,0)
  lut.SetHueRange(0,0)
  lut.SetTableRange(table_range)
  lut.SetValueRange(0,1)
  lut.SetRampToLinear()
  lut.Build()
  return lut

def hot_cmap(table_range=(0,255)):
  '''Create a VTK look up table similar to matlab's hot.

  *Parameters*
  
  **table_range**: 2 values tuple (default: (0,255))
  start and end values for the table range.

  *Returns*

  A vtkLookupTable from white to red.
  '''
  lut = vtk.vtkLookupTable()
  lutNum = 64
  lut.SetNumberOfTableValues(lutNum)
  lut.Build()
  lut.SetTableValue(0,0.041667,0.000000,0.000000,1.0)
  lut.SetTableValue(1,0.083333,0.000000,0.000000,1.0)
  lut.SetTableValue(2,0.125000,0.000000,0.000000,1.0)
  lut.SetTableValue(3,0.166667,0.000000,0.000000,1.0)
  lut.SetTableValue(4,0.208333,0.000000,0.000000,1.0)
  lut.SetTableValue(5,0.250000,0.000000,0.000000,1.0)
  lut.SetTableValue(6,0.291667,0.000000,0.000000,1.0)
  lut.SetTableValue(7,0.333333,0.000000,0.000000,1.0)
  lut.SetTableValue(8,0.375000,0.000000,0.000000,1.0)
  lut.SetTableValue(9,0.416667,0.000000,0.000000,1.0)
  lut.SetTableValue(10,0.458333,0.000000,0.000000,1.0)
  lut.SetTableValue(11,0.500000,0.000000,0.000000,1.0)
  lut.SetTableValue(12,0.541667,0.000000,0.000000,1.0)
  lut.SetTableValue(13,0.583333,0.000000,0.000000,1.0)
  lut.SetTableValue(14,0.625000,0.000000,0.000000,1.0)
  lut.SetTableValue(15,0.666667,0.000000,0.000000,1.0)
  lut.SetTableValue(16,0.708333,0.000000,0.000000,1.0)
  lut.SetTableValue(17,0.750000,0.000000,0.000000,1.0)
  lut.SetTableValue(18,0.791667,0.000000,0.000000,1.0)
  lut.SetTableValue(19,0.833333,0.000000,0.000000,1.0)
  lut.SetTableValue(20,0.875000,0.000000,0.000000,1.0)
  lut.SetTableValue(21,0.916667,0.000000,0.000000,1.0)
  lut.SetTableValue(22,0.958333,0.000000,0.000000,1.0)
  lut.SetTableValue(23,1.000000,0.000000,0.000000,1.0)
  lut.SetTableValue(24,1.000000,0.041667,0.000000,1.0)
  lut.SetTableValue(25,1.000000,0.083333,0.000000,1.0)
  lut.SetTableValue(26,1.000000,0.125000,0.000000,1.0)
  lut.SetTableValue(27,1.000000,0.166667,0.000000,1.0)
  lut.SetTableValue(28,1.000000,0.208333,0.000000,1.0)
  lut.SetTableValue(29,1.000000,0.250000,0.000000,1.0)
  lut.SetTableValue(30,1.000000,0.291667,0.000000,1.0)
  lut.SetTableValue(31,1.000000,0.333333,0.000000,1.0)
  lut.SetTableValue(32,1.000000,0.375000,0.000000,1.0)
  lut.SetTableValue(33,1.000000,0.416667,0.000000,1.0)
  lut.SetTableValue(34,1.000000,0.458333,0.000000,1.0)
  lut.SetTableValue(35,1.000000,0.500000,0.000000,1.0)
  lut.SetTableValue(36,1.000000,0.541667,0.000000,1.0)
  lut.SetTableValue(37,1.000000,0.583333,0.000000,1.0)
  lut.SetTableValue(38,1.000000,0.625000,0.000000,1.0)
  lut.SetTableValue(39,1.000000,0.666667,0.000000,1.0)
  lut.SetTableValue(40,1.000000,0.708333,0.000000,1.0)
  lut.SetTableValue(41,1.000000,0.750000,0.000000,1.0)
  lut.SetTableValue(42,1.000000,0.791667,0.000000,1.0)
  lut.SetTableValue(43,1.000000,0.833333,0.000000,1.0)
  lut.SetTableValue(44,1.000000,0.875000,0.000000,1.0)
  lut.SetTableValue(45,1.000000,0.916667,0.000000,1.0)
  lut.SetTableValue(46,1.000000,0.958333,0.000000,1.0)
  lut.SetTableValue(47,1.000000,1.000000,0.000000,1.0)
  lut.SetTableValue(48,1.000000,1.000000,0.062500,1.0)
  lut.SetTableValue(49,1.000000,1.000000,0.125000,1.0)
  lut.SetTableValue(50,1.000000,1.000000,0.187500,1.0)
  lut.SetTableValue(51,1.000000,1.000000,0.250000,1.0)
  lut.SetTableValue(52,1.000000,1.000000,0.312500,1.0)
  lut.SetTableValue(53,1.000000,1.000000,0.375000,1.0)
  lut.SetTableValue(54,1.000000,1.000000,0.437500,1.0)
  lut.SetTableValue(55,1.000000,1.000000,0.500000,1.0)
  lut.SetTableValue(56,1.000000,1.000000,0.562500,1.0)
  lut.SetTableValue(57,1.000000,1.000000,0.625000,1.0)
  lut.SetTableValue(58,1.000000,1.000000,0.687500,1.0)
  lut.SetTableValue(59,1.000000,1.000000,0.750000,1.0)
  lut.SetTableValue(60,1.000000,1.000000,0.812500,1.0)
  lut.SetTableValue(61,1.000000,1.000000,0.875000,1.0)
  lut.SetTableValue(62,1.000000,1.000000,0.937500,1.0)
  lut.SetTableValue(63,1.000000,1.000000,1.000000,1.0)
  lut.SetRange(table_range)
  return lut

def add_hklplane_to_grain(hklplane, grid, orientation, origin=(0, 0, 0), 
  opacity=1.0, show_normal=False, normal_length=1.0):
  rot_plane = vtk.vtkPlane()
  rot_plane.SetOrigin(origin)
  # rotate the plane by setting the normal
  Bt = orientation.orientation_matrix().transpose()
  n_rot = numpy.dot(Bt, hklplane.normal()/numpy.linalg.norm(hklplane.normal()))
  rot_plane.SetNormal(n_rot)
  #print '[hkl] normal direction expressed in sample coordinate system is: ', n_rot
  if show_normal:
    return add_plane_to_grid(rot_plane, grid, origin, opacity=opacity)
  else:
    return add_plane_to_grid_with_normal(rot_plane, grid, origin, \
    opacity=opacity, normal_length=normal_length)

def add_plane_to_grid(plane, grid, origin, opacity=0.3):
  '''Add a 3d plane inside another object.
  
  This function adds a plane inside another object described by a mesh 
  (vtkunstructuredgrid). The method is to use a vtkCutter with the mesh 
  as input and the plane as the cut function. An actor is returned.
  This may be used directly to add hkl planes inside a lattice cell or 
  a grain.

  *Parameters*
  
  **plane**: A VTK implicit function describing the plane to add.

  **grid**: A VTK unstructured grid in which the plane is to be added.
  
  **opacity**: Opacity value of the plane actor.
  
  *Returns*
  
  A VTK actor.
  '''
  # cut the unstructured grid with the plane
  planeCut = vtk.vtkCutter()
  if vtk.vtkVersion().GetVTKMajorVersion() > 5:
    planeCut.SetInputData(grid)
  else:
    planeCut.SetInput(grid)
  planeCut.SetCutFunction(plane)

  cutMapper = vtk.vtkPolyDataMapper()
  cutMapper.SetInputConnection(planeCut.GetOutputPort())
  cutActor = vtk.vtkActor()
  cutActor.SetMapper(cutMapper)
  cutActor.GetProperty().SetOpacity(opacity)
  return cutActor
  
def add_plane_to_grid_with_normal(plane, grid, origin, opacity=0.3, normal_length=1.0):
  '''Add a 3d plane and display its normal inside another object.
  
  This function adds a plane inside another object described by a mesh 
  (vtkunstructuredgrid). It basicall call `add_plane_to_grid` and also 
  add a 3d arrow to display the plane normal.

  *Parameters*
  
  **plane**: A VTK implicit function describing the plane to add.

  **grid**: A VTK unstructured grid in which the plane is to be added.
  
  **opacity**: Opacity value of the plane actor.
  
  **normal_length**: The length of the plane normal vector.
  
  *Returns*
  
  A VTK assembly with the plane and the normal.
  '''
  assembly = vtk.vtkAssembly()
  planeActor = add_plane_to_grid(plane, grid, origin, opacity=opacity)
  assembly.AddPart(planeActor)
  # add an arrow to display the normal to the plane
  arrowActor = unit_arrow_3d(origin, normal_length*numpy.array(plane.GetNormal()), make_unit=False)
  assembly.AddPart(arrowActor)
  return assembly
  
def axes_actor(length = 1.0, axisLabels = True):
  axes = vtk.vtkAxesActor()
  axes.SetTotalLength(length, length, length)
  axes.SetShaftTypeToCylinder()
  axes.SetCylinderRadius(0.02)
  if axisLabels == True:
    axes.SetXAxisLabelText('x')
    axes.SetYAxisLabelText('y')
    axes.SetZAxisLabelText('z')
    axprop = vtk.vtkTextProperty()
    axprop.SetColor(0, 0, 0)
    axprop.SetFontSize(1)
    axprop.SetFontFamilyToArial()
    axes.GetXAxisCaptionActor2D().SetCaptionTextProperty(axprop)
    axes.GetYAxisCaptionActor2D().SetCaptionTextProperty(axprop)
    axes.GetZAxisCaptionActor2D().SetCaptionTextProperty(axprop)
  else:
    axes.SetAxisLabels(0)
  return axes

def grain_3d(grain, hklplanes=None, show_normal=False, \
  plane_opacity=1.0, show_orientation=False):
  assembly = vtk.vtkAssembly()
  # create mapper
  mapper = vtk.vtkDataSetMapper()
  if vtk.vtkVersion().GetVTKMajorVersion() > 5:
    mapper.SetInputData(grain.vtkmesh)
  else:
    mapper.SetInput(grain.vtkmesh)
  mapper.ScalarVisibilityOff() # we use the grain id for chosing the color
  lut = rand_cmap(N=2048, first_is_black = True, table_range=(0,2047))
  grain_actor = vtk.vtkActor()
  grain_actor.GetProperty().SetColor(lut.GetTableValue(grain.id)[0:3])
  grain_actor.SetMapper(mapper)
  assembly.AddPart(grain_actor)
  # add all hkl planes
  if hklplanes != None:
    for hklplane in hklplanes:
      # the grain has its center of mass at the origin
      origin = (0., 0., 0.)
      hklplaneActor = add_hklplane_to_grain(hklplane, grain.vtkmesh, \
        grain.orientation, origin, opacity=plane_opacity, \
        show_normal=show_normal, normal_length=50.)
      assembly.AddPart(hklplaneActor)
  if show_orientation:
    grain_actor.GetProperty().SetOpacity(0.3)
    local_orientation = add_local_orientation_axes(grain.orientation, axes_length=30)
    # add local orientation to the grain actor
    assembly.AddPart(local_orientation)
  return assembly

# deprecated, will be removed soon
def add_grain_to_3d_scene(grain, hklplanes, show_orientation=False):
  orientation = grain.orientation
  assembly = vtk.vtkAssembly()
  # create mapper
  print 'creating grain actor'
  mapper = vtk.vtkDataSetMapper()
  if vtk.vtkVersion().GetVTKMajorVersion() > 5:
    mapper.SetInputData(grain.vtkmesh)
  else:
    mapper.SetInput(grain.vtkmesh)
  mapper.ScalarVisibilityOff() # we use the grain id for chosing the color
  lut = rand_cmap(N=2048, first_is_black = True, table_range=(0,2047))
  grain_actor = vtk.vtkActor()
  grain_actor.GetProperty().SetColor(lut.GetTableValue(grain.id)[0:3])
  grain_actor.SetMapper(mapper)
  assembly.AddPart(grain_actor)
  # add all hkl planes and local grain orientation actor
  if show_orientation:
    grain_actor.GetProperty().SetOpacity(0.3)
    local_orientation = add_HklPlanes_with_orientation_in_grain(grain, hklplanes)
    # add local orientation to the grain actor
    assembly.AddPart(local_orientation)
  return assembly

def add_local_orientation_axes(orientation, axes_length=30):
  # use a vtkAxesActor to display the crystal orientation
  local_orientation = vtk.vtkAssembly()
  axes = axes_actor(length = axes_length, axisLabels = False)
  transform = vtk.vtkTransform()
  transform.Identity()
  transform.RotateZ(orientation.phi1())
  transform.RotateX(orientation.Phi())
  transform.RotateZ(orientation.phi2())
  axes.SetUserTransform(transform)
  local_orientation.AddPart(axes)
  return local_orientation

def add_HklPlanes_with_orientation_in_grain(grain, \
  hklplanes=[]):
  '''
  Add some plane actors corresponding to a list of (hkl) planes to 
  a grain actor.
  '''
  # use a vtkAxesActor to display the crystal orientation
  local_orientation = vtk.vtkAssembly()
  grain_axes = axes_actor(length = 30, axisLabels = False)
  transform = vtk.vtkTransform()
  transform.Identity()
  transform.RotateZ(grain.orientation.phi1())
  transform.RotateX(grain.orientation.Phi())
  transform.RotateZ(grain.orientation.phi2())
  grain_axes.SetUserTransform(transform)
  local_orientation.AddPart(grain_axes)
  # add all hkl planes to the grain
  for hklplane in hklplanes:
    hklplaneActor = add_hklplane_to_grain(hklplane, grain.vtkmesh, \
      grain.orientation)
    local_orientation.AddPart(hklplaneActor)
  return local_orientation
  
def unit_arrow_3d(start, vector, color=orange, make_unit=True):
  n = numpy.linalg.norm(vector)
  arrowSource = vtk.vtkArrowSource()
  # We build a local direct base with X being the unit arrow vector
  X = vector/n
  arb = numpy.array([1,0,0]) # used numpy here, could used the vtkMath module as well...
  Z = numpy.cross(X, arb)
  Y = numpy.cross(Z, X)
  m = vtk.vtkMatrix4x4()
  m.Identity()
  m.DeepCopy((1, 0, 0, start[0],
              0, 1, 0, start[1],
              0, 0, 1, start[2],
              0, 0, 0, 1))
  # Create the direction cosine matrix
  if make_unit: n = 1
  for i in range(3):
    m.SetElement(i, 0, n*X[i]);
    m.SetElement(i, 1, n*Y[i]);
    m.SetElement(i, 2, n*Z[i]);
  t = vtk.vtkTransform()
  t.Identity()
  t.Concatenate(m)
  transArrow = vtk.vtkTransformFilter()
  transArrow.SetInputConnection(arrowSource.GetOutputPort())
  transArrow.SetTransform(t)
  mapper = vtk.vtkPolyDataMapper()
  mapper.SetInputConnection(transArrow.GetOutputPort())
  arrowActor = vtk.vtkActor()
  arrowActor.SetMapper(mapper)
  arrowActor.GetProperty().SetColor(color)
  return arrowActor

def lattice_points(lattice, origin=[0., 0., 0.]):
  '''
  Create a vtk representation of a the lattice points.

  A vtkPoints instance is used to store the lattice points, including 
  the points not on the lattice corners according to the system 
  centering (may be P, I, F for instance).

  *Parameters*
  
  **lattice**: the `Lattice` instance from which to construct the points.
  
  **origin**: cartesian coordinates of the origin.
  
  *Returns*
  
  A vtkPoints with all the lattice points ordered such that the first 8 
  points describe the lattice cell.
  '''
  print lattice
  m = lattice._matrix
  print m
  [A, B, C] = m #lattice._matrix
  O = origin

  # create the eight points based on the lattice matrix
  points = vtk.vtkPoints()
  points.InsertNextPoint(O)
  points.InsertNextPoint(O+A)
  points.InsertNextPoint(O+A+B)
  points.InsertNextPoint(O+B)
  points.InsertNextPoint(O+C)
  points.InsertNextPoint(O+C+A)
  points.InsertNextPoint(O+C+A+B)
  points.InsertNextPoint(O+C+B)
  # use the point basis if it contain several atoms
  if len(lattice._basis) > 1:
    for point in lattice._basis[1:]:
      points.InsertNextPoint(O + point[0]*A + point[1]*B + point[2]*C)
  if lattice._centering == 'P':
    pass # nothing to do
  elif lattice._centering == 'I':
    points.InsertNextPoint(O+0.5*C+0.5*A+0.5*B)
  elif lattice._centering == 'A':
    points.InsertNextPoint(O+0.5*B+0.5*C)
    points.InsertNextPoint(O+0.5*B+0.5*C+A)
  elif lattice._centering == 'B':
    points.InsertNextPoint(O+0.5*C+0.5*A)
    points.InsertNextPoint(O+0.5*C+0.5*A+B)
  elif lattice._centering == 'C':
    points.InsertNextPoint(O+0.5*A+0.5*B)
    points.InsertNextPoint(O+0.5*A+0.5*B+C)
  elif lattice._centering == 'F':
    points.InsertNextPoint(O+0.5*A+0.5*B)
    points.InsertNextPoint(O+0.5*A+0.5*B+C)
    points.InsertNextPoint(O+0.5*B+0.5*C)
    points.InsertNextPoint(O+0.5*B+0.5*C+A)
    points.InsertNextPoint(O+0.5*C+0.5*A)
    points.InsertNextPoint(O+0.5*C+0.5*A+B)
  return points
  
def lattice_grid(lattice, origin=[0., 0., 0.]):
  '''
  Create a mesh representation of a crystal lattice.

  A vtkUnstructuredGrid instance is used with one hexaedron element
  corresponding to the lattice system.

  *Parameters*
  
  **lattice**: the `Lattice` instance from which to construct the grid.
  
  **origin**: cartesian coordinates of the origin.
  
  *Returns*
  
  A vtkUnstructuredGrid with one hexaedron cell representing the crystal 
  lattice.
  '''
  points = lattice_points(lattice, origin=[0., 0., 0.])

  # ids list
  Ids = vtk.vtkIdList()
  Ids.InsertNextId(0)
  Ids.InsertNextId(1)
  Ids.InsertNextId(2)
  Ids.InsertNextId(3)
  Ids.InsertNextId(4)
  Ids.InsertNextId(5)
  Ids.InsertNextId(6)
  Ids.InsertNextId(7)

  # build the unstructured grid with one cell
  grid = vtk.vtkUnstructuredGrid()
  grid.Allocate(1, 1)
  grid.InsertNextCell(12, Ids) # 12 is hexaedron cell type
  grid.SetPoints(points)
  return grid

def hexagonal_lattice_grid(lattice, origin=[0., 0., 0.]):
  [A, B, C] = lattice._matrix
  O = origin
  points = vtk.vtkPoints()
  points.InsertNextPoint(O)
  points.InsertNextPoint(O+A)
  points.InsertNextPoint(O+A-B)
  points.InsertNextPoint(O-2*B)
  points.InsertNextPoint(O-2*B-A)
  points.InsertNextPoint(O-B-A)
  points.InsertNextPoint(O+C)
  points.InsertNextPoint(O+A+C)
  points.InsertNextPoint(O+A-B+C)
  points.InsertNextPoint(O-2*B+C)
  points.InsertNextPoint(O-2*B-A+C)
  points.InsertNextPoint(O-B-A+C)

  ids = vtk.vtkIdList()
  ids.InsertNextId(0)
  ids.InsertNextId(1)
  ids.InsertNextId(2)
  ids.InsertNextId(3)
  ids.InsertNextId(4)
  ids.InsertNextId(5)
  ids.InsertNextId(6)
  ids.InsertNextId(7)
  ids.InsertNextId(8)
  ids.InsertNextId(9)
  ids.InsertNextId(10)
  ids.InsertNextId(11)
  # build the unstructured grid with one cell
  grid = vtk.vtkUnstructuredGrid()
  grid.Allocate(1, 1)
  grid.InsertNextCell(16, ids) # 16 is hexagonal prism cell type
  grid.SetPoints(points)
  return grid

def lattice_edges(grid, tubeRadius=0.02):
  '''
  Create the 3D representation of crystal lattice edges.

  *Parameters*
  
  **grid**: vtkUnstructuredGrid
  The vtkUnstructuredGrid instance representing the crystal lattice.

  **tubeRadius**: float
  Radius of the tubes representing the atomic bonds (default: 0.02).

  *Returns*

  The method return a vtk actor for lattice edges.
  '''
  Edges = vtk.vtkExtractEdges()
  if vtk.vtkVersion().GetVTKMajorVersion() > 5:
    Edges.SetInputData(grid)
  else:
    Edges.SetInput(grid)
  Tubes = vtk.vtkTubeFilter()
  Tubes.SetInputConnection(Edges.GetOutputPort())
  Tubes.SetRadius(tubeRadius)
  Tubes.SetNumberOfSides(6)
  Tubes.UseDefaultNormalOn()
  Tubes.SetDefaultNormal(.577, .577, .577)
  # Create the mapper and actor to display the cell edges.
  TubeMapper = vtk.vtkPolyDataMapper()
  TubeMapper.SetInputConnection(Tubes.GetOutputPort())
  Edges = vtk.vtkActor()
  Edges.SetMapper(TubeMapper)
  return Edges

def lattice_vertices(grid, sphereRadius=0.1):
  '''
  Create the 3D representation of crystal lattice atoms.

  *Parameters*
  
  **grid**: vtkUnstructuredGrid
  The vtkUnstructuredGrid instance representing the crystal lattice.

  **sphereRadius**: float
  Size of the spheres representing the atoms (default: 0.1).

  *Returns*

  The method return a vtk actor for lattice vertices.
  '''
  # Create a sphere to use as a glyph source for vtkGlyph3D.
  Sphere = vtk.vtkSphereSource()
  Sphere.SetRadius(sphereRadius)
  Sphere.SetPhiResolution(20)
  Sphere.SetThetaResolution(20)
  Vertices = vtk.vtkGlyph3D()
  if vtk.vtkVersion().GetVTKMajorVersion() > 5:
    Vertices.SetInputData(grid)
  else:
    Vertices.SetInput(grid)
  Vertices.SetSourceConnection(Sphere.GetOutputPort())
  # Create a mapper and actor to display the glyphs.
  SphereMapper = vtk.vtkPolyDataMapper()
  SphereMapper.SetInputConnection(Vertices.GetOutputPort())
  SphereMapper.ScalarVisibilityOff()
  Vertices = vtk.vtkActor()
  Vertices.SetMapper(SphereMapper)
  Vertices.GetProperty().SetDiffuseColor(blue)
  return Vertices
  
def lattice_3d(lattice, sphereRadius=0.1, tubeRadius=0.02):
  '''
  Create the 3D representation of a crystal lattice.

  The lattice edges are shown using a vtkTubeFilter and the atoms are 
  displayed using spheres. Both tube and sphere radius can be controlled.

  .. code-block:: python

    l = Lattice.cubic(1.0)
    cubic = lattice_3d(l)
    ren = vtk.vtkRenderer()
    ren.AddActor(cubic)
    render(ren, display=True)

  .. figure:: _static/lattice_3d.png
      :width: 300 px
      :height: 300 px
      :alt: lattice_3d
      :align: center

      A 3D view of a cubic lattice.

  *Parameters*
  
  **lattice**: Lattice
  The Lattice instance representing the crystal lattice.

  **sphereRadius**: float
  Size of the spheres representing the atoms (default: 0.1).

  **tubeRadius**: float
  Radius of the tubes representing the atomic bonds (default: 0.02).

  *Returns*

  The method return a vtk assembly combining lattice edges and vertices.
  '''
  grid = lattice_grid(lattice)
  edges = lattice_edges(grid, tubeRadius=tubeRadius)
  vertices = lattice_vertices(grid, sphereRadius=sphereRadius)
  assembly = vtk.vtkAssembly()
  assembly.AddPart(edges)
  assembly.AddPart(vertices)
  return assembly

def lattice_3d_with_planes(lattice, hklplanes, crystal_orientation=None, \
  show_atoms=True, show_normal=True, plane_opacity=1.0):
  '''
  Create the 3D representation of a crystal lattice.
  HklPlanes can be displayed within the lattice cell with their normals.
  A single vtk actor in form of an assembly is returned.
  Crystal orientation can also be provided which rotates the whole assembly appropriately.

  .. code-block:: python

    l = Lattice.cubic(1.0)
    o = Orientation.from_euler((344.0, 125.0, 217.0))

    grid = lattice_grid(l)
    hklplanes = Hklplane.get_family('111')
    cubic = lattice_3d_with_planes(grid, hklplanes, crystal_orientation=o, \\
      show_normal=True, plane_opacity=0.5)
    ren = vtk.vtkRenderer()
    ren.AddActor(cubic)
    render(ren, display=True)

  .. figure:: _static/cubic_crystal_3d.png
     :width: 300 px
     :alt: lattice_3d_with_planes
     :align: center

     A 3D view of a cubic lattice with all four 111 planes displayed.

  *Parameters*

  **hklplanes**: list of `pymicro.crystal.lattice.HklPlane`
  A list of the hkl planes to add to the lattice.

  **crystal_orientation**: Orientation
  The crystal orientation with respect to the sample coordinate system
  (default: None).
  
  **show_atoms** bool
  A boolean controling if the atoms are shown (default: True)

  **show_normal** bool
  A boolean controling if the slip plane normals are shown (default: True)
  
  **plane_opacity** float in [0., 1.0]
  A float number controlling the slip plane opacity.

  *Returns*

  The method return a vtkAssembly that can be directly added to a renderer.
  '''
  # get grid corresponding to the crystal lattice
  grid = lattice_grid(lattice)
  (a, b, c) = lattice._lengths
  
  # an assembly is used to gather all the actors together
  assembly = vtk.vtkAssembly()

  # display all the hkl planes (with normal)
  for hklplane in hklplanes:
    origin = (a/2, b/2, c/2)
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(hklplane.normal())
    hklplaneActor = add_plane_to_grid(plane, grid, origin, opacity=plane_opacity)
    assembly.AddPart(hklplaneActor)
    if show_normal:
      # add an arrow to display the normal to the plane
      arrowActor = unit_arrow_3d(origin, a*hklplane.normal(), make_unit=False)
      assembly.AddPart(arrowActor)
  
  Edges = lattice_edges(grid, tubeRadius=0.02*a)
  Vertices = lattice_vertices(grid, sphereRadius=0.1*a)
  # add the two actors to the renderer
  assembly.AddPart(Edges)
  if show_atoms: assembly.AddPart(Vertices)

  # finally, apply crystal orientation to the lattice
  assembly.SetOrigin(a/2, b/2, c/2)
  assembly.AddPosition(-a/2, -b/2, -c/2)
  if crystal_orientation != None:
    apply_orientation_to_actor(assembly, crystal_orientation)
  return assembly

def apply_orientation_to_actor(actor, orientation):
  '''
  Transform the actor assembly using the specified Orientation.
  The three euler angles are used according to Bunge's convention.
  '''
  transform = vtk.vtkTransform()
  transform.Identity()
  transform.RotateZ(orientation.phi1())
  transform.RotateX(orientation.Phi())
  transform.RotateZ(orientation.phi2())
  actor.SetUserTransform(transform)

def load_STL_actor(name, verbose=False):
  '''Read a STL file and return the corresponding vtk actor.
  '''
  if verbose: print 'adding part: %s' % name
  part = vtk.vtkSTLReader()
  part.SetFileName(name + ".STL")
  part.Update()
  partMapper = vtk.vtkPolyDataMapper()
  partMapper.SetInputConnection(part.GetOutputPort())
  partActor = vtk.vtkActor()
  partActor.SetMapper(partMapper)
  return partActor
  
def read_image_data(file_name, size, header_size=0, data_type='uint8', verbose=False):
  '''
  vtk helper function to read a 3d data file.
  The size is needed in the form (x, y, z) as well a string describing
  the data type in numpy format (uint8 is assumed by default).
  Lower file left and little endian are assumed.

  *Parameters*

  **file_name**: the name of the file to read.

  **size**: a sequence of three number describing the size of the 3d data set
  
  **header_size**: size of the header to skip in bytes (0 by default)
  
  **data_type**: a string describing the data type in numpy format ('uint8' by default)
  
  **verbose**: verbose mode (False by default)
  
  *Returns*
  
  A VTK data array
  '''
  vtk_type = to_vtk_type(data_type)
  if verbose:
    print 'reading scan %s with size %dx%dx%d using vtk type %d' % \
      (file_name, size[0], size[1], size[2], vtk_type)
  reader = vtk.vtkImageReader2() # 2 is faster
  reader.SetDataScalarType(vtk_type)
  reader.SetFileDimensionality(3)
  reader.SetHeaderSize(header_size)
  reader.SetDataByteOrderToLittleEndian()
  reader.FileLowerLeftOn()
  reader.SetDataExtent (0, size[0]-1, 0, size[1]-1, 0, size[2]-1)
  reader.SetNumberOfScalarComponents(1)
  reader.SetDataOrigin(0, 0, 0)
  reader.SetFileName(file_name)
  reader.Update()
  data = reader.GetOutput()
  return data

def data_outline(data, corner=False, color=black):
  '''
  vtk helper function to draw a bounding box around a volume.
  '''
  if corner:
    outlineFilter = vtk.vtkOutlineCornerFilter()
  else:
    outlineFilter = vtk.vtkOutlineFilter()
  if vtk.vtkVersion().GetVTKMajorVersion() > 5:
    outlineFilter.SetInputData(data)
  else:
    outlineFilter.SetInput(data)
  outlineMapper = vtk.vtkPolyDataMapper()
  outlineMapper.SetInputConnection(outlineFilter.GetOutputPort())
  outline = vtk.vtkActor()
  outline.SetMapper(outlineMapper)
  outline.GetProperty().SetColor(color)
  return outline

def box_3d(size=(100, 100, 100), line_color=black):
  '''
  vtk helper function to draw a box of a given size.
  '''
  l = Lattice.orthorombic(size[0], size[1], size[2])
  grid = lattice_grid(l, origin=[0., 0., 0.])
  edges = vtk.vtkExtractEdges()
  if vtk.vtkVersion().GetVTKMajorVersion() > 5:
    edges.SetInputData(grid)
  else:
    edges.SetInput(grid)
  mapper = vtk.vtkPolyDataMapper()
  mapper.SetInputConnection(edges.GetOutputPort())
  box = vtk.vtkActor()
  box.SetMapper(mapper)
  box.GetProperty().SetColor(line_color)
  return box

def line_3d(start_point, end_point):
  '''
  vtk helper function to draw a line in a 3d scene.
  '''
  linePoints = vtk.vtkPoints()
  linePoints.SetNumberOfPoints(2)
  linePoints.InsertPoint(0, start_point[0], start_point[1], start_point[2])
  linePoints.InsertPoint(1, end_point[0], end_point[1], end_point[2])
  aLine = vtk.vtkLine()
  aLine.GetPointIds().SetId(0, 0)
  aLine.GetPointIds().SetId(1, 1)
  aLineGrid = vtk.vtkUnstructuredGrid()
  aLineGrid.Allocate(1, 1)
  aLineGrid.InsertNextCell(aLine.GetCellType(), aLine.GetPointIds())
  aLineGrid.SetPoints(linePoints)
  aLineMapper = vtk.vtkDataSetMapper()
  if vtk.vtkVersion().GetVTKMajorVersion() > 5:
    aLineMapper.SetInputData(aLineGrid)
  else:
    aLineMapper.SetInput(aLineGrid)
  aLineActor = vtk.vtkActor()
  aLineActor.SetMapper(aLineMapper)
  return aLineActor
  
def contourFilter(data, value, color=grey, diffuseColor=grey, opacity=1.0, discrete=False):
  if discrete:
    contour = vtk.vtkDiscreteMarchingCubes()
  else:
    contour = vtk.vtkContourFilter()
  if vtk.vtkVersion().GetVTKMajorVersion() > 5:
    contour.SetInputData(data)
  else:
    contour.SetInput(data)
  contour.SetValue(0, value)
  contour.Update()
  normals = vtk.vtkPolyDataNormals()
  normals.SetInputConnection(contour.GetOutputPort())
  normals.SetFeatureAngle(60.0)
  mapper = vtk.vtkPolyDataMapper()
  mapper.ScalarVisibilityOff()
  mapper.SetInputConnection(normals.GetOutputPort())
  mapper.Update()
  actor = vtk.vtkActor()
  actor.SetMapper(mapper)
  print 'setting actor color to',color
  actor.GetProperty().SetColor(color)
  actor.GetProperty().SetDiffuseColor(diffuseColor)
  actor.GetProperty().SetSpecular(.4)
  actor.GetProperty().SetSpecularPower(10)
  actor.GetProperty().SetOpacity(opacity)
  return actor

def map_data_with_clip(data, lut = gray_cmap(), cell_data=True):
  '''This method construct an actor to map a 3d dataset.

  1/8 of the data is clipped out to have a better view of the interior.
  It requires a fair amount of memory so downsample your data if you can
  (it may not be visible at all on the resulting image).
  
  .. code-block:: python

    data = read_image_data(im_file, size)
    ren = vtk.vtkRenderer()
    actor = map_data_with_clip(data)
    ren.AddActor(actor)
    render(ren, display=True)

  .. figure:: _static/pa66gf30_clip_3d.png
     :width: 300 px
     :alt: pa66gf30_clip_3d
     :align: center

     A 3D view of a polyamid sample with reinforcing glass fibers.
  
  *Parameters*

  **data**: the dataset to map, in VTK format.
  
  **lut**: VTK look up table (default: `gray_cmap`).

  **cell_data**: boolean to map cell data or point data if False (True by default)
  
  *Returns*

  The method return a vtkActor that can be directly added to a renderer.
  '''  
  size = 1 + numpy.array(data.GetExtent()[1::2])
  # implicit function
  bbox = vtk.vtkBox()
  bbox.SetXMin(size[0]/2., -1, size[2]/2.)
  bbox.SetXMax(size[0], size[1]/2., size[2])
  return map_data(data, bbox, lut = lut)

def map_data(data, function, lut = gray_cmap(), cell_data=True):
  '''This method construct an actor to map a 3d dataset.

  It requires a fair amount of memory so downsample your data if you can
  (it may not be visible at all on the resulting image).
  
  *Parameters*

  **data**: the dataset to map, in VTK format.
  
  **function**: VTK implicit function where to map the data.
  
  **lut**: VTK look up table (default: `gray_cmap`).

  **cell_data**: boolean to map cell data or point data if False (True by default)
  
  *Returns*

  The method return a vtkActor that can be directly added to a renderer.
  '''  
  # use extract geometry filter to access the data
  extract = vtk.vtkExtractGeometry()
  if vtk.vtkVersion().GetVTKMajorVersion() > 5:
    extract.SetInputData(data)
  else:
    extract.SetInput(data)
  extract.ExtractInsideOff()
  extract.ExtractBoundaryCellsOn()
  extract.SetImplicitFunction(function)

  mapper = vtk.vtkDataSetMapper()
  mapper.ScalarVisibilityOn()
  mapper.SetLookupTable(lut)
  mapper.UseLookupTableScalarRangeOn()
  if cell_data:
    mapper.SetScalarModeToUseCellData()
  else:
    mapper.SetScalarModeToUsePointData()
  mapper.SetColorModeToMapScalars()
  if vtk.vtkVersion().GetVTKMajorVersion() > 5:
    mapper.SetInputConnection(extract.GetOutputPort())
    # with VTK 6, since SetInputData does not create a pipeline, we can also use:
    #extract.Update()
    #mapper.SetInputData(extract.GetOutput())
  else:
    mapper.SetInput(extract.GetOutput())
  mapper.Update()
  actor = vtk.vtkActor()
  actor.SetMapper(mapper)
  return actor
  
def setup_camera(size=(100, 100, 100)):
  '''Setup the camera with usual viewing parameters.

  The camera is looking at the center of the data with the Z-axis vertical.

  *Parameters*
  
  **size**: the size of the 3d data set (100x100x100 by default).

  '''
  cam = vtk.vtkCamera()
  cam.SetViewUp(0, 0, 1)
  cam.SetPosition(2*size[0], -2*size[1], 2*size[2])
  cam.SetFocalPoint(0.5*size[0], 0.5*size[1], 0.5*size[2])
  cam.SetClippingRange(1, 10*max(size))
  return cam

def render(ren, ren_size=(600, 600), display=True, save=False, name='render_3d.png'):
  '''Render the VTK scene in 3D.
  
  Given a `vtkRenderer`, this function does the actual 3D rendering. It 
  can be used to display the scene interactlively and/or save a still 
  image in png format.
  
  *Parameters*
  
  **ren**: the VTK renderer with containing all the actors.
  
  **ren_size**: a tuple with two value to set the size of the image in 
  pixels (defalut 600x600).
  
  **display**: a boolean to control if the scene has to be displayed 
  interactively to the user (default True).
  
  **save**: a boolean to to control if the scene has to be saved as a 
  png image (default False).
  
  **name**: a string to used when saving the scene as an image (default 
  is 'render_3d.png').
  '''
  # Create a window for the renderer
  if save:
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(ren_size)
    # capture the display and write a png image
    w2i = vtk.vtkWindowToImageFilter()
    writer = vtk.vtkPNGWriter()
    w2i.SetInput(renWin)
    w2i.Update()
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.SetFileName(name)
    renWin.Render()
    writer.Write()
  if display:
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(ren_size)
    # Start the initialization and rendering
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    renWin.Render()
    iren.Initialize()
    iren.Start()

def show_data(data, map_scalars=False, lut=None):
  '''Create a 3d actor representing a numpy array.
  
  Given a 3d numpy array, this function compute the skin of the volume. 
  The scalars can be mapped to the created surface and the colormap 
  adjusted.
  
  *Parameters*
  
  **data**: a numpy array.
  
  **data**: bool.
  map the scalar in the data array to the created surface.
  
  **lut**: vtk lookup table
  the colormap used to map the scalars.
  
  Returns a vtk actor that can be added to a rendered to show the 
  3d array.
  '''
  size = data.shape
  vtk_data_array = numpy_support.numpy_to_vtk(numpy.ravel(data, order='F'), deep=1)
  grid = vtk.vtkUniformGrid()
  grid.SetSpacing(1, 1, 1)
  if vtk.vtkVersion().GetVTKMajorVersion() > 5:
    grid.SetScalarType(to_vtk_type(data.dtype), vtk.vtkInformation())
  else:
    grid.SetScalarType(to_vtk_type(data.dtype))
  grid.SetExtent(0, size[0], 0, size[1], 0, size[2]) # for cell data
  grid.GetCellData().SetScalars(vtk_data_array)
  visible = numpy_support.numpy_to_vtk(numpy.ravel(data > 0, order='F').astype(numpy.uint8), deep=1)
  grid.SetCellVisibilityArray(visible)

  # use extract geometry filter to access the data
  extract = vtk.vtkExtractGeometry()
  #extract = vtk.vtkExtractVOI() # much faster but seems not to work with blanking
  if vtk.vtkVersion().GetVTKMajorVersion() > 5:
    extract.SetInputData(grid)
  else:
    extract.SetInput(grid)
  extract.ExtractInsideOn()
  extract.ExtractBoundaryCellsOn()
  bbox = vtk.vtkBox()
  bbox.SetXMin(0, 0, 0)
  bbox.SetXMax(size[0], size[1], size[2])
  extract.SetImplicitFunction(bbox)

  mapper = vtk.vtkDataSetMapper()
  mapper.ScalarVisibilityOff()
  if map_scalars:
    mapper.ScalarVisibilityOn()
    mapper.UseLookupTableScalarRangeOn()
    mapper.SetScalarModeToUseCellData();
    mapper.SetColorModeToMapScalars();
    if not lut:
      # default to the usual gray colormap
      lut = gray_cmap()
    mapper.SetLookupTable(lut)
  if vtk.vtkVersion().GetVTKMajorVersion() > 5:
    mapper.SetInputConnection(extract.GetOutputPort())
  else:
    mapper.SetInput(extract.GetOutput())
  mapper.Update()
  actor = vtk.vtkActor()
  actor.SetMapper(mapper)
  actor.GetProperty().SetSpecular(.4)
  actor.GetProperty().SetSpecularPower(10)
  actor.GetProperty().SetOpacity(1.0)
  return actor

def show_grains(data):
  '''Create a 3d actor of all the grains in a labeled numpy array.
  
  Given a 3d numpy array, this function compute the skin of all the 
  grains (labels > 0). the background is assumed to be zero and is 
  removed. The actor produced is colored by the grain ids using the 
  random color map, see `rand_cmap`.
  
  *Parameters*
  
  **data**: a labeled numpy array.
  
  Returns a vtk actor that can be added to a rendered to show all the 
  grains colored by their id.  
  '''
  grain_lut = rand_cmap(N=2048, first_is_black = True, table_range=(0,2047))
  grains = show_data(data, map_scalars=True, lut=grain_lut)
  return grains

def xray_arrow():
  xrays_arrow = vtk.vtkArrowSource()
  xrays_mapper = vtk.vtkPolyDataMapper()
  xrays_mapper.SetInputConnection(xrays_arrow.GetOutputPort())
  xrays = vtk.vtkActor()
  xrays.SetMapper(xrays_mapper)
  return xrays
  
def slits(size, x_slits=0):
  '''Create a 3d schematic represenation of X-ray slits.
  
  The 3d represenation is made of 4 corners of the given size along 
  the Y and Z axes.
  
  **Parameters**:
  
  *size*: a (X,Y,Z) tuple giving the size of the illuminated volume.
  The first value of the tuple is not used.
  
  *x_slits*: position of the slits along the X axis (0 be default).
  
  **Returns**:
  
  A vtk assembly of the 4 corners representing the slits.
  '''
  slits = vtk.vtkAssembly()
  corner_points = numpy.empty((3, 3, 4), dtype=numpy.float)
  corner_points[:, 0, 0] = [x_slits, -0.6*size[1]/2, -size[2]/2]
  corner_points[:, 1, 0] = [x_slits, -size[1]/2, -size[2]/2]
  corner_points[:, 2, 0] = [x_slits, -size[1]/2, -0.6*size[2]/2]
  corner_points[:, 0, 1] = [x_slits, -0.6*size[1]/2, size[2]/2]
  corner_points[:, 1, 1] = [x_slits, -size[1]/2, size[2]/2]
  corner_points[:, 2, 1] = [x_slits, -size[1]/2, 0.6*size[2]/2]
  corner_points[:, 0, 2] = [x_slits, 0.6*size[1]/2, -size[2]/2]
  corner_points[:, 1, 2] = [x_slits, size[1]/2, -size[2]/2]
  corner_points[:, 2, 2] = [x_slits, size[1]/2, -0.6*size[2]/2]
  corner_points[:, 0, 3] = [x_slits, 0.6*size[1]/2, size[2]/2]
  corner_points[:, 1, 3] = [x_slits, size[1]/2, size[2]/2]
  corner_points[:, 2, 3] = [x_slits, size[1]/2, 0.6*size[2]/2]
  print corner_points
  for c in range(4):
    linePoints = vtk.vtkPoints()
    linePoints.SetNumberOfPoints(3)
    linePoints.InsertPoint(0, corner_points[:, 0, c])
    linePoints.InsertPoint(1, corner_points[:, 1, c])
    linePoints.InsertPoint(2, corner_points[:, 2, c])
    line1 = vtk.vtkLine()
    line1.GetPointIds().SetId(0, 0)
    line1.GetPointIds().SetId(1, 1)
    line2 = vtk.vtkLine()
    line2.GetPointIds().SetId(0, 1)
    line2.GetPointIds().SetId(1, 2)
    slitCorner1Grid = vtk.vtkUnstructuredGrid()
    slitCorner1Grid.Allocate(2, 1)
    slitCorner1Grid.InsertNextCell(line1.GetCellType(), line1.GetPointIds())
    slitCorner1Grid.InsertNextCell(line2.GetCellType(), line2.GetPointIds())
    slitCorner1Grid.SetPoints(linePoints)
    slitCorner1Mapper = vtk.vtkDataSetMapper()
    if vtk.vtkVersion().GetVTKMajorVersion() > 5:
      slitCorner1Mapper.SetInputData(slitCorner1Grid)
    else:
      slitCorner1Mapper.SetInput(slitCorner1Grid)
    slitCorner1Actor = vtk.vtkActor()
    slitCorner1Actor.SetMapper(slitCorner1Mapper)
    slitCorner1Actor.GetProperty().SetLineWidth(2.0)
    slitCorner1Actor.GetProperty().SetDiffuseColor(black)
    slits.AddPart(slitCorner1Actor)
  return slits

def pin_hole(inner_radius=100, outer_radius=200):
  pin_hole = vtk.vtkAssembly()
  disc = vtk.vtkDiskSource()
  disc.SetCircumferentialResolution(50)
  disc.SetInnerRadius(inner_radius)
  disc.SetOuterRadius(outer_radius)
  disc_mapper = vtk.vtkPolyDataMapper()
  disc_mapper.SetInputConnection(disc.GetOutputPort())
  discActor = vtk.vtkActor()
  discActor.SetMapper(disc_mapper)
  discActor.GetProperty().SetColor(black)
  pin_hole.AddPart(discActor)
  pin_hole.RotateY(90)
  return pin_hole
    
def zone_plate(thk=50, sep=25, n_rings=5):
  '''Create a 3d schematic represenation of a Fresnel zone plate.
  
  The 3d represenation is made of a number or concentric rings separated 
  by a specific distance which control the X-ray focalisation.
  
  **Parameters**:
  
  *thk*: ring thickness (50 by default).
  
  *sep*: ring spacing (25 by default).
  
  **Returns**:
  
  A vtk assembly of the rings composing the Fresnel zone plate.
  '''
  zone_plate = vtk.vtkAssembly()
  for i in range(n_rings):
    disc = vtk.vtkDiskSource()
    disc.SetCircumferentialResolution(50)
    disc.SetInnerRadius(i*(thk+sep))
    disc.SetOuterRadius((i+1)*thk + i*sep)
    disc_mapper = vtk.vtkPolyDataMapper()
    disc_mapper.SetInputConnection(disc.GetOutputPort())
    discActor = vtk.vtkActor()
    discActor.SetMapper(disc_mapper)
    zone_plate.AddPart(discActor)
  zone_plate.RotateY(90)
  return zone_plate

def grid_vol_view(scan):
  s_size = scan[:-4].split('_')[-2].split('x')
  s_type = scan[:-4].split('_')[-1]
  size = [int(s_size[0]), int(s_size[1]), int(s_size[2])]
  # prepare a uniform grid to receive the image data
  uGrid = vtk.vtkUniformGrid()
  uGrid.SetExtent(0,size[0],0,size[1],0,size[2])
  uGrid.SetOrigin(0,0,0)
  uGrid.SetSpacing(1,1,1)
  uGrid.SetScalarType(to_vtk_type(s_type))
  # read the actual image data
  print 'reading scan %s with size %dx%dx%d using type %d' % \
    (scan, size[0], size[1], size[2], to_vtk_type(s_type))
  reader = vtk.vtkImageReader2() # 2 is faster
  reader.SetDataScalarType(to_vtk_type(s_type))
  reader.SetFileDimensionality(3)
  reader.SetHeaderSize(0)
  reader.SetDataByteOrderToLittleEndian()
  reader.FileLowerLeftOn()
  reader.SetDataExtent (0, size[0]-1, 0, size[1]-1, 0, size[2]-1)
  reader.SetNumberOfScalarComponents(1)
  reader.SetDataOrigin(0, 0, 0)
  reader.SetFileName(scan)
  reader.Update()
  data = reader.GetOutput()
  # expose the image data array
  array = data.GetPointData().GetScalars()
  uGrid.GetCellData().SetScalars(array)
  uGrid.SetCellVisibilityArray(array)
  # create random lut
  lut = rand_cmap(N=2048, first_is_black = True, table_range=(0,2047)) 
  # prepare the implicit function
  bbox = vtk.vtkBox()
  bbox.SetXMin(0,0,0)
  bbox.SetXMax(size[0],size[1],size[2])
  # use extract geometry filter to clip data
  extract = vtk.vtkExtractGeometry()
  extract.SetInput(uGrid)
  #extract.ExtractInsideOn()
  extract.SetImplicitFunction(bbox)
  extract.ExtractBoundaryCellsOn()
  # create mapper
  print 'creating actors'
  mapper = vtk.vtkDataSetMapper()
  mapper.SetLookupTable(lut)
  mapper.SetInput(extract.GetOutput())
  mapper.UseLookupTableScalarRangeOn()
  mapper.SetScalarModeToUseCellData();
  mapper.SetColorModeToMapScalars();
  actor = vtk.vtkActor()
  actor.SetMapper(mapper)
  # set up camera
  cam = vtk.vtkCamera()
  cam.SetViewUp(0, 0, 1)
  cam.SetPosition(size[0], -size[1], 200)
  cam.SetFocalPoint(size[0]/2, size[1]/2, size[2]/2)
  cam.Dolly(0.6)
  cam.SetClippingRange(0,1000)
  # add axes actor
  l = 0.5*numpy.mean(size)
  axes = axes_actor(length = l, axisLabels = True)
  # Create renderer
  ren = vtk.vtkRenderer()
  ren.SetBackground(1.0, 1.0, 1.0)
  ren.AddActor(actor)
  #ren.AddActor(outline)
  ren.AddViewProp(axes);
  ren.SetActiveCamera(cam)
  
  # Create a window for the renderer
  renWin = vtk.vtkRenderWindow()
  renWin.AddRenderer(ren)
  renWin.SetSize(600, 600)
  # Start the initialization and rendering
  iren = vtk.vtkRenderWindowInteractor()
  iren.SetRenderWindow(renWin)
  renWin.Render()
  iren.Initialize()
  iren.Start()
  print 'done'

def vol_view(scan):
  #TODO change from scan name to numpy array
  s_size = scan[:-4].split('_')[-2].split('x')
  s_type = scan[:-4].split('_')[-1]
  size = [int(s_size[0]), int(s_size[1]), int(s_size[2])]
  print 'reading scan %s with size %dx%dx%d using type %d' % \
    (scan, size[0], size[1], size[2], to_vtk_type(s_type))
  reader = vtk.vtkImageReader2() # 2 is faster
  reader.SetDataScalarType(to_vtk_type(s_type))
  reader.SetFileDimensionality(3)
  reader.SetHeaderSize(0)
  reader.SetDataByteOrderToLittleEndian()
  reader.FileLowerLeftOn()
  reader.SetDataExtent (0, size[0]-1, 0, size[1]-1, 0, 100)#size[2]-1)
  reader.SetNumberOfScalarComponents(1)
  reader.SetDataOrigin(0, 0, 0)
  reader.SetFileName(scan)
  data = reader.GetOutput()
  # threshold to remove background  
  print 'thresholding to remove background'
  thresh = vtk.vtkThreshold()
  if vtk.vtkVersion().GetVTKMajorVersion() > 5:
    thresh.SetInputData(data)
  else:
    thresh.SetInput(data)
  #thresh.SetInputConnection(data)
  thresh.Update()
  thresh.ThresholdByUpper(1.0)
  thresh.SetInputArrayToProcess(1, 0, 0, 0, "ImageFile")
  # create random lut
  lut = rand_cmap(N=2048, first_is_black = True, table_range=(0,2047)) 
  # create mapper
  print 'creating actors'
  mapper = vtk.vtkDataSetMapper()
  mapper.SetLookupTable(lut)
  mapper.SetInputConnection(thresh.GetOutputPort())
  #mapper.SetInput(data)
  mapper.UseLookupTableScalarRangeOn()
  mapper.SetScalarModeToUsePointData();
  mapper.SetColorModeToMapScalars();
  actor = vtk.vtkActor()
  actor.SetMapper(mapper)
  # set up camera
  cam = vtk.vtkCamera()
  cam.SetViewUp(0, 0, 1)
  cam.SetPosition(400, -400, 300)
  cam.SetFocalPoint(size[0], size[1], size[2])
  cam.SetClippingRange(20,1000)
  # add axes actor
  l = min(size)
  axes = axes_actor(length = l, axisLabels = True)
  # Create renderer
  ren = vtk.vtkRenderer()
  ren.SetBackground(1.0, 1.0, 1.0)
  ren.AddActor(actor)
  #ren.AddActor(outline)
  ren.AddViewProp(axes);
  ren.SetActiveCamera(cam)
  
  # Create a window for the renderer
  renWin = vtk.vtkRenderWindow()
  renWin.AddRenderer(ren)
  renWin.SetSize(600, 600)
  # Start the initialization and rendering
  iren = vtk.vtkRenderWindowInteractor()
  iren.SetRenderWindow(renWin)
  renWin.Render()
  iren.Initialize()
  iren.Start()
  print 'done'

def ask_for_map_file(dir,scan_name):
    list={}; i = 0
    print 'no map file was specified, please chose from the following file available'
    for file in os.listdir(dir):
        if file.startswith(scan_name + '.'):
            i += 1
            list[i]=file
            print ' * ', file, '[',i,']'
    if i==0:
        sys.exit('no matching map file could be located, exiting...')
    r = raw_input('chose file by entering the coresponding number [1]: ')
        
    if r == '':
        return list[1]
    else:
        try:
            ir = int(r)
        except:
            sys.exit('not a number, exiting...')
        else:
            if int(r) < i+1:
                return list[int(r)]
            else:
                sys.exit('wrong entry, exiting...')

