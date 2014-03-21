import os
import sys
import vtk
from vtk.util.colors import *
import numpy
# see if some of the stuff needs to be moved to the Microstructure module
from PyMicro.crystal.lattice import HklPlane
#from PyMicro.crystal.microstructure import * 

def to_vtk_type(type):
  if type == 'uint8':
    return vtk.VTK_UNSIGNED_CHAR
  elif type == 'uint16':
    return vtk.VTK_UNSIGNED_SHORT
  
def rand_cmap(N=256, first_is_black = False, table_range=(0,255)):
  '''create a look up table with random color.
     The first color can be enforced to black and usually figure out the background.
     The random seed is fixed to consistently produce the same colormap. '''
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
  '''write out the random color map in paraview xml format. '''
  numpy.random.seed(13)
  rand_colors = numpy.random.rand(N,3)
  if first_is_black:
    rand_colors[0] = [0., 0., 0.] # enforce black background
  print '<ColorMap name="random" space="RGB">'
  for i in range(N):
    print '<Point x="%d" o="1" r="%8.6f" g="%8.6f" b="%8.6f"/>' % (i, rand_colors[i][0], rand_colors[i][1], rand_colors[i][2])
  print '</ColorMap>'

def gray_cmap(table_range=(0,255)):
  '''create a black and white colormap.'''
  lut = vtk.vtkLookupTable()
  lut.SetSaturationRange(0,0)
  lut.SetHueRange(0,0)
  lut.SetTableRange(table_range)
  lut.SetValueRange(0,1)
  lut.SetRampToLinear()
  lut.Build()
  return lut

def hot_cmap(table_range=(0,255)):
  '''create a look up table similar to matlab's hot.'''
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

def add_hklplane_to_grain(hklplane, grid, euler, origin=(0, 0, 0)):
  rot_plane = vtk.vtkPlane()
  rot_plane.SetOrigin(origin)
  # rotate the plane by setting the normal
  rot_normal = numpy.array([0., 0., 0.])
  transform = vtk.vtkTransform()
  transform.Identity()
  transform.RotateZ(euler.phi1)
  transform.RotateX(euler.Phi)
  transform.RotateZ(euler.phi2)
  matrix = vtk.vtkMatrix4x4()
  matrix = transform.GetMatrix()
  for i in range(3):
    rot_normal[0] += hklplane.normal[0] * matrix.GetElement(0, i);
    rot_normal[1] += hklplane.normal[1] * matrix.GetElement(1, i);
    rot_normal[2] += hklplane.normal[2] * matrix.GetElement(2, i);
  rot_plane.SetNormal(rot_normal)
  return add_plane_to_grid(rot_plane, grid, origin)
    
def add_plane_to_grid(plane, grid, origin):
  '''
  vtk helper function to add a plane inside another object
  described by a mesh (vtkunstructuredgrid).
  The method is to use a vtkCutter with the mesh as input and the plane 
  as the cut function. An actor in returned.
  This may be used directly to add hkl planes inside a lattice cell.
  '''
  # cut the crystal with the plane
  planeCut = vtk.vtkCutter()
  planeCut.SetInput(grid)
  planeCut.SetCutFunction(plane)
  #print planeCut.GetOutput()

  cutMapper = vtk.vtkPolyDataMapper()
  cutMapper.SetInputConnection(planeCut.GetOutputPort())
  cutActor = vtk.vtkActor()
  cutActor.SetMapper(cutMapper)
  cutActor.GetProperty().SetOpacity(0.3)
  return cutActor
  
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
    
def add_HklPlane_with_orientation_in_grain(grain):
  # use a vtkAxesActor to display the crystal orientation
  local_orientation = vtk.vtkAssembly()
  grain_axes = axes_actor(length = 30, axisLabels = False)
  transform = vtk.vtkTransform()
  transform.Identity()
  transform.RotateZ(grain.orientation.phi1)
  transform.RotateX(grain.orientation.Phi)
  transform.RotateZ(grain.orientation.phi2)
  grain_axes.SetUserTransform(transform)
  local_orientation.AddPart(grain_axes)
  # add hkl plane
  hklplane = HklPlane(1, 1, 1)
  hklplaneActor = add_hklplane_to_grain(hklplane, grain.vtkmesh, \
    grain.orientation)
  local_orientation.AddPart(hklplaneActor)
  return local_orientation
  
def unit_arrow_3d(start, vector, color=orange):
  arrowSource = vtk.vtkArrowSource()
  # We build a local direct base with X being the unit arrow vector
  X = vector/numpy.linalg.norm(vector)
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
  for i in range(3):
    m.SetElement(i, 0, X[i]);
    m.SetElement(i, 1, Y[i]);
    m.SetElement(i, 2, Z[i]);
  #print m
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
  
def lattice_grid(lattice, origin=[0., 0., 0.]):
  [A, B, C] = lattice.matrix
  O = origin

  # create the eight points based on the lattice matrix
  Points = vtk.vtkPoints()
  Points.InsertNextPoint(O)
  Points.InsertNextPoint(O+A)
  Points.InsertNextPoint(O+A+B)
  Points.InsertNextPoint(O+B)
  Points.InsertNextPoint(O+C)
  Points.InsertNextPoint(O+C+A)
  Points.InsertNextPoint(O+C+A+B)
  Points.InsertNextPoint(O+C+B)

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
  grid.SetPoints(Points)
  return grid

def lattice_3d(grid, tubeRadius=0.02, sphereRadius=0.1):
  Edges = vtk.vtkExtractEdges()
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

  # Create a sphere to use as a glyph source for vtkGlyph3D.
  Sphere = vtk.vtkSphereSource()
  Sphere.SetRadius(sphereRadius)
  Sphere.SetPhiResolution(20)
  Sphere.SetThetaResolution(20)
  Vertices = vtk.vtkGlyph3D()
  Vertices.SetInput(grid)
  Vertices.SetSource(Sphere.GetOutput())
  # Create a mapper and actor to display the glyphs.
  SphereMapper = vtk.vtkPolyDataMapper()
  SphereMapper.SetInputConnection(Vertices.GetOutputPort())
  SphereMapper.ScalarVisibilityOff()
  Vertices = vtk.vtkActor()
  Vertices.SetMapper(SphereMapper)
  Vertices.GetProperty().SetDiffuseColor(blue)
  return Edges, Vertices

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
  matrix = vtk.vtkMatrix4x4()
  matrix = transform.GetMatrix()
  actor.SetUserTransform(transform)

def read_image_data(file_name, size, data_type='uint8'):
  '''
  vtk helper function to read a 3d data file.
  The size is needed in the form (x, y, z) as well a string describing
  the data type in numpy format (uint8 is assumed by default).
  Lower file left and little endian are assumed.
  '''
  reader = vtk.vtkImageReader2() # 2 is faster
  reader.SetDataScalarType(to_vtk_type(data_type))
  reader.SetFileDimensionality(3)
  reader.SetHeaderSize(0)
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
  outlineFilter.SetInput(data)
  outlineMapper = vtk.vtkPolyDataMapper()
  outlineMapper.SetInputConnection(outlineFilter.GetOutputPort())
  outline = vtk.vtkActor()
  outline.SetMapper(outlineMapper)
  outline.GetProperty().SetColor(color)
  return outline

def contourByDiscreteMarchingCubes(data, value, color='grey', diffuseColor='grey', opacity=1.0):
  contour = vtk.vtkDiscreteMarchingCubes()
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

def render(ren, ren_size=(600, 600), display=True, name='render_3d.png'):
  # Create a window for the renderer
  renWin = vtk.vtkRenderWindow()
  renWin.AddRenderer(ren)
  renWin.SetSize(ren_size)
  if display:
    # Start the initialization and rendering
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    renWin.Render()
    iren.Initialize()
    iren.Start()
  else:
    # capture the display and write a png image
    w2i = vtk.vtkWindowToImageFilter()
    writer = vtk.vtkPNGWriter()
    w2i.SetInput(renWin)
    w2i.Update()
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.SetFileName(name)
    renWin.Render()
    writer.Write()

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

