import vtk
from vtk.util.colors import peacock, tomato

# parameters
params={ \
        'volume_name' : 'PA_20x20x100.raw', \
        #'volume_name' : 'PA_100x100x100.raw', \
        #'volume_name' : 'PA_0001crop.raw', \
        'vtk_dir' : '/home/proudhon/esrf/PA/vtk/', \
        'raw_dir' : '/home/proudhon/esrf/PA/raw/', \
        'shrink' : 2, \
}

reader = vtk.vtkImageReader()
reader.SetDataScalarType(vtk.VTK_UNSIGNED_CHAR)
reader.SetFileDimensionality(3)
reader.SetDataExtent (0, 19, 0, 19, 0, 99)
#reader.SetDataExtent (0, 99, 0, 99, 0, 99)
reader.SetDataSpacing(1, 1, 1)
reader.SetNumberOfScalarComponents(1)
reader.SetDataByteOrderToBigEndian()
reader.SetFileName(params['raw_dir'] + params['volume_name'])

# downsample the data to yield better performances
#shrinkFactor = params['shrink']
#shrink = vtk.vtkImageShrink3D()
#shrink.SetShrinkFactors(shrinkFactor, shrinkFactor, shrinkFactor)
#shrink.SetInput(reader.GetOutput())
#shrink.AveragingOff()

gf = vtk.vtkContourFilter()
#gf.SetInput(shrink.GetOutput())
gf.SetInput(reader.GetOutput())
gf.SetValue(0,100.)
normals = vtk.vtkPolyDataNormals()
normals.SetInputConnection(gf.GetOutputPort())
normals.SetFeatureAngle(45)


# implicit function to clip and close the surface
box = vtk.vtkBox()
box.SetBounds(0., 19., 0., 19., 10., 90.)
#box.SetBounds(10., 90., 10., 90., 10., 90.)
plane = vtk.vtkPlane()
plane.SetOrigin(50., 50., 90.)
plane.SetNormal(0, 0, -1)

clipper = vtk.vtkClipPolyData()
clipper.SetInputConnection(normals.GetOutputPort())
clipper.SetClipFunction(box)
#clipper.GenerateClippedOutputOn()
clipper.InsideOutOn()
clipMapper = vtk.vtkPolyDataMapper()
clipMapper.SetInputConnection(clipper.GetOutputPort())
clipMapper.ScalarVisibilityOff()
backProp = vtk.vtkProperty()
backProp.SetDiffuseColor(peacock)
clipActor = vtk.vtkActor()
clipActor.SetMapper(clipMapper)
clipActor.GetProperty().SetColor(tomato)
clipActor.SetBackfaceProperty(backProp)

cutEdges = vtk.vtkCutter()
cutEdges.SetInputConnection(normals.GetOutputPort())
cutEdges.SetCutFunction(box)
#cutEdges.GenerateCutScalarsOn()
#cutEdges.SetValue(0, 0.5)
cutStrips = vtk.vtkStripper()
cutStrips.SetInputConnection(cutEdges.GetOutputPort())
cutStrips.Update()
cutPoly = vtk.vtkPolyData()
cutPoly.SetPoints(cutStrips.GetOutput().GetPoints())
cutPoly.SetPolys(cutStrips.GetOutput().GetLines())

# Triangle filter is robust enough to ignore the duplicate point at
# the beginning and end of the polygons and triangulate them.
cutTriangles = vtk.vtkTriangleFilter()
cutTriangles.SetInput(cutPoly)
cutMapper = vtk.vtkPolyDataMapper()
cutMapper.SetInput(cutPoly)
cutMapper.SetInputConnection(cutTriangles.GetOutputPort())
cutActor = vtk.vtkActor()
cutActor.SetMapper(cutMapper)
cutActor.GetProperty().SetColor(peacock)

# write out triangles (do not use triangle stripper!)
writer = vtk.vtkPolyDataWriter()
writer.SetInputConnection(gf.GetOutputPort())
writer.SetFileTypeToASCII()
writer.SetFileName(params['vtk_dir'] + 'mesh_holes.vtk')
writer.Update()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInput(gf.GetOutput())

# Create the vtk actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(1, 1, 1)
#actor.GetProperty().SetRepresentationToWireframe()

# Create renderer
ren = vtk.vtkRenderer()
ren.SetBackground(0.329412, 0.34902, 0.427451) #Paraview blue
#ren.AddActor(actor)
ren.AddActor(clipActor)
ren.AddActor(cutActor)

# Create a window for the renderer of size 250x250
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(250, 250)

# Set an user interface interactor for the render window
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Start the initialization and rendering
iren.Initialize()
renWin.Render()
iren.Start()

print 'done'
