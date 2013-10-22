import vtk

# parameters
params={ \
        'volume_name' : 'grains_c.raw', \
        'vtk_dir' : '/home/proudhon/esrf/gt/vtk/', \
        'raw_dir' : '/home/proudhon/esrf/gt/raw/', \
        'shrink' : 8, \
}

reader = vtk.vtkImageReader()
reader.SetDataScalarType(vtk.VTK_UNSIGNED_CHAR)
reader.SetFileDimensionality(3)
reader.SetDataExtent (0, 59, 0, 59, 0, 98)
reader.SetDataSpacing(1, 1, 1)
reader.SetNumberOfScalarComponents(1)
reader.SetDataByteOrderToBigEndian()
reader.SetFileName(params['raw_dir'] + params['volume_name'])

# downsample the data to yield better performances
shrinkFactor = params['shrink']
shrink = vtk.vtkImageShrink3D()
shrink.SetShrinkFactors(shrinkFactor, shrinkFactor, shrinkFactor)
shrink.SetInput(reader.GetOutput())
shrink.AveragingOff()

connect = vtk.vtkConnectivityFilter()
connect.SetInput(shrink.GetOutput())
connect.ColorRegionsOn()
connect.ScalarConnectivityOn()
connect.SetScalarRange(7., 8.)
connect.SetExtractionModeToAllRegions()
#connect.SetExtractionModeToSpecifiedRegions()
#connect.AddSpecifiedRegion(0)
#connect.AddSpecifiedRegion(10)
#connect.AddSpecifiedRegion(20)
#connect.AddSpecifiedRegion(30)
#connect.AddSpecifiedRegion(50)
#connect.AddSpecifiedRegion(130)
#connect.AddSpecifiedRegion(150)
#connect.AddSpecifiedRegion(170)
#connect.AddSpecifiedRegion(200)
#connect.SetExtractionModeToLargestRegion()
connect.Update()
print 'number of regions: ' + str(connect.GetNumberOfExtractedRegions())
print 'scalar range: ' + str(connect.GetScalarRange())

gf = vtk.vtkGeometryFilter()
gf.SetInputConnection(connect.GetOutputPort())

mapper = vtk.vtkPolyDataMapper()
mapper.SetInput(gf.GetOutput())

# Create the vtk actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)
#actor.GetProperty().SetColor(1, 1, 1)
actor.GetProperty().SetRepresentationToWireframe()

# Create renderer
ren = vtk.vtkRenderer()
ren.SetBackground(0.329412, 0.34902, 0.427451) #Paraview blue
ren.AddActor(actor)

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
