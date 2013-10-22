import vtk
import os

# parameters
params={ \
        'volume_name' : 'grains_b.raw', \
        'vtk_dir' : '/home/proudhon/esrf/gt/vtk/', \
        'raw_dir' : '/home/proudhon/esrf/gt/raw/', \
        'shrink' : 4, \
        'grain_id_start' : 53, \
        'grain_id_end' : 53, \
        'decimation' : 0.5, \
}

reader = vtk.vtkImageReader()
reader.SetDataScalarType(vtk.VTK_UNSIGNED_CHAR)
reader.SetFileDimensionality(3)
reader.SetDataExtent (0, 301, 0, 301, 0, 445)
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

#thr = 58
r = range(params['grain_id_start'], params['grain_id_end']+1)
for thr in r:
    print 'meshing grain ' + str(thr)
    thres = vtk.vtkImageThreshold()
    thres.SetInput(reader.GetOutput())
    #thres.SetInputConnection(shrink.GetOutputPort())
    thres.ThresholdBetween(thr, thr)
    thres.SetInValue(255)
    thres.SetOutValue(0)
    
    mcubes = vtk.vtkMarchingCubes()
    mcubes.SetInput(thres.GetOutput())
    mcubes.ComputeNormalsOff()
    mcubes.ComputeGradientsOff()
    mcubes.ComputeScalarsOff()
    mcubes.SetValue(0,128)
    
    # decimation
    decimator = vtk.vtkDecimatePro()
    decimator.SetInputConnection(mcubes.GetOutputPort())
    decimator.SetFeatureAngle(45)
    decimator.PreserveTopologyOn()
    decimator.SetTargetReduction(params['decimation'])
    
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(decimator.GetOutputPort())
    smoother.SetNumberOfIterations(1)
    #smoother.setRelaxationFactor(0.1)
    smoother.SetFeatureAngle(45)
    smoother.BoundarySmoothingOn()
    
    # compute normals for smooth rendering
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(mcubes.GetOutputPort())
    normals.SetFeatureAngle(45)
    
    # write out triangles (do not use triangle stripper!)
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputConnection(decimator.GetOutputPort())
    writer.SetFileTypeToASCII()
    writer.SetFileName(params['vtk_dir'] + 'mesh_grain_' + str(thr) + '.vtk')
    writer.Update()
    
    # create the vtk->geof input file
    fname = params['vtk_dir'] + 'mesh_grain_' + str(thr) + '.inp'
    print 'writing file ' + fname
    f = open(fname, 'w')
    f.write('****mesher\n')
    f.write(' ***mesh mesh_grain_' + str(thr) + '\n')
    f.write(' **import_vtkpolydata\n')
    f.write('  *vtk_file_name mesh_grain_' + str(thr) + '.vtk\n')
    f.write('****return\n')
    f.close()
    
    # now mesh the grain
    print 'meshing grain ' + str(thr)
    #cmd = 'ls ' + params['vtk_dir'] + ';Zrun -m ' + fname
    cmd = 'cd ' + params['vtk_dir'] + ';Zrun -m mesh_grain_' + str(thr) + '.inp'
    print 'cmd = ' + cmd
    os.system(cmd)

# now write the union file
fname = params['vtk_dir'] + 'union.inp'
print 'writing file ' + fname
f = open(fname, 'w')
f.write('****mesher\n')
f.write(' ***mesh grains\n')
for thr in r:
    f.write('  **union\n')
    f.write('   *add mesh_grain_' + str(thr) + '\n')
    f.write('   *elset grain_' + str(thr) + '\n')
f.write('****return\n')
f.close()

## Take the isosurface data and create geometry
#mapper = vtk.vtkPolyDataMapper()
##mapper.SetInput(mcubes.GetOutput())
#mapper.SetInput(normals.GetOutput())
#mapper.ScalarVisibilityOff()
#
## Take the isosurface data and create geometry
#actor = vtk.vtkLODActor()
#actor.SetNumberOfCloudPoints(1000000)
#actor.SetMapper(mapper)
#actor.GetProperty().SetColor(1, 1, 1)
#
## Create renderer
#ren = vtk.vtkRenderer()
#ren.SetBackground(0.329412, 0.34902, 0.427451) #Paraview blue
#ren.AddActor(actor)
#
## Create a window for the renderer of size 250x250
#renWin = vtk.vtkRenderWindow()
#renWin.AddRenderer(ren)
#renWin.SetSize(250, 250)
#
## Set an user interface interactor for the render window
#iren = vtk.vtkRenderWindowInteractor()
#iren.SetRenderWindow(renWin)
#
## Start the initialization and rendering
#iren.Initialize()
#renWin.Render()
#iren.Start()

print 'done'
