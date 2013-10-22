import vtk
import os

# parameters
params={ \
        'volume_name' : 'grains_c.raw', \
        'vtk_dir' : '/home/proudhon/esrf/gt/vtk/', \
        'raw_dir' : '/home/proudhon/esrf/gt/raw/', \
        'shrink' : 2, \
        'grain_id_start' : 1, \
        'grain_id_end' : 79, \
}

reader = vtk.vtkImageReader()
reader.SetDataScalarType(vtk.VTK_UNSIGNED_CHAR)
reader.SetFileDimensionality(3)
#reader.SetDataExtent (0, 301, 0, 301, 0, 98)
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

# open up the union file for writting
fname = params['vtk_dir'] + 'grains/union.inp'
uf = open(fname, 'w')
uf.write('****mesher\n')
uf.write(' ***mesh grains\n')

r = range(params['grain_id_start'], params['grain_id_end']+1)
r = [7, 8, 17, 35, 74]
for thr in r:
    print 'contouring grain ' + str(thr)
    filter = vtk.vtkDiscreteMarchingCubes()
    filter.SetInput(shrink.GetOutput())
    filter.ComputeNormalsOff()
    filter.ComputeGradientsOff()
    filter.ComputeScalarsOff()
    filter.SetValue(0,thr)

    #dataset = vtk.vtkImplicitDataSet()
    #dataset.SetDataSet(shrink.GetOutput())

    #window = vtk.vtkImplicitWindowFunction()
    #window.SetImplicitFunction(dataset)
    #window.SetImplicitFunction(shrink.GetOutput())
    #window.SetWindowRange(thr-0.5, thr+0.5)
    
    #clip = vtk.vtkClipVolume()
    #clip.SetInput(shrink.GetOutput())
    #clip.SetClipFunction(window)
    #clip.SetValue(0.0)
    #clip.GenerateClippedOutputOff()
    #clip.Mixed3DCellGenerationOff()

    #filter = vtk.vtkGeometryFilter()
    #filter.SetInputConnection(clip.GetOutputPort())
    
    # write out triangles (do not use triangle stripper!)
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputConnection(filter.GetOutputPort())
    writer.SetFileTypeToASCII()
    writer.SetFileName(params['vtk_dir'] + 'grains/mesh_grain_' + str(thr) + '.vtk')
    writer.Update()
    
    # create the vtk->geof input file
    fname = params['vtk_dir'] + 'grains/mesh_grain_' + str(thr) + '.inp'
    print 'writing file ' + fname
    f = open(fname, 'w')
    f.write('****mesher\n')
    f.write(' ***mesh mesh_grain_' + str(thr) + '\n')
    f.write(' **import_vtkpolydata\n')
    f.write('  *vtk_file_name mesh_grain_' + str(thr) + '.vtk\n')
    f.write('****return\n')
    f.close()
    
    # now create the geof for this grain
    print 'meshing grain ' + str(thr)
    #cmd = 'ls ' + params['vtk_dir'] + ';Zrun -m ' + fname
    cmd = 'cd ' + params['vtk_dir'] + 'grains/;Zrun -m mesh_grain_' + str(thr) + '.inp'
    print 'cmd = ' + cmd
    os.system(cmd)

    if os.path.exists(params['vtk_dir'] + 'grains/mesh_grain_' + str(thr) + '.geof'):

        # create the yams_ghs3d input file
        fname = params['vtk_dir'] + 'grains/mesh_inside_grain_' + str(thr) + '.inp'
        print 'writing file ' + fname
        f = open(fname, 'w')
        f.write('****mesher\n')
        f.write(' ***mesh mesh_inside_grain_' + str(thr) + '\n')
        f.write(' **open mesh_grain_' + str(thr) + '.geof\n')
        f.write(' **yams_ghs3d\n')
        f.write('  *optim_style 0\n')
        #f.write('  *absolu\n')
        #f.write('  *min_size ' + str(params['shrink']*2.) + '\n')
        f.write('****return\n')
        f.close()

        # contribute to the union file
        uf.write('  **union\n')
        uf.write('   *add mesh_grain_' + str(thr) + '\n')
        uf.write('   *elset grain_' + str(thr) + '\n')

    else:
        print 'label ' + str(thr) + ' does not seem to be present...'
    
# now close the union file
uf.write('****return\n')
uf.close()

print 'done'
