import vtk
import os
import pickle
import numpy
from numpy import uint8
from pylab import *
from vtk_utils import ask_for_map_file
from vtk.util.colors import peacock, tomato

def cube_clip(data, params):
    gf = vtk.vtkContourFilter()
    gf.SetInput(data.GetOutput())
    gf.SetValue(0,100.)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(gf.GetOutputPort())
    normals.SetFeatureAngle(45)

    # implicit function to clip and close the surface
    box = vtk.vtkBox()
    box.SetBounds(data.GetDataExtent())
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

def map_isosurface(points, mapper, params):
    if params['map'] == None:
        s = params['volume_name'][0:len(params['volume_name'])-4] + '.raw'
        params['map'] = params['map_dir'] + ask_for_map_file(params['map_dir'], s)
    if not os.path.exists(params['map']):
        print 'map file',params['map'],'cannot be located'
        print 'aborting isosurface mapping'
        return
    f = file(params['map'])
    map = pickle.load(f)
    f.close()
    (NX, NY, NZ) = points.GetDimensions()
    max = numpy.max(map)
    print (NY, NX, NZ), 'min=', numpy.min(map), 'max=', max
    
    # build array
    array = vtk.vtkFloatArray()
    array.SetName("MAP")
    array.SetNumberOfTuples(NX*NY*NZ)
    array.SetNumberOfComponents(1)
    for z in range(0, NZ):
        for y in range(0, NY):
            for x in range(0, NX):
                id = z*NX*NY + y*NX + x
                array.SetValue(id, map[y*params['shrink'], x*params['shrink']])
                #array.SetValue(id, z) 
    
    points.GetPointData().AddArray(array)
    
    # crack mapper
    lut = vtk.vtkLookupTable()
    if params['hue_range'] != None:
        print params['hue_range']
        lut.SetHueRange(params['hue_range'][0], params['hue_range'][1])
    else:
        #lut.SetHueRange(0.0, 0.667)
        lut.SetHueRange(0.5, -0.5)
    mapper.ScalarVisibilityOn()
    if params['scalar_range'] != None:
        print params['scalar_range']
        mapper.SetScalarRange(params['scalar_range'][0], params['scalar_range'][1])
    else:
        mapper.SetScalarRange(numpy.min(map)/2, numpy.max(map)/2)
    #mapper.SetLookupTable(lut)
    mapper.SetScalarModeToUsePointFieldData()
    mapper.ColorByArrayComponent("MAP", 0)
    
def parse_labels(label_file):    
    f = open(label_file, 'r')
    # particle are ordered in the file starting with number 2.
    # we use a list where item 0 is for labels and item 1 is None.
    # 'nb'
    # 'xg'
    # 'yg'
    # 'zg'
    # 'volume'
    # 'surface'
    # 'sphericity'
    # 'I1'
    # 'I2'
    # 'I3'
    # 'vI1x'
    # 'vI1y'
    # 'vI1z'
    # 'vI2x'
    # 'vI2y'
    # 'vI2z'
    # 'vI3x'
    # 'vI3y'
    # 'vI3z'
    # 'a'
    # 'b'
    # 'c'
    # 'Fab'
    # 'Fac'
    # 'Fbc'
    # 'border'
    labels = []
    print 'parsing labels'
    # the first line contains fields
    line = f.readline()
    labels.append(line.split())
    labels.append(None)
    for line in f.readlines():
        labels.append(line.split(' '))
    print 'found',len(labels)-2,'particles'
    return labels
    
def map_with_labels(points, mapper, params):
    # load label volume
    if params['labels_name'] == None:
        params['labels_name'] = params['vtk_dir'] + params['volume_name'].replace('_bin', '_label')
        print params['labels_name']
        if not os.path.exists(params['labels_name']):
            print 'no labelled volume could be located, disabling label mapping...'
            return
    # parse label file
    labels = parse_labels(params['labels_name'].replace('.vtk', '.txt'))
    v = vtk.vtkStructuredPointsReader()
    v.SetFileName(params['labels_name'])
    v.Update()
    
    # chose mapping parameter (4 for volume)
    N = 20
    
    shrinkFactor = params['shrink']
    shrink = vtk.vtkImageShrink3D()
    shrink.SetShrinkFactors(shrinkFactor, shrinkFactor, shrinkFactor)
    shrink.SetInput(v.GetOutput())
    # averaging must be off to preserve labels
    shrink.AveragingOff()
    shrink.Update()

    (NX, NY, NZ) = points.GetDimensions()
    NT = NX * NY * NZ
    print (NY, NX, NZ)
    
    mapper.ScalarVisibilityOn()
    # figure out the scalar range
    m = int(float(labels[3][N]))
    for i in range(4, len(labels)):
        vali = int(float(labels[i][N]))
        if vali > m:
            m = vali
    print 'scalar range = 0 --',ceil(m/2)
    mapper.SetScalarRange(0, ceil(m/2))
    
    # build array
    array = vtk.vtkFloatArray()
    array.SetName("LABEL")
    array.SetNumberOfTuples(NT)
    array.SetNumberOfComponents(1)
    count=0
    while (count < NT):
        if count % (NT / 100) == 0:
            print count / (NT / 100),'% done'
        label = shrink.GetOutput().GetPointData().GetScalars().GetValue(count)
        if label > 1:
            array.SetValue(count, float(labels[label][N])) 
        count +=1
        
    points.GetPointData().AddArray(array)
    
    # crack mapper
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfColors(64)
    lut.SetHueRange(0.667, 0.0)
    mapper.SetLookupTable(lut)
    mapper.SetScalarModeToUsePointFieldData()
    mapper.ColorByArrayComponent("LABEL", 0)

def render_isosurface(data, params):
    iso = vtk.vtkContourFilter()
    iso.SetInputConnection(data.GetOutputPort())
    iso.SetValue(0, params['thr'])
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(iso.GetOutputPort())
    normals.SetFeatureAngle(45)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    if params['map_isosurface']:
        print 'isosurface mapping requested'
        map_isosurface(data.GetOutput(), mapper, params)
    elif params['map_with_labels']:
        print 'label mapping requested'
        map_with_labels(data.GetOutput(), mapper, params)

    crack = vtk.vtkActor()
    crack.SetMapper(mapper)
    return crack

def damage_rendering(**args):    
    # parameters
    params={ \
            'volume_name' : '', \
            'vtk_dir' : 'vtk/', \
            'raw_dir' : 'raw/', \
            'shrink' : 8, \
            'map_isosurface' : False, \
            'hue_range' : None, \
            'scalar_range' : None, \
            'thr' : 100, \
            'map' : None, \
            'map_dir' : 'map/', \
            'map_with_labels' : False, \
            'labels_name' : None, \
            'save_png' : False, \
            'save_png_file' : None, \
            'display' : True, \
            }

    # override defaults with user choices        
    keys = args.keys()
    for kw in keys:
        print kw, ':', args[kw]
        params[kw] = args[kw]

    # Create the renderer, the render window, and the interactor. The
    # renderer draws into the render window, the interactor enables mouse-
    # and keyboard-based interaction with the scene.
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    
    v = vtk.vtkStructuredPointsReader()
    v.SetFileName(params['vtk_dir'] + params['volume_name'])
    # actually read the file
    v.Update()
    
    # We downsample the data a bit to yield better performances
    shrinkFactor = params['shrink']
    shrink = vtk.vtkImageShrink3D()
    shrink.SetShrinkFactors(shrinkFactor, shrinkFactor, shrinkFactor)
    shrink.SetInput(v.GetOutput())
    if params['map_with_labels']:
        # averaging must be off to preserve label mapping
        shrink.AveragingOff()
    else:
        shrink.AveragingOn()
    shrink.Update()
    
    # extract the crack
    crack = render_isosurface(shrink, params)

#    # extract the porosity
#    params2 = params
#    params2['volume_name'] = 'tm2_porosity_scaled_bin_.vtk'
#    params2['map_with_labels'] = True
#    w = vtk.vtkStructuredPointsReader()
#    w.SetFileName(params2['vtk_dir'] + params2['volume_name'])
#    w.Update()
#    print w.GetOutput()
#    shrink2 = vtk.vtkImageShrink3D()
#    shrink2.SetShrinkFactors(shrinkFactor, shrinkFactor, shrinkFactor)
#    shrink2.SetInput(w.GetOutput())
#    if params2['map_with_labels']:
#        # averaging must be off to preserve label mapping
#        shrink2.AveragingOff()
#    else:
#        shrink2.AveragingOn()
#    shrink2.Update()
#    porosity = render_isosurface(shrink2, params2)
    
    # An outline provides context around the data.
    outlineData = vtk.vtkOutlineFilter()
    outlineData.SetInput(v.GetOutput())
    mapOutline = vtk.vtkPolyDataMapper()
    mapOutline.SetInputConnection(outlineData.GetOutputPort())
    outline = vtk.vtkActor()
    outline.SetMapper(mapOutline)
    outline.GetProperty().SetColor(0, 0, 0)
    
    # this seems to be a good orientation for b4 scans
    #cam.SetViewUp(0.1, 0.5, -1)
    #cam.SetPosition(-0.5, 1.0, -0.5)
    #cam.SetFocalPoint(-1, -1, 1)
    
#    ren.AddActor(porosity)
    ren.AddActor(outline)
    ren.AddActor(crack)
    #ren.SetActiveCamera(cam)
    ren.ResetCamera()
    #cam.Dolly(1.25)
    
    ren.SetBackground(1, 1, 1)
    renWin.SetSize(800, 600)
    
    if params['save_png']:
        renderLarge = vtk.vtkRenderLargeImage()
        renderLarge.SetInput(ren)
        renderLarge.SetMagnification(4)    
        writer = vtk.vtkPNGWriter()
        writer.SetInputConnection(renderLarge.GetOutputPort())
        if params['save_png_file'] == None:
            writer.SetFileName(params['vtk_dir'] + params['volume_name'][:-4] + '.png')
        else:
            writer.SetFileName(params['vtk_dir'] + params['save_png_file'])
        writer.Write()
    
    if params['display']:
        # Interact with the data.
        iren.Initialize()
        renWin.Render()
        iren.Start()

if __name__ == '__main__':
#    vdir = '/home/proudhon/esrf/thilo/tm4/vtk/'
#    rdir = '/home/proudhon/esrf/thilo/tm4/raw/'
#    volname = 'tm4_scaled_bin_.vtk'
    #volname = 'tm4_crop_label_.vtk'
    
#    vdir = '/home/proudhon/esrf/thilo/tm2/vtk/'
#    rdir = '/home/proudhon/esrf/thilo/tm2/raw/'
#    #volname = 'tm2_bin_turned_scaled_.vtk'
#    volname = 'tm2_full_scaled_adj_bin_.vtk'

#    vdir = '/home/proudhon/esrf/s3/vtk/'
#    rdir = '/home/proudhon/esrf/s3/raw/'
#    volname = 's3_853k_bin_.vtk'

    vdir = '/home/proudhon/esrf/thilo/init_2_TL_T8/vtk/'
    rdir = '/home/proudhon/esrf/thilo/init_2_TL_T8/raw/'
    volname = 'crack_vols7-8_bin_.vtk'

    damage_rendering(volume_name=volname, vtk_dir=vdir, raw_dir=rdir, shrink=4, \
             map_with_labels=False, map_isosurface=True, \
             scalar_range=[0, 1000], save_png=True, \
             #hue_range=[-0.5, 0.5], scalar_range=[100, 237], save_png=True, \
             #hue_range=[0.0, 0.667], scalar_range=[0, 1000], save_png=False, \
             #map='/home/proudhon/esrf/thilo/init_2_TL_T8/map/crack_vols7-8_bin_.cod')
             map='/home/proudhon/esrf/thilo/init_2_TL_T8/map/crack_vols7-8_bin_.pos')
