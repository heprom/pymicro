#!/usr/bin/env python
import os
import vtk
from vtk.util import numpy_support
from pymicro.crystal.microstructure import *
from pymicro.view.vtk_utils import *
from pymicro.view.scene3d import Scene3D

calc_name = 'CT10_PROPAG100'
calc_filename = os.path.join('..', 'data', '%s.vtu' % calc_name)
field = 'sig22'
field_min = 0.
field_max = 400.
fmt = '%.1f'
lut = jet_cmap(table_range=(field_min, field_max))

reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(calc_filename)
reader.Update()
calc = reader.GetOutput()  # the vtkUnstructuredGrid object
bounds = np.array(calc.GetBounds())
size = bounds[1::2] - bounds[0::2]

# we need to convert the data to vtkPolyData
poly_data = vtk.vtkGeometryFilter()
poly_data.SetInputData(calc)
poly_data.Update()
poly_data.GetOutput().GetPointData().SetActiveVectors('U')

# deform the mesh using the displacement field
warp = vtk.vtkWarpVector()
warp.SetInputConnection(poly_data.GetOutputPort())
#warp.SetInputArrayToProcess(1, 0, 0, 0, 'U')
warp.SetScaleFactor(5)
warp.Update()
warp.GetOutput().GetCellData().SetActiveScalars(field)  # must be after call to Update

probe_mapper = vtk.vtkPolyDataMapper()
probe_mapper.SetInputConnection(warp.GetOutputPort())
probe_mapper.SetLookupTable(lut)
#probe_mapper.UseLookupTableScalarRangeOn()
probe_mapper.SetScalarModeToUseCellData()
probe_mapper.ScalarVisibilityOn()
probe_mapper.SetScalarRange(field_min, field_max)
probeActor = vtk.vtkActor()
probeActor.SetMapper(probe_mapper)

# color bar
bar = color_bar(field, lut, fmt, num_labels=5, font_size=20)

# create axes actor
axes = axes_actor(5.0, fontSize=60)
apply_translation_to_actor(axes, (-15, 0, 0))

# create the 3D scene
base_name = os.path.splitext(__file__)[0]
s3d = Scene3D(display=False)
s3d.add(probeActor)
s3d.add(axes)
s3d.add(bar)
s3d.name = base_name
s3d.renWin.SetSize(800, 800)

cam = setup_camera(size=size)
cam.SetFocalPoint(0., -3., 0.)
cam.SetViewUp(0., 1., 0.)
cam.SetPosition([-45., 20., 55.])

s3d.set_camera(cam)
s3d.render()

# thumbnail for the image gallery
from matplotlib import image

image_name = base_name + '.png'
image.thumbnail(image_name, 'thumb_' + image_name, 0.2)

