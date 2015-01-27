#!/usr/bin/env python
from pymicro.view.vtk_utils import *
from pymicro.file.file_utils import HST_info, HST_read
from vtk.util.colors import white, grey, black, lamp_black
import os, vtk
import numpy as np

if __name__ == '__main__':
  data_dir = '../data'
  scan_name = 'pa6_teg11_e_crop.raw'
  scan_path = os.path.join(data_dir, scan_name)
  data = HST_read(scan_path, data_type='uint8', verbose=True)
  size = data.shape
  vtk_data_array = numpy_support.numpy_to_vtk(np.ravel(data, order='F'), deep=1)

  size = np.shape(data)
  grid = vtk.vtkUniformGrid()
  grid.SetExtent(0, size[0], 0, size[1], 0, size[2])
  grid.SetSpacing(1, 1, 1)
  if vtk.vtkVersion().GetVTKMajorVersion() > 5:
    grid.SetScalarType(vtk.VTK_UNSIGNED_CHAR, vtk.vtkInformation())
  else:
    grid.SetScalarType(vtk.VTK_UNSIGNED_CHAR)
  grid.GetCellData().SetScalars(vtk_data_array)

  print 'creating renderer'
  ren = vtk.vtkRenderer()
  ren.SetBackground(1.0, 1.0, 1.0)

  print 'adding renderer'
  actor = map_data_with_clip(grid)
  actor.GetProperty().SetSpecular(.4)
  actor.GetProperty().SetSpecularPower(10)
  actor.GetProperty().SetOpacity(1.0)
  ren.AddActor(actor)

  print 'adding bounding box'
  outline = data_outline(grid)
  outline.GetProperty().SetColor(black)
  ren.AddActor(outline)

  print 'adding XYZ axes'
  axes = axes_actor(length = 100, axisLabels = True)
  axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(grey)
  ren.AddViewProp(axes);

  print 'setting up camera'
  cam = setup_camera(size=size)
  cam.SetClippingRange(1, 5000)
  ren.SetActiveCamera(cam)
  image_name = os.path.splitext(__file__)[0] + '.png'
  print 'writting %s' % image_name
  render(ren, save=True, display=False, ren_size=(800,800), name=image_name)

  from matplotlib import image
  image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
