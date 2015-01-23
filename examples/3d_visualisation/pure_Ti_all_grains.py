#!/usr/bin/env python
import os, vtk
from pymicro.file.file_utils import edf_read, HST_read
from pymicro.view.vtk_utils import show_grains, box_3d, axes_actor, setup_camera, render

if __name__ == '__main__':
  data_dir = '../data'
  scan_name = 'pure_Ti_216x216x141_uint16.raw'
  scan_path = os.path.join(data_dir, scan_name)
  data = HST_read(scan_path, autoparse_filename=True)
  size = data.shape
  print 'done reading, volume size is ', size

  # Create renderer
  ren = vtk.vtkRenderer()
  ren.SetBackground(1.0, 1.0, 1.0)

  # add all the grains
  from pymicro.view.vtk_utils import show_data
  grains = show_grains(data)
  ren.AddActor(grains)

  # add outline
  outline = box_3d(size=size, line_color=(0., 0., 0.))
  ren.AddActor(outline)

  # add axes actor
  axes = axes_actor(0.5*size[0])
  ren.AddViewProp(axes);

  cam = setup_camera(size=(size))
  cam.SetPosition(2.0*size[0], 0.0*size[1], 2.0*size[2])
  cam.Dolly(0.75)
  ren.SetActiveCamera(cam)
  image_name = os.path.splitext(__file__)[0] + '.png'
  print 'writting %s' % image_name
  render(ren, save=True, display=False, ren_size=(800,800), name=image_name)

  from matplotlib import image
  image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
