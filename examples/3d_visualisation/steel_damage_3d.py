import os, vtk, numpy as np
from vtk.util import numpy_support
from vtk.util.colors import *
from pymicro.file.file_utils import HST_read, HST_write, HST_info
from pymicro.view.vtk_utils import *

'''
Create a 3d scene showing a damaged tension steel sample.
The sample outline is made semi-transparent and cavities are shown
in blue. The axes are labeled (L,T,S) accordingly to the material
directions.
'''
print('reading volume...')
data_dir = '../data'
scan_name = 'steel_bin_431x431x246_uint8'
scan_path = os.path.join(data_dir, scan_name)
infos = HST_info(scan_path + '.raw.info')
volsize = np.array([infos['x_dim'], infos['y_dim'], infos['z_dim']])

print(volsize)
grid = read_image_data(scan_path + '.raw', volsize, header_size=0, data_type='uint8')

print('setting actors...')
damage = contourFilter(grid, 255, opacity=1.0, discrete=True, color=blue, diffuseColor=blue)
skin = contourFilter(grid, 155, opacity=0.05, discrete=True)
outline = data_outline(grid)

# Create renderer
ren = vtk.vtkRenderer()
ren.SetBackground(1.0, 1.0, 1.0)
ren.AddActor(outline)
ren.AddActor(skin)
ren.AddActor(damage)

print('setting up LTS axes')
axes = axes_actor(length=100)
axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(grey)
ax1 = 'L';
ax2 = 'S';
ax3 = 'T'
axes.SetXAxisLabelText(ax1)
axes.SetYAxisLabelText(ax2)
axes.SetZAxisLabelText(ax3)
ren.AddViewProp(axes);

print('generating views...')
cam = setup_camera(size=(volsize))
ren.SetActiveCamera(cam)
ren.AddViewProp(axes)
cam.SetFocalPoint(0.5 * volsize)
cam.SetPosition(500 + volsize)
cam.SetViewUp(0, 0, 1)

image_name = os.path.splitext(__file__)[0] + '.png'
print('writting %s' % image_name)
render(ren, ren_size=(800, 800), save=True, display=False, name=image_name)

from matplotlib import image

image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
