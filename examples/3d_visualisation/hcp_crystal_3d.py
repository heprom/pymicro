from math import sqrt
import numpy as np
from pymicro.view.scene3d import Scene3D
from pymicro.view.vtk_utils import *
from pymicro.crystal.lattice import HklPlane
from pymicro.crystal.microstructure import Orientation

'''
Create a 3d scene with a hexagonal crystal lattice.
The usual hexagonal prism is displayed with the unit cell highlighted.
'''
# Create the 3D scene
base_name = os.path.splitext(__file__)[0]
s3d = Scene3D(display=False, ren_size=(800, 800), name=base_name)

# hexagonal lattice
a = 0.295  # nm
c = 0.468  # nm
l = Lattice.hexagonal(a, c)
l._basis = [(0., 0., 0.), (2. / 3, 1. / 3, 1. / 2)]  # hexagonal compact basis

hcp1 = vtk.vtkAssembly()
grid = lattice_grid(l)
Vertices1 = lattice_vertices(grid, sphereRadius=0.1 * a)
Vertices1.GetProperty().SetColor(0, 0, 0)
# hcp1.AddPart(Edges1)
hcp1.AddPart(Vertices1)
hcp1.RotateZ(180)
offset = l.matrix[0] + l.matrix[1]
hcp1.SetOrigin(offset)
hcp1.AddPosition(-offset)
s3d.add(hcp1)

hcp2 = vtk.vtkAssembly()
grid = lattice_grid(l)
Vertices2 = lattice_vertices(grid, sphereRadius=0.1 * a)
Vertices2.GetProperty().SetColor(0, 0, 0)
# hcp2.AddPart(Edges2)
hcp2.AddPart(Vertices2)
hcp2.RotateZ(60)
offset = l.matrix[0]
hcp2.SetOrigin(-offset)
hcp2.AddPosition(offset)
s3d.add(hcp2)

hcp3 = vtk.vtkAssembly()
grid = lattice_grid(l)
Vertices3 = lattice_vertices(grid, sphereRadius=0.102 * a)
mapper = vtk.vtkDataSetMapper()
if vtk.vtkVersion().GetVTKMajorVersion() > 5:
    mapper.SetInputData(grid)
else:
    mapper.SetInput(grid)
ShadowedCell = vtk.vtkActor()
ShadowedCell.SetMapper(mapper)
ShadowedCell.GetProperty().SetOpacity(0.3)
# hcp3.AddPart(Edges3)
hcp3.AddPart(Vertices3)
hcp3.AddPart(ShadowedCell)
hcp3.RotateZ(-60)
offset = l.matrix[1]
hcp3.SetOrigin(-offset)
hcp3.AddPosition(offset)
s3d.add(hcp3)

grid = hexagonal_lattice_grid(l)
Edges = vtk.vtkExtractEdges()
if vtk.vtkVersion().GetVTKMajorVersion() > 5:
    Edges.SetInputData(grid)
else:
    Edges.SetInput(grid)
Tubes = vtk.vtkTubeFilter()
Tubes.SetInputConnection(Edges.GetOutputPort())
Tubes.SetRadius(0.02 * a)
Tubes.SetNumberOfSides(6)
Tubes.UseDefaultNormalOn()
Tubes.SetDefaultNormal(.577, .577, .577)
# Create the mapper and actor to display the cell edges.
TubeMapper = vtk.vtkPolyDataMapper()
TubeMapper.SetInputConnection(Tubes.GetOutputPort())
hcp_edges = vtk.vtkActor()
hcp_edges.SetMapper(TubeMapper)
offset = 1 * l.matrix[0] + 2 * l.matrix[1]
hcp_edges.SetOrigin(-offset)
hcp_edges.AddPosition(offset)
s3d.add(hcp_edges)

# set up camera and render
cam = setup_camera(size=(a, a, c))
center = 1 * (l.matrix[0] + l.matrix[1]) + 0.5 * l.matrix[2]
cam.SetFocalPoint(center)
cam.SetPosition(4 * a, -2 * a, 2.5 * a)
cam.Dolly(0.9)
s3d.set_camera(cam)
s3d.render()

# thumbnail for the image gallery
from matplotlib import image

image_name = base_name + '.png'
image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
