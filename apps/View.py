import os
import sys
import numpy as np
from pymicro.view.scene3d import Scene3D
from pymicro.view.vtk_utils import *
from pymicro.crystal.microstructure import Grain, Orientation
from pymicro.crystal.lattice import Lattice


class View():
    def __init__(self, args):
        '''Init a View window.'''
        print(args)
        # create the 3D scene
        s3d = Scene3D(display=True, ren_size=(800, 800))
        args_type = type(args)
        if isinstance(args, list):
            if len(args) == 1:
                print('Please specify the file representing the 3D object to view')
                sys.exit(1)
            elif len(args) == 2:
                file_path = args[1]
            else:
                print('Please use only one parameter (the path to the file representing the 3D object to view)')
                sys.exit(1)
            (path, ext) = os.path.splitext(file_path)
            ext = ext.strip('.')
            print(ext)
            if ext in ['stl', 'STL']:
                actor = load_STL_actor(path, ext)
            else:
                print('Unrecognized file extenstion: %s' % ext)
                sys.exit(1)
        elif isinstance(args, Grain):
            actor = grain_3d(args)
        elif isinstance(args, Orientation):
            l = Lattice.cubic(1.0)
            (a, b, c) = l._lengths
            grid = lattice_grid(l)
            actor = lattice_edges(grid)
            actor.SetOrigin(a / 2, b / 2, c / 2)
            actor.AddPosition(-a / 2, -b / 2, -c / 2)
            apply_orientation_to_actor(actor, args)
        elif isinstance(args, Lattice):
            (a, b, c) = args._lengths
            actor = lattice_3d(args)
            actor.SetOrigin(a / 2, b / 2, c / 2)
            actor.AddPosition(-a / 2, -b / 2, -c / 2)
        bounds = actor.GetBounds()
        size = (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])  # bounds[1::2]
        print(size)
        axes = axes_actor(np.mean(size), fontSize=60)
        s3d.add(axes)
        s3d.add(actor)
        cam = setup_camera(size)
        cam.SetFocalPoint(0.5 * (bounds[0] + bounds[1]), 0.5 * (bounds[2] + bounds[3]), 0.5 * (bounds[4] + bounds[5]))
        s3d.set_camera(cam)
        s3d.render()


if __name__ == "__main__":
    View(sys.argv)
