import os
import sys
import numpy as np
import vtk

from pymicro.view.scene3d import Scene3D
from pymicro.view.vtk_utils import *
from pymicro.crystal.microstructure import Grain, Orientation
from pymicro.crystal.lattice import Lattice


class View:
    """Quick visualisation of various instances at use in pymicro.

    This class provides a way to quickly visualise an instance of a particular class.
    A 3D scene is then created with default settings and the actor added to the scene.
    The following types are supported:

     * Grain: the `grain_3d` method is called to create the actor;
     * Orientation: A cube is created and rotated according to the passive orientation matrix;
     * Lattice: the `lattice_3d` method is called to create the actor;
     * numpy array: the `show_array` method is called to create the actor;
     * vtk actor: the actor is directly added to the scene.

    The key_pressed_callback is activated so it is possible to save an image using the 's' key.
    """

    def __init__(self, args):
        '''Init a new View window.'''
        #print(args)
        # create the 3D scene
        s3d = Scene3D(display=True, ren_size=(800, 800))
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
        elif isinstance(args, np.ndarray):
            actor = show_array(args)
        elif isinstance(args, vtk.vtkActor):
            actor = args
        else:
            raise ValueError('unsupported object type: {0}'.format(type(args)))
        bounds = actor.GetBounds()
        size = (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])  # bounds[1::2]
        print(size)
        axes = axes_actor(length=np.mean(size), fontSize=60)
        s3d.add(axes)
        s3d.add(actor)
        cam = setup_camera(size)
        cam.SetFocalPoint(0.5 * (bounds[0] + bounds[1]), 0.5 * (bounds[2] + bounds[3]), 0.5 * (bounds[4] + bounds[5]))
        s3d.set_camera(cam)
        s3d.render(key_pressed_callback=True)


if __name__ == "__main__":
    View(sys.argv)
