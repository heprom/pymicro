from pymicro.view.scene3d import Scene3D
from pymicro.view.vtk_utils import *
from pymicro.crystal.microstructure import Grain, Orientation
from pymicro.crystal.lattice import Lattice


class View:
    """Quick visualisation of various instances at use in pymicro.

    This class provides a way to quickly visualise an instance of a particular class.
    A 3D scene is then created with default settings and the actor added to the scene.
    The following types are supported:

     * str: the `load_STL_actor` method is called to create the actor;
     * Grain: the `grain_3d` method is called to create the actor;
     * Orientation: A cube is created and rotated according to the passive orientation matrix;
     * Lattice: the `lattice_3d` method is called to create the actor;
     * numpy array: the `show_array` method is called to create the actor;
     * vtk actor: the actor is directly added to the scene.

    The key_pressed_callback is activated so it is possible to save an image using the 's' key.
    """

    def __init__(self, arg):
        """Init a new View window.

        :param arg: a descriptor of the object to view, it can be an instance of `Grain`, `Orientation`, `Lattice`,
        a vtkActor, a 3D numpy array or the path to a STL file.
        """
        # create the 3D scene
        s3d = Scene3D(display=True, ren_size=(800, 800))
        if isinstance(arg, str):
            (path, ext) = os.path.splitext(arg)
            ext = ext.strip('.')
            print(ext)
            if ext in ['stl', 'STL']:
                actor = load_STL_actor(path, ext)
            else:
                print('Unrecognized file extension: %s' % ext)
                sys.exit(1)
        elif isinstance(arg, Grain):
            actor = grain_3d(arg)
        elif isinstance(arg, Orientation):
            l = Lattice.cubic(1.0)
            (a, b, c) = l._lengths
            grid = lattice_grid(l)
            actor = lattice_edges(grid)
            actor.SetOrigin(a / 2, b / 2, c / 2)
            actor.AddPosition(-a / 2, -b / 2, -c / 2)
            apply_orientation_to_actor(actor, arg)
        elif isinstance(arg, Lattice):
            (a, b, c) = arg._lengths
            actor = lattice_3d(arg)
            actor.SetOrigin(a / 2, b / 2, c / 2)
            actor.AddPosition(-a / 2, -b / 2, -c / 2)
        elif isinstance(arg, np.ndarray):
            if arg.ndim != 3:
                print('Only 3D arrays can be viewed with this method.')
                sys.exit(1)
            actor = show_array(arg)
        elif isinstance(arg, vtk.vtkActor):
            actor = arg
        else:
            raise ValueError('unsupported object type: {0}'.format(type(arg)))
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
    if len(sys.argv) > 2:
        print('Please use only one parameter (the path to the STL file representing the 3D object to view)')
        sys.exit(1)
    View(sys.argv[1])
