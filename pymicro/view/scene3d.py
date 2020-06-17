import vtk, sys, os
import numpy as np

class Scene3D:
    """A class to manage a 3D scene using VTK actors.

    Each instance of this class has its own `vtkRenderer`, it can be used
    to display the scene interactlively and/or save a still image in png
    format. The actual 3D rendering is done by calling the `render` method.
    """

    def __init__(self, display=True, ren_size=(600, 600), name=None, background=(1., 1., 1.)):
        """Initialization called when creating a new `Scene3D` object.

        :param display: a boolean to control if the scene has to be displayed
        interactively to the user (default True). If True, a frame counter
        is used when saving images using the 's' key pressed callback. If
        False a single image is save using the base name.

        :param ren_size: a tuple with two value to set the size of the image
        in pixels (default 600x600).

        :param name: a string to used to describe the scene, it is used in
        particular when saving the scene as an image. If not set, the file
        name of the Python script will be used or 'scene_3d' if run
        interactively.

        :param background: the background of the scene (white by default).
        """
        ren = vtk.vtkRenderer()
        ren.SetBackground(background)
        self.renderer = ren
        # Create a window for the renderer
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.renderer)
        self.renWin.SetSize(ren_size)

        self.display = display
        if name == None:
            if '__file__' in globals():
                self.name = os.path.splitext(__file__)[0]
            else:
                self.name = 'scene_3d'
        else:
            self.name = name
        self.frame_counter = 0
        self.verbose = True

    def add(self, actor):
        """Add a given actor to the 3D scene.

        *Parameters*

        **actor** a VTK actor to add to the renderer.
        """
        self.renderer.AddActor(actor)

    def get_renderer(self):
        """Get the vtk renderer attached to this 3d scene."""
        return self.renderer

    def set_camera(self, cam):
        """Set the camera for the 3D scene.

        *Parameters*

        **cam** a VTK camera to attach to the renderer.
        """
        self.renderer.SetActiveCamera(cam)

    def get_frame_as_array(self):
        """render the 3d scene and export it as a numpy array.

        This can be useful to display the image in a plot for instance. We use the `vtkBMPWriter` to generate the
        buffer as an array of bytes. The numpy array is then created from this buffer and reshaped to the appropriate
        RGB image size.

        :return: a numpy array representing the RGB rendered image.
        """
        self.renWin.SetOffScreenRendering(1)
        self.renWin.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.renWin)
        w2i.Update()
        bmp_shape = self.renWin.GetSize() + (3,)
        writer = vtk.vtkBMPWriter()
        writer.SetWriteToMemory(1)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        data = bytes(memoryview(writer.GetResult()))
        self.renWin.SetOffScreenRendering(0)
        # create the numpy array from the bytes buffer (leave off the first 54 bytes corresponding to the BMP header)
        array_bgr = np.frombuffer(data[54:], dtype=np.uint8).reshape(bmp_shape)
        b = array_bgr[::-1, :, 0].T
        g = array_bgr[::-1, :, 1].T
        r = array_bgr[::-1, :, 2].T
        return np.array([r, g, b]).transpose(2, 1, 0)

    def get_frame(self):
        """Generate a frame from the vtkRenderer instance of the 3d scene.
        
        :return: the image as a string buffer.
        """
        self.renWin.SetOffScreenRendering(1)
        self.renWin.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.renWin)
        w2i.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetWriteToMemory(1)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        try:
            data = str(buffer(writer.GetResult()))
        except NameError:
            data = bytes(memoryview(writer.GetResult()))
        self.renWin.SetOffScreenRendering(0)
        return data

    def save_frame(self):
        """Render the 3D scene and save a png image.

        When using the internal frame counter, it is incremented by 1 each
        time this method is called."""
        w2i = vtk.vtkWindowToImageFilter()
        writer = vtk.vtkPNGWriter()
        w2i.SetInput(self.renWin)
        w2i.Update()
        writer.SetInputConnection(w2i.GetOutputPort())
        if self.display:
            file_name = '%s_%04d.png' % (self.name, self.frame_counter)
        else:
            file_name = '%s.png' % self.name
        if self.verbose:
            print('writing still image ' + file_name)
        writer.SetFileName(file_name)
        self.renWin.Render()
        writer.Write()
        self.frame_counter += 1
        del writer, w2i

    def print_camera_settings(self):
        """Print out the active camera settings."""
        cam = self.renderer.GetActiveCamera()
        print('Camera settings:')
        print('  * position:        %s' % (cam.GetPosition(),))
        print('  * focal point:     %s' % (cam.GetFocalPoint(),))
        print('  * up vector:       %s' % (cam.GetViewUp(),))
        print('  * clipping range:  %s' % (cam.GetClippingRange(),))

    def pymicro_callback(self, obj, event):
        """Standard key pressed callback to attach to the 3d scene.

        This fuction can be used directly to be attached to the rendering
        window `vtkRenderWindowInteractor`. It handles user events by
        pressing keys:

         *s save a png image of the scene
         *c print the current camera settings
         *q exit the interactive rendering
        """
        key = obj.GetKeySym()
        if key == 's':
            self.save_frame()
        elif key == 'c':
            self.print_camera_settings()
        elif key == 'q':
            if self.verbose:
                print("Bye, thanks for using pymicro.")
            sys.exit(0)

    def render(self, key_pressed_callback=None):
        """Render the VTK scene in 3D.

        This function does the actual 3D rendering using the `vtkRenderer`
        of the object. It can be used to display the scene interactlively
        and/or save a still image in png format.

        *Parameters*

        **key_pressed_callback** a function (functions are first class variables)
        called in interactive mode when a key is pressed.
        """
        if self.display:
            # start the initialization and rendering
            iren = vtk.vtkRenderWindowInteractor()
            iren.SetRenderWindow(self.renWin)
            if key_pressed_callback:
                iren.AddObserver("KeyPressEvent", self.pymicro_callback)
            self.renWin.Render()
            iren.Initialize()
            iren.Start()
        else:
            self.save_frame()

    @staticmethod
    def from_experiment(experiment, show_lab_frame=True):
        """Create a 3D scene associated with an experimental setup.
        
        :param experiment: an instance of the `Experiment` class with all the setup parameters.
        :return: a `Scene3d` instance ready to display.
        """
        from pymicro.view.vtk_utils import box_3d, axes_actor, apply_translation_to_actor, detector_3d
        s3d = Scene3D()
        if show_lab_frame:
            # display the coordinate axes
            axes = axes_actor()
            s3d.add(axes)
        # display the sample
        bb = experiment.sample.geo.get_bounding_box()
        sample_bb = box_3d(origin=bb[0], size=bb[1])
        apply_translation_to_actor(sample_bb, experiment.sample.position)
        s3d.add(sample_bb)

        # display the detectors
        for i in range(experiment.get_number_of_detectors()):
            det = detector_3d(experiment.detectors[i], show_axes=True, see_reference=False)
            s3d.add(det)
        return s3d