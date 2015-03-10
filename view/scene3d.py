import vtk, sys

class Scene3D:
  '''A class to manage a 3D scene using VTK actors.

  Each instance of this class has its own `vtkRenderer`, it can be used 
  to display the scene interactlively and/or save a still image in png 
  format. The actual 3D rendering is done by calling the `render` method.
  '''

  def __init__(self, ren_size=(600, 600), display=True, save=False, name='scene_3d', use_frame_counter=False):
    '''Initialization called when creating a new `Scene3D` object.

    *Parameters*
    
    **ren_size**: a tuple with two value to set the size of the image in 
    pixels (defalut 600x600).
    
    **display**: a boolean to control if the scene has to be displayed 
    interactively to the user (default True).
    
    **save**: a boolean to to control if the scene has to be saved as a 
    png image (default False).
    
    **name**: a string to used to describe the scene, it is used in 
    particular when saving the scene as an image (default is 'scene_3d').
    '''
    ren = vtk.vtkRenderer()
    ren.SetBackground(1., 1., 1.)
    self.renderer = ren
    self.display = display
    self.save = save
    self.ren_size = ren_size
    self.name = name
    self.use_frame_counter = use_frame_counter
    self.frame_counter = 0
    self.verbose = True

  def add(self, actor):
    '''Add a given actor to the 3D scene.

    *Parameters*
    
    **actor** a VTK actor to add to the renderer.
    '''
    self.renderer.AddActor(actor)

  def set_camera(self, cam):
    '''Set the camera for the 3D scene.

    *Parameters*
    
    **cam** a VTK camera to attach to the renderer.
    '''
    self.renderer.SetActiveCamera(cam)

  def save_frame(self):
    '''Render the 3D scene and save a png image.
    
    When using the internal frame counter, it is incremented by 1 each
    time this method is called.'''
    # Create a window for the renderer
    self.renWin = vtk.vtkRenderWindow()
    self.renWin.AddRenderer(self.renderer)
    self.renWin.SetSize(self.ren_size)
    w2i = vtk.vtkWindowToImageFilter()
    writer = vtk.vtkPNGWriter()
    w2i.SetInput(self.renWin)
    w2i.Update()
    writer.SetInputConnection(w2i.GetOutputPort())
    if self.use_frame_counter:
      file_name = '%s_%04d.png' % (self.name, self.frame_counter)
    else:
      file_name = '%s.png' % self.name
    if self.verbose:
      print 'writing still image ' + file_name
    writer.SetFileName(file_name)
    self.renWin.Render()
    writer.Write()
    self.frame_counter += 1

  def print_camera_settings(self):
    '''Print out the active camera settings.'''
    cam = self.renderer.GetActiveCamera()
    print 'Camera settings:'
    print '  * position:        %s' % (cam.GetPosition(),)
    print '  * focal point:     %s' % (cam.GetFocalPoint(),)
    print '  * up vector:       %s' % (cam.GetViewUp(),)
    print '  * clipping range:  %s' % (cam.GetViewUp(),)

  def pymicro_callback(self, obj, event):
    '''Standard key pressed callback to attach to the 3d scene.
    
    This fuction can be used directly to be attached to the rendering
    window `vtkRenderWindowInteractor`. It handles user events by 
    pressing keys:
    
     *s save a png image of the scene
     *c print the current camera settings
     *q exit the interactive rendering
    '''
    key = obj.GetKeySym()
    if key == 's':
      self.save_frame()
    elif key == 'c':
      self.print_camera_settings()
    elif key == 'q':
      if self.verbose:
        print "Bye, thanks for using pymicro."
      sys.exit()

  def render(self, key_pressed_callback=None):
    '''Render the VTK scene in 3D.
    
    This function does the actual 3D rendering using the `vtkRenderer` 
    of the object. It can be used to display the scene interactlively 
    and/or save a still image in png format.
    
    *Parameters*
    
    **key_pressed_callback** a function (functions are first class variables)
    called in interactive mode when a key is pressed.
    '''
    if self.save:
      self.save_frame()
    if self.display:
      # Create a window for the renderer
      self.renWin = vtk.vtkRenderWindow()
      self.renWin.AddRenderer(self.renderer)
      self.renWin.SetSize(self.ren_size)
      # Start the initialization and rendering
      iren = vtk.vtkRenderWindowInteractor()
      iren.SetRenderWindow(self.renWin)
      if key_pressed_callback:
        iren.AddObserver("KeyPressEvent", key_pressed_callback)
      self.renWin.Render()
      iren.Initialize()
      iren.Start()

