'''The vtk_anim module define a set of classes to generate 3d 
   animations with vtk in the form of a series of png images.
'''
import vtk
import os, numpy as np

class vtkAnimationScene:

  def __init__(self, ren, ren_size=(600, 600)):
    self.timer_count = 0
    self.timer_incr = 1
    self.timer_end = -1 # run until 'q' is pressed
    self.save_image = False
    self.prefix = 'prefix'
    self.verbose = False
    self.anims = []
    # Create a window for the renderer
    self.renWin = vtk.vtkRenderWindow()
    self.renWin.AddRenderer(ren)
    self.renWin.SetSize(ren_size)
    # Start the initialization and rendering
    self.iren = vtk.vtkRenderWindowInteractor()
    self.iren.SetRenderWindow(self.renWin)
    self.renWin.Render()
    self.iren.Initialize()

  def add_animation(self, anim):
    anim.scene = self
    anim.save_image = self.save_image
    anim.prefix = self.prefix
    anim.verbose = self.verbose
    self.anims.append(anim)
    self.iren.AddObserver('TimerEvent', anim.execute)
  
  def write_image(self):
    # capture the display and write a png image
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(self.iren.GetRenderWindow())
    # the next two liines fix some opacity problems but slow things down...
    #self.renWin.Render()
    #self.iren.Render()
    writer = vtk.vtkPNGWriter()
    writer.SetInputConnection(w2i.GetOutputPort())
    file_name = os.path.join(self.prefix, '%s_%03d.png' % (self.prefix, self.timer_count))
    writer.SetFileName(file_name)
    writer.Write()

  def execute(self, iren, event):
    self.timer_count += self.timer_incr
    if self.verbose:
      print 'animation scene timer_count=', self.timer_count
    if self.timer_end > 0 and self.timer_count > self.timer_end:
      print 'end of animation loop, exiting...'
      self.iren.ExitCallback()
    else:
      self.iren.Render()
    if self.save_image:
      self.write_image()

  def render(self):
    if self.save_image and not os.path.exists(self.prefix):
      os.mkdir(self.prefix) # create a folder to store the images
    timerId = self.iren.CreateRepeatingTimer(100); # time in ms
    self.iren.AddObserver('TimerEvent', self.execute)
    self.iren.Start()
    
'''
Abstract class for all vtk animation stuff.
'''
class vtkAnimation:
  
  def __init__(self, t):
    self.scene = None
    self.time_anim_starts = t
    self.time_anim_ends = t + 10
    self.verbose = False
 
  def pre_execute(self):
    if self.verbose:
      print self.__repr__()
    if self.scene.timer_count < self.time_anim_starts or self.scene.timer_count > self.time_anim_ends:
      return 0
    else:
      return 1

  def post_execute(self, iren, event):
    pass

  def __repr__(self):
    out = [self.__class__.__name__,
      ' timer: ' + str(self.scene.timer_count),
      ' anim starts at: ' + str(self.time_anim_starts),
      ' anim ends at: ' + str(self.time_anim_ends)]
    return '\n'.join(out)

class vtkAnimCameraAroundZ(vtkAnimation):
  '''
  Animate the camera around the vertical axis.
  
  This class can be used to generate a series of images (default 36)
  while the camera rotate around the vertical axis (defined by the 
  camera SetViewUp method)?
  '''
  
  def __init__(self, t, cam, turn=360):
    '''Initialize the animation.
    
    The animation perform a full turn in 36 frames by default.
    '''
    print 'init vtkAnimCameraAroundZ'
    vtkAnimation.__init__(self, t)
    self.turn = turn
    self.time_anim_ends = t + abs(self.turn)/10
    print 'time_anim_starts', self.time_anim_starts
    print 'time_anim_ends', self.time_anim_ends
    print 'turn', self.turn
    self.camera = cam
 
  def execute(self, iren, event):
    '''Execute method called to rotate the camera.'''
    do = vtkAnimation.pre_execute(self)
    if not do: return
    t1 = self.time_anim_starts
    t2 = self.time_anim_ends
    r = self.turn / (t2 - t1)
    if self.scene.verbose: print 'moving azimuth by', r
    self.camera.Azimuth(r)
    vtkAnimation.post_execute(self, iren, event)
    
class vtkRotateActorAroundZAxis(vtkAnimation):
  
  def __init__(self, t = 0):
    vtkAnimation.__init__(self, t)
    self.time_anim_ends = t + 360
    self.actor = None
    self.actor_position = (0., 0., 0.)
 
  def execute(self, iren, event):
    '''instruction block exectued when a TimerEvent is captured by the vtkRotateActorAroundZAxis.
    
    If the time is not in [start, end] it just returns. Otherwise the 
    transform matrix corresponding to the rotation around the Z-axis is 
    computed and applied to the actor.
    '''
    do = vtkAnimation.pre_execute(self)
    if not do: return
    t1 = self.time_anim_starts
    t2 = self.time_anim_ends
    #t = (self.scene.timer_count - self.time_anim_starts) * np.pi / 180.
    angle = (self.scene.timer_count - t1) / float(t2 - t1) * 2 * np.pi
    X = ([np.cos(angle), np.sin(angle), 0])
    print self.scene.timer_count, X, angle * 180 / np.pi
    Z = np.array([0,0,1])
    Y = np.cross(Z, X)
    m = vtk.vtkMatrix4x4()
    m.Identity()
    # Create the direction cosine matrix
    for i in range(3):
      m.SetElement(i, 0, X[i]);
      m.SetElement(i, 1, Y[i]);
      m.SetElement(i, 2, Z[i]);
    t = vtk.vtkTransform()
    t.Identity()
    t.Concatenate(m)
    t.PostMultiply()
    t.Translate(self.actor_position)
    self.actor.SetUserTransform(t)
    vtkAnimation.post_execute(self, iren, event)

class vtkAnimCameraToZ(vtkAnimation):
  
  def __init__(self, t, cam):
    vtkAnimation.__init__(self, t)
    self.camera = cam
 
  def execute(self, iren, event):
    do = vtkAnimation.pre_execute(self)
    if not do: return
    t1 = self.time_anim_starts
    t2 = self.time_anim_ends
    angle = 90 - (t2 - self.scene.timer_count)/float(t2 - t1) * (90 - 15)
    if self.verbose: print self.scene.timer_count, self.camera.GetPosition(), angle
    self.camera.SetPosition(0, -2*np.cos(angle*np.pi/180.), 2*np.sin(angle*np.pi/180.))
    vtkAnimation.post_execute(self, iren, event)

class vtkZoom(vtkAnimation):
  
  def __init__(self, t, cam, zoom):
    vtkAnimation.__init__(self, t)
    self.camera = cam
    self.zoom = zoom
    self.timer_end = t + 10

  def execute(self, iren, event):
    do = vtkAnimation.pre_execute(self)
    if not do: return
    t1 = self.time_anim_starts
    t2 = self.time_anim_ends
    z = 1 + (self.zoom - 1) * (self.scene.timer_count - t1) / float(t2 - t1)
    if self.verbose: print 'zooming to', z
    self.camera.Zoom(z)
    vtkAnimation.post_execute(self, iren, event)

class vtkSetVisibility(vtkAnimation):
  
  def __init__(self, t, actor, visible = 1, max_opacity = 1, gradually=False):
    vtkAnimation.__init__(self, t)
    self.actor = actor
    self.visible = visible
    self.gradually = gradually
    self.max_opacity = max_opacity
 
  def execute(self, iren, event):
    do = vtkAnimation.pre_execute(self)
    if not do: return
    if not self.gradually:
      self.actor.SetVisibility(self.visible)
    else:
      t1 = self.time_anim_starts
      t2 = self.time_anim_ends
      if self.scene.timer_count >= t1 and self.scene.timer_count <= t2: # useless to test this (do == 1 here)
        if self.actor.GetVisibility() == 0:
          self.actor.SetVisibility(1) # make the actor visible
        if self.visible:
          opacity = self.max_opacity * (1 - (t2 - self.scene.timer_count) / float(t2 - t1))
        else:
          opacity = self.max_opacity * (t2 - self.scene.timer_count) / float(t2 - t1)
        if self.verbose: print 'opacity=',opacity
        # change the opacity for each actor in the assemby
        collection = vtk.vtkPropCollection()
        self.actor.GetActors(collection);
        for i in range(collection.GetNumberOfItems()):
          collection.GetItemAsObject(i).GetProperty().SetOpacity(opacity)
    vtkAnimation.post_execute(self, iren, event)

class vtkMoveActor(vtkAnimation):
  
  def __init__(self, t, actor, motion):
    vtkAnimation.__init__(self, t)
    self.actor = actor
    if self.actor.GetUserTransform() == None:
      if self.verbose: print 'setting initial 4x4 matrix'
      t = vtk.vtkTransform()
      t.Identity()
      self.actor.SetUserTransform(t)
    self.motion = np.array(motion)

  def execute(self, iren, event):
    do = vtkAnimation.pre_execute(self)
    if not do: return
    t1 = self.time_anim_starts
    t2 = self.time_anim_ends
    d = self.motion / (t2 - t1)
    if self.verbose: print 'will move actor by', d
    self.actor.GetUserTransform().Translate(d)
    vtkAnimation.post_execute(self, iren, event)

class vtkAnimLine(vtkAnimation):
  
  def __init__(self, points, t1, t2):
    vtkAnimation.__init__(self, t1)
    self.time_anim_line_end = t2
    self.line_points = points
    self.p0 = np.array(self.line_points.GetPoint(0))
    self.p1 = np.array(self.line_points.GetPoint(1))
    self.grid = None
    self.actor = None
    self.pole = None

  def execute(self, iren, event):
    do = vtkAnimation.pre_execute(self)
    if not do: return
    t1 = self.time_anim_starts
    t2 = self.time_anim_ends
    #if self.scene.timer_count >= t1 and self.scene.timer_count <= t2:
    self.actor.SetVisibility(1)
    point = self.p1 + (t2 - self.scene.timer_count)/float(t2 - t1) * (self.p0 - self.p1)
    self.line_points.SetPoint(1, point)
    if point[2] <= 0 and self.pole != None:
      self.pole.SetVisibility(1)
    self.grid.Modified()
    vtkAnimation.post_execute(self, iren, event)

if __name__ == '__main__':
  cam = vtk.vtkCamera()
  anim = vtkAnimCameraAroundZ(cam, 10)
  anim.verbose = True
