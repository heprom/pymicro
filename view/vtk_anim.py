import vtk
import os, numpy as np

class vtkAnimationScene():

  def __init__(self, ren, ren_size=(600, 600)):
    self.timer_count = 0
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
    anim.save_image = self.save_image
    anim.prefix = self.prefix
    anim.verbose = self.verbose
    self.anims.append(anim)
    self.iren.AddObserver('TimerEvent', anim.execute)
  
  def render(self):
    if self.save_image and not os.path.exists(self.prefix):
      os.mkdir(self.prefix) # create a folder to store the images
    timerId = self.iren.CreateRepeatingTimer(100); # time in ms
    self.iren.Start()
    
'''
Abstract class for all vtk animation stuff.
'''
class vtkAnimation():
  
  def __init__(self, t):
    self.timer_count = 0
    self.timer_incr = 1
    self.time_anim_starts = t
    self.time_anim_ends = t + 10
    self.save_image = False
    self.prefix = 'prefix'
    self.verbose = False
 
  def write_image(self, iren):
    # capture the display and write a png image
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(iren.GetRenderWindow())
    writer = vtk.vtkPNGWriter()
    writer.SetInputConnection(w2i.GetOutputPort())
    file_name = os.path.join(self.prefix, '%s_%d.png' % (self.prefix, self.timer_count))
    writer.SetFileName(file_name)
    writer.Write()

  def pre_execute(self):
    if self.verbose:
      print self.__repr__()
    if self.timer_count < self.time_anim_starts or self.timer_count > self.time_anim_ends:
      self.timer_count += self.timer_incr
      return 0
    else:
      return 1

  def post_execute(self, iren, event):
    iren.Render()
    if self.save_image:
      self.write_image(iren)
    self.timer_count += self.timer_incr

  def __repr__(self):
    out = [self.__class__.__name__,
      ' timer: ' + str(self.timer_count),
      ' anim starts at: ' + str(self.time_anim_starts),
      ' anim ends at: ' + str(self.time_anim_ends)]
    return '\n'.join(out)

class vtkAnimCameraAroundZ(vtkAnimation):
  
  def __init__(self, cam, t):
    print 'init vtkAnimCameraAroundZ'
    vtkAnimation.__init__(self, t)
    self.time_anim_ends = t + 36
    print self.timer_incr
    print self.time_anim_starts
    print self.time_anim_ends
    self.camera = cam
 
  def execute(self, iren, event):
    do = vtkAnimation.pre_execute(self)
    if not do: return
    t1 = self.time_anim_starts
    t2 = self.time_anim_ends
    r = 360 / (t2 - t1)
    self.camera.Azimuth(r)
    vtkAnimation.post_execute(self, iren, event)
    
class vtkRotateActorAroundZAxis(vtkAnimation):
  
  def __init__(self, t = 5):
    vtkAnimation.__init__(self, t)
    self.actor = None
    self.actor_position = (0., 0., 0.)
 
  def execute(self, iren, event):
    do = vtkAnimation.pre_execute()
    if not do: return
    t = self.timer_count * np.pi / 180.
    X = ([np.cos(t), np.sin(t), 0])
    print self.timer_count, X
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
    angle = 90 - (t2 - self.timer_count)/float(t2 - t1) * (90 - 15)
    if self.verbose: print self.timer_count, self.camera.GetPosition(), angle
    self.camera.SetPosition(0, -2*np.cos(angle*np.pi/180.), 2*np.sin(angle*np.pi/180.))
    vtkAnimation.post_execute(self, iren, event)

class vtkSetVisibility(vtkAnimation):
  
  def __init__(self, t, actor, visible = 1, gradually=False):
    vtkAnimation.__init__(self, t)
    self.actor = actor
    self.visible = visible
    self.gradually = gradually
 
  def execute(self, iren, event):
    do = vtkAnimation.pre_execute(self)
    if not do: return
    if not self.gradually:
      self.actor.SetVisibility(self.visible)
    else:
      t1 = self.time_anim_starts
      t2 = self.time_anim_ends
      if self.timer_count >= t1 and self.timer_count <= t2:
        if self.actor.GetVisibility() == 0:
          self.actor.SetVisibility(1) # make the actor visible
        if self.visible:
          opacity = 1 - (t2 - self.timer_count)/float(t2 - t1)
        else:
          opacity = (t2 - self.timer_count)/float(t2 - t1)
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
    #if self.timer_count >= t1 and self.timer_count <= t2:
    self.actor.SetVisibility(1)
    point = self.p1 + (t2 - self.timer_count)/float(t2 - t1) * (self.p0 - self.p1)
    self.line_points.SetPoint(1, point)
    if point[2] <= 0 and self.pole != None:
      self.pole.SetVisibility(1)
    self.grid.Modified()
    vtkAnimation.post_execute(self, iren, event)

if __name__ == '__main__':
  cam = vtk.vtkCamera()
  anim = vtkAnimCameraAroundZ(cam, 10)
  anim.verbose = True
