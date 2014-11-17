import vtk

class vtkTimerCallback():
   def __init__(self):
     self.timer_count = 0
 
   def started(self, obj, event):
     print 'STARTED!'
     
   def execute(self, obj, event):
     print event
     print self.timer_count
     self.actor.SetPosition(0.1*self.timer_count, 0.1*self.timer_count,0);
     iren = obj
     iren.GetRenderWindow().Render()
     self.timer_count += 1
  
if __name__ == '__main__':
  iren = vtk.vtkRenderWindowInteractor()
  ren = vtk.vtkRenderer()
  renWin = vtk.vtkRenderWindow()
  renWin.SetSize(600, 600)
  renWin.SetMultiSamples(0)
  iren.SetRenderWindow(renWin)
  renWin.AddRenderer(ren)
  renWin.Render()
  
  # Create an Animation Scene
  scene = vtk.vtkAnimationScene()
  scene.SetModeToSequence()
  scene.SetLoop(0)
  scene.SetFrameRate(5)
  scene.SetStartTime(3)
  scene.SetEndTime(20)

  # Create an Animation Cue.
  cue1 = vtk.vtkAnimationCue()
  cue1.SetStartTime(5)
  cue1.SetEndTime(23)
  scene.AddCue(cue1)

  # Create a sphere
  sphereSource = vtk.vtkSphereSource()
  sphereSource.SetCenter(0.0, 0.0, 0.0)
  sphereSource.SetRadius(1)
 
  # Create a mapper and actor
  mapper = vtk.vtkPolyDataMapper()
  mapper.SetInputConnection(sphereSource.GetOutputPort())
  actor = vtk.vtkActor()
  actor.SetMapper(mapper)
  prop = actor.GetProperty()
  # Add the actor to the scene
  ren.AddActor(actor)
  ren.SetBackground(1,1,1) # Background color white
  
  iren.Initialize()
  cb = vtkTimerCallback()
  cb.actor = actor
  iren.AddObserver('TimerEvent', cb.execute)
  #iren.AddObserver('StartEvent', cb.started)
  timerId = iren.CreateRepeatingTimer(100);
  #scene.Play()
  #scene.Stop()
  iren.Start()

  print 'done'
