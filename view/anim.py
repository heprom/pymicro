import vtk
from vtk.util.colors import *
import numpy

class vtkTimerCallback():
  def __init__(self):
    self.timer_count = 0
 
  def execute(self,obj,event):
    t = self.timer_count * numpy.pi / 180.
    X = ([numpy.cos(t), numpy.sin(t), 0])
    print self.timer_count, X
    arb = numpy.array([0,0,1])
    Z = numpy.cross(X, arb)
    Y = numpy.cross(Z, X)
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
    self.actor.SetUserTransform(t)
    iren = obj
    iren.GetRenderWindow().Render()
    self.timer_count += 1
 
def main():
   #Create a sphere or cube
   object = vtk.vtkCubeSource()
   object.SetCenter(0.0, 0.0, 0.0)
   #object.SetRadius(5)
 
   #Create a mapper and actor
   mapper = vtk.vtkPolyDataMapper()
   mapper.SetInputConnection(object.GetOutputPort())
   actor = vtk.vtkActor()
   actor.SetMapper(mapper)
   actor.GetProperty().SetColor(orange)
 
   # Setup a renderer, render window, and interactor
   renderer = vtk.vtkRenderer()
   renderWindow = vtk.vtkRenderWindow()
   #renderWindow.SetWindowName("Test")
 
   renderWindow.AddRenderer(renderer);
   renderWindowInteractor = vtk.vtkRenderWindowInteractor()
   renderWindowInteractor.SetRenderWindow(renderWindow)
 
   #Add the actor to the scene
   renderer.AddActor(actor)
   renderer.SetBackground(1,1,1) # Background color white
 
   #Render and interact
   renderWindow.Render()
 
   # Initialize must be called prior to creating timer events.
   renderWindowInteractor.Initialize()
 
   # Sign up to receive TimerEvent
   cb = vtkTimerCallback()
   cb.actor = actor
   renderWindowInteractor.AddObserver('TimerEvent', cb.execute)
   timerId = renderWindowInteractor.CreateRepeatingTimer(100);
 
   #start the interaction and timer
   renderWindowInteractor.Start()
 
 
if __name__ == '__main__':
   main()
