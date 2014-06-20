import vtk
import os, numpy

class vtkRotateActorAroundZAxis():
  
  def __init__(self):
    self.timer_count = 0
    self.timer_incr = 2
    self.prefix = 'prefix'
    self.actor = None
    self.display = True
    self.actor_position = (0., 0., 0.)
 
  def execute(self, iren, event):
    if self.timer_count >= 359: return
    t = self.timer_count * numpy.pi / 180.
    X = ([numpy.cos(t), numpy.sin(t), 0])
    print self.timer_count, X
    Z = numpy.array([0,0,1])
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
    t.PostMultiply()
    t.Translate(self.actor_position)
    self.actor.SetUserTransform(t)
    iren.Render()
    if not self.display:
      # capture the display and write a png image
      w2i = vtk.vtkWindowToImageFilter()
      w2i.SetInput(iren.GetRenderWindow())
      writer = vtk.vtkPNGWriter()
      writer.SetInputConnection(w2i.GetOutputPort())
      file_name = os.path.join(self.prefix, '%s_%d.png' % (self.prefix, self.timer_count))
      writer.SetFileName(file_name)
      writer.Write()
    self.timer_count += self.timer_incr

