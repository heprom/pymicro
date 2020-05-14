'''The vtk_anim module define a set of classes to generate 3d 
   animations with vtk in the form of a series of png images.
'''
import vtk
import os
import numpy as np

from pymicro.view.vtk_utils import set_opacity

class vtkAnimationScene:
    def __init__(self, ren, ren_size=(600, 600)):
        self.timer_count = 0
        self.timer_incr = 1
        self.timer_end = -1  # run until 'q' is pressed
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
        # the next two lines fix some opacity problems but slow things down...
        # self.renWin.Render()
        # self.iren.Render()
        writer = vtk.vtkPNGWriter()
        writer.SetInputConnection(w2i.GetOutputPort())
        file_name = os.path.join(self.prefix, '%s_%03d.png' % (self.prefix, self.timer_count))
        writer.SetFileName(file_name)
        writer.Write()

    def execute(self, iren, event):
        self.timer_count += self.timer_incr
        if self.verbose:
            print('animation scene timer_count=', self.timer_count)
        if self.timer_end > 0 and self.timer_count > self.timer_end:
            print('end of animation loop, exiting...')
            self.iren.ExitCallback()
        else:
            self.iren.Render()
        if self.save_image:
            self.write_image()

    def render(self):
        if self.save_image and not os.path.exists(self.prefix):
            os.mkdir(self.prefix)  # create a folder to store the images
        timerId = self.iren.CreateRepeatingTimer(100)  # time in ms
        self.iren.AddObserver('TimerEvent', self.execute)
        self.iren.Start()

    def render_at(self, time=0.):
        if self.save_image and not os.path.exists(self.prefix):
            os.mkdir(self.prefix)  # create a folder to store the images
        self.timer_count = time
        self.iren.CreateOneShotTimer(100)  # time in ms
        self.iren.AddObserver('TimerEvent', self.execute)
        self.iren.Start()


'''
Abstract class for all vtk animation stuff.
'''


class vtkAnimation:
    def __init__(self, t, duration=10):
        self.scene = None
        self.time_anim_starts = t
        self.time_anim_ends = t + duration
        self.verbose = False

    def pre_execute(self):
        if self.verbose:
            print(self.__repr__())
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
    camera SetViewUp method).
    '''

    def __init__(self, t, cam, turn=360):
        '''Initialize the animation.

        The animation perform a full turn in 36 frames by default.
        '''
        print('init vtkAnimCameraAroundZ')
        vtkAnimation.__init__(self, t)
        self.turn = turn
        self.time_anim_ends = t + abs(self.turn) / 10
        print('time_anim_starts', self.time_anim_starts)
        print('time_anim_ends', self.time_anim_ends)
        print('turn', self.turn)
        self.camera = cam

    def execute(self, iren, event):
        '''Execute method called to rotate the camera.'''
        do = vtkAnimation.pre_execute(self)
        if not do: return
        t1 = self.time_anim_starts
        t2 = self.time_anim_ends
        r = self.turn / (t2 - t1)
        if self.scene.verbose:
            print('moving azimuth by', r)
        self.camera.Azimuth(r)
        vtkAnimation.post_execute(self, iren, event)


class vtkRotateActorAroundAxis(vtkAnimation):

    def __init__(self, t=0, duration=10, axis=(0., 0., 1.), angle=360):
        vtkAnimation.__init__(self, t, duration)
        self._actor = None
        self.axis = axis
        self.angle = angle

    def set_actor(self, actor):
        """Set the actor for this animation.

        This also keep a record of the actor initial transformation matrix.

        :param vtkActor actor: the actor on which to apply the animation.
        """
        self._actor = actor
        # keep track of the initial user transform matrix
        transform = actor.GetUserTransform()
        if not transform:
            transform = vtk.vtkTransform()
            transform.Identity()
            actor.SetUserTransform(transform)
        self.user_transform_matrix = actor.GetUserTransform().GetMatrix()

    def execute(self, iren, event):
        """instruction block executed when a TimerEvent is captured by the vtkRotateActorAroundAxis.

        If the time is not in [start, end] nothing is done. Otherwise the transform matrix corresponding
        to the 3D rotation is applied to the actor.

        The transform matrix for this increment is the result of the multiplication of the rotation matrix
        for the current angle with the initial 4x4 matrix before any rotation (we keep a record of this in
        the `user_transform_matrix` attribute).

        :param vtkRenderWindowInteractor iren: the vtk render window interactor.
        :param event: the captures event.
        """
        do = vtkAnimation.pre_execute(self)
        if not do:
            return
        t1 = self.time_anim_starts
        t2 = self.time_anim_ends
        angle = (self.scene.timer_count - t1) / float(t2 - t1) * self.angle
        from pymicro.crystal.microstructure import Orientation
        om = Orientation.Axis2OrientationMatrix(self.axis, angle)
        m = vtk.vtkMatrix4x4()  # row major order, 16 elements matrix
        m.Identity()
        for j in range(3):
            for i in range(3):
                m.SetElement(j, i, om[i, j])
        # compute the transformation matrix for this increment
        t = vtk.vtkTransform()
        t.SetMatrix(self.user_transform_matrix)
        t.Concatenate(m)
        self._actor.SetUserTransform(t)
        vtkAnimation.post_execute(self, iren, event)


class vtkRotateActorAroundZAxis(vtkRotateActorAroundAxis):

    def __init__(self, t=0):
        vtkRotateActorAroundAxis.__init__(self, t, duration=360, axis=(0., 0., 1.), angle=360)


class vtkAnimCameraToZ(vtkAnimation):
    def __init__(self, t, cam):
        vtkAnimation.__init__(self, t)
        self.camera = cam

    def execute(self, iren, event):
        do = vtkAnimation.pre_execute(self)
        if not do: return
        t1 = self.time_anim_starts
        t2 = self.time_anim_ends
        angle = 90 - (t2 - self.scene.timer_count) / float(t2 - t1) * (90 - 15)
        if self.verbose:
            print(self.scene.timer_count, self.camera.GetPosition(), angle)
        self.camera.SetPosition(0, -2 * np.cos(angle * np.pi / 180.), 2 * np.sin(angle * np.pi / 180.))
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
        if self.verbose:
            print('zooming to', z)
        self.camera.Zoom(z)
        vtkAnimation.post_execute(self, iren, event)


class vtkSetVisibility(vtkAnimation):
    def __init__(self, t, actor, visible=1, max_opacity=1, gradually=False):
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
            set_opacity(self.actor, 1)
        else:
            t1 = self.time_anim_starts
            t2 = self.time_anim_ends
            if self.scene.timer_count >= t1 and self.scene.timer_count <= t2:  # useless to test this (do == 1 here)
                if self.actor.GetVisibility() == 0:
                    self.actor.SetVisibility(1)  # make the actor visible
                if self.visible:
                    opacity = self.max_opacity * (1 - (t2 - self.scene.timer_count) / float(t2 - t1))
                else:
                    opacity = self.max_opacity * (t2 - self.scene.timer_count) / float(t2 - t1)
                if self.verbose:
                    print('opacity=', opacity)
                # change the opacity for each actor in the assembly
                set_opacity(self.actor, opacity)
        vtkAnimation.post_execute(self, iren, event)


class vtkMoveActor(vtkAnimation):
    def __init__(self, t, actor, motion):
        vtkAnimation.__init__(self, t)
        self.actor = actor
        if self.actor.GetUserTransform() == None:
            if self.verbose:
                print('setting initial 4x4 matrix')
            t = vtk.vtkTransform()
            t.Identity()
            self.actor.SetUserTransform(t)
        self.motion = np.array(motion).astype(float)

    def execute(self, iren, event):
        do = vtkAnimation.pre_execute(self)
        if not do: return
        t1 = self.time_anim_starts
        t2 = self.time_anim_ends
        d = self.motion / (t2 - t1)
        if self.verbose:
            print('will move actor by', d)
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
        # if self.scene.timer_count >= t1 and self.scene.timer_count <= t2:
        self.actor.SetVisibility(1)
        point = self.p1 + (t2 - self.scene.timer_count) / float(t2 - t1) * (self.p0 - self.p1)
        self.line_points.SetPoint(1, point)
        if point[2] <= 0 and self.pole != None:
            self.pole.SetVisibility(1)
        self.grid.Modified()
        vtkAnimation.post_execute(self, iren, event)


class vtkUpdateText(vtkAnimation):

    def __init__(self, text_actor, str_method, t=0, duration=10):
        vtkAnimation.__init__(self, t, duration)
        self.actor = text_actor
        self.str_method = str_method

    def execute(self, iren, event):
        do = vtkAnimation.pre_execute(self)
        if not do:
            return
        t1 = self.time_anim_starts
        t2 = self.time_anim_ends
        updated_text = self.str_method()  #self.scene.timer_count, t1, t2)
        self.actor.GetMapper().SetInput(updated_text)
        vtkAnimation.post_execute(self, iren, event)

if __name__ == '__main__':
    cam = vtk.vtkCamera()
    anim = vtkAnimCameraAroundZ(cam, 10)
    anim.verbose = True
