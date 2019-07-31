#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, wx, pickle
import thread
from numpy import shape, linspace, zeros_like, pi, cos, sin, tan
from math import atan
from matplotlib import cm

from pymicro.apps.wxPlotPanel import PlotPanel
from pymicro.diffraction.waxd import *


class ImPanel(PlotPanel):
    '''The ImPanel extends PlotPanel to draw the image in a matplotlib
    canvas.

    .. note::
      Right now, drawing the image is very long (several seconds),
      we need to find a way around that.'''

    def __init__(self, parent, image=None, color=None, dpi=None, **kwargs):
        '''Initialize a new ImPanel. This Set some references on the parent,
        the image to draw and call PlotPanel __init__ method.'''
        # initialize Panel
        self.parent = parent
        self.im = image
        self.draw_ref = False
        self.cmap = cm.jet
        self.max = 65536
        # default controls
        self.update_controls(shape(self.im)[0] / 8, shape(self.im)[0] / 4, shape(self.im)[0] / 2, shape(self.im)[0] / 2,
                             -45, 45, 100)
        # initiate plotter
        PlotPanel.__init__(self, parent, **kwargs)
        self.SetColor((255, 255, 255))

    def update_controls(self, rmin, rmax, xc, yc, theta1, theta2, steps, ref_pixels=None):
        self.rmin = rmin
        self.rmax = rmax
        self.xc = xc
        self.yc = yc
        self.theta1 = theta1
        self.theta2 = theta2
        self.steps = steps
        self.ref_pixels = ref_pixels

    def SetDrawRef(self, bool):
        self.draw_ref = bool

    def SetCmap(self, map):
        current = self.cmap
        if map == 'gray':
            self.cmap = cm.gray
        elif map == 'jet':
            self.cmap = cm.jet
        elif map == 'hot':
            self.cmap = cm.hot
        else:
            self.cmap == cm.jet
        # redraw if cmap has changed
        if not (current == self.cmap):
            self.draw()

    def SetColorMax(self, max):
        self.max = max

    def draw(self):
        '''Draw image data and profile controls on top of the image.'''
        if not hasattr(self, 'subplot'):
            self.subplot = self.figure.add_subplot(111)
        self.subplot.imshow(self.im, cmap=self.cmap, origin='lower')
        ###gca().get_images()[0].set_clim(0, self.max)
        del self.subplot.lines[:]
        # draw a cross at (xc, yc)
        xv = (self.xc, self.xc)
        yv = (self.yc - 10, self.yc + 10)
        xh = (self.xc - 10, self.xc + 10)
        yh = (self.yc, self.yc)
        self.subplot.plot(xv, yv, 'y', xh, yh, 'y')
        theta_min = self.theta1 * pi / 180.
        theta_max = self.theta2 * pi / 180.
        thetas = linspace(theta_min, theta_max, self.steps)
        self.subplot.plot(self.xc + self.rmin * cos(thetas), \
                          self.yc + self.rmin * sin(thetas), 'y', linewidth=2)
        self.subplot.plot(self.xc + self.rmax * cos(thetas), \
                          self.yc + self.rmax * sin(thetas), 'y', linewidth=2)
        self.subplot.plot((self.xc + self.rmin * cos(theta_min), self.xc + self.rmax * cos(theta_min)), \
                          (self.yc + self.rmin * sin(theta_min), self.yc + self.rmax * sin(theta_min)), 'y',
                          linewidth=2)
        self.subplot.plot((self.xc + self.rmin * cos(theta_max), self.xc + self.rmax * cos(theta_max)), \
                          (self.yc + self.rmin * sin(theta_max), self.yc + self.rmax * sin(theta_max)), 'y',
                          linewidth=2)
        # if self.draw_ref and self.ref_pixels != None:
        if self.draw_ref:
            self.subplot.plot(self.xc + self.ref_pixels * cos(thetas), self.yc + self.ref_pixels * sin(thetas), 'r',
                              linewidth=2)
        extent = (0, shape(self.im)[0], 0, shape(self.im)[1])
        self.subplot.axis(extent)
        self.Layout()


class ProfilePanel(PlotPanel):
    '''The ProfilePanel extends PlotPanel to display radial profile data in matplotlib canvas.'''

    def __init__(self, parent, color=None, dpi=None, **kwargs):
        # initialize Panel
        self.parent = parent
        self.profile = None
        self.pix2angle = False
        # initiate plotter
        PlotPanel.__init__(self, parent, **kwargs)
        self.SetColor((255, 255, 255))

    def update_profile(self, profile):
        self.profile = profile
        self.draw()

    def SetPix2Angle(self, bool, ref_angle, ref_pixels):
        self.pix2angle = bool
        self.ref_angle = ref_angle
        self.ref_pixels = ref_pixels

    def draw(self):
        '''Draw radial profile data.'''
        if not hasattr(self, 'subplot'):
            self.subplot = self.figure.add_subplot(111)
        if self.profile == None:
            return
        if self.pix2angle:
            # self.subplot.xlabel('$\theta$')
            x = self.profile[0]
            angles = zeros_like(x)
            for i in range(len(x)):
                angles[i] = atan(tan(self.ref_angle * pi / 180.) * x[i] / self.ref_pixels) * 180. / pi
            self.subplot.plot(angles, self.profile[1])
        else:
            self.subplot.plot(self.profile[0], self.profile[1])
        self.Layout()


class wxRadialProfileFrame(wx.Frame):
    def __init__(self, im_file=''):
        wx.Frame.__init__(self, None, wx.ID_ANY, title=u"wx Radial Profile")
        self.SetSize((800, 600))
        self.statusbar = self.CreateStatusBar()
        self.im = None
        if im_file != None:
            self.OnLoadImage(im_file)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        # child 1: image panel
        leftPanel = wx.Panel(self, wx.ID_ANY)
        leftBox = wx.BoxSizer(wx.VERTICAL)
        leftPanel.SetSizer(leftBox)
        self.imPanel = ImPanel(leftPanel, image=self.im, style=wx.SUNKEN_BORDER)
        cmapPanel = wx.Panel(leftPanel, wx.ID_ANY)
        leftBox.Add(self.imPanel, 1, wx.EXPAND | wx.ALL, 5)
        cmapBox = wx.BoxSizer(wx.HORIZONTAL)
        cmapBox.Add(wx.StaticText(cmapPanel, wx.ID_ANY, 'Colormap:'), 0, wx.ALIGN_CENTER)
        self.cmapCombo = wx.ComboBox(cmapPanel, id=421, choices=['gray', 'jet', 'hot'], style=wx.CB_DROPDOWN)
        self.cmapCombo.SetSelection(1)
        self.Bind(wx.EVT_COMBOBOX, self.OnCmapSelected, id=421)
        cmapBox.Add(self.cmapCombo, 0, wx.ALIGN_CENTER)  # wx.EXPAND | wx.ALL, 5)
        cmapPanel.SetSizer(cmapBox)
        self.maxim_sc = wx.SpinCtrl(cmapPanel, wx.ID_ANY, min=1, max=12000)
        self.Bind(wx.EVT_SPINCTRL, self.OnColorScaleUpdate)
        cmapBox.Add(self.maxim_sc, 0, wx.ALIGN_CENTER)  # wx.EXPAND | wx.ALL, 5)
        leftBox.Add(cmapPanel, 0, wx.EXPAND | wx.ALL, 5)

        sizer.Add(leftPanel, 1, wx.EXPAND | wx.ALL, 5)
        # sizer.Add(wx.StaticText(dummyPanel, -1, 'Colormap:'), 0, wx.EXPAND | wx.ALL, 5)

        # child 2: profile panel
        profilePanel = wx.Panel(self, -1)
        # profilePanel.SetBackgroundColour(wx.LIGHT_GREY)
        sizer.Add(profilePanel, 1, wx.EXPAND)

        vbox = wx.BoxSizer(wx.VERTICAL)
        # ** design control panel
        controlPanel = wx.Panel(profilePanel, -1)
        # we use a StaticBox and a StaticBoxSizer to reproup controls
        staticBox = wx.StaticBox(controlPanel, -1, 'Controls', (5, 5))
        sbSizer = wx.StaticBoxSizer(staticBox, wx.VERTICAL)
        grid = wx.GridSizer(5, 4, 3, 3)
        sbSizer.Add(grid, 0, wx.EXPAND)
        # we use spin spin controls to adjust radius and center coords
        self.rmin_sc = wx.SpinCtrl(controlPanel, -1, min=1, max=shape(self.im)[0])
        self.rmax_sc = wx.SpinCtrl(controlPanel, -1, min=1, max=shape(self.im)[0])
        self.xcenter_sc = wx.SpinCtrl(controlPanel, -1, min=1, max=shape(self.im)[0])
        self.ycenter_sc = wx.SpinCtrl(controlPanel, -1, min=1, max=shape(self.im)[1])
        self.theta1_sc = wx.SpinCtrl(controlPanel, -1, min=-180, max=180)
        self.theta2_sc = wx.SpinCtrl(controlPanel, -1, min=-180, max=180)
        self.steps_sc = wx.SpinCtrl(controlPanel, -1, min=0, max=500, initial=100)
        self.ref_radius = wx.SpinCtrl(controlPanel, -1, min=0, max=90)
        self.ref_pixels = wx.SpinCtrl(controlPanel, -1, min=0, max=shape(self.im)[1])
        self.highlight = wx.CheckBox(controlPanel, 410, '&Highlight')
        ## WARN: we now use the number of steps instead of the step angle (wx.SpinCtrl only handle int)
        self.Bind(wx.EVT_SPINCTRL, self.CtrlUpdate)
        grid.AddMany([(wx.StaticText(controlPanel, -1, 'R min:'), 0, wx.ALIGN_CENTER),
                      (wx.StaticText(controlPanel, -1, 'R max:'), 0, wx.ALIGN_CENTER),
                      (wx.StaticText(controlPanel, -1, 'X center:'), 0, wx.ALIGN_CENTER),
                      (wx.StaticText(controlPanel, -1, 'Y center:'), 0, wx.ALIGN_CENTER),
                      (self.rmin_sc, 0, wx.EXPAND),
                      (self.rmax_sc, 0, wx.EXPAND),
                      (self.xcenter_sc, 0, wx.EXPAND),
                      (self.ycenter_sc, 0, wx.EXPAND),
                      (wx.StaticText(controlPanel, -1, 'Phi 1:'), 0, wx.ALIGN_CENTER),
                      (wx.StaticText(controlPanel, -1, 'Phi 2:'), 0, wx.ALIGN_CENTER),
                      (wx.StaticText(controlPanel, -1, 'Steps:'), 0, wx.ALIGN_CENTER),
                      (wx.StaticText(controlPanel, -1, 'Oversampling:'), 0, wx.ALIGN_CENTER),
                      (self.theta1_sc, 0, wx.EXPAND),
                      (self.theta2_sc, 0, wx.EXPAND),
                      (self.steps_sc, 0, wx.EXPAND),
                      (wx.StaticText(controlPanel, -1, '2'), 0, wx.ALIGN_CENTER),
                      (wx.Button(controlPanel, 20, '&Clear'), 0, wx.EXPAND),
                      (wx.Button(controlPanel, 21, '&Save'), 0, wx.EXPAND),
                      (wx.Button(controlPanel, 22, '&Quit'), 0, wx.EXPAND)])
        sbSizer.Add(wx.CheckBox(controlPanel, 401, 'Convert pixels to angles'), 0, wx.EXPAND)
        self.Bind(wx.EVT_CHECKBOX, self.Pix2angleChecked, id=401)
        self.Bind(wx.EVT_CHECKBOX, self.Highlight, id=410)
        grid2 = wx.GridSizer(2, 3, 3, 3)
        sbSizer.Add(grid2, 0, wx.EXPAND)
        grid2.AddMany([(wx.StaticText(controlPanel, -1, 'Ref (angle):'), 0, wx.ALIGN_CENTER),
                       (self.ref_radius, 0, wx.EXPAND),
                       (wx.StaticText(controlPanel, -1, ''), 0, wx.ALIGN_CENTER),
                       (wx.StaticText(controlPanel, -1, 'Ref (pixels):'), 0, wx.ALIGN_CENTER),
                       (self.ref_pixels, 0, wx.EXPAND),
                       (self.highlight, 0, wx.ALIGN_CENTER)
                       ])
        controlPanel.SetSizer(sbSizer)
        self.Bind(wx.EVT_BUTTON, self.OnClear, id=20)
        self.Bind(wx.EVT_BUTTON, self.OnSave, id=21)
        self.Bind(wx.EVT_BUTTON, self.OnQuit, id=22)
        vbox.Add(controlPanel, 0, wx.EXPAND)

        # ** design plot panel
        self.plotPanel = ProfilePanel(profilePanel, -1)
        vbox.Add(self.plotPanel, 1, wx.EXPAND | wx.ALL, 5)
        plotPanelButtons = wx.Panel(profilePanel, -1)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(wx.Button(plotPanelButtons, 30, 'Radial profile'), 1, 3)
        hbox.Add(wx.Button(plotPanelButtons, 31, 'Phi profile'), 1, 3)
        hbox.Add(wx.Button(plotPanelButtons, 32, 'Clear'), 1, 3)
        hbox.Add(wx.Button(plotPanelButtons, 33, 'Save profile'), 1, 3)
        plotPanelButtons.SetSizer(hbox)
        self.Bind(wx.EVT_BUTTON, self.OnCompute, id=30)
        self.Bind(wx.EVT_BUTTON, self.OnComputePhiProfile, id=31)
        self.Bind(wx.EVT_BUTTON, self.OnClearProfile, id=32)
        self.Bind(wx.EVT_BUTTON, self.OnSaveProfile, id=33)
        vbox.Add(plotPanelButtons, 0, wx.EXPAND | wx.ALL, 5)
        self.gauge = wx.Gauge(profilePanel, wx.ID_ANY, 100)
        vbox.Add(self.gauge, 0, wx.EXPAND | wx.ALL, 5)

        # do some initialization
        if os.path.exists(self.path + '.par'):
            print('found radial profile parameters')
            params = open_radial_profile_params(self.path + '.par')
            if params[0] != None: self.rmin_sc.SetValue(params[0])
            if params[1] != None: self.rmax_sc.SetValue(params[1])
            if params[2] != None: self.xcenter_sc.SetValue(params[2])
            if params[3] != None: self.ycenter_sc.SetValue(params[3])
            if params[4] != None: self.theta1_sc.SetValue(params[4])
            if params[5] != None: self.theta2_sc.SetValue(params[5])
            if params[6] != None: self.steps_sc.SetValue(params[6])
            if params[7] != None: self.ref_radius.SetValue(params[7])
            if params[8] != None: self.ref_pixels.SetValue(params[8])
            self.CtrlUpdate(None)
            if params[7] == None or params[8] == None:
                self.Pix2angleChecked(None)
        else:
            self.CtrlDefaults()

        # final settings
        profilePanel.SetSizer(vbox)
        self.SetSizer(sizer)
        # self.Centre()

    def OnLoadImage(self, im_file):
        self.path = im_file
        print('self.path=', self.path)
        # read diffraction image
        if self.path.endswith('.edf'):
            # self.im = edf_readf(im_file, 2300)
            from scipy.signal import medfilt2d
            self.im = medfilt2d(edf_readf(im_file, 2300), 5)
            for i in range(2300):
                for j in range(2300):
                    if self.im[i, j] > 12000.: self.im[i, j] = 12000.
        elif self.path.endswith('.dat'):
            self.im = np.genfromtxt(im_file)
        else:
            # self.im = rawmar_read(im_file, 2300)
            from pymicro.file.file_utils import HST_read
            self.im = HST_read(im_file, data_type=np.uint16, dims=(2048, 2048, 1))[:, :, 0]
        self.statusbar.SetStatusText('Image loaded from %s ' % self.path)

    def OnCmapSelected(self, event):
        print(self.cmapCombo.GetValue())
        self.imPanel.SetCmap(self.cmapCombo.GetValue())

    def OnColorScaleUpdate(self, event):
        print(self.maxim_sc.GetValue())
        self.imPanel.SetColorMax(self.maxim_sc.GetValue())

    def CtrlDefaults(self):
        self.rmin_sc.SetValue(shape(self.im)[0] / 8)
        self.rmax_sc.SetValue(shape(self.im)[0] / 4)
        self.xcenter_sc.SetValue(shape(self.im)[0] / 2)
        self.ycenter_sc.SetValue(shape(self.im)[1] / 2)
        self.theta1_sc.SetValue(-45)
        self.theta2_sc.SetValue(45)
        self.steps_sc.SetValue(100)
        self.ref_radius.SetValue(30)
        self.ref_pixels.SetValue(1000)

    def Pix2angleChecked(self, event):
        if event == None:
            enable = False
        else:
            enable = event.IsChecked()
        self.ref_radius.Enable(enable)
        self.ref_pixels.Enable(enable)
        self.highlight.Enable(enable)
        self.plotPanel.SetPix2Angle(enable, self.ref_radius.GetValue(), \
                                    self.ref_pixels.GetValue())

    def Highlight(self, event):
        enable = event.IsChecked()
        self.imPanel.SetDrawRef(enable)
        self.imPanel.draw()

    def CtrlUpdate(self, event):
        self.imPanel.update_controls(self.rmin_sc.GetValue(), self.rmax_sc.GetValue(), \
                                     self.xcenter_sc.GetValue(), self.ycenter_sc.GetValue(), \
                                     self.theta1_sc.GetValue(), self.theta2_sc.GetValue(), \
                                     self.steps_sc.GetValue(), self.ref_pixels.GetValue())
        self.imPanel.draw()

    def OnClear(self, event):
        self.CtrlDefaults()
        self.CtrlUpdate(event)

    def OnSave(self, event):
        print('saving radial profile parameters to ' + self.path + '.par')
        save_radial_profile_params(self.path + '.par', self.rmin_sc.GetValue(), self.rmax_sc.GetValue(), \
                                   self.xcenter_sc.GetValue(), self.ycenter_sc.GetValue(), \
                                   self.theta1_sc.GetValue(), self.theta2_sc.GetValue(), \
                                   self.steps_sc.GetValue(), self.ref_radius.GetValue(), \
                                   self.ref_pixels.GetValue(), self.im)

    def OnQuit(self, event):
        self.Close()

    def OnProgress(self, progress):
        self.gauge.SetValue(int(progress * 100))
        self.gauge.Update()

    def OnCompute(self, event):
        '''  Callback to compute profile.

        The radial profile is computed with the parameters read from
        the UI controls. The progress bar is updated in a separate thread.
        '''
        self.profile = radial_profile(self.rmin_sc.GetValue(), self.rmax_sc.GetValue(), \
                                      self.xcenter_sc.GetValue(), self.ycenter_sc.GetValue(), \
                                      self.theta1_sc.GetValue(), self.theta2_sc.GetValue(), \
                                      self.steps_sc.GetValue(), self.im, \
                                      lambda progress: self.OnProgress(progress))
        self.plotPanel.update_profile(self.profile)

    def OnComputePhiProfile(self, event):
        '''  Callback to compute phi profile.

        The phi profile is computed with the parameters read from
        the UI controls. The progress bar is updated in a separate thread.
        '''
        self.profile = phi_profile(self.rmax_sc.GetValue() - 100., self.rmax_sc.GetValue(), \
                                   self.xcenter_sc.GetValue(), self.ycenter_sc.GetValue(), \
                                   self.theta1_sc.GetValue(), self.theta2_sc.GetValue(), \
                                   self.steps_sc.GetValue(), self.im, \
                                   lambda progress: self.OnProgress(progress))
        print(self.profile)
        self.plotPanel.update_profile(self.profile)

    def OnClearProfile(self, event):
        self.plotPanel.subplot.cla()

    def OnSaveProfile(self, event):
        print('saving radial profile to ' + self.path + '.profile')
        if self.plotPanel.pix2angle:
            x = self.profile[0]
            angles = zeros_like(x)
            for i in range(len(x)):
                angles[i] = atan(
                    tan(self.ref_radius.GetValue() * pi / 180.) * x[i] / self.ref_pixels.GetValue()) * 180. / pi
            pickle.dump([angles, self.profile[1]], file(self.path + '.profile', 'w'), 1)
        else:
            pickle.dump(self.profile, file(self.path + '.profile', 'w'), 1)


class wxRadialProfile(wx.App):
    def __init__(self, im_file):
        self.path = im_file
        wx.App.__init__(self)

    def OnInit(self):
        '''Init wxRadialProfile application.'''
        frame = wxRadialProfileFrame(im_file=self.path)
        frame.Show()
        # set the main window
        self.SetTopWindow(frame)
        return True


if __name__ == '__main__':
    data_dir = '/home/proudhon/data/20080545/nr_04/'
    name = 'nr_04_096'
    name = 'nr_04_098'
    im_file = data_dir + name + '.image'
    data_dir = '/home/proudhon/data/20090552/2010_0203/'
    # im_file = data_dir + 'marTest_2010-02-03_13-28-27nxs1_test_1.edf'
    # im_file = data_dir + 'nr_seq1_2010-02-03_17-52-14nxs2_nr_seq1_76.edf'
    im_file = data_dir + 'nr_seq2_2010-02-03_22-01-50nxs2_nr_seq1_70.edf'

    # create application instance
    app = wxRadialProfile(im_file)
    # start main event loop
    app.MainLoop()
