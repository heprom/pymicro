#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, wx, numpy as np
from matplotlib import pyplot, colors, cm

from pymicro.apps.wxPlotPanel import PlotPanel
from pymicro.file.file_utils import edf_read
from pymicro.view.vtk_utils import vol_view, grid_vol_view


class ImPanel(PlotPanel):
    '''The ImPanel extends PlotPanel to draw the image in a matplotlib
    canvas.

    .. note::
      Right now, drawing the image is very long (several seconds),
      we need to find a way around that.'''

    def __init__(self, parent, image=None, color=None, dpi=None, **kwargs):
        '''Initialize a new ImPanel. This Set some references on the parent,
        the image to draw and call PlotPanel __init__ method.'''
        # consistently create the same random colors
        np.random.seed(13)
        rand_colors = np.random.rand(2048, 3)
        rand_colors[0] = [0., 0., 0.]  # enforce black background
        self.rand_cmap = colors.ListedColormap(rand_colors)
        # initialize Panel
        self.parent = parent
        self.im = image
        self.cmap = self.rand_cmap
        # initiate plotter
        PlotPanel.__init__(self, parent, **kwargs)
        self.SetColor((255, 255, 255))

    def draw(self):
        '''Helper function to draw image data.'''
        if not hasattr(self, 'subplot'):
            self.subplot = self.figure.add_subplot(111)
        if self.im != None:
            self.subplot.axis('off')
            self.subplot.imshow(self.im, cmap=self.cmap, origin='upper', interpolation='nearest', clim=[0, 2047])
            self.Layout()

    def SetImage(self, image):
        self.im = image
        self.draw()

    def SetCmap(self, map):
        current = self.cmap
        if map == 'polycrystal':
            self.cmap = self.rand_cmap
        elif map == 'gray':
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


class wxVolumeViewerFrame(wx.Frame):
    def __init__(self, im_file=''):
        wx.Frame.__init__(self, None, wx.ID_ANY, title=u"Volume Viewer")
        self.SetSize((800, 600))
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        # image panel
        leftPanel = wx.Panel(self, wx.ID_ANY)
        leftBox = wx.BoxSizer(wx.VERTICAL)
        leftPanel.SetSizer(leftBox)
        self.imPanel = ImPanel(leftPanel, style=wx.SUNKEN_BORDER)
        cmapPanel = wx.Panel(leftPanel, wx.ID_ANY)
        leftBox.Add(self.imPanel, 1, wx.EXPAND | wx.ALL, 5)
        cmapBox = wx.BoxSizer(wx.HORIZONTAL)
        cmapBox.Add(wx.StaticText(cmapPanel, wx.ID_ANY, ' Colormap:'), 0, wx.ALIGN_CENTER)
        self.cmapCombo = wx.ComboBox(cmapPanel, id=421, choices=['polycrystal', 'gray', 'jet', 'hot'],
                                     style=wx.CB_DROPDOWN)
        self.cmapCombo.SetSelection(0)
        self.Bind(wx.EVT_COMBOBOX, self.OnCmapSelected, id=421)
        cmapBox.Add(self.cmapCombo, 0, wx.ALIGN_CENTER)
        cmapPanel.SetSizer(cmapBox)
        cmapBox.Add(wx.StaticText(cmapPanel, wx.ID_ANY, ' Slice:'), 0, wx.ALIGN_CENTER)
        self.maxim_sc = wx.SpinCtrl(cmapPanel, wx.ID_ANY, min=0, max=10)
        self.Bind(wx.EVT_SPINCTRL, self.OnSliceUpdate)
        cmapBox.Add(self.maxim_sc, 0, wx.ALIGN_CENTER)
        cmapBox.Add(wx.Button(cmapPanel, 20, '3D &View'), 0, wx.ALIGN_CENTER)
        leftBox.Add(cmapPanel, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(leftPanel, 1, wx.EXPAND | wx.ALL, 5)
        # final settings
        self.Bind(wx.EVT_BUTTON, self.On3dView, id=20)
        if im_file != None:
            self.OnLoadImage(im_file)
        self.SetSizer(sizer)

    def OnLoadImage(self, im_file):
        self.path = im_file
        print('self.path=', self.path)
        # read 3d image
        h, self.vol = edf_read(im_file, header_size=0, verbose=True, \
                               autoparse_filename=True, return_header=True)
        print('min in image= %d, max in image=%d' % (np.min(self.vol), np.max(self.vol)))
        self.maxim_sc.SetRange(0, np.shape(self.vol)[2] - 1)
        print(np.shape(self.vol))
        self.imPanel.SetImage(self.vol[:, :, 0])

    def OnCmapSelected(self, event):
        self.imPanel.SetCmap(self.cmapCombo.GetValue())

    def OnSliceUpdate(self, event):
        self.imPanel.SetImage(self.vol[:, :, self.maxim_sc.GetValue()])

    def On3dView(self, event):
        print('launching 3d vol viewer')
        # grid_vol_view(self.path)
        vol_view(self.path)


class wxVolumeViewer(wx.App):
    def __init__(self, im_file):
        self.path = im_file
        wx.App.__init__(self)

    def OnInit(self):
        '''Init wxVolumeViewer application.'''
        frame = wxVolumeViewerFrame(im_file=self.path)
        frame.Show()
        # set the main window
        self.SetTopWindow(frame)
        return True


if __name__ == '__main__':
    data_dir = '/home/proudhon/anr/AFGRAP/AGREGATS/RAW/'
    # name = 'dct_test1_304x304x10_uint8.raw'
    name = 'F30_crop0_2-2-2_80x80x58_uint8.raw'
    im_file = os.path.join(data_dir, name)
    # create application instance
    app = wxVolumeViewer(im_file)
    # start main event loop
    app.MainLoop()
