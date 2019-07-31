#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, wx, os, re, numpy as np
from pymicro.apps.wxPlotPanel import PlotPanel
from matplotlib import pyplot, colors, cm
import matplotlib.image as mpimg
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
        # initialize Panel
        self.parent = parent
        self.im = image
        self.cmap = cm.gray
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


class wxImageViewerFrame(wx.Frame):
    def __init__(self, images=[]):
        self.image_list = images
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
        self.cmapCombo = wx.ComboBox(cmapPanel, id=421, choices=['gray', 'jet', 'hot'], style=wx.CB_DROPDOWN)
        self.cmapCombo.SetSelection(0)
        self.Bind(wx.EVT_COMBOBOX, self.OnCmapSelected, id=421)
        cmapBox.Add(self.cmapCombo, 0, wx.ALIGN_CENTER)
        cmapPanel.SetSizer(cmapBox)
        cmapBox.Add(wx.StaticText(cmapPanel, wx.ID_ANY, ' Image:'), 0, wx.ALIGN_CENTER)
        self.maxim_sc = wx.SpinCtrl(cmapPanel, wx.ID_ANY, min=0, max=len(self.image_list) - 1)
        self.Bind(wx.EVT_SPINCTRL, self.OnSliceUpdate)
        cmapBox.Add(self.maxim_sc, 0, wx.ALIGN_CENTER)
        self.cur_image_name = wx.StaticText(cmapPanel, wx.EXPAND | wx.TE_READONLY, '')
        cmapBox.Add(self.cur_image_name, 0, wx.EXPAND | wx.ALL, 5)
        leftBox.Add(cmapPanel, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(leftPanel, 1, wx.EXPAND | wx.ALL, 5)
        # final settings
        if self.image_list:
            self.OnLoadImage(self.image_list[0])
        self.SetSizer(sizer)

    def OnLoadImage(self, im_file):
        self.path = im_file
        print('self.path=', self.path)
        # read image depending on file extension (only .png .tif and .edf supported)
        if self.path.endswith('.edf'):
            self.im = edf_read(im_file).transpose()
        elif self.path.endswith('.png'):
            self.im = mpimg.imread(im_file)
        elif self.path.endswith('.tif'):
            self.im = TiffFile(im_file).asarray()
        else:
            print('Only png, tif and edf images are supported for the moment')
            sys.exit(1)
        print('min in image= %d, max in image=%d' % (np.min(self.im), np.max(self.im)))
        print(np.shape(self.im))
        self.cur_image_name.SetLabel(im_file)
        self.imPanel.SetImage(self.im)

    def OnCmapSelected(self, event):
        self.imPanel.SetCmap(self.cmapCombo.GetValue())

    def OnSliceUpdate(self, event):
        self.OnLoadImage(self.image_list[self.maxim_sc.GetValue()])


class ImageViewer(wx.App):
    def __init__(self, wdir='.', pattern='.png$'):
        self.wdir = wdir
        self.images = []
        for file in os.listdir(self.wdir):
            if re.search(pattern, file):
                self.images.append(os.path.join(wdir, file))
        print(self.images)
        if not self.images:
            print('No image found, please verify your pattern or your working directory')
            sys.exit(1)
        wx.App.__init__(self)
        # start main event loop
        self.MainLoop()

    def OnInit(self):
        '''Init the ImageViewer application.'''
        frame = wxImageViewerFrame(self.images)
        frame.Show()
        # set the main window
        self.SetTopWindow(frame)
        return True


if __name__ == '__main__':
    ImageViewer(wdir='/home/proudhon/data/tomo/rawdata/ma1921/Pb_balls_316LN_', pattern='.edf')
    # ImageViewer(wdir = '/home/proudhon/data')
