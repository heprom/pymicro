#!/usr/bin/env python
import sys
import os
import re

import numpy as np
from matplotlib.figure import Figure
import matplotlib.image as mpimg

from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from pymicro.file.file_utils import edf_read
from pymicro.file.tifffile import TiffFile
from pymicro.view.vol_utils import hist

class ImageViewerForm(QMainWindow):
  
    def __init__(self, parent=None, image=None):
        QMainWindow.__init__(self, parent)
        self.data = image
        self.cmap = 'gray'
        self.create_main_frame()
        self.on_draw()

    def create_main_frame(self):
        self.main_frame = QWidget()

        #self.file_label = QtGui.QLabel()
        self.fig = Figure((10.0, 8.0), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()

        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)

        hbox = QHBoxLayout()
        self.cmap_label = QLabel('color map')
        hbox.addWidget(self.cmap_label)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItem('gray')
        self.cmap_combo.addItem('jet')
        self.cmap_combo.addItem('hot')
        self.cmap_combo.activated[str].connect(self.on_cmap_selected) 
        hbox.addWidget(self.cmap_combo)
        
        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)  # the matplotlib canvas
        vbox.addWidget(self.mpl_toolbar)
        vbox.addLayout(hbox)
        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

    def set_image(self, image):
      self.data = image
      self.on_draw()

    def on_cmap_selected(self, text):
      self.cmap = str(text)
      self.on_draw()

    def on_draw(self):
        if not hasattr(self.fig, 'subplot'):
          self.axes = self.fig.add_subplot(111)
        self.axes.imshow(self.data, cmap=self.cmap, origin='upper', interpolation='nearest', clim=[np.min(self.data), np.max(self.data)])
        self.canvas.draw()

    def on_key_press(self, event):
        if event.key == 'i':
          print('* image infos:')
          print('shape is (%dx%d)' % (np.shape(self.data)[0], np.shape(self.data)[1]))
          print('min in image= %d, max in image=%d' % (np.min(self.data), np.max(self.data)))
        if event.key == 'h':
          # plot the histogram
          hist(self.data, data_range=(np.min(self.data), np.max(self.data)), show=True)
        # implement the default mpl key press events described at
        # http://matplotlib.org/users/navigation_toolbar.html#navigation-keyboard-shortcuts
        key_press_handler(event, self.canvas, self.mpl_toolbar)


class ImageViewer(QApplication):

  def __init__(self, args):
    '''Init the ImageViewer application.'''
    print(args)
    self.wdir = '.'
    pattern = '.png$'
    if len(args) > 1:
      self.wdir = args[1]
    if len(args) > 2:
      pattern = args[2]
    self.images = []
    for f in os.listdir(self.wdir):
      if re.search(pattern, f):
        self.images.append(os.path.join(self.wdir, f))
    print self.images
    if not self.images:
      print('No image found, please verify your pattern or your working directory')
      sys.exit(1)
    self.load_image(self.images[0])
    app = QApplication(['Pymicro Image Viewer'])
    form = ImageViewerForm(image = self.im)
    form.show()
    app.exec_()

  def load_image(self, im_file):
    self.path = im_file
    print 'self.path=%s' % self.path
    # read image depending on file extension
    if self.path.endswith('.edf'):
      self.im = edf_read(im_file).transpose()
    elif self.path.endswith('.png'):
      self.im = mpimg.imread(im_file)
    elif self.path.endswith('.tif'):
      self.im = TiffFile(im_file).asarray()
    else:
      print('Only png, tif and edf images are supported for the moment')
      sys.exit(1)

if __name__ == "__main__":
  #ImageViewer(wdir = '/home/proudhon/data/tomo/rawdata/ma1921/Pb_balls_316LN_', pattern='.edf')
  ImageViewer(sys.argv)
