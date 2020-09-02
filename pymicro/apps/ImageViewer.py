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

from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QHBoxLayout, QVBoxLayout, QComboBox, QLabel, QCheckBox
from PyQt5.QtCore import Qt

from pymicro.file.file_utils import edf_read
from pymicro.external.tifffile import TiffFile
from pymicro.view.vol_utils import hist


class PlotWidget(QWidget):
    def __init__(self, parent=None, image=None, toolbar=True):
        QWidget.__init__(self, parent)
        self.data = image
        self.fliplr = False
        self.flipud = False
        self.dpi = 100
        self.cmap = 'gray'
        self.toolbar = toolbar
        self.create_main_widget()
        if self.data is not None:
            self.on_draw()

    def create_main_widget(self):
        print(self.data)
        self.fig = Figure((10.0, 8.0), dpi=self.dpi)
        self.axes = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()
        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)  # the matplotlib canvas
        if self.toolbar:
            self.mpl_toolbar = NavigationToolbar(self.canvas, self)
            vbox.addWidget(self.mpl_toolbar)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.setLayout(vbox)

    def set_image(self, image):
        self.data = image
        self.on_draw()

    def set_fliplr(self, state):
        self.fliplr = state
        self.on_draw()

    def set_flipud(self, state):
        self.flipud = state
        self.on_draw()

    def on_key_press(self, event):
        if event.key == 'i':
            print('* image infos:')
            print('shape is (%dx%d)' % (np.shape(self.data)[0], np.shape(self.data)[1]))
            print('min in image= %d, max in image=%d' % (np.min(self.data), np.max(self.data)))
        elif event.key == 'right':
            # load next image
            self.parent().parent().on_next_image()
        elif event.key == 'left':
            # load previous image
            self.parent().parent().on_prev_image()
        elif event.key == 'h':
            # plot the image histogram
            hist(self.data, data_range=(np.min(self.data), np.max(self.data)), show=True)
        # implement the default mpl key press events described at
        # http://matplotlib.org/users/navigation_toolbar.html#navigation-keyboard-shortcuts
        key_press_handler(event, self.canvas, self.mpl_toolbar)

    def on_draw(self):
        # andle image flips
        data = self.data
        if self.fliplr:
            data = np.fliplr(data)
        if self.flipud:
            data = np.flipud(data)
        self.axes.imshow(data, cmap=self.cmap, origin='upper', interpolation='nearest',
                         clim=[np.min(self.data), np.max(self.data)])
        self.canvas.draw()


class ImageViewerForm(QMainWindow):
    def __init__(self, parent=None, images=None, image_type='path'):
        QMainWindow.__init__(self, parent)
        self.image_type = image_type  # can be either path or data
        self.images = images
        self.index = 0
        self.create_main_frame(self.get_image_data())
        self.plot_widget.on_draw()

    def get_image_number(self):
        if self.image_type == 'path':
            return len(self.images)
        elif self.image_type == 'data':
            return self.images.shape[2]

    def create_main_frame(self, image):
        print('create_main_frame')
        self.main_frame = QWidget()
        vbox = QVBoxLayout()

        # create the plot widget
        self.plot_widget = PlotWidget(image=image, parent=self, toolbar=True)
        vbox.addWidget(self.plot_widget)

        # create the flip check boxes bar
        hbox_flips = QHBoxLayout()
        cblr = QCheckBox('Flip L/R', self)
        cblr.stateChanged.connect(self.plot_widget.set_fliplr)
        hbox_flips.addWidget(cblr)
        cbud = QCheckBox('Flip U/D', self)
        hbox_flips.addWidget(cbud)
        cbud.stateChanged.connect(self.plot_widget.set_flipud)
        vbox.addLayout(hbox_flips)

        # create the color map bar
        hbox = QHBoxLayout()
        self.cmap_label = QLabel('Image color map')
        hbox.addWidget(self.cmap_label)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItem('gray')
        self.cmap_combo.addItem('jet')
        self.cmap_combo.addItem('hot')
        self.cmap_combo.activated[str].connect(self.on_cmap_selected)
        hbox.addWidget(self.cmap_combo)
        vbox.addLayout(hbox)
        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

    def on_cmap_selected(self, text):
        self.plot_widget.cmap = str(text)
        self.plot_widget.on_draw()

    def on_prev_image(self):
        self.index -= 1
        if self.index < 0: self.index += self.get_image_number()
        # get the previous image
        data = self.get_image_data()
        self.plot_widget.set_image(data)

    def on_next_image(self):
        self.index = (self.index + 1) % self.get_image_number()
        # get the next image
        data = self.get_image_data()
        self.plot_widget.set_image(data)

    def get_image_data(self):
        if self.image_type == 'path':
            data = self.load_image(self.images[self.index])
        elif self.image_type == 'data':
            data = self.images[:, :, self.index].T
        return data

    def load_image(self, image_path):
        print('loading image from %s' % image_path)
        # read image depending on file extension
        if image_path.endswith('.edf'):
            image = edf_read(image_path).transpose()
        elif image_path.endswith('.png'):
            image = mpimg.imread(image_path)
        elif image_path.endswith('.tif'):
            image = TiffFile(image_path).asarray()  # should we transpose here?
        else:
            print('Only png, tif and edf images are supported for the moment')
            image = np.zeros((5, 5), dtype=np.uint8)  # dummy image
        return image


class ImageViewer(QApplication):
    def __init__(self, args):
        '''Init the ImageViewer application.'''
        app = QApplication(['Pymicro Image Viewer'])
        # print(args)
        print(type(args[0]))
        # check if first arg is a numpy array
        if type(args) == np.ndarray:  # this may be replaced by a 'data' keyword...
            form = ImageViewerForm(images=args, image_type='data')
        else:
            # check if first arg contains '='
            if args[0].find('=') < 0:
                start = 1  # app launched from the command line, first arg is just the name of the file
            else:
                start = 0
            # parse the list of args into a dictionary
            d = dict(map(lambda x: x.split('='), args[start:]))
            print('received args:', d)
            if 'wdir' in d:
                self.wdir = d['wdir']
            else:
                self.wdir = '.'
            if 'pattern' in d:
                pattern = d['pattern']
            else:
                pattern = '.png$'
            image_list = []
            file_list = os.listdir(self.wdir)
            file_list.sort()
            for f in file_list:
                if re.search(pattern, f):
                    image_list.append(os.path.join(self.wdir, f))
            if not image_list:
                print('No image found, please verify your pattern or your working directory')
                sys.exit(1)
            form = ImageViewerForm(images=image_list, image_type='path')
        form.show()
        app.exec_()


if __name__ == "__main__":
    ImageViewer(['wdir=/home/proudhon/data/tomo/rawdata/ma1921/Pb_balls_316LN_', 'pattern=.edf'])
    #ImageViewer(sys.argv)
    # a = np.zeros((100, 50, 10))
    # a[50, :, 1] = 1
    # ImageViewer(a)
