import unittest
import numpy as np
import os
from pymicro.file.tifffile import imsave, imread

class TiffTests(unittest.TestCase):

  def setUp(self):
    print 'testing the Tifffile module'
    self.data = (255*np.random.rand(5, 301, 219)).astype(np.uint8)
    imsave('temp.tif', self.data)

  def test_imread(self):
    image = imread('temp.tif')
    print 'image size is', np.shape(image)
    np.testing.assert_array_equal(image, self.data)

  def tearDown(self):
    os.remove('temp.tif')

if __name__ == '__main__':
  unittest.main()


