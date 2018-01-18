import unittest
import numpy as np
import os
from pymicro.file.file_utils import *
from pymicro.external.tifffile import imsave, imread


class file_utils_Tests(unittest.TestCase):
    def setUp(self):
        print 'testing the file_utils module'
        self.data = (255 * np.random.rand(20, 30, 10)).astype(np.uint8)
        HST_write(self.data, 'temp_20x30x10_uint8.raw')

    def test_HST_info(self):
        infos = HST_info('temp_20x30x10_uint8.raw.info')
        self.assertEqual(infos['x_dim'], 20)
        self.assertEqual(infos['y_dim'], 30)
        self.assertEqual(infos['z_dim'], 10)

    def test_HST_read(self):
        data = HST_read('temp_20x30x10_uint8.raw', autoparse_filename=True)
        self.assertEqual(data.shape[0], 20)
        self.assertEqual(data.shape[1], 30)
        self.assertEqual(data.shape[2], 10)

    def test_HST_read_mmap(self):
        data = HST_read('temp_20x30x10_uint8.raw', autoparse_filename=True, mmap=True)
        self.assertEqual(data.shape[0], 20)
        self.assertEqual(data.shape[1], 30)
        self.assertEqual(data.shape[2], 10)

    def tearDown(self):
        os.remove('temp_20x30x10_uint8.raw')
        os.remove('temp_20x30x10_uint8.raw.info')


class TiffTests(unittest.TestCase):
    def setUp(self):
        print 'testing the Tifffile module'
        self.data = (255 * np.random.rand(5, 301, 219)).astype(np.uint8)
        imsave('temp.tif', self.data)

    def test_imread(self):
        image = imread('temp.tif')
        print 'image size is', np.shape(image)
        np.testing.assert_array_equal(image, self.data)

    def tearDown(self):
        os.remove('temp.tif')


if __name__ == '__main__':
    unittest.main()
