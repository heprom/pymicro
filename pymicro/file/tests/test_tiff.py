import unittest
import numpy as np
import os
from pymicro.file.file_utils import *
from pymicro.external.tifffile import imsave, imread
from config import PYMICRO_EXAMPLES_DATA_DIR


class file_utils_Tests(unittest.TestCase):

    def setUp(self):
        print('testing the file_utils module')
        self.data = (255 * np.random.rand(20, 30, 10)).astype(np.uint8)
        HST_write(self.data, 'temp_20x30x10_uint8.raw')
        data = np.zeros((3, 4, 5), dtype=bool)
        data[0, 1, 2] = True
        data[1, 2, 3] = True
        data[2, 3, 4] = True
        HST_write(data, 'test_bool_write.raw', pack_binary=True)
        HST_write(data, 'test_bool_write_as_uint8.raw', pack_binary=False)
        # craft a custom info file without the DATA_TYPE info
        f = open('temp.info', 'w')
        f.write('! PyHST_SLAVE VOLUME INFO FILE\n')
        f.write('NUM_X = 953\n')
        f.write('NUM_Y = 542\n')
        f.write('NUM_Z = 104\n')
        f.close()

    def test_edf_info(self):
        infos = edf_info(os.path.join(PYMICRO_EXAMPLES_DATA_DIR, 'sam8_dct0_cen_full0000.edf'))
        self.assertEqual(infos['DataType'], 'FloatValue')

    def test_edf_read(self):
        im = edf_read(os.path.join(PYMICRO_EXAMPLES_DATA_DIR, 'sam8_dct0_cen_full0000.edf'))
        self.assertEqual(im.shape[0], 2048)
        self.assertEqual(im.shape[1], 2048)

    def test_HST_info(self):
        infos = HST_info('temp_20x30x10_uint8.raw.info')
        self.assertEqual(infos['x_dim'], 20)
        self.assertEqual(infos['y_dim'], 30)
        self.assertEqual(infos['z_dim'], 10)
        infos = HST_info('temp.info')
        self.assertEqual(infos['x_dim'], 953)
        self.assertEqual(infos['y_dim'], 542)
        self.assertEqual(infos['z_dim'], 104)

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

    def test_write_append(self):
        n = 5
        HST_write(self.data, 'test_append.raw', mode='w')
        for i in range(n - 1):
            HST_write(self.data, 'test_append.raw', mode='a')
        size = os.path.getsize('test_append.raw')
        self.assertEqual(size, n * np.prod(self.data.shape))
        os.remove('test_append.raw')
        os.remove('test_append.raw.info')

    def test_pack_binary(self):
        infos = HST_info('test_bool_write.raw.info')
        self.assertEqual(infos['x_dim'], 3)
        self.assertEqual(infos['y_dim'], 4)
        self.assertEqual(infos['z_dim'], 5)
        self.assertEqual(infos['data_type'], 'PACKED_BINARY')
        bin_data = HST_read('test_bool_write.raw')
        self.assertEqual(bin_data.dtype, np.uint8)
        self.assertEqual(bin_data[0, 1, 2], True)
        self.assertEqual(bin_data[0, 1, 1], False)
        self.assertEqual(bin_data[1, 2, 3], True)
        self.assertEqual(bin_data[2, 3, 4], True)

    def test_write_bool_array(self):
        bin_data = HST_read('test_bool_write_as_uint8.raw')
        self.assertEqual(bin_data.dtype, np.uint8)
        size = os.path.getsize('test_bool_write_as_uint8.raw')
        self.assertEqual(size, 60)

    def tearDown(self):
        os.remove('temp_20x30x10_uint8.raw')
        os.remove('temp_20x30x10_uint8.raw.info')
        os.remove('test_bool_write_as_uint8.raw')
        os.remove('test_bool_write_as_uint8.raw.info')
        os.remove('test_bool_write.raw')
        os.remove('test_bool_write.raw.info')
        os.remove('temp.info')


class TiffTests(unittest.TestCase):
    def setUp(self):
        print('testing the Tifffile module')
        self.data = (255 * np.random.rand(5, 301, 219)).astype(np.uint8)
        imsave('temp.tif', self.data)

    def test_imread(self):
        image = imread('temp.tif')
        print('image size is', np.shape(image))
        np.testing.assert_array_equal(image, self.data)

    def tearDown(self):
        os.remove('temp.tif')


if __name__ == '__main__':
    unittest.main()
