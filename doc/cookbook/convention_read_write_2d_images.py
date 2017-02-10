from pymicro.file.file_utils import HST_read
import os, numpy as np
from matplotlib import pyplot as plt, cm

data_dir = '../../examples/data'
scan_name = 'mousse_250x250x250_uint8'
scan_path = os.path.join(data_dir, scan_name + '.raw')

vol = HST_read(scan_path, autoparse_filename=True, zrange=range(100, 101))
print 'shape vol:', np.shape(vol)
ext = 'png'  # choose here png or jpg for instance
print 'saving slice 100 as a %s image' % ext
plt.imsave('%s.%s' % (scan_name, ext), np.transpose(vol[:, :, 0]), cmap=cm.gray)

print 'now reading the 2d image'
im = np.transpose(plt.imread('%s.%s' % (scan_name, ext))[:, :, 0])
im_uint8 = (255 * im).astype(np.uint8)
print 'size of 2d image is:', im.shape
print 'in raw data: pixel value at [124,108] is %d, at [157,214] is %d' % (vol[124, 108, 0], vol[157, 214, 0])
print 'in 2d image: pixel value at [124,108] is %d, at [157,214] is %d' % (im_uint8[124, 108], im_uint8[157, 214])

print 'plotting'
fig = plt.figure(1, figsize=(15, 5))
fig.add_subplot(131)
plt.imshow(np.transpose(vol[:, :, 0]), cmap=cm.gray)
plt.title('100th slice of 3d volume (transposed)')
fig.add_subplot(132)
plt.imshow(im.T, cmap=cm.gray)
plt.title('2d %s image plotted (transposed)' % ext)
fig.add_subplot(133)
plt.plot(vol[:, 108, 0], label='raw data')
plt.plot(im_uint8[:, 108], label='%s image' % ext)
plt.xlabel('x coordinate')
plt.ylabel('data value')
plt.title('profile along y=108')
plt.legend()
plt.subplots_adjust(left=0.05, right=0.95)
plt.savefig('%s_2d_slices.png' % scan_name)
