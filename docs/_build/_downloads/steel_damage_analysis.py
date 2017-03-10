import os, numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt, cm
from pymicro.file.file_utils import HST_read, HST_write, HST_info

data_dir = '../../examples/data'
scan_name = 'steel_431x431x246_uint8'
scan_path = os.path.join(data_dir, scan_name + '.raw')

print('reading volume...')
data = HST_read(scan_path, header_size=0)
plt.figure(1, figsize=(10, 5))
plt.subplot(121)
plt.imshow(data[:, :, 87].transpose(), interpolation='nearest', cmap=cm.gray)
print('rotating volume...')
data = ndimage.rotate(data, 15.5, axes=(1, 0), reshape=False)
plt.subplot(122)
plt.imshow(data[:, :, 87].transpose(), interpolation='nearest', cmap=cm.gray)
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
plt.savefig(scan_name + '_data.png')

print('binarizing volume...')
data_bin = np.where(np.greater(data, 100), 0, 255).astype(np.uint8)

print('labeling cavities...')
label_im, nb_labels = ndimage.label(data_bin)

plt.figure(2, figsize=(10, 5))
plt.subplot(121)
plt.imshow(data_bin[:, :, 87].transpose(), interpolation='nearest', cmap=cm.gray)
plt.subplot(122)
plt.imshow(label_im[:, :, 87].transpose(), interpolation='nearest', cmap=cm.jet)
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
plt.savefig(scan_name + '_label.png')

print('nb of labels: %s' % nb_labels)
sizes = ndimage.sum(data_bin / 255, label_im, range(1, nb_labels + 1))

# simple ring removal artifact
coms = ndimage.measurements.center_of_mass(data_bin / 255, labels=label_im, \
                                           index=range(1, nb_labels + 1))
rings = 0
for i in range(nb_labels):
    com_x = round(coms[i][0])
    com_y = round(coms[i][1])
    com_z = round(coms[i][2])
    if not label_im[com_x, com_y, com_z] == i + 1:
        print 'likely found a ring artifact at (%d, %d, %d) for label = %d, value is %d' \
              % (com_x, com_y, com_z, (i + 1), label_im[com_x, com_y, com_z])
        data_bin[label_im == (i + 1)] = 0
        rings += 1
print 'removed %d rings artifacts' % rings

print('labeling and using a fixed color around the specimen')
# the outside is by far the largest label here
mask_outside = (sizes >= ndimage.maximum(sizes))
print('inverting the image so that outside is now 1')
data_inv = 1 - label_im
outside = mask_outside[data_inv]
outside = ndimage.binary_closing(outside, iterations=3)
# fix the image border
outside[0:3, :, :] = 1;
outside[-3:, :, :] = 1;
outside[:, 0:3, :] = 1;
outside[:, -3:, :] = 1;
outside[:, :, 0:3] = 1;
outside[:, :, -3:] = 1
data_bin[outside] = 155

plt.figure(3, figsize=(10, 5))
plt.subplot(121)
plt.imshow(data_inv[:, :, 87].transpose(), interpolation='nearest', cmap=cm.gray)
plt.clim(0, 1)
plt.subplot(122)
plt.imshow(data_bin[:, :, 87].transpose(), interpolation='nearest', cmap=cm.gray)
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
plt.savefig(scan_name + '_outside.png')

print('saving data...')
HST_write(data_bin, scan_name + '_bin.raw')
