import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.colors import colorConverter
import matplotlib as mpl


def hist(data, nb_bins=256, data_range=(0, 255), show=True, save=False, prefix='data', density=False):
    '''Histogram of a data array.

    Compute and plot the gray level histogram of the provided data array.

    *Parameters*

    **data**: the data array to analyse.

    **nb_bins**: the number of bins in the histogram.

    **data_range**: the data range to use in the histogram, (0,255) by default.

    **show**: boolean to display the figure using pyplot (defalut True).

    **save**: boolean to save the figure using pyplot (defalut False).

    **prefix**: a string to use in the file name when saving the histogram
    as an image (defaut 'data').

    **density**: a boolean to control wether the histogram is plotted
    using the probability density function, see numpy histogram function
    for more details (default False).

    .. figure:: _static/HDPE_0001-2_512x512x512_uint8_hist.png
        :width: 600 px
        :height: 400 px
        :alt: HDPE_0001-2_512x512x512_uint8_hist.png
        :align: center

        Gray level histogram computed on a 512x512x512 8 bits image.
    '''
    print 'computing gray level histogram'
    hist, bin_edges = np.histogram(data, bins=nb_bins, range=data_range, density=density)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.figure(figsize=(6, 4))
    if density:
        plt.bar(bin_centers, 100 * hist, width=1, fill=True, color='g', edgecolor='g')
        plt.ylabel('Probability (%)')
    else:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.bar(bin_centers, hist, width=1, fill=True, color='g', edgecolor='g')
        plt.ylabel('Counts')
    plt.xlim(data_range)
    plt.xlabel('Gray level value')
    if save: plt.savefig(prefix + '_hist.png', format='png')
    if show: plt.show()


def flat(img, ref, dark):
    '''Apply flat field correction to an image.

    :param np.array img: A 2D array representing the image to correct.
    :param np.array ref: The reference image (without the sample), same shape as the image to correct.
    :param np.array dark: A 2D numpy array representing the dark image (thermal noise of the camera).
    :returns np.array float: the flat field corrected image (between 0 and 1) as a float32 numpy array.
    '''
    flat = (img - dark).astype(np.float32) / (ref - dark).astype(np.float32)
    return flat


def auto_min_max(data, cut=0.0002, nb_bins=256, verbose=False):
    '''Compute the min and max values in a numpy data array.

    The min and max values are calculated based on the histogram of the
    array which is limited at both end by the cut parameter (0.0002 by
    default). This means 99.98% of the values are within the [min, max]
    range.

    :param data: the data array to analyse.
    :param float cut: the cut off value to use on the histogram (0.0002 by default).
    :param int nb_bins: number of bins to use in the histogram (256 by default).
    :param bool verbose: activate verbose mode (False by default).
    :returns tuple (min, max): a tuple containing the min and max values.
    '''
    n, bins = np.histogram(data, bins=nb_bins, density=False)
    min = 0.0
    max = 0.0
    total = np.sum(n)
    p = np.cumsum(n)
    for i in range(nb_bins):
        if p[i] > total * cut:
            min = bins[i]
            if verbose:
                print('min = %f (i=%d)' % (min, i))
            break
    for i in range(nb_bins):
        if total - p[nb_bins - 1 - i] > total * cut:
            max = bins[nb_bins - 1 - i]
            if verbose:
                print('max = %f' % max)
            break
    return min, max


def recad(data, min, max):
    '''Cast a numpy array into 8 bit data type.

    This function change the data type of a numpy array into 8 bit. The
    data values are interpolated from [min, max] to [0, 255]. Both min
    and max values may be chosen manually or computed by the function
    `find_min_max`.

    :param data: the data array to cast to uint8.
    :param float min: value to use as the minimum (will be 0 in the casted array).
    :param float max: value to use as the maximum (will be 255 in the casted array).
    :returns data_uint8: the data array casted to uint8.
    '''
    low_values_indices = data < min
    data[low_values_indices] = min
    large_values_indices = data > max
    data[large_values_indices] = max
    data_uint8 = (255 * (data - min) / (max - min)).astype(np.uint8)
    return data_uint8


def alpha_cmap(color='red', opacity=1.0):
    '''Creating a particular colormap with transparency.

    Only values equal to 255 will have a non zero alpha channel.
    This is typically used to overlay a binary result on initial data.

    :param color: the color to use for non transparent values (ie. 255).
    :param float opacity: opacity value to use for visible pixels.
    :returns mycmap: a fully transparent colormap except for 255 values.
    '''
    color1 = colorConverter.to_rgba('white')
    color2 = colorConverter.to_rgba(color)
    mycmap = mpl.colors.LinearSegmentedColormap.from_list('my_cmap', [color1, color2], 256)
    mycmap._init()
    alphas = np.zeros(mycmap.N + 3)
    alphas[255:] = opacity  # show only values at 255
    mycmap._lut[:, -1] = alphas
    return mycmap


def stitch(image_stack, nh=2, nv=1, pattern='E', hmove=None, vmove=None, adjust_bc=False, adjust_bc_nbins=256,
           verbose=False, show=True, save=False, save_name='stitch', save_ds=1, check=None):
    '''Stich a series of images together.
    
    :param int nh: number of images to stitch horizontally.
    :param int nv: number of images to stitch vertically.
    :param str pattern: stitching pattern ('E' or 'S').
    :param tuple hmove: horizontal move between 2 images (first image width by default).
    :param tuple vmove: vertical move between 2 images (first image height by default).
    :param bool adjust_bc: adjust gray levels by comparing the histograms in the overlapping regions (False by default).
    :param int adjust_bc_nbins: numbers of bins to use tin the histograms when adjusting the gray levels (256 by default).
    :param bool verbose: activate verbose mode (False by default).
    :param bool show: show the stitched image (True by default).
    :param bool save: flag to save the image as png to the disk (False by default).
    :param str save_name: name to use to save the stitched image (stitch by default).
    :param int save_ds: downsampling factor using when saving the stitched image (1 by default).
    :param int check: trigger plotting for the given image if set.
    :returns full_im: the stitched image in [x, y] form.
    '''
    assert pattern in ['E', 'S']
    # is motion not set, assume the full size of the first image
    if hmove is None:
        hmove = (image_stack[0].shape[0], 0)
    if vmove is None:
        vmove = (0, image_stack[0].shape[1])
    H = np.array(hmove).astype(int)
    V = np.array(vmove).astype(int)
    image_size = np.array(image_stack[0].shape)
    if adjust_bc:
        # make sure the images are overlapping
        assert image_size[0] > H[0]
        assert image_size[1] > V[1]
        bch = image_size[0] - H[0]  # horizontal overlap
        bcv = image_size[1] - V[1]  # vertical overlap
        if verbose:
            print('overlapping is {0} pixels horizontally and {1} pixels vertically'.format(bch, bcv))
    data_type = image_stack[0].dtype
    full_im = np.zeros((image_size[0] + (nh - 1) * H[0] + nv * V[0],
                        image_size[1] + (nv - 1) * V[1] + nh * H[1]),
                       dtype=data_type)
    if verbose:
        print('image_size', image_size)
        print('image data type', data_type)
        print('full image_size', full_im.shape)
    for i in range(len(image_stack)):
        # compute indices
        x = i % nh
        y = i // nh
        if pattern == 'S' and (y % 2) == 1:
            x = nh - 1 - x
        im = image_stack[i]
        xp, yp = x * H + y * V + image_size // 2
        if adjust_bc:
            if i == 0:
                # this is the reference
                pass
            else:
                # isolate the overlapping regions
                if i % nh != 0:
                    x1s = xp - image_size[0] // 2
                    x1e = xp - image_size[0] // 2 + bch
                    y1s = yp - image_size[1] // 2
                    y1e = yp - image_size[1] // 2 + image_size[1]
                    region1 = full_im[x1s:x1e, y1s:y1e]
                    region2 = im[:bch, :]
                else:
                    x1s = xp - image_size[0] // 2
                    x1e = xp - image_size[0] // 2 + image_size[0]
                    y1s = yp - image_size[1] // 2
                    y1e = yp - image_size[1] // 2 + bcv
                    region1 = full_im[x1s:x1e, y1s:y1e]
                    region2 = im[:, :bcv]
                if i == check:
                    fig = plt.figure()
                    ax1 = fig.add_subplot(121 if i % nh != 0 else 211)
                    im1 = ax1.imshow(region1.T, cmap=cm.gray, vmin=0, vmax=65535)
                    fig.colorbar(im1, ax=ax1)
                    ax1.set_title('region1')
                    ax2 = fig.add_subplot(122 if i % nh != 0 else 212)
                    im2 = ax2.imshow(region2.T, cmap=cm.gray, vmin=0, vmax=65535)
                    fig.colorbar(im2, ax=ax2)
                    ax2.set_title('region2')
                    plt.show()
                region1 = region1.flatten()
                region2 = region2.flatten()
                # work out the histograms
                data_type_min = np.iinfo(data_type).min
                data_type_max = np.iinfo(data_type).max
                hist1, bin_edges = np.histogram(region1, bins=adjust_bc_nbins, density=False,
                                                range=(data_type_min, data_type_max))
                hist2, bin_edges = np.histogram(region2, bins=adjust_bc_nbins, density=False,
                                                range=(data_type_min, data_type_max))
                if i == check:
                    hist(region1, data_range=(data_type_min, data_type_max), nb_bins=adjust_bc_nbins)
                    hist(region2, data_range=(data_type_min, data_type_max), nb_bins=adjust_bc_nbins)
                # compute the difference not taking into account the edges of the histograms
                max1 = np.argmax(hist1[1:-2])
                max2 = np.argmax(hist2[1:-2])
                diff = (data_type_max - data_type_min) / adjust_bc_nbins * (max1 - max2)
                if verbose:
                    print('matching max of histograms in both images: {0:d} and {1:d}, adding {2:d}'.format(max1, max2, diff))
                # keep a reference of saturated value (we do not want to change those)
                im_is_min = im == data_type_min
                im_is_max = im == data_type_max
                # shift all the gray levels, making sure not to overflow the image type range
                im = im.astype(float) + diff
                im[im < data_type_min] = data_type_min
                im[im > data_type_max] = data_type_max
                im = im.astype(data_type)
                # reset saturated values
                im[im_is_min] = data_type_min
                im[im_is_max] = data_type_max
        full_im[xp - image_size[0] // 2:xp - image_size[0] // 2 + image_size[0],
            yp - image_size[1] // 2:yp - image_size[1] // 2 + image_size[1]] = im
    if save:
        plt.imsave('{0:s}.png'.format(save_name), full_im[::save_ds, ::save_ds].T,
                   cmap=cm.gray)
    if show:
        plt.imshow(full_im[::save_ds, ::save_ds].T, interpolation='nearest', cmap=cm.gray)
        plt.axis('off')
        plt.show()
    return full_im

def compute_affine_transform(fixed, moving):
    '''Compute the affine transform by point set registration.

    The affine transform is the composition of a translation and a linear map.
    The two lists of points must be of the same length and the order of the points should match.

    :param list fixed: a list of the reference points.
    :param list moving: a list of the moving points to register on the fixed point.
    :returns translation, linear_map: the computed translation and linear map affine transform. 

    Thanks to Will Lenthe for helping with this code.
    '''
    assert len(fixed) == len(moving)
    fixed_centroid = np.average(fixed, 0)
    moving_centroid = np.average(moving, 0)
    # offset every point by the center of mass of all the points in the set
    fixed_from_centroid = fixed - fixed_centroid
    moving_from_centroid = moving - moving_centroid
    covariance = moving_from_centroid.T.dot(fixed_from_centroid)
    variance = moving_from_centroid.T.dot(moving_from_centroid)
    # compute the full affine transform: translation + linear map
    linear_map = np.linalg.inv(variance).dot(covariance).T
    translation = fixed_centroid - linear_map.dot(moving_centroid)
    return translation, linear_map


class AxShowPixelValue:
    '''A simple class that wraps a pyplot ax and modify its coordinate
    formatter to show the pixel value.'''

    def __init__(self, ax):
        '''Create a new AxShowPixelValue instance with a reference to the
        given pyplot ax.'''
        self.ax = ax
        self.ax.format_coord = self.format_coord

    def imshow(self, array, **kwargs):
        '''Wraps pyplot imshow function for the ax selected and modify
        the coordinate formatter to show the pixel value as well.
        All the usual parameters can be passed to imshow.
        '''
        self.array = array
        self.ax.imshow(self.array, **kwargs)

    def format_coord(self, x, y):
        '''A function to modify the coordinate formatter of pyplot to
        display the pixel value. This is usually called when moving the
        mouse above the plotted image.
        '''
        n_rows, n_cols = self.array.shape
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < n_cols and row >= 0 and row < n_rows:
            z = self.array[row, col]
            return 'x=%1.1f, y=%1.1f, z=%1.1f' % (x, y, z)
        else:
            return 'x=%1.1f, y=%1.1f' % (x, y)
