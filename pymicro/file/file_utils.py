import os, sys
import numpy as np
import struct


def read_image_sequence(data_dir, prefix, num_images, start_index=0, image_format='png', zero_padding=0, crop=None, verbose=False):
    '''Read a series of images into a list of numpy arrays.
    
    :param str data_dir: directory where the image files are located.
    :param str prefix: a string to construct the image file names.
    :param int num_images: the number of images to read.
    :param int start_index: the index to start loading the images (0 by default).
    :param str image_format: can be tif or png (png by default).
    :param int zero_padding: number of zero to use in zero padding (0 by default).
    :param list crop: bounds to crop the images (None by default) 
    :param bool verbose: activate verbose mode (False by default).
    :return: the list of the images read from the disk.
    '''
    # build the numbering pattern
    pat = '0%dd' % zero_padding
    fmt = '{0:s}{1:' + pat + '}.{2:s}'
    image_stack = []
    if image_format == 'tif':
        from pymicro.external.tifffile import TiffFile
    elif image_format == 'png':
        from matplotlib import pyplot as plt
    for i in range(start_index, start_index + num_images):
        image_path = os.path.join(data_dir, fmt.format(prefix, i, image_format))
        if verbose:
            print('loading image %s' % image_path)
        if image_format == 'tif':
            im = TiffFile(image_path).asarray()
        elif image_format == 'png':
            im = (np.mean(plt.imread(image_path)[:, :, :3], axis=2) * 255).astype(np.uint8)
        if crop:
            im = im[crop[2]:crop[3], crop[0]:crop[1]]
        image_stack.append(im.T)
    return image_stack


def unpack_header(header):
    """Unpack an ascii header.

    Form a string with the read binary data and then split it into string
    tokens which are put in a dictionnary.

    :param str header: the ascii header to unpack.
    :return: a dictionnary with the (key, value) fields contained in the header.
    """
    header_values = {}
    for line in header.split('\n'):
        tokens = line.split('=')
        if len(tokens) > 1:
            header_values[tokens[0].strip()] = tokens[1].split(';')[0].strip()
    return header_values


def edf_info(file_name, header_size=None, verbose=False):
    """Read and return informations contained in the header of a .edf file.

    Edf files always start with a header (of variable length) containing
    information about the file such as acquisition conditions, image
    dimensions... This function reads a certain amount of bytes of a given
    file as ascii data and unpack it.
    If not specified, the header size is determined automatically by
    substracting the data size (read as ascii at the begining of the file)
    to the total file size.

    :param str file_name: the name of the edf file to read.
    :param int header_size: number of bytes to read as a multiple of 512 (None by default).
    :param bool verbose: flag to activate verbose mode.
    :return: a dictionary containing the file information.
    """
    try:
        f = open(file_name, 'r', encoding='latin-1')
    except TypeError:
        f = open(file_name, 'r')  # fall back
    if header_size is None:
        # guess the header size by peeking at the first chunk of 512 bytes
        header_values = unpack_header(f.read(512))
        total_file_size = os.path.getsize(file_name)
        payload_size = int(header_values['Size'].split('.')[0])
        header_size = total_file_size - payload_size
        if verbose:
            print('determined header size is %d bytes' % header_size)
        f.seek(0)
    header = f.read(header_size)
    f.close()
    return unpack_header(header)


def edf_read(file_name, verbose=False):
    """Read an edf file.

    edf stands for ESRF data file. It has a variable header size which is
    a multiple of 512 bytes and contains the image meta in ASCII format
    (eg. image size, data type, motor positions).

    The ascii header is parsed automatically by `edf_info` to retreive the
    image size and data type. Depending on the information enclosed in the
    header, this function may return a 1d, 2d or 3d array.
    ::

      >>> im = edf_read('radio.edf')
      >>> im.shape
      (2048, 2048)

    :param str file_name: the name of the edf file to read.
    :param bool verbose: flag to activate verbose mode.
    :return: a numpy array containing the data
    """
    header_values = edf_info(file_name, verbose=verbose)
    f = open(file_name, 'r')
    data_type = esrf_to_numpy_datatype(header_values['DataType'])
    if verbose:
        print(header_values['DataType'], data_type)
    # get the payload size
    payload_size = int(header_values['Size'].split('.')[0])
    # get the image size from the ascii header
    dim_1 = int(header_values['Dim_1'].split('.')[0])
    try:
        dim_2 = int(header_values['Dim_2'].split('.')[0])
    except KeyError:
        if verbose:
            print('Dim_2 not defined in header')
        dim_2 = None
    try:
        dim_3 = int(header_values['Dim_3'].split('.')[0])
    except KeyError:
        if verbose:
            print('Dim_3 not defined in header')
        dim_3 = None
    # now read binary data
    header_size = os.path.getsize(file_name) - payload_size
    f.seek(header_size)
    payload = np.fromfile(f, dtype=data_type)
    if dim_1 and dim_2 and dim_3:
        data = np.reshape(payload, (dim_3, dim_2, dim_1)).transpose(2, 1, 0)
    elif dim_1 and dim_2:
        data = np.reshape(payload, (dim_2, dim_1)).transpose(1, 0)
    else:
        data = np.reshape(payload, (dim_1))
    f.close()
    # pay attention to byte order
    if header_values['ByteOrder'] == 'HighByteFirst':
        data = data.byteswap()
    return data


def esrf_to_numpy_datatype(data_type):
    return {
        'UnsignedByte': np.uint8,
        'UnsignedShort': np.uint16,
        'UnsignedLong': np.uint32,
        'SignedInteger': np.int32,
        'FloatValue': np.float32,
        'DoubleValue': np.float64,
    }.get(data_type, np.uint16)


def numpy_to_esrf_datatype(data_type):
    return {
        np.uint8: 'UnsignedByte',
        np.uint16: 'UnsignedShort',
        np.uint32: 'UnsignedLong',
        np.int32: 'SignedInteger',
        np.float32: 'FloatValue',
        np.float64: 'DoubleValue',
        np.dtype('uint8'): 'UnsignedByte',
        np.dtype('uint16'): 'UnsignedShort',
        np.dtype('uint32'): 'UnsignedLong',
        np.dtype('int32'): 'SignedInteger',
        np.dtype('float32'): 'FloatValue',
        np.dtype('float64'): 'DoubleValue',
        float: 'DoubleValue',
    }.get(data_type, 'UnsignedShort')


def edf_write(data, file_name, header_size=1024):
    """Write a binary edf file with the appropriate header.

    This function write a (x,y,z) 3D dataset to the disk.
    The file is written as a Z-stack. It means that the first nx*ny bytes
    represent the first slice and so on...

    :param ndarray data: the data array to write to the file.
    :param str file_name: the file name to use.
    :param int header_size: the size of te header (a multiple of 512).
    """
    # get current time
    from time import gmtime, strftime
    today = strftime('%d-%b-%Y', gmtime())
    size = np.shape(data)
    print('data size in pixels is ', size)
    nbytes = np.prod(size) * data.dtype.itemsize
    print('opening', file_name, 'for writing')
    # craft an ascii header of the appropriate size
    f = open(file_name, 'wb')
    head = '{\n'
    head += 'HeaderID       = EH:000001:000000:000000 ;\n'
    head += 'Image          = 1 ;\n'
    head += 'ByteOrder      = LowByteFirst ;\n'
    head += 'DataType       = %13s;\n' % numpy_to_esrf_datatype(data.dtype)
    print('using data type %s' % numpy_to_esrf_datatype(data.dtype))
    head += 'Dim_1          = %4s;\n' % size[0]
    if len(size) > 1: head += 'Dim_2          = %4s;\n' % size[1]
    if len(size) > 2: head += 'Dim_3          = %4s;\n' % size[2]
    head += 'Size           = %9s;\n' % nbytes
    head += 'Date           = ' + today + ' ;\n'
    for i in range(header_size - len(head) - 2):
        head += ' '
    head += '}\n'
    f.write(head.encode('utf-8'))
    if len(data.shape) == 3:
        s = np.ravel(data.transpose(2, 1, 0)).tostring()
    elif len(data.shape) == 2:
        s = np.ravel(data.transpose(1, 0)).tostring()
    else:
        s = np.ravel(data).tostring()
    f.write(s)
    f.close()


def HST_info(info_file):
    """Read the given info file and returns a dictionary containing the data size and type.
    
    .. note::
    
       The first line of the file must begin by ! PyHST or directly by NUM_X. 
       Also note that if the data type is not specified, it will not be present in the dictionary.
    
    :param str info_file: path to the ascii file to read.
    :return: a dictionary with the values for x_dim, y_dim, z_dim and data_type if needed.
    """
    info_values = {}
    f = open(info_file, 'r')
    # the first line must contain PyHST or NUM_X
    line = f.readline()
    if line.startswith('! PyHST'):
        # read an extra line
        line = f.readline()
    elif line.startswith('NUM_X'):
        pass
    else:
        sys.exit('The file does not seem to be a PyHST info file')
    info_values['x_dim'] = int(line.split()[2])
    info_values['y_dim'] = int(f.readline().split()[2])
    info_values['z_dim'] = int(f.readline().split()[2])
    try:
        info_values['data_type'] = f.readline().split()[2]
    except IndexError:
        pass
    return info_values


def HST_read(scan_name, zrange=None, data_type=np.uint8, verbose=False,
             header_size=0, autoparse_filename=False, dims=None, mmap=False, pack_binary=False):
    '''Read a volume file stored as a concatenated stack of binary images.

    The volume size must be specified by dims=(nx, ny, nz) unless an associated
    .info file is present in the same location to determine the volume
    size. The data type is unsigned short (8 bits) by default but can be set
    to any numpy type (32 bits float for example).

    The autoparse_filename can be activated to retreive image type and
    size:
    ::

      HST_read(myvol_100x200x50_uint16.raw, autoparse_filename=True)

    will read the 3d image as unsigned 16 bits with size 100 x 200 x 50.

    .. note::

      If you use this function to read a .edf file written by
      matlab in +y+x+z convention (column major order), you may want to
      use: np.swapaxes(HST_read('file.edf', ...), 0, 1)

    :param str scan_name: path to the binary file to read.
    :param zrange: range of slices to use.
    :param data_type: numpy data type to use.
    :param bool verbose: flag to activate verbose mode.
    :param int header_size: number of bytes to skeep before reading the payload.
    :param bool autoparse_filename: flag to parse the file name to retreive the dims and data_type automatically.
    :param tuple dims: a tuple containing the array dimensions.
    :param bool mmap: activate the memory mapping mode.
    :param bool pack_binary: this flag should be true when reading a file written with the binary packing mode.
    '''
    if autoparse_filename:
        s_type = scan_name[:-4].split('_')[-1]
        data_type = np.dtype(s_type)
        s_size = scan_name[:-4].split('_')[-2].split('x')
        dims = (int(s_size[0]), int(s_size[1]), int(s_size[2]))
        if verbose:
            print('auto parsing filename: data type is set to', data_type)
    if dims is None:
        infos = HST_info(scan_name + '.info')
        [nx, ny, nz] = [infos['x_dim'], infos['y_dim'], infos['z_dim']]
        if 'data_type' in infos:
            if infos['data_type'] == 'PACKED_BINARY':
                pack_binary = True
                data_type = np.uint8
            else:
                data_type = np.dtype(infos['data_type'].lower())  # overwrite defaults with .info file value
    else:
        (nx, ny, nz) = dims
    if zrange is None:
        zrange = range(0, nz)
    if verbose:
        print('data type is', data_type)
        print('volume size is %d x %d x %d' % (nx, ny, len(zrange)))
        if pack_binary:
            print('unpacking binary data from single bytes (8 values per byte)')
    if mmap:
        data = np.memmap(scan_name, dtype=data_type, mode='c', shape=(len(zrange), ny, nx))
    else:
        f = open(scan_name, 'rb')
        f.seek(header_size + np.dtype(data_type).itemsize * nx * ny * zrange[0])
        if verbose:
            print('reading volume... from byte %d' % f.tell())
        # read the payload
        payload = f.read(np.dtype(data_type).itemsize * len(zrange) * ny * nx)
        if pack_binary:
            data = np.unpackbits(np.fromstring(payload, data_type))[:len(zrange) * ny * nx]
        else:
            data = np.fromstring(payload, data_type)
        # convert the payload into actual 3D data
        data = np.reshape(data.astype(data_type), (len(zrange), ny, nx), order='C')
        f.close()
    # HP 10/2013 start using proper [x,y,z] data ordering
    data_xyz = data.transpose(2, 1, 0)
    return data_xyz


def rawmar_read(image_name, size, verbose=False):
    '''Read a square 2D image plate MAR image.

    These binary images are typically obtained from the marcvt utility.

    .. note::

       This method assume Big endian byte order.
    '''
    data = HST_read(image_name, dims=(1, size, size), header=4600,
                    data_type=np.uint16, verbose=verbose)[:, :, 0]
    return data


def HST_write(data, file_name, mode='w', verbose=True, pack_binary=False):
    '''Write data as a raw binary file.

    This function write a (x,y,z) 3D dataset to the disk. The actual data type is used, you can convert your data array 
    on the fly using data.astype if you want to change the type. 
    The file is written as a Z-stack. It means that the first nx*ny bytes written represent the first slice and so on...
    For binary data files (stored in memory as integer or bool data type), binary packing mode can be activated which 
    stores 8 values on each byte (saving 7/8 of the disk space).

    A .info file containing the volume size and data type is also written.

    :param data: the 3d array to write to the disk in [x, y, z] form.
    :param str file_name: the name of the file to write, including file extension.
    :param char mode: file write mode, change to 'a' to append to a file.
    :param bool verbose: flag to activate verbose mode.
    :param bool pack_binary: flag to activate binary packing.
    '''
    if data.dtype == np.bool:
        print('casting bool array to uint8, you may consider using binary packing to save disk space.')
        data = data.astype(np.uint8)
    (nx, ny, nz) = data.shape
    if verbose:
        print('opening %s for writing in mode %s' % (file_name, mode))
        print('volume size is %dx%dx%d' % (nx, ny, nz))
        print('data type is %s' % data.dtype)
    f = open(file_name, mode + 'b')
    # HP 11/2013 swap axes according to read function
    if pack_binary:
        s = np.packbits(data.astype(np.uint8).transpose(2, 1, 0)).tostring()
    else:
        s = np.ravel(data.transpose(2, 1, 0)).tostring()
    f.write(s)
    f.close()
    if verbose:
        print('writing .info file')
    f = open(file_name + '.info', mode)
    f.write('! PyHST_SLAVE VOLUME INFO FILE\n')
    f.write('NUM_X = %4d\n' % nx)
    f.write('NUM_Y = %4d\n' % ny)
    f.write('NUM_Z = %4d\n' % nz)
    if pack_binary:
        f.write('DATA_TYPE = PACKED_BINARY\n')
    else:
        f.write('DATA_TYPE = %s\n' % data.dtype)
    f.close()
    if verbose:
        print('done with writing')


def recad_vol(vol_filename, min, max, verbose=False):
    '''Recad a 32 bit vol file into 8 bit raw file.

    This function reads a 3D volume file into a numpy float32 array and
    applies the `recad` function with the [min, max] range. The result is
    saved into a .raw file with the same name as the input file.

    In verbose mode, a piture to compare mid slices in the two volumes
    and another one to compare the histograms are saved.

    .. note::

       To read the vol file, the presence of a .info file is
       assumed, see `HST_read`.

    *Parameters*

    **vol_filename**: the path to the binary vol file.

    **min**: value to use as the minimum (will be 0 in the casted array).

    **max**: value to use as the maximum (will be 255 in the casted array).

    **verbose**: activate verbose mode (False by default).
    '''
    prefix = vol_filename[:-4]
    infos = HST_info(vol_filename + '.info')
    vol_size = [infos['x_dim'], infos['y_dim'], infos['z_dim']]
    data = HST_read(vol_filename, type=np.float32)
    data_uint8 = recad(data, min, max)
    if verbose:
        plt.figure(1, figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(data[:, :, vol_size[2] // 2])
        plt.title('float image')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(data_uint8[:, :, vol_size[2] // 2])
        plt.title('uint8 image')
        plt.axis('off')
        plt.savefig('%sslices.pdf' % (prefix), format='pdf')
        plt.figure(2)
        plt.clf()
        plt.subplot(211)
        plt.title('Gray level histograms from float to uint8')
        n, bins, patches = plt.hist(data.ravel(), bins=256, histtype='stepfilled', facecolor='green')
        plt.figure(2)
        plt.subplot(212)
        plt.hist(data_uint8.ravel(), bins=256, histtype='stepfilled', facecolor='green')
        plt.savefig('%shist.pdf' % (prefix), format='pdf')
    HST_write(data_uint8, '%s.raw' % prefix)


def Vtk_write(data, fname):
    '''Write a data array into old style (V3.0) VTK format.

    An ascii header is written to which the binary data is appended.

    .. note::

       The header assumes uint8 data type.
    '''
    (nz, ny, nx) = data.shape
    print('opening', fname, 'for writing')
    print('volume size is %d x %d x %d' % (nx, ny, nz))
    # write header
    f = open(fname, 'w')
    f.write('# vtk DataFile Version3.0\n')
    f.write(fname[:-4] + '\n')
    f.write('BINARY\n');
    f.write('DATASET STRUCTURED_POINTS\n');
    f.write('DIMENSIONS ' + str(nx) + ' ' + str(ny) + ' ' + str(nz) + '\n');
    f.write('SPACING 1.0 1.0 1.0\n');
    f.write('ORIGIN 0.0 0.0 0.0\n');
    f.write('POINT_DATA ' + str(nx * ny * nz) + '\n');
    f.write('SCALARS volume_scalars unsigned_char 1\n');
    f.write('LOOKUP_TABLE default\n')
    f.close()
    # append binary data
    f = open(fname, 'ab')
    s = np.ravel(data).tostring()
    f.write(s)
    f.close()
    print('done with writing')
