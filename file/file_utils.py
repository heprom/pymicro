import os, sys
import numpy as np
import struct

def unpack_header(h):
  '''Unpack an ascii header.
  
  Form a string with the read binary data and then split it into string 
  tokens which are put in a dictionnary.
  '''
  nbytes = len(h)
  nbits = 8 * nbytes
  s = struct.unpack(str(nbits) + 'c', h)
  tmp_header = ''
  for i in range(nbits):
    tmp_header += s[i]
  header_values = {}
  for line in tmp_header.split('\n'):
    tokens = line.split('=')
    if len(tokens)>1:
      header_values[tokens[0].strip()] = tokens[1].split(';')[0].strip()
  return header_values
  
def edf_info(file_name, header_size=None):
  '''Read and return informations contained in the header of a .edf file.
  
  Edf files always start with a header (of variable length) containing 
  informations about the file such as acquisition conditions, image 
  dimensions... This function reads a certain amount of bytes of a given 
  file as ascii data and unpack it.
  If not specified, the header size is determined automatically by 
  substracting the data size (read as ascii at the begining of the file) 
  to the total file size.

  *Parameters*

  **header_size**: number of bytes to read (None by default).
  
  Returns a dictionnary containing the file informations.
  '''
  f = open(file_name, 'r')
  if header_size == None:
    # guess the header size by peeking at the first chunk of 512 bytes
    h = np.fromstring(f.read(512))
    header_values = unpack_header(h)
    total_file_size = os.path.getsize(file_name)
    payload_size = int(header_values['Size'])
    header_size = total_file_size - payload_size
    print 'determined header size is:', header_size
    f.seek(0)
  h = np.fromstring(f.read(header_size))
  f.close()
  return unpack_header(h)  
    
def edf_read(file_name):
  '''Read an edf file.

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

  *Parameters*

  **file_name**: the name of the edf file to read.
  
  '''
  header_values = edf_info(file_name)
  f = open(file_name, 'r')
  data_type = esrf_to_numpy_datatype(header_values['DataType'])
  # get the payload size
  payload_size = int(header_values['Size'])
  # get the image size from the ascii header
  dim_1 = int(header_values['Dim_1'].split('.')[0])
  try:
    dim_2 = int(header_values['Dim_2'].split('.')[0])
    dim_3 = int(header_values['Dim_3'].split('.')[0])
  except:
    pass
  # now read binary data
  header_size = os.path.getsize(file_name) - payload_size
  f.seek(header_size)
  payload = np.fromstring(f.read(payload_size), data_type)
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
        'FloatValue': np.float32,
        'DoubleValue': np.float64,
        }.get(data_type, np.uint16)

def numpy_to_esrf_datatype(data_type):
    return {
        np.uint8: 'UnsignedByte',
        np.uint16: 'UnsignedShort',
        np.uint32: 'UnsignedLong',
        np.float32: 'FloatValue',
        np.float64: 'DoubleValue',
        }.get(data_type, 'UnsignedShort')

def edf_write(data, fname, type=np.uint16, header_size=1024):
  '''Write a binary edf file with the appropriate header.
  
  This function write a (x,y,z) 3D dataset to the disk.
  The file is written as a Z-stack. It means that the first nx*ny bytes 
  represent the first slice and so on...
  '''
  # get current time
  from time import gmtime, strftime
  today = strftime('%d-%b-%Y', gmtime())
  size = np.shape(data)
  print 'data size in pixels is ', size
  nbytes = np.prod(size) * np.dtype(type).itemsize
  print 'opening',fname,'for writing'
  # craft an ascii header of the appropriate size
  f = open(fname, 'wb')
  head = '{\n'
  head += 'HeaderID       = EH:000001:000000:000000 ;\n'
  head += 'Image          = 1 ;\n'
  head += 'ByteOrder      = LowByteFirst ;\n'
  head += 'DataType       = %13s;\n' % numpy_to_esrf_datatype(type)
  head += 'Dim_1          = %4s;\n' % size[0]
  if len(size) > 1: head += 'Dim_2          = %4s;\n' % size[1]
  if len(size) > 2: head += 'Dim_3          = %4s;\n' % size[2]
  head += 'Size           = %9s;\n' % nbytes
  head += 'Date           = ' + today + ' ;\n'
  for i in range(header_size - len(head) - 2):
    head += ' '
  head += '}\n'
  f.write(head)
  s = np.ravel(data.transpose(2,1,0)).astype(type).tostring()
  f.write(s)
  f.close()

def HST_info(info_file):
    '''Read the given info file and returns the volume size.
    
    Note that the first line of the file must begin by ! PyHST
    or directly by NUM_X.
    '''
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
    x_dim = int(line.split()[2])
    y_dim = int(f.readline().split()[2])
    z_dim = int(f.readline().split()[2])
    return [x_dim, y_dim, z_dim]

def HST_read(scan_name, zrange=None, data_type=np.uint8, verbose=False, \
  header_size=0, dims=None):
  '''Read a volume file stored as a concatenated stack of binary images.
  
  The volume size must be specified by dims=(nx,ny,nz) unless an associated 
  .info file is present in the same location to determine the volume 
  size. The data type is unsigned short (8 bits) by default but can be set 
  to any numpy type (32 bits float for example).
  
  The autoparse_filename can be activated to retreive image type and 
  size:
  ::
  
    HST_read(myvol_100x200x50_uint16.raw, autoparse_filename=True)

  will read the 3d image as unsigned 16 bits with size 100 x 200 x 50.

  ..note:: if you use this function to read a .edf file written by 
    matlab in +y+x+z convention (column major order), you may want to 
    use: np.swapaxes(HST_read('file.edf', ...), 0, 1)

  '''
  if autoparse_filename == True:
    s_type = file_name[:-4].split('_')[-1]
    if s_type == 'uint8':
      data_type = np.uint8
    elif s_type == 'uint16':
      data_type = np.uint16
    s_size = file_name[:-4].split('_')[-2].split('x')
    (dim_1, dim_2, dim_3) = (int(s_size[0]), int(s_size[1]), int(s_size[2]))
    if verbose: print 'autoparsing filename: data type is set to', type
  if verbose: print 'data type is',type
  if dims == None:
    [nx, ny, nz] = HST_info(scan_name + '.info')
  else:
    (nx, ny, nz) = dims
  if zrange == None:
      zrange = range(0, nz)
  if verbose: print 'volume size is ', nx, 'x', ny, 'x', len(zrange)
  f = open(scan_name, 'rb')
  data = np.empty((ny, nx, len(zrange)), dtype=type)
  if verbose: print 'reading volume... from byte ',f.tell()
  f.seek(header_size + np.dtype(data_type).itemsize * nx * ny * zrange[0])
  data = np.reshape(np.fromstring( \
      f.read(np.dtype(data_type).itemsize * len(zrange) * ny * nx), \
      data_type).astype(data_type), (len(zrange), ny, nx), order='C')
  f.close()
  # HP 10/2013 start using proper [x,y,z] data ordering
  data_xyz = data.transpose(2,1,0)
  return data_xyz

def rawmar_read(image_name, size, verbose=False):
  '''Read a square 2D image plate MAR image.
  
  These binary images are typically obtained from the marcvt utility.

  ..note:: This method assume Big endian byte order.
  '''
  data = HST_read(image_name, dims = (1, size, size), header=4600, \
    type=np.uint16, verbose=verbose)[:,:,0]
  return data

def HST_write(data, fname):
  '''Write binary raw file.
  
  This function write a (x,y,z) 3D dataset to the disk.
  The file is written as a Z-stack. It means that the first nx*ny bytes 
  represent the first slice and so on...
  This function is deprecated and its use should be replaced by the use of edf_write.
  '''
  (nx, ny, nz) = data.shape
  print 'opening',fname,'for writing'
  print 'volume size is ', nx, 'x', ny, 'x', nz
  f = open(fname, 'wb')
  # HP 11/2013 swap axes according to read function
  s = np.ravel(data.transpose(2,1,0)).tostring()
  f.write(s)
  f.close()
  print 'writing .info file'
  f = open(fname + '.info', 'w')
  f.write('! PyHST_SLAVE VOLUME INFO FILE\n')
  f.write('NUM_X = ' + str(nx) + '\n')
  f.write('NUM_Y = ' + str(ny) + '\n')
  f.write('NUM_Z = ' + str(nz) + '\n')
  f.close()
  print 'done with writing'

def recad_vol(vol_filename, min, max, verbose=False):
  '''Recad a 32 bit vol file into 8 bit raw file.
  
  This function reads a 3D volume file into a numpy float32 array and
  applies the `recad` function with the [min, max] range. The result is 
  saved into a .raw file with the same name as the input file.

  In verbose mode, a piture to compare mid slices in the two volumes 
  and another one to compare the histograms are saved.
  
  ..note:: To read the vol file, the presence of a .info file is 
    assumed, see `HST_read`.

  *Parameters*

  **vol_filename**: the path to the binary vol file.
  
  **min**: value to use as the minimum (will be 0 in the casted array).

  **max**: value to use as the maximum (will be 255 in the casted array).

  **verbose**: activate verbose mode (False by default).
  '''
  prefix = vol_filename[:-4]
  vol_size = HST_info(vol_filename + '.info')
  data = HST_read(vol_filename, type=np.float32)  
  data_uint8 = recad(data, min, max)
  if verbose:
    plt.figure(1, figsize=(10,5))
    plt.subplot(121)
    plt.imshow(data[:,:,vol_size[2]//2])
    plt.title('float image')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(data_uint8[:,:,vol_size[2]//2])
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
  
  ..note:: the header assumes uint8 data type.
  '''
  (nz, ny, nx) = data.shape
  print 'opening',fname,'for writing'
  print 'volume size is ', nx, 'x', ny, 'x', nz
  # write header
  f = open(fname, 'w')
  f.write('# vtk DataFile Version3.0\n')
  f.write(fname[:-4]+'\n')
  f.write('BINARY\n');
  f.write('DATASET STRUCTURED_POINTS\n');
  f.write('DIMENSIONS '+str(nx)+' '+str(ny)+' '+str(nz)+'\n');
  f.write('SPACING 1.0 1.0 1.0\n');
  f.write('ORIGIN 0.0 0.0 0.0\n');
  f.write('POINT_DATA '+str(nx*ny*nz)+'\n');
  f.write('SCALARS volume_scalars unsigned_char 1\n');
  f.write('LOOKUP_TABLE default\n')
  f.close()
  # append binary data
  f = open(fname, 'ab')
  s = np.ravel(data).tostring()
  f.write(s)
  f.close()
  print 'done with writing'

def read_dif(Nx, Ny, file, shift=False):
  '''Read the result of an FFT computation in a csv format.
  
     These fiels are basically produced by **process diffraction
     within a z-set post_processing.
     
     Returns an image array containing the data.
  '''
  import csv                                             
  reader = csv.reader(open(file, "r"))
  im = np.zeros((Ny, Nx), dtype=np.float)
  max = 0
  xmax = -1
  ymax = -1
  for row in reader:
    [x, y, a, b] = row
    xx = Nx/2 + int(x)
    yy = Ny/2 + int(y)
    if xx >= Nx:
      xx -= Nx
    if yy >= Ny:
      yy -= Ny
    im[yy, xx] = np.sqrt(float(a)*float(a) + float(b)*float(b))
    if im[yy, xx] > max:
      max = im[yy, xx]
      xmax = xx
      ymax = yy
  print xmax,ymax

  if shift:
    # find the image shift
    (jsh, ish) = maximum_position(im)
    ish -= Nx/2
    jsh -= Ny/2
    print 'image shift is:',ish,jsh
    shift_im = np.zeros_like(im)
    for i in range(Nx):
      ishift = i + ish
      if ishift >= Nx:
        ishift -= Nx
      if ishift < 0:
        ishift += Nx
      for j in range(Ny):
        jshift = j + jsh
        if jshift >= Ny:
          jshift -= Ny
        if jshift < 0:
          jshift += Ny
        shift_im[j, i] = im[jshift, ishift]
    im = shift_im
  return im
