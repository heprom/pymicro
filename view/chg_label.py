from HST_utils import HST_read, HST_write
from crack_utils import ask_for_file
from pylab import zeros_like, zeros

def parameters(**args):
    params={ \
            'scan_name' : '', \
            'raw_dir' : 'raw/', \
            'old' : 0, \
            'new' : 255, \
            }

    # override defaults with user choices        
    keys = args.keys()
    for kw in keys:
        print kw, ':', args[kw]
        params[kw] = args[kw]
        
    if params['scan_name'] == '':
        params['scan_name'] = ask_for_file(params['raw_dir'], ['raw'])
    print 'scan name is ', params['scan_name']
    return params

def chg_label(**args):
    '''Change a label within a 3D volume.

    Usage:
    chg_label(scan_name='/path/to/myscan.raw',old=10,new=20)
    '''

    # delegate parameters setup
    params = parameters(**args)
    
    # now read the volume
    data = HST_read(params['raw_dir'] + params['scan_name'])
    (nz, ny, nx) = data.shape
    for z in range(0, nz):
        for y in range(0, ny):
            for x in range(0, nx):
                if data[z, y, x] == params['old']:
                    #print 'found data to be modified at x=', x, ' y=', y, ' z=', z
                    data[z, y, x] = params['new']

    print data[1, 30, 30]
    print 'writing modified data'
    HST_write(data, params['raw_dir'] + params['scan_name'][:-4] + '_mod_.raw')
    print 'done'

def extend_rect_layer(**args):

    # delegate parameters setup
    params = parameters(**args)

    # now read the volume
    data = HST_read(params['raw_dir'] + params['scan_name'])
    (nz, ny, nx) = data.shape
    y = 44
    for z in range(0, nz):
      for x in range(5, 45):
        print data[z, y, x]
        data[z, y+1, x] = data[z, y, x] + 100
        data[z, y+2, x] = data[z, y, x] + 100
        data[z, y+3, x] = data[z, y, x] + 100

    print 'writing modified data'
    HST_write(data, params['raw_dir'] + params['scan_name'][:-4] + '_mod_.raw')
    print 'done'

def first_non_zero_neighbor(array, x, y):
  '''Return a (i,j) tuple corresponding to the closest non-zero
     neighbor.

     Browse array in a spiral-like manner starting from (x,y) seed
     and find the closest (or first) non-zero value in array.
     Browsing starts right then down then left then right and etc...
  '''
  (ny, nx) = array.shape
  #print 'shape=',array.shape,' x=',x,' y=',y
  if (x+1 < nx):
    if array[x+1, y] != 0:
      return (x+1, y)
  if (x+1 < nx) & (y+1 < ny):
    if array[x+1, y+1] != 0:
      return (x+1, y+1)
  # n is the loop number
  n = 1
  while (n<max(nx,ny)):
    #print '** n=',n
    if y+n < ny:
      for l in range(n-1,-n-1,-1):
        if (x+l < 0) | (x+l >= nx):
          continue
        #print 'l=',l,' x=',x+l,' y=',y+n 
        if array[x+l, y+n] != 0:
          return (x+l, y+n)
    if x-n >= 0:
      for u in range(n-1,-n-1,-1):
        if (y+u < 0) | (y+u >= ny):
          continue
        #print 'u=',u,' x=',x-n,' y=',y+u 
        if array[x-n, y+u] != 0:
          return (x-n, y+u)
    if y-n >= 0:
      for r in range(-n+1,n+2):
        if (x+r < 0) | (x+r >= nx):
          continue
        #print 'r=',r,' x=',x+r,' y=',y-n 
        if array[x+r, y-n] != 0:
          return (x+r, y-n)
    if x+n+1 < nx:
      for d in range(-n+1,n+2):
        if (y+d < 0) | (y+d >= ny):
          continue
        #print 'd=',d,' x=',x+n+1,' y=',y+d 
        if array[x+n+1, y+d] != 0:
          return (x+n+1, y+d)
    n += 1
  return None

def extend_layer(**args):

  # delegate parameters setup
  params = parameters(**args)

  # now read the volume
  data = HST_read(params['raw_dir'] + params['scan_name'])
  (nz, ny, nx) = data.shape
  new_data = zeros_like(data)
  z=0
  print 'z=',z
  for y in range(0, ny/4):
    print 'y=',y
    for x in range(0, nx):
      # look only at black voxels
      if data[z, y, x] != 0:
        new_data[z, y, x] = data[z, y, x]
        continue
      # find closest non zero voxel (in current xy plane)
      (yy,xx) = first_non_zero_neighbor(data[z,:,:],x,y)
      print 'found non zero data at x=',xx,' yy=',yy, ' val=',data[z, yy, xx]
      new_data[z, y, x] = data[z, yy, xx] + 100
      #dist = nx**2 + ny**2
      #for yy in range(0, ny/4):
      #  for xx in range(0, nx):
      #    if (data[z, yy, xx] != 0) & ((yy-y)**2+(xx-x)**2 < dist):
      #      new_data[z, y, x] = data[z, yy, xx] + 100
      #      dist = (yy-y)**2+(xx-x)**2

  print 'writing modified data'
  HST_write(new_data, params['raw_dir'] + params['scan_name'][:-4] + '_mod_.raw')
  print 'done'

def grow(array, val):
  (ny, nx) = array.shape
  new_array = zeros_like(array)
  for y in range(1, ny-1):
    #print 'y=',y
    for x in range(1, nx-1):
      if array[y, x] != 0:
        new_array[y, x] = array[y, x]
        # change neighbor values
        if array[y+1, x] == 0: new_array[y+1, x] = array[y, x] + val
        if array[y-1, x] == 0: new_array[y-1, x] = array[y, x] + val
        if array[y, x+1] == 0: new_array[y, x+1] = array[y, x] + val
        if array[y, x-1] == 0: new_array[y, x-1] = array[y, x] + val
        if array[y+1, x+1] == 0: new_array[y+1, x+1] = array[y, x] + val
        if array[y+1, x-1] == 0: new_array[y+1, x-1] = array[y, x] + val
        if array[y-1, x+1] == 0: new_array[y-1, x+1] = array[y, x] + val
        if array[y-1, x-1] == 0: new_array[y-1, x-1] = array[y, x] + val
  return new_array

def grey_level_grow(**args):

  # delegate parameters setup
  params = parameters(**args)

  # now read the volume
  data = HST_read(params['raw_dir'] + params['scan_name'])
  (nz, ny, nx) = data.shape
  new_data = zeros_like(data)
  for z in range(0,nz):
    print ' * z=',z
    new_data[z,:,:] = grow(data[z,:,:],100)
    for i in range(0,5):
      new_data[z,:,:] = grow(new_data[z,:,:],0)

  print 'writing modified data'
  HST_write(new_data, params['raw_dir'] + params['scan_name'][:-4] + '_mod_.raw')
  print 'done'

if __name__ == '__main__':
    #chg_label(raw_dir='/home/proudhon/esrf/gt/raw/', old=0, new=200)
    #extend_layer(raw_dir='/home/proudhon/esrf/gt/raw/')
    #first_non_zero_neighbor(zeros((5,5)),2,2)
    grey_level_grow(raw_dir='/home/proudhon/esrf/gt/raw/')
