import numpy as np
#from numpy import array, empty, uint8, uint16, uint32, float32, fromstring, reshape, zeros, linspace, sqrt, pi, cos, sin, exp
from scipy import ndimage

def edf_readf(image_name, size):
    '''Read a 2D image plate MAR pattern in EDF file format.
    
    These images are typically obtained from passerelle (extracted 
    from the nexus data file).
    This method assume Big endian byte order.'''
    nx = size
    ny = size
    print 'image size is ', nx, 'x', ny
    f = open(image_name, 'rb')
    data = empty((ny, nx), dtype=type)
    print 'reading EDF image...'
    f.seek(1024)
    data = reshape(np.fromstring(f.read(ny * nx * 4), \
        np.uint32).astype(float32), (ny, nx))
    f.close()
    return data

def edf_read(image_name, size):
    '''Read a 2D image plate MAR pattern in EDF file format.
    
    These images are typically obtained from passerelle (extracted 
    from the nexus data file).
    This method assume Big endian byte order.'''
    nx = size
    ny = size
    print 'image size is ', nx, 'x', ny
    f = open(image_name, 'rb')
    data = empty((ny, nx), dtype=type)
    print 'reading EDF image...'
    f.seek(1024)
    data = reshape(np.fromstring(f.read(ny * nx * 4), \
        np.uint32).astype(np.uint32), (ny, nx))
    f.close()
    return data

def rawmar_read(image_name, size):
    '''Read a 2D image plate MAR pattern.
    
    These images are typically obtained from the marcvt utility.
    That this method assume Big endian byte order.'''
    nx = size
    ny = size
    print 'image size is ', nx, 'x', ny
    f = open(image_name, 'rb')
    data = empty((ny, nx), dtype=type)
    print 'reading image...'
    f.seek(4600)
    data = reshape(np.fromstring(f.read(ny * nx * 2), \
        np.uint16).astype(np.uint16), (ny, nx))
    f.close()
    return data

def interp_im_along_line(im, jvals, ivals):
  ''' perform image data interpolation along a line
      using numpy ndimage module.
  '''
  coords = np.array([jvals, ivals])
  vals = ndimage.map_coordinates(im, coords)
  return vals

def gaussian(x, mean, stddev):
  y = (1/np.sqrt(2*np.pi*stddev))*np.exp(-((x-mean)**2)/(2*stddev))
  return y

def gauss_pic(x, I0, mean, stddev):
  y = np.zeros_like(x)
  alpha = I0*np.sqrt(2*np.pi*stddev)
  for i in range(len(x)):
    y[i] = alpha*gaussian(x[i], mean, stddev)
  return y

def pearson7(x, I0, x0, delta2teta, m):
  ''' PEARSON VII
      I0 hauteur du pic
      x0 position centre pic
      delta2teta largeur a mi hauteur
      m facteur de forme.
  '''
  I = I0/(1+4*(2**(1/m)-1)*((x-x0)/delta2teta)**2)**m
  return I

def pearson7asym(x, I0, x0, delta2tetag, mg, delta2tetad, md):
  ''' PEARSON VII asymetrique
      I0 hauteur du pic
      x0 position centre pic
      delta2tetag largeur a mi hauteur gauche
      mg facteur de forme gauche
      delta2tetad largeur a mi hauteur droite
      md facteur de forme droite
  '''
  I = np.zeros_like(x)
  for i in range(len(x)):
    if x[i]<=x0:
      I[i]=pearson7(x[i],I0,x0,delta2tetag,mg)
    else:
      I[i]=pearson7(x[i],I0,x0,delta2tetad,md)
  return I

def pearson7asym2(x, I0, x0, delta2tetag, mg, delta2tetad, md):
  ''' PEARSON VII asymetrique
      I0 hauteur du pic
      x0 position centre pic
      delta2tetag largeur a mi hauteur gauche
      mg facteur de forme gauche
      delta2tetad largeur a mi hauteur droite
      md facteur de forme droite
  '''
  if x<=x0:
    I=pearson7(x,I0,x0,delta2tetag,mg)
  else:
    I=pearson7(x,I0,x0,delta2tetad,md)
  return I

def save_radial_profile_params(path, _rmin, _rmax, _xc, _yc, _theta_min, \
  _theta_max, _steps, _ref_radius, _ref_pixels, _im):
  f = open(path, 'w')
  f.write('radial profile parameters\n')
  f.write('rmin ' + str(_rmin) + '\n')
  f.write('rmax ' + str(_rmax) + '\n')
  f.write('xc ' + str(_xc) + '\n')
  f.write('yc ' + str(_yc) + '\n')
  f.write('theta_min ' + str(_theta_min) + '\n')
  f.write('theta_max ' + str(_theta_max) + '\n')
  f.write('steps ' + str(_steps) + '\n')
  f.write('ref_radius ' + str(_ref_radius) + '\n')
  f.write('ref_pixels ' + str(_ref_pixels) + '\n')
  f.close()

def open_radial_profile_params(path):
  rmin = None; rmax = None; xc = None; yc = None; theta_min = None; 
  theta_max = None; steps = None; ref_radius = None; ref_pixels=None
  f = open(path, 'r')
  try:
    f.readline() # skip the first line
    line = f.readline()
    rmin = float(line.split(' ')[1])
    line = f.readline()
    rmax = float(line.split(' ')[1])
    line = f.readline()
    xc = float(line.split(' ')[1])
    line = f.readline()
    yc = float(line.split(' ')[1])
    line = f.readline()
    theta_min = float(line.split(' ')[1])
    line = f.readline()
    theta_max = float(line.split(' ')[1])
    line = f.readline()
    steps = float(line.split(' ')[1])
    line = f.readline()
    ref_radius = float(line.split(' ')[1])
    line = f.readline()
    ref_pixels = float(line.split(' ')[1])
  except IndexError:
    print 'Error reading parameter file, some of the parameters may no be initialized properly...'
  return rmin, rmax, xc, yc, theta_min, theta_max, steps, ref_radius, ref_pixels

def phi_profile(_rmin, _rmax, _xc, _yc, _phi_min, _phi_max, _steps, _im, callback = None):
  ''' Angular integration of the WAXD pattern.
  '''  
  _int_vals = np.zeros(_steps)
  for i, phi_i in enumerate(np.linspace(_phi_min, _phi_max, _steps)):
    if callback:
      progress = abs((phi_i - _phi_min)/(_phi_max - _phi_min))
      callback(progress)
    _rphi = (180. + phi_i) * np.pi / 180.
    xs = _xc + _rmin * np.cos(_rphi)
    ys = _yc - _rmin * np.sin(_rphi)
    xe = _xc + _rmax * np.cos(_rphi)
    ye = _yc - _rmax * np.sin(_rphi)
    ivals = np.linspace(xs, xe, 2*(_rmax-_rmin))
    jvals = np.linspace(ys, ye, 2*(_rmax-_rmin))
    vals = interp_im_along_line(_im, jvals, ivals)
    _int_vals[i] = sum(vals)/2.
  return [np.linspace(_phi_min, _phi_max, _steps), _int_vals]
  
def radial_profile(_rmin, _rmax, _xc, _yc, _phi_min, _phi_max, _steps, _im, callback = None):
  ''' Radial integration of the WAXD pattern.

      Integrate the WAXD pattern on an angular sector
      defined be the center (_xc, _yc), the radius range 
      _rmin to _rmax and the angles _phi_min and _phi_max.
      A callback flag is used to notify GUI functions 
      (like to update a progress bar).
  '''
  _int_vals = np.zeros(2*(_rmax-_rmin))
  for phi_i in np.linspace(_phi_min, _phi_max, _steps):
    if callback:
      progress = abs((phi_i - _phi_min)/(_phi_max - _phi_min))
      callback(progress)
    #_rtheta = theta_i * np.pi / 180.
    #_rtheta = (90 + theta_i) * np.pi / 180.
    _rphi = (180 + phi_i) * np.pi / 180.
    xs = _xc + _rmin * np.cos(_rphi)
    ys = _yc - _rmin * np.sin(_rphi)
    xe = _xc + _rmax * np.cos(_rphi)
    ye = _yc - _rmax * np.sin(_rphi)
    ivals = np.linspace(xs, xe, 2*(_rmax-_rmin))
    jvals = np.linspace(ys, ye, 2*(_rmax-_rmin))
    vals = interp_im_along_line(_im, jvals, ivals)
    _int_vals += vals * float(_phi_max - _phi_min) / _steps
  return [np.linspace(_rmin, _rmax, 2*(_rmax-_rmin)), _int_vals]

def element_max(tt):
    i=0
    max=0
    #_imax=0    
    while i < len(tt):
       if tt[i] > max :
          max, imax=tt[i], i
       i=i+1
    return imax,max 

def element_min(tt):
    min=element_max(tt)[1]
    i=0
    while i < len(tt):
       if tt[i] < min :
          min, imin=tt[i], i
       i=i+1
    return imin,min
      
