import os, numpy as np
from matplotlib import pyplot as plt, cm, rcParams
rcParams.update({'font.size': 12})
rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
rcParams['image.interpolation'] = 'nearest'
from pymicro.file.file_utils import HST_read, HST_write


class Detector2d:
  '''Class to handle 2D detectors.
  
  2D detectors produce array like images and may have different geometries.
  This abstract class regroup the generic method for those kind of detectors.
  '''
  def __init__(self):
    self.xcen = 0.
    self.ycen = 0.
    self.mask_flag = 0 # use a mask
    self.mask_size_increase = 0
    self.image_path = None
    self.save_path = '.'
    self.correction = 'none' # could be none, bg, flat

  def azimuthal_regroup(self, two_theta_mini=None, two_theta_maxi=None, two_theta_step=None, psi_min=None, psi_max=None, write_txt=False, output_image=False):
    # assign default values if needed
    if not two_theta_mini: two_theta_mini = self.two_thetas.min()
    if not two_theta_maxi: two_theta_maxi = self.two_thetas.max()
    if not two_theta_step: two_theta_step = 1./mar.calib
    nbOfBins = int((two_theta_maxi - two_theta_mini) / two_theta_step)
    print '* Azimuthal regroup (two theta binning)'
    print '  delta range = [%.1f-%.1f] with a %g deg step (%d bins)' % (two_theta_mini, two_theta_maxi, two_theta_step, nbOfBins)

    bin_edges = np.linspace(two_theta_mini, two_theta_maxi, 1 + nbOfBins)
    two_theta_values = bin_edges[:-1] + 0.5 * two_theta_step
    intensityResult = np.zeros(nbOfBins); # this will be the summed intensity
    pointsCounted = np.zeros(nbOfBins); # this will be the number of pixel contributing to each point

    # calculating bin indices for each pixel
    binIndices = np.floor((self.two_thetas - two_theta_mini) / two_theta_step).astype(np.int16)
    binIndices[self.two_thetas > two_theta_maxi] = -1
    # mark out pixels with negative intensity
    binIndices[self.corr_data < 0] = -1
    # mark out pixels outside of phi range which is [0,phi_max] and [180-phi_max,180]
    if psi_min:
      binIndices[(self.psis < psi_min)] = -1
    if psi_max:
      binIndices[(self.psis > psi_max) & (self.psis < 180 - psi_max)] = -1
    for ii in range(nbOfBins):
        intensityResult[ii] = self.corr_data[binIndices == ii].sum()
        pointsCounted[ii] = (binIndices == ii).sum()
    intensityResult /= pointsCounted

    if output_image:
      print self.image_path
      print os.path.basename(self.image_path)
      print os.path.splitext(os.path.basename(self.image_path))
      output_image_path = os.path.join(self.save_path, \
        'AR_%s.png' % os.path.splitext(os.path.basename(self.image_path))[0])
      plt.imsave(output_image_path, binIndices, vmin=0, vmax=nbOfBins)

    if write_txt:
      if not self.save_path:
        self.save_path = os.path.dirname(self.image_path)
      print "writing text file"
      np.savetxt(os.path.join(self.save_path, \
        'Int_%s_2theta_profile.txt' % os.path.splitext(os.path.basename(self.image_path))[0]), \
        (two_theta_values, intensityResult, pointsCounted), \
        header = '# delta (deg) -- norm intensity -- points counted', \
        fmt='%.6e')
    return two_theta_values, intensityResult, pointsCounted

  def sagital_regroup(self, two_theta_mini=None, two_theta_maxi=None, psi_min=None, psi_max=None, psi_step=None, write_txt=False, output_image=False):
    # assign default values if needed
    if not two_theta_mini: two_theta_mini = self.two_thetas.min()
    if not two_theta_maxi: two_theta_maxi = self.two_thetas.max()
    if not psi_step: psi_step = 1./mar.calib
    nbOfBins = int((psi_max - psi_min) / psi_step)
    print '* Sagital regroup (psi binning)'
    print '  psi range = [%.1f-%.1f] with a %g deg step (%d bins)' % (psi_min, psi_max, psi_step, nbOfBins)

    bin_edges = np.linspace(psi_min, psi_max, 1 + nbOfBins)
    psi_values = bin_edges[:-1] + 0.5 * psi_step
    intensityResult = np.zeros(nbOfBins); # this will be the summed intensity
    pointsCounted = np.zeros(nbOfBins); # this will be the number of pixel contributing to each point

    # calculating bin indices for each pixel
    binIndices = np.floor((self.psis - psi_min) / psi_step).astype(np.int16)
    binIndices[self.psis > psi_max] = -1
    # mark out pixels with negative intensity
    binIndices[self.corr_data < 0] = -1
    # mark out pixels outside of psi range [-psi_max, psi_max]
    if two_theta_mini:
      binIndices[(self.two_thetas < two_theta_mini)] = -1
    if two_theta_maxi:
      binIndices[(self.two_thetas > two_theta_maxi)] = -1
    for ii in range(nbOfBins):
        intensityResult[ii] = self.corr_data[binIndices == ii].sum()
        pointsCounted[ii] = (binIndices == ii).sum()
    intensityResult /= pointsCounted

    if output_image:
      print self.image_path
      print os.path.basename(self.image_path)
      print os.path.splitext(os.path.basename(self.image_path))
      output_image_path = os.path.join(self.save_path, \
        'AR_%s.png' % os.path.splitext(os.path.basename(self.image_path))[0])
      plt.imsave(output_image_path, binIndices, vmin=0, vmax=nbOfBins)

    if write_txt:
      if not self.save_path:
        self.save_path = os.path.dirname(self.image_path)
      print "writing text file"
      np.savetxt(os.path.join(self.save_path, \
        'Int_%s_psi_profile.txt' % os.path.splitext(os.path.basename(self.image_path))[0]), \
        (two_theta_values, intensityResult, pointsCounted), \
        header = '# psi (deg) -- norm intensity -- points counted', \
        fmt='%.6e')
    return psi_values, intensityResult, pointsCounted

class Mar165(Detector2d):
  '''Class to handle a rayonix marccd165.
  
  The image plate marccd 165 detector produce 16 unsigned (2048, 2048) images.
  '''
  def __init__(self):
    Detector2d.__init__(self)
    self.xcen = 1024.
    self.ycen = 1024.
    self.calib = 1. # pixel by degree
    self.ref = np.ones((2048, 2048), dtype=np.uint16)
    self.dark = np.zeros((2048, 2048), dtype=np.uint16)
    self.bg = np.zeros((2048, 2048), dtype=np.uint16)

  def load_image(self, image_path):
    print('loading image %s' % image_path)
    self.image_path = image_path
    self.data = HST_read(self.image_path, data_type=np.uint16, dims=(2048, 2048, 1))[:,:,0].astype(np.float32)
    self.compute_corrected_image()
    
  def compute_corrected_image(self):
    if self.correction == 'none':
      self.corr_data = self.data
    elif self.correction == 'bg':
      self.corr_data = self.data - self.bg
    elif self.correction == 'flat':
      self.corr_data = (self.data - self.dark).astype(np.float32) / (self.ref - self.dark).astype(np.float32)

  def compute_geometry(self):
    '''Calculate an array of the image size with the (2theta, psi) for each pixel.'''

  def compute_TwoTh_Psi_arrays(self):
    '''Calculate two arrays (2theta, psi) TwoTheta and Psi angles arrays corresponding to repectively 
    the vertical and the horizontal pixels.
    '''
    deg2rad = np.pi / 180.
    inv_deg2rad = 1. / deg2rad 
    # distance xpad to sample, in pixel units
    distance = self.calib / np.tan(1.0 * deg2rad)
    x = np.linspace(0, 2047, 2048)
    y = np.linspace(0, 2047, 2048)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt((xx - self.xcen)**2 + (yy - self.ycen)**2)
    self.two_thetas = np.arctan(r/distance) * inv_deg2rad
    self.psis = np.arccos((xx - self.xcen) / r) * inv_deg2rad

class Xpad(Detector2d):
  '''Class to handle Xpad like detectors.
  
  Xpad are pixel detectors made with stacked array of silicon chips.
  Between each chip are 2 double pixels with a 2.5 times bigger size 
  and which need to be taken into account.
  
  .. note::
  
     This code is heavily inspired by the early version of C. Mocuta, 
     scientist on the DiffAbs beamline at Soleil synchrotron.

  .. warning::

     only tested with Xpad S140 for now...
    
  '''
  
  def __init__(self):
    Detector2d.__init__(self)
    self.numberOfModules = 2
    self.numberOfChips = 7
    # chip dimension, in pixels (X = horiz, Y = vertical)
    self.chip_sizeX = 80
    self.chip_sizeY = 120
    self.pixel_size = 0.13 # actual size of a pixel in mm
    self.factorIdoublePixel = 2.64; # this is the intensity correction factor for the double pixels
    self.deltaOffset = 13; # detector offset on the delta axis
    self.calib = 85.62 # pixels in 1 deg.
    self.XcenDetector = 451.7 + 5*3
    self.YcenDetector = 116.0 # position of direct beam on xpad at del=gam=0
    self.orientation = 'horizontal' # either 'horizontal' or 'vertical'
    self.verbose = True
    
  def load_image(self, image_path):
    self.image_path = image_path
    # check the use of using [y, x] array instead of [x, y]
    self.data = HST_read(self.image_path, data_type=np.uint16, dims=(560, 240, 1))[:,:,0].astype(np.float32).transpose()
    self.compute_corrected_image()
    if self.orientation == 'vertical':
      self.data = self.data.transpose()
      self.corr_data = self.corr_data.transpose()

  def compute_geometry(self):
    '''Calculate the array with the corrected geometry (double pixels).'''

    lines_to_remove_array = (0, -3); # adding 3 more lines, corresponding to the double pixels on the last and 1st line of the modules
    #calculate the total number of lines to remove from the image
    lines_to_remove = 0; #initialize to 0 for calculating the sum. For xpad 3.2 these lines (negative value) will be added
    for i in range (0, self.numberOfModules):
        lines_to_remove +=  lines_to_remove_array[i]

    #size of the resulting (corrected) image
    image_corr1_sizeY = self.numberOfModules * self.chip_sizeY - lines_to_remove; 
    image_corr1_sizeX = (self.numberOfChips-1)*3 + self.numberOfChips * self.chip_sizeX; # considers the 2.5x pixels

    # calculate the corrected x coordinates
    newX_array = np.zeros(image_corr1_sizeX) # contains the new x coordinates
    newX_Ifactor_array = np.zeros(image_corr1_sizeX) # contains the mult. factor to apply for each x coordinate
    for x in range(0, 79): # this is the 1st chip (index chip = 0)
      newX_array[x] = x; 
      newX_Ifactor_array[x] = 1 # no change in intensity

    newX_array[79] = 79; newX_Ifactor_array[79] = 1/self.factorIdoublePixel;
    newX_array[80] = 79; newX_Ifactor_array[80] = 1/self.factorIdoublePixel;
    newX_array[81] = 79; newX_Ifactor_array[81] = -1

    for indexChip in range (1, 6):
      temp_index0 = indexChip * 83
      for x in range(1, 79): # this are the regular size (130 um) pixels
        temp_index = temp_index0 + x;
        newX_array[temp_index] = x + 80*indexChip;
        newX_Ifactor_array[temp_index] = 1; # no change in intensity
      newX_array[temp_index0] = 80*indexChip; newX_Ifactor_array[temp_index0] = 1/self.factorIdoublePixel; # 1st double column
      newX_array[temp_index0-1] = 80*indexChip; newX_Ifactor_array[temp_index0-1] = 1/self.factorIdoublePixel;
      newX_array[temp_index0+79] = 80*indexChip+79; newX_Ifactor_array[temp_index0+79] = 1/self.factorIdoublePixel; # last double column
      newX_array[temp_index0+80] = 80*indexChip+79; newX_Ifactor_array[temp_index0+80] = 1/self.factorIdoublePixel;
      newX_array[temp_index0+81] = 80*indexChip+79; newX_Ifactor_array[temp_index0+81] = -1;

    for x in range (6*80+1, 560): # this is the last chip (index chip = 6)
      temp_index = 18 + x;
      newX_array[temp_index] = x; 
      newX_Ifactor_array[temp_index] = 1; # no change in intensity

    newX_array[497] = 480; newX_Ifactor_array[497] = 1/self.factorIdoublePixel;
    newX_array[498] = 480; newX_Ifactor_array[498] = 1/self.factorIdoublePixel;

    newY_array = np.zeros(image_corr1_sizeY); # correspondance oldY - newY
    newY_array_moduleID = np.zeros(image_corr1_sizeY); # will keep trace of module index

    newYindex = 0;
    for moduleIndex in range (0, self.numberOfModules):
      for chipY in range (0, self.chip_sizeY):
        y = chipY + self.chip_sizeY*moduleIndex;
        newYindex = y - lines_to_remove_array[moduleIndex]*moduleIndex;
        newY_array[newYindex] = y;
        newY_array_moduleID[newYindex] = moduleIndex;
    #plt.plot(newX_array)
    #plt.plot(newY_array)
    #plt.plot(newY_array_moduleID)
    #plt.plot(newX_Ifactor_array)
    #plt.show()
    return newX_array, newY_array, newX_Ifactor_array
  
  def compute_corrected_image(self):
    newX_array, newY_array, newX_Ifactor_array = self.compute_geometry()
    image_corr1_sizeX = len(newX_array)
    image_corr1_sizeY = len(newY_array)
    thisCorrectedImage = np.zeros((image_corr1_sizeY, image_corr1_sizeX))
    for y in range (0, image_corr1_sizeY): # correct for double pixels
        yold = newY_array[y];
        for x in range (0, image_corr1_sizeX):
            xold = newX_array[x]
            Ifactor = newX_Ifactor_array[x]
            if (Ifactor > 0):
                #print "pos"
                thisCorrectedImage[y, x] = self.data[yold, xold]*Ifactor
            if(Ifactor < 0):
                #print "neg"
                thisCorrectedImage[y, x] = (self.data[yold, xold-1]+self.data[yold, xold+1])/2.0/self.factorIdoublePixel
                    
    # correct the double lines (last and 1st line of the modules, at their junction)
    lineIndex1 = self.chip_sizeY-1; # last line of module1 = 119, is the 1st line to correct
    lineIndex5 = lineIndex1 + 3 +1; # 1st line of module2 (after adding the 3 empty lines), becomes the 5th line tocorrect
    lineIndex2 = lineIndex1+1; lineIndex3 = lineIndex1+2; lineIndex4 = lineIndex1+3; 
    for x in range (0, image_corr1_sizeX):
        i1 = thisCorrectedImage[lineIndex1, x]; i5 = thisCorrectedImage[lineIndex5, x];
        i1new = i1/self.factorIdoublePixel; i5new = i5/self.factorIdoublePixel; i3 = (i1new+i5new)/2.0;
        thisCorrectedImage[lineIndex1, x] = i1new; thisCorrectedImage[lineIndex2, x] = i1new;
        thisCorrectedImage[lineIndex3, x] = i3;
        thisCorrectedImage[lineIndex5, x] = i5new; thisCorrectedImage[lineIndex4, x] = i5new

    if self.mask_flag == 1:
      double_pixel_mask = np.zeros_like(thisCorrectedImage)
      hlist = ((0,4+self.mask_size_increase), (77-self.mask_size_increase,85+self.mask_size_increase),  (160-self.mask_size_increase,168+self.mask_size_increase), (243-self.mask_size_increase,250+self.mask_size_increase), (326-self.mask_size_increase,332+self.mask_size_increase), (410-self.mask_size_increase,417+self.mask_size_increase), (492-self.mask_size_increase,498+self.mask_size_increase), (573-self.mask_size_increase,577))
      print hlist
      for (xLineStart, xLineEnd) in hlist:
        double_pixel_mask[:,xLineStart:xLineEnd+1] = True
      vlist = ((118, 125),)
      for (yLineStart, yLineEnd) in vlist:
        double_pixel_mask[yLineStart:yLineEnd+1,:] = True
      self.corr_data = np.ma.array(thisCorrectedImage, mask = double_pixel_mask)
    else:
      self.corr_data = thisCorrectedImage

  def compute_TwoTh_Psi_arrays(self, diffracto_delta, diffracto_gamma):
    '''Computes TwoTheta and Psi angles arrays corresponding to repectively 
    the vertical and the horizontal pixels.
    
    *Parameters*
    
    **diffracto_delta**: diffractometer value of the delta axis
    
    **diffracto_gamma**: diffractometer value of the gamma axis
    
    .. note::

      This assume the detector is perfectly aligned with the delta and 
      gamma axes (which should be the case).
    '''
    deg2rad = np.pi / 180
    inv_deg2rad = 1 / deg2rad 
    # distance xpad to sample, in pixel units
    distance = self.calib / np.tan(1.0 * deg2rad)

    diffracto_delta_rad = (diffracto_delta + self.deltaOffset) * deg2rad; 
    sindelta = np.sin(diffracto_delta_rad)
    cosdelta = np.cos(diffracto_delta_rad)
    singamma = np.sin(diffracto_gamma * deg2rad)
    cosgamma = np.cos(diffracto_gamma * deg2rad); 

    (image_corr1_sizeX, image_corr1_sizeY) = self.corr_data.shape
    twoThArray = np.zeros_like(self.corr_data)
    psiArray = np.zeros_like(self.corr_data)
    
    #converting to 2th-psi
    for x in range (0, image_corr1_sizeX):
        for y in range (0, image_corr1_sizeY):
            corrX = distance # for xpad3.2 like
            corrZ = self.YcenDetector - y # for xpad3.2 like
            corrY = self.XcenDetector - x # sign is reversed
            tempX = corrX
            tempY = corrZ*(-1.0)
            tempZ = corrY
            #apply here the rotation matrixes as follows: delta rotation as Ry + gamma rotation as Rz
            #        (cosTH  -sinTH  0)            ( cosTH  0  sinTH)
            #   Rz = (sinTH   cosTH  0)       Ry = (   0    1    0  )
            #        (  0       0    1)            (-sinTH  0  cosTH)
            #apply Ry(-delta); sin = -1 sign; cos = +1 sign
            x1 = tempX*cosdelta - tempZ*sindelta
            y1 = tempY
            z1 = tempX*sindelta + tempZ*cosdelta
            #apply Rz(-gamma); due to geo consideration on the image, the gamma rotation should be negative for gam>0
            #apply the same considerations as for the delta, and keep gam values positive	
            corrX = x1*cosgamma + y1*singamma
            corrY = -x1*singamma + y1*cosgamma
            corrZ = z1
            #calculate the square values and normalization
            corrX2 = corrX*corrX
            corrY2 = corrY*corrY
            corrZ2 = corrZ*corrZ
            norm = np.sqrt(corrX2 + corrY2+ corrZ2)
            #calculate the corresponding angles
            #delta = angle between vector(corrX, corrY, corrZ) and the vector(1,0,0)
            thisdelta = np.arccos(corrX / norm) * inv_deg2rad
            #psi = angle between vector(0, corrY, corrZ) and the vector(0,1,0) *** NOT properly calculated *** but the approx should be rather good, since corrX ~0 (from -7 to +7 pixels)
            #valid only for gam = del = 0 and flat detector
            sign = 1;
            if(corrZ < 0):
                sign = -1
            cos_psi_rad = corrY/np.sqrt(corrY2 + corrZ2)
            psi = np.arccos(cos_psi_rad)*inv_deg2rad*sign
            if(psi<0):
              psi += 360
            psi -= 90
            twoThArray[x, y] = thisdelta
            psiArray[x, y] = psi
    self.two_thetas = twoThArray
    self.psis = psiArray
    return twoThArray, psiArray
    
if __name__ == '__main__':

  xpad = Xpad()
  xpad.mask_flag = 0
  # calib
  xpad.XcenDetector = 271.0 + 3*3
  xpad.YcenDetector = 116.0
  xpad.deltaOffset = 15.0
  # read xpad image from nxs file
  import os, tables
  scanNo = 18
  file_name = 's1_10keV_%d.nxs' % scanNo
  path = 'D:/explCM/test/data/'
  pathSave = 'D:/explCM/test/data/exploited/'
  f = tables.openFile(path + file_name)
  print 'looking at file = %s' % (file_name)
  fileNameRoot2 = f.root._v_groups.keys()[0]
  command = "f.root.__getattr__(\""+str(fileNameRoot2)+"\")"
  test_data = eval(command + ".scan_data.data_06.read()")[0]
  #from scipy import ndimage
  #test_data = ndimage.median_filter(test_data, 3)

  fig = plt.figure()
  fig.add_subplot(211)
  plt.imshow(test_data, clim=(0,50))
  print test_data.shape
  corr_data = xpad.compute_corrected_image(test_data)
  fig.add_subplot(212)
  plt.imshow(corr_data, clim=(0,50))
  # compute 2theta, psi values for this image
  two_thetas, psis = xpad.compute_TwoTh_Psi_arrays(5.0, 0.0, corr_data)
  print two_thetas
  print '\n min',two_thetas.min()
  print '\n max',two_thetas.max()

  # the result
  deltaMini = 20.0
  deltaMaxi = 23.0
  stepDelta = 0.02
  nbOfBins = int((deltaMaxi-deltaMini)/stepDelta)+1; 
  print "   Azimuthal regroup"
  print "   deltaMini=%f  deltaMaxi=%f  nbOfBins=%d  stepDelta=%f" %(deltaMini, deltaMaxi, nbOfBins, stepDelta)

  deltaResult = np.zeros(nbOfBins); # generate the tables for radial integration, this is delta
  intensityResult = np.zeros(nbOfBins); # this will be the summed intensity
  pointsCounted = np.zeros(nbOfBins); # this will be the number of points contributing to each intensity point
  intensityResultTemp = np.zeros(nbOfBins); #this will be the temp corrected intensity (temporary for plot)

  # calculating binned data
  print "   2th binning"
  binIndices = np.floor((two_thetas - deltaMini) / stepDelta).astype(np.uint16)
  print binIndices
  # mark out pixels with negative intensity
  binIndices[corr_data < 0] = -1
  for ii in range(nbOfBins):
      intensityResult[ii] = corr_data[binIndices == ii].sum()
      pointsCounted[ii] = (binIndices == ii).sum()
  print '** normalizing, mean point counted:', np.mean(pointsCounted)
  intensityResult /= pointsCounted
   
  print "writing text file"
  np.savetxt(os.path.join(pathSave, 'Int_c1_exptime_air_4.txt'), \
    (deltaResult, intensityResult, pointsCounted), \
    header = '# delta (deg) -- norm intensity -- points counted', \
    fmt='%.6e')

  plt.figure()
  plt.imshow(two_thetas, cmap=cm.jet)
  plt.colorbar()
  #plt.plot(two_thetas[:,120], label='two theta')
  #plt.plot(psis[280,:], label='gamma')
  plt.show()
