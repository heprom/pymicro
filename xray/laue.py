import numpy as np
from pymicro.crystal.lattice import HklPlane
from pymicro.xray.xray_utils import *

def select_lambda(hkl, orientation, Xu=np.array([1., 0., 0.]), verbose=False):
  '''
  Compute the wavelength corresponding to the first order reflection 
  of a given lattice plane.
  
  :param hkl: The given latice plane.
  :param Xu: The unit vector of the incident X-ray beam (default along the X-axis).
  :param bool verbose: activate verbose mode (default False).
  :returns tuple: A tuple of the wavelength value and the corresponding Bragg angle.
  '''
  (h, k, l) = hkl.miller_indices()
  dhkl = hkl.interplanar_spacing()
  Gc = hkl.scattering_vector()
  Bt = orientation.orientation_matrix().transpose()
  Gs = Bt.dot(Gc)
  # compute the theta angle between X and the scattering vector which is theta + pi/2
  theta = np.arccos(np.dot(Xu, Gs/np.linalg.norm(Gs))) - np.pi/2
  # find the lambda selected by this hkl plane
  the_lambda = lambda_nm_to_keV(2*dhkl*np.sin(theta))
  if verbose:
    print('\n** Looking at plane %d%d%d **' % (h, k, l))
    print('scattering vector in laboratory CS', Gs)    
    print('glancing angle between incident beam and the hkl plane is %.1f deg' % (theta*180/np.pi))
    print('corresponding lambda is %.1f' % the_lambda)
  return (the_lambda, theta)

def build_list(lattice=None, max_miller=3):
  '''Create a list of all HklPlanes.
  This should move to the HklPlane class as a static method.'''
  # build a list of all hkl planes
  hklplanes = []
  indices = range(-max_miller, max_miller+1)
  for h in indices:
    for k in indices:
      for l in indices:
        if h == k == l == 0: # skip (0, 0, 0)
          continue
        if (h % 2 == 0 & k % 2 == 0 and l % 2 == 0) or (h % 2 == 1 & k % 2 == 1 and l % 2 == 1):
          # take only all odd or all even hkl indices
          hklplanes.append(HklPlane(h, k, l, lattice))
  return hklplanes

def compute_ellpisis(orientation, detector, det_distance, uvw):
  det_size_pixel = np.array([1950, 1456])
  Bt = orientation.orientation_matrix().transpose()
  ZA = Bt.dot(uvw.direction())
  if ZA[0] < 0:
    ZA *= -1 # make sur the ZA vector is going forward
  psi = np.arccos(np.dot(ZA, np.array([1., 0., 0.])))
  ZAD = det_distance*ZA/ZA[0]
  eta = np.pi/2 - np.arctan(ZAD[1]/ZAD[2])
  print('angle psi (deg) is', psi*180/np.pi)
  print('angle eta (deg) is', eta*180/np.pi)
  print('ZA cross the det plane at', ZAD)
  (uc, vc) = (0.5*det_size_pixel[0] - ZAD[1]/detector.pixel_size, 0.5*det_size_pixel[1] - ZAD[2]/detector.pixel_size)
  print('ZA crosses the detector at (%.3f,%.3f) mm or (%d,%d) pixels' % (ZAD[1], ZAD[2], uc, vc))
  e = np.tan(psi) # this assume the incident beam is normal to the detector
  print('ellipse eccentricity is %f' % e)
  '''
  # wrong formulae from Amoros (The Laue Method)...
  a = det_distance*np.tan(psi)/detector.pixel_size
  b = det_distance*np.sin(psi)/detector.pixel_size
  '''
  # the proper equation is:
  a = 0.5*det_distance*np.tan(2*psi)/detector.pixel_size
  b = 0.5*det_distance*np.tan(2*psi)*np.sqrt(1 - np.tan(psi)**2)/detector.pixel_size
  print('ellipsis major and minor half axes are a=%.1f and b=%.1f' % (a, b))
  # use a parametrc curve to plot the ellipsis
  t = np.linspace(0., 2*np.pi, 101)
  print(t)
  print(t[50], np.pi)
  x = a*np.cos(t)
  y = b*np.sin(t)
  data = np.array([x,y])
  R = np.array([[np.cos(eta), -np.sin(eta)], [np.sin(eta), np.cos(eta)]])
  data = np.dot(R, data) # rotate our ellipse
  data[0] += 0.5*det_size_pixel[0] - a*np.cos(eta)
  data[1] += 0.5*det_size_pixel[1] - a*np.sin(eta)
  return data
  
def diffracted_vector(hkl, orientation, min_theta=0.1):
  Bt = orientation.orientation_matrix().transpose()
  (h, k, l) = hkl.miller_indices()
  # this hkl plane will select a particular lambda
  (the_lambda, theta) = select_lambda(hkl, orientation, verbose=True)
  if abs(theta) < min_theta*np.pi/180: # skip angles < min_theta deg
    return None
  lambda_nm = 1.2398 / the_lambda
  X = np.array([1., 0., 0.])/lambda_nm
  Gs = Bt.dot(hkl.scattering_vector())
  print('bragg angle for %d%d%d reflection is %.1f' % (h, k, l, hkl.bragg_angle(the_lambda)*180/np.pi))
  assert (abs(theta - hkl.bragg_angle(the_lambda)) < 1e-6) # verify than theta_bragg is equal to the glancing angle
  # compute diffracted direction
  K = X + Gs
  return K
  
def compute_Laue_pattern(orientation, detector, det_distance, hklplanes=None, inverted=False):
  r_spot = 10 # pixels
  det_size_pixel = np.array([1840, 1456])
  #det_size_pixel = np.array([1950, 1456])
  det_size_mm = det_size_pixel * detector.pixel_size # mm
  detector.data = np.zeros(det_size_pixel, dtype=np.uint8)
  val = np.iinfo(detector.data.dtype.type).max # 255 here
  # show direct beam
  detector.data[0.5*det_size_pixel[0] - 2*r_spot:0.5*det_size_pixel[0] + 2*r_spot, 0.5*det_size_pixel[1] - 2*r_spot:0.5*det_size_pixel[1] + 2*r_spot] = val

  for hkl in hklplanes:
    K = diffracted_vector(hkl, orientation)
    if K is None:
      continue
    Ku = K/np.linalg.norm(K)
    (u, v) = (det_distance*K[1]/K[0], det_distance*K[2]/K[0]) # unit is mm
    (up, vp) = (0.5*det_size_pixel[0] - u/detector.pixel_size, 0.5*det_size_pixel[1] - v/detector.pixel_size) # unit is pixel on the detector
    if abs(up) > 1e6 or abs(vp) > 1e6:
      continue
    print('diffracted beam will hit the detector at (%.3f,%.3f) mm or (%d,%d) pixels' % (u, v, up, vp))
    # mark corresponding pixels on the image detector
    if up >= 0 and vp >= 0 and up < det_size_pixel[0] and vp < det_size_pixel[1]:
      detector.data[up-r_spot:up+r_spot, vp-r_spot:vp+r_spot] = val
  if inverted:
    print('inverting image')
    detector.data = np.invert(detector.data)
  return detector.data

if __name__ == '__main__':
  from matplotlib import pyplot as plt, cm, rcParams
  rcParams.update({'font.size': 12})
  rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

  from pymicro.xray.detectors import Varian2520
  det_distance = 100
  detector = Varian2520()
  detector.pixel_size = 0.127 # mm
  det_size_pixel = np.array((1950, 1456))
  det_size_mm = det_size_pixel * detector.pixel_size # mm

  # orientation of our single crystal
  from pymicro.crystal.lattice import *
  ni = Lattice.from_symbol('Ni')
  from pymicro.crystal.microstructure import *
  phi1 = 10 # deg
  phi = 20 # deg
  phi2 = 10 # deg
  orientation = Orientation.from_euler([phi1, phi, phi2])

  #'''
  #uvw = HklDirection(1, 1, 0, ni)
  uvw = HklDirection(5, 1, 0, ni)
  ellipsis_data = compute_ellpisis(orientation, detector, det_distance, uvw)
  #'''
  hklplanes = build_list(lattice=ni, max_miller=8)

  # compute image for the cube orientation
  det_distance = 300
  orientation = Orientation.cube()

  '''
  # euler angles from EBSD
  euler1 = [286.3, 2.1, 70.5]
  euler2 = [223.7, 2.8, 122.2]
  eulers = [euler1, euler2]
  o_ebsd = Orientation.from_euler(eulers[0])
  # apply the transformation matrix corresponding to putting the sample with X being along the X-ray direction
  Tr = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
  orientation = Orientation(np.dot(o_ebsd.orientation_matrix(), Tr.T))

  # compute the list of zone axes having a glancing angle < 45 deg
  max_miller = 5
  # max glancing angle in the vertical direction
  max_glangle = np.arctan(0.5*det_size_pixel[1]*detector.pixel_size/det_distance)*180./np.pi
  zoneaxes = []
  indices = range(-max_miller, max_miller+1)
  Bt = orientation.orientation_matrix().transpose()
  for u in indices:
    for v in indices:
      for w in indices:
        if u == v == w == 0: # skip (0, 0, 0)
          continue
        uvw = HklDirection(u, v, w, ni)
        ZA = Bt.dot(uvw.direction())
        psi = np.arccos(np.dot(ZA, np.array([1., 0., 0.])))*180./np.pi
        print('angle between incident beam and zone axis is %.1f' % psi)
        if psi < max_glangle:
          zoneaxes.append(HklDirection(u, v, w, ni))
  for ZA in zoneaxes:
    print(ZA.miller_indices())
  print(len(zoneaxes))
  '''
  image = compute_Laue_pattern(orientation, detector, det_distance, hklplanes)
  plt.imshow(image.T, cmap=cm.gray)
  plt.plot(ellipsis_data[0], ellipsis_data[1], 'b-')
  plt.title(r'Simulated Laue pattern, cube orientation, $d={0}$ mm'.format(det_distance))
  #plt.savefig('Laue_plot_cube_distance_%02d.pdf' % det_distance)
  plt.show()

  '''
  # simple loop to vary the detector distance
  for i in range(6):
    det_distance = 50 + 10*i # mm
    # compute image
    image = compute_Laue_pattern(orientation, detector, det_distance, hklplanes)
    plt.imshow(image.T, cmap=cm.gray)
    plt.title(r'Simulated Laue pattern, $d={0}$ mm, ($\varphi_1={1}^\circ$, $\phi={2}^\circ$, $\varphi_2={3}^\circ$)'.format(det_distance, phi1, phi, phi2))
    plt.savefig('Laue_plot_distance_%02d' % i)
  
  # another loop to vary the crystal orientation
  for i in range(20):
    phi2 = 2*i # mm
    orientation = Orientation.from_euler([phi1, phi, phi2])
    # compute image
    image = compute_Laue_pattern(orientation, detector, det_distance, hklplanes)
    plt.imshow(image.T, cmap=cm.gray)
    plt.title(r'Simulated Laue pattern, $d={0}$ mm, ($\varphi_1={1}^\circ$, $\phi={2}^\circ$, $\varphi_2={3}^\circ$)'.format(det_distance, phi1, phi, phi2))
    plt.savefig('Laue_plot_phi2_%02d' % i)
  '''
