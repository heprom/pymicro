import numpy as np
from pymicro.crystal.lattice import HklPlane
from pymicro.xray.xray_utils import *
from pymicro.xray.dct import add_to_image

def select_lambda(hkl, orientation, Xu=np.array([1., 0., 0.]), verbose=False):
    '''
    Compute the wavelength corresponding to the first order reflection
    of a given lattice plane.

    :param hkl: The given lattice plane.
    :param orientation: The orientation of the crystal lattice.
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
    theta = np.arccos(np.dot(Xu, Gs / np.linalg.norm(Gs))) - np.pi / 2
    # find the lambda selected by this hkl plane
    the_energy = lambda_nm_to_keV(2 * dhkl * np.sin(theta))
    if verbose:
        print('\n** Looking at plane %d%d%d **' % (h, k, l))
        print('scattering vector in laboratory CS', Gs)
        print('glancing angle between incident beam and the hkl plane is %.1f deg' % (theta * 180 / np.pi))
        print('corresponding energy is %.1f' % the_energy)
    return (the_energy, theta)


def build_list(lattice=None, max_miller=3):
    '''Create a list of all HklPlanes.
    This should move to the HklPlane class as a static method.'''
    # build a list of all hkl planes
    hklplanes = []
    indices = range(-max_miller, max_miller + 1)
    for h in indices:
        for k in indices:
            for l in indices:
                if h == k == l == 0:  # skip (0, 0, 0)
                    continue
                hklplanes.append(HklPlane(h, k, l, lattice))
                # if (h % 2 == 0 & k % 2 == 0 and l % 2 == 0) or (h % 2 == 1 & k % 2 == 1 and l % 2 == 1):
                #  # take only all odd or all even hkl indices
                #  hklplanes.append(HklPlane(h, k, l, lattice))
    return hklplanes


def compute_ellipsis(orientation, detector, det_distance, uvw, verbose=False):
    '''
    Compute the ellipsis associated with the given zone axis. All lattice
    planes sharing this zone axis will diffract along that ellipse.

    :param orientation: The crystal orientation.
    :param detector: An instance of the Detector2D class.
    :param float det_distance: the object - detector distance (along the X-axis).
    :param uvw: An instance of the HklDirection representing the zone axis.
    :param bool verbose: activate verbose mode (default False).
    :returns data: the (u, v) data (in pixel unit) representing the ellipsis.
    '''
    Bt = orientation.orientation_matrix().transpose()
    ZA = Bt.dot(uvw.direction())
    ZAD = det_distance * ZA / ZA[0]  # ZAD is pointing towards X>0
    X = np.array([1., 0., 0.])
    # psi = np.arccos(np.dot(ZAD/np.linalg.norm(ZAD), X))
    from math import atan2, pi
    psi = atan2(np.linalg.norm(np.cross(ZA, X)), np.dot(ZA, X))  # use cross product
    eta = pi / 2 - atan2(ZAD[1],
                         ZAD[2])  # atan2(y, x) compute the arc tangent of y/x considering the sign of both y and x
    (uc, vc) = (detector.ucen - ZAD[1] / detector.pixel_size, detector.vcen - ZAD[2] / detector.pixel_size)
    e = np.tan(psi)  # this assume the incident beam is normal to the detector
    '''
    # wrong formulae from Amoros (The Laue Method)...
    a = det_distance*np.tan(psi)/detector.pixel_size
    b = det_distance*np.sin(psi)/detector.pixel_size
    '''
    # the proper equation is:
    a = abs(0.5 * det_distance * np.tan(2 * psi) / detector.pixel_size)
    b = abs(0.5 * det_distance * np.tan(2 * psi) * np.sqrt(1 - np.tan(psi) ** 2) / detector.pixel_size)
    if verbose:
        print('angle psi (deg) is', psi * 180 / np.pi)
        print('angle eta (deg) is', eta * 180 / np.pi)
        print('ZA cross the det plane at', ZAD)
        print('ZA crosses the detector at (%.3f,%.3f) mm or (%d,%d) pixels' % (ZAD[1], ZAD[2], uc, vc))
        print('ellipse eccentricity is %f' % e)
        print('ellipsis major and minor half axes are a=%.1f and b=%.1f' % (a, b))
    # use a parametric curve to plot the ellipsis
    t = np.linspace(0., 2 * np.pi, 101)
    x = a * np.cos(t)
    y = b * np.sin(t)
    data = np.array([x, y])
    # rotate the ellipse
    R = np.array([[np.cos(eta), -np.sin(eta)], [np.sin(eta), np.cos(eta)]])
    data = np.dot(R, data)  # rotate our ellipse
    # move one end of the great axis to the direct beam position
    data[0] += detector.ucen - a * np.cos(eta)
    data[1] += detector.vcen - a * np.sin(eta)
    return data


def diffracted_vector(hkl, orientation, min_theta=0.1, verbose=False):
    Bt = orientation.orientation_matrix().transpose()
    (h, k, l) = hkl.miller_indices()
    # this hkl plane will select a particular lambda
    (the_energy, theta) = select_lambda(hkl, orientation, verbose=verbose)
    if abs(theta) < min_theta * np.pi / 180:  # skip angles < min_theta deg
        return None
    lambda_nm = 1.2398 / the_energy
    X = np.array([1., 0., 0.]) / lambda_nm
    Gs = Bt.dot(hkl.scattering_vector())
    if verbose:
        print('bragg angle for %d%d%d reflection is %.1f' % (h, k, l, hkl.bragg_angle(the_energy) * 180 / np.pi))
    assert (abs(theta - hkl.bragg_angle(the_energy)) < 1e-6)  # verify than theta_bragg is equal to the glancing angle
    # compute diffracted direction
    K = X + Gs
    return K


def compute_Laue_pattern(orientation, detector, hklplanes=None, spectrum=None, spectrum_thr=0.,
                         r_spot=5, color_spots_by_energy=False, inverted=False, show_direct_beam=False):
    '''
    Compute a transmission Laue pattern. The data array of the given
    `Detector2d` instance is initialized with the result.
    
    The incident beam is assumed to be along the X axis: (1, 0, 0). The crystal can have any orientation and uses 
    an instance of the `Orientation` class. The `Detector2d` instance holds all the geometry (detector size and 
    position).

    :param orientation: The crystal orientation.
    :param detector: An instance of the Detector2d class.
    :param list hklplanes: A list of the lattice planes to include in the pattern.
    :param spectrum: A two columns array of the spectrum to use for the calculation.
    :param float spectrum_thr: The threshold to use to determine if a wave length is contributing or not.
    :param int r_spot: Size of the spots on the detector in pixel (5 by default)
    :param bool color_spots_by_energy: Flag to color the diffraction spots in the image by the diffracted beam energy (in keV).
    :param bool inverted: A flag to control if the pattern needs to be inverted.
    :param bool show_direct_beam: A flag to control if the direct beam is shown.
    :returns: the computed pattern as a numpy array.
    '''
    detector.data = np.zeros(detector.size, dtype=np.uint8)
    # create a small square image for one spot
    spot = np.ones((2 * r_spot + 1, 2 * r_spot + 1), dtype=detector.data.dtype)
    max_val = np.iinfo(detector.data.dtype.type).max  # 255 here
    if show_direct_beam:
        add_to_image(detector.data, max_val * spot, (detector.ucen, detector.vcen))

    if spectrum is not None:
        print('using spectrum')
        indices = np.argwhere(spectrum[:, 1] > spectrum_thr)
        E_min = float(spectrum[indices[0], 0])
        E_max = float(spectrum[indices[-1], 0])
        lambda_min = lambda_keV_to_nm(E_max)
        lambda_max = lambda_keV_to_nm(E_min)
        print('energy bounds: [{0:.1f}, {1:.1f}] keV'.format(E_min, E_max))

    for hkl in hklplanes:
        (the_energy, theta) = select_lambda(hkl, orientation, verbose=False)
        if spectrum is not None:
            if abs(the_energy) < E_min or abs(the_energy) > E_max:
                #print('skipping reflection {0:s} which would diffract at {1:.1f}'.format(hkl.miller_indices(), abs(the_energy)))
                continue
            #print('including reflection {0:s} which will diffract at {1:.1f}'.format(hkl.miller_indices(), abs(the_energy)))
        K = diffracted_vector(hkl, orientation)
        if K is None or np.dot([1., 0., 0.], K) == 0:
            continue  # skip diffraction // to the detector
        R = detector.project_along_direction(K, origin=[0., 0., 0.])
        (u, v) = detector.lab_to_pixel(R)
        if u >= 0 and u < detector.size[0] and v >= 0 and v < detector.size[1]:
            print('diffracted beam will hit the detector at (%.3f, %.3f) mm or (%d, %d) pixels' % (R[1], R[2], u, v))
            print('diffracted beam energy is {0}'.format(abs(the_energy)))
        # mark corresponding pixels on the image detector
        if color_spots_by_energy:
            add_to_image(detector.data, abs(the_energy) * spot.astype(float), (u, v))
        else:
            add_to_image(detector.data, max_val * spot, (u, v))
    if inverted:
        print('inverting image')
        detector.data = np.invert(detector.data)
    return detector.data

def gnomonic_projection(detector):
    '''This function carries out the gnomonic projection of the detector image.
    
    The data must be of uint8 type (between 0 and 255) with diffraction spots equals to 255.
    The function create a new detector instance (think of it as a virtual detector) located at the same position 
    as the given detector and with an inverse pixel size. The gnomonic projection is stored into this new detector data.
    
    :param RegArrayDetector2d detector: the detector instance with the data from which to compute the projection.
    :returns RegArrayDetector2d gnom: A virtual detector with the gnomonic projection as its data.
    '''
    assert detector.data.dtype == np.uint8
    dif = detector.data == 255
    u = np.linspace(0, detector.size[0] - 1, detector.size[0])
    v = np.linspace(0, detector.size[1] - 1, detector.size[1])
    vv, uu = np.meshgrid(v, u)
    r_pxl = np.sqrt((uu - detector.ucen) ** 2 + (vv - detector.vcen) ** 2)  # pixel
    r = r_pxl * detector.pixel_size  # distance from the incident beam to the pxl in mm
    theta = 0.5 * np.arctan(r / detector.ref_pos[0])  # bragg angle rad
    p = detector.ref_pos[0] * np.tan(np.pi / 2 - theta)  # distance from the incident beam to the gnomonic projection mm

    # compute gnomonic projection coordinates in mm
    u_dif = (uu[dif] - detector.ucen) * detector.pixel_size  # mm, wrt detector center
    v_dif = (vv[dif] - detector.vcen) * detector.pixel_size  # mm, wrt detector center
    ug_mm = - u_dif * p[dif] / r[dif]  # mm, wrt detector center
    vg_mm = - v_dif * p[dif] / r[dif]  # mm, wrt detector center
    print('first projected spot in uv (mm):', (ug_mm[0], vg_mm[0]))
    print('last projected spot in uv (mm):', (ug_mm[-1], vg_mm[-1]))
    print('first projected vector in XYZ (mm):', (detector.ref_pos[0], -ug_mm[0], -vg_mm[0]))
    print('last projected vector in XYZ (mm):', (detector.ref_pos[0], -ug_mm[-1], -vg_mm[-1]))

    # create 2d image of the gnomonic projection because the gnomonic projection is bigger than the pattern
    from pymicro.xray.detectors import RegArrayDetector2d
    gnom = RegArrayDetector2d(size=np.array(detector.size))
    gnom.ref_pos = detector.ref_pos
    gnom.pixel_size = 1. / detector.pixel_size  # mm

    gnom.data = np.zeros(gnom.size, dtype=np.uint8)
    u_gnom = np.linspace(0, gnom.size[0] - 1, gnom.size[0])
    v_gnom = np.linspace(0, gnom.size[1] - 1, gnom.size[1])
    vv_gnom, uu_gnom = np.meshgrid(v_gnom, u_gnom)

    # TODO use a function to go from mm to pixel coordinates
    u_px = gnom.ucen + u_dif / gnom.pixel_size  # pixel, wrt (top left corner of gnom detector)
    v_px = gnom.vcen + v_dif / gnom.pixel_size  # pixel, wrt (top left corner of gnom detector)
    ug_px = gnom.ucen + ug_mm / gnom.pixel_size  # pixel, wrt (top left corner of gnom detector)
    vg_px = gnom.vcen + vg_mm / gnom.pixel_size  # pixel, wrt (top left corner of gnom detector)
    print(ug_px)
    print(vg_px)

    # remove pixel outside of the ROI (TODO use masked numpy array)
    vg_px[ug_px > gnom.size[0] - 1] = gnom.vcen  # change vg_px first when testing on ug_px
    ug_px[ug_px > gnom.size[0] - 1] = gnom.ucen
    vg_px[ug_px < 0] = gnom.vcen
    ug_px[ug_px < 0] = gnom.ucen

    ug_px[vg_px > gnom.size[1] - 1] = gnom.ucen  # change ug_px first when testing on vg_px
    vg_px[vg_px > gnom.size[1] - 1] = gnom.vcen
    ug_px[vg_px < 0] = gnom.ucen
    vg_px[vg_px < 0] = gnom.vcen

    print(ug_px.min())
    assert (ug_px.min() >= 0)
    assert (ug_px.max() < gnom.size[0])
    assert (vg_px.min() >= 0)
    assert (vg_px.max() < gnom.size[1])

    # convert pixel coordinates to integer
    ug_px = ug_px.astype(np.uint16)
    vg_px = vg_px.astype(np.uint16)
    print('after conversion to uint16:')
    print(ug_px)
    print(vg_px)

    gnom.data[ug_px, vg_px] = 1
    return gnom

if __name__ == '__main__':
    from matplotlib import pyplot as plt, cm, rcParams

    rcParams.update({'font.size': 12})
    rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

    from pymicro.xray.detectors import Varian2520

    det_distance = 100
    detector = Varian2520()
    detector.pixel_size = 0.127  # mm
    det_size_pixel = np.array((1950, 1456))
    det_size_mm = det_size_pixel * detector.pixel_size  # mm

    # orientation of our single crystal
    from pymicro.crystal.lattice import *

    ni = Lattice.from_symbol('Ni')
    from pymicro.crystal.microstructure import *

    phi1 = 10  # deg
    phi = 20  # deg
    phi2 = 10  # deg
    orientation = Orientation.from_euler([phi1, phi, phi2])

    # '''
    # uvw = HklDirection(1, 1, 0, ni)
    uvw = HklDirection(5, 1, 0, ni)
    ellipsis_data = compute_ellipsis(orientation, detector, det_distance, uvw)
    # '''
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
    # plt.savefig('Laue_plot_cube_distance_%02d.pdf' % det_distance)
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
