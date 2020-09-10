"""The laue module provide helpers functions to work with polychromatic X-ray diffraction.
"""
import numpy as np
from math import cos, sin, tan, atan2, pi
from pymicro.crystal.lattice import HklPlane, Symmetry, HklDirection, HklObject
from pymicro.crystal.microstructure import Orientation
from pymicro.xray.xray_utils import lambda_nm_to_keV, lambda_keV_to_nm
from pymicro.xray.dct import add_to_image
import os


def select_lambda(hkl, orientation, Xu=(1., 0., 0.), verbose=False):
    """
    Compute the wavelength corresponding to the first order reflection
    of a given lattice plane in the specified orientation.

    :param hkl: The given lattice plane.
    :param orientation: The orientation of the crystal lattice.
    :param Xu: The unit vector of the incident X-ray beam (default along the X-axis).
    :param bool verbose: activate verbose mode (default False).
    :returns tuple: A tuple of the wavelength value (keV) and the corresponding Bragg angle (radian).
    """
    (h, k, l) = hkl.miller_indices()
    d_hkl = hkl.interplanar_spacing()
    Gc = hkl.scattering_vector()
    gt = orientation.orientation_matrix().transpose()
    Gs = gt.dot(Gc)
    # compute the theta angle between X and the scattering vector which is theta + pi/2
    theta = np.arccos(np.dot(Xu, Gs / np.linalg.norm(Gs))) - pi / 2
    # find the lambda selected by this hkl plane
    the_energy = lambda_nm_to_keV(2 * d_hkl * np.sin(theta))
    if verbose:
        print('\n** Looking at plane %d%d%d **' % (h, k, l))
        print('scattering vector in laboratory frame = %s' % Gs)
        print('glancing angle between incident beam and the hkl plane is %.1f deg' % (theta * 180 / pi))
        print('corresponding energy is %.1f keV' % the_energy)
    return the_energy, theta


def build_list(lattice=None, max_miller=3, extinction=None, Laue_extinction=False, max_keV=120.):
    hklplanes = []
    indices = range(-max_miller, max_miller + 1)
    for h in indices:
        for k in indices:
            for l in indices:
                if h == k == l == 0:  # skip (0, 0, 0)
                    continue
                if not extinction :
                    hklplanes.append(HklPlane(h, k, l, lattice))
                if extinction == 'FCC':
                    # take plane if only all odd or all even hkl indices
                    if (h % 2 == 0 and k % 2 == 0 and l % 2 == 0) or (h % 2 == 1 and k % 2 == 1 and l % 2 == 1):
                        hklplanes.append(HklPlane(h, k, l, lattice))
                if extinction == 'BCC':
                    # take plane only if the sum indices is even
                    if ((h**2 + k**2 + l**2) % 2 == 0):
                        hklplanes.append(HklPlane(h, k, l, lattice))
    if Laue_extinction is True:
        lam_min = lambda_keV_to_nm(max_keV)
        val = 2. * lattice._lengths[0] / lam_min # lattice have to be cubic !
        print('Limit value is %d' % val)
        for hkl in hklplanes:
            (h, k, l) = hkl.miller_indices()
            test = h ** 2 + k ** 2 + l ** 2
            if val < test: # TODO check the test
                hklplanes.remove(HklPlane(h, k, l, lattice))

    return hklplanes


def compute_ellipsis(orientation, detector, uvw, Xu=(1., 0., 0.), n=101, verbose=False):
    """
    Compute the ellipsis associated with the given zone axis. 
    
    The detector is supposed to be normal to the X-axis, but the incident wave vector can be specified.
    All lattice planes sharing this zone axis will diffract along that ellipse.

    :param orientation: The crystal orientation.
    :param detector: An instance of the Detector2D class.
    :param uvw: An instance of the HklDirection representing the zone axis.
    :param Xu: The unit vector of the incident X-ray beam (default along the X-axis).
    :param int n: number of poits used to define the ellipse.
    :param bool verbose: activate verbose mode (default False).
    :returns data: the (Y, Z) data (in mm unit) representing the ellipsis.
    """
    gt = orientation.orientation_matrix().transpose()
    za = gt.dot(uvw.direction())  # zone axis unit vector in lab frame
    OA = detector.project_along_direction(za)  # vector from the origin to projection of the zone axis onto the detector
    OC = detector.project_along_direction(Xu)  # C is the intersection of the direct beam with the detector
    ON = detector.project_along_direction(detector.w_dir)  # N is the intersection of the X-axis with the detector plane
    CA = OA - OC
    NA = OA - ON
    # psi = np.arccos(np.dot(ZAD/np.linalg.norm(ZAD), Xu))
    psi = atan2(np.linalg.norm(np.cross(za, Xu)), np.dot(za, Xu))  # use cross product instead of arccos
    nu = atan2(np.linalg.norm(np.cross(za, detector.w_dir)), np.dot(za, detector.w_dir))
    e = np.sin(nu) / np.cos(psi)  # ellipis excentricity (general case), e reduces to tan(psi) when the incident beam is normal to the detector (nu = psi)
    '''
    The major axis direction is given by the intersection of the plane defined by ON and OA and by the detector 
    plane. It does not depends on Xu, like the cone angle. If the detector is perpendicular to the X-axis, 
    this direction is given by NA. both N and A are located on the major axis.
    '''
    eta = pi / 2 - atan2(NA[1],
                         NA[2])  # atan2(y, x) compute the arc tangent of y/x considering the sign of both y and x
    (uc, vc) = (detector.ucen - OA[1] / detector.pixel_size, detector.vcen - OA[2] / detector.pixel_size)
    '''
    # wrong formulae from Amoros (The Laue Method)...
    a = ON * np.tan(psi)
    b = ON * np.sin(psi)
    # the proper equation for normal incidence are (Cruickshank 1991):
    a = abs(0.5 * ON * np.tan(2 * psi))
    b = abs(0.5 * ON * np.tan(2 * psi) * np.sqrt(1 - np.tan(psi) ** 2))
    this equation does not hold if the incident beam is not normal with the detector !
    the general equation is:
    2.a / ON = tan(nu) + tan(psi - nu) + sin(psi) / (cos(nu) * cos(psi + nu))
    this reduces when nu = psi to 2.a / ON = np.tan(2.psi)
    '''
    #factor = tan(nu) + tan(psi - nu) + sin(psi) / (cos(nu) * cos(psi + nu))
    #a = 0.5 * np.linalg.norm(ON) * factor
    # a = abs(0.5 * np.linalg.norm(ON) * np.tan(2 * psi))
    # use the factorized version of the factor calculated in the article
    factor = cos(psi) * sin(psi) / (cos(psi + nu) * cos(psi - nu))
    a = np.linalg.norm(ON) * factor
    b = a * np.sqrt(1 - e ** 2)
    if verbose:
        print('angle nu (deg) is %.3f' % (nu * 180 / pi))
        print('angle psi (deg) is %.3f' % (psi * 180 / pi))
        print('angle eta (deg) is %.3f' % (eta * 180 / pi))
        print('direct beam crosses the det plane at %s' % OC)
        print('zone axis crosses the det plane at %s' % OA)
        print('zone axis crosses the detector at (%.3f,%.3f) mm or (%d,%d) pixels' % (OA[1], OA[2], uc, vc))
        print('ellipse eccentricity is %f' % e)
        print('ellipsis major and minor half axes are a=%.3f and b=%.3f' % (a, b))
    # use a parametric curve to plot the ellipsis
    t = np.linspace(0., 2 * pi, n)
    x = a * np.cos(t)
    y = b * np.sin(t)
    data = np.array([x, y])
    # rotate the ellipse
    R = np.array([[np.cos(eta), -np.sin(eta)], [np.sin(eta), np.cos(eta)]])
    data = np.dot(R, data)  # rotate our ellipse
    # move one end of the great axis to the direct beam position
    '''
    NB: in the general case, point C (intersection of the direct beam and the detector) is on the ellipse 
    but not lying on the great axis. Point A and N still lie on the major axis, so the translation can be determined. 
    '''
    NX = np.linalg.norm(ON) * tan(psi - nu)
    if verbose:
        print('****************** NX = %.1f' % NX)
    data[0] += (a - NX) * np.cos(eta)
    data[1] += (a - NX) * np.sin(eta)
    #data[0] += ON[1] + a * np.cos(eta)
    #data[1] += ON[2] + a * np.sin(eta)
    yz_data = np.empty((n, 2), dtype=float)
    for i in range(n):
        yz_data[i, 0] = data[0, i]
        yz_data[i, 1] = data[1, i]
    return yz_data


def diffracted_vector(hkl, orientation, Xu=(1., 0., 0.), min_theta=0.1, use_friedel_pair=True, verbose=False):
    """Compute the diffraction vector for a given reflection of a crystal.

    This method compute the diffraction vector.
    The incident wave vector is along the X axis by default but can be changed by specifying any unit vector.

    :param hkl: an instance of the `HklPlane` class.
    :param orientation: an instance of the `Orientation` class.
    :param Xu: The unit vector of the incident X-ray beam (default along the X-axis).
    :param float min_theta: the minimum considered Bragg angle (in radian).
    :param bool use_friedel_pair: also consider the Friedel pairs of the lattice plane. 
    :param bool verbose: a flag to activate verbose mode.
    :return: the diffraction vector as a numpy array.
    """
    gt = orientation.orientation_matrix().transpose()
    (h, k, l) = hkl.miller_indices()
    # this hkl plane will select a particular lambda
    (the_energy, theta) = select_lambda(hkl, orientation, Xu=Xu, verbose=verbose)
    if the_energy < 0:
        if use_friedel_pair:
            # use the Friedel pair of this lattice plane
            hkl = hkl.friedel_pair()
            (h, k, l) = hkl.miller_indices()
            if verbose:
                print('switched to friedel pair (%d,%d,%d)' % (h, k, l))
            (the_energy, theta) = select_lambda(hkl, orientation, Xu=Xu, verbose=verbose)
        else:
            return None
    assert(theta >= 0)
    if theta < min_theta * pi / 180:  # skip angles < min_theta deg
        return None
    lambda_nm = 1.2398 / the_energy
    X = np.array(Xu) / lambda_nm
    Gs = gt.dot(hkl.scattering_vector())
    if verbose:
        print('bragg angle for %d%d%d reflection is %.1f' % (h, k, l, hkl.bragg_angle(the_energy) * 180 / pi))
    assert (abs(theta - hkl.bragg_angle(the_energy)) < 1e-6)  # verify than theta_bragg is equal to the glancing angle
    # compute diffracted direction
    K = X + Gs
    return K


def diffracted_intensity(hkl, I0=1.0, symbol='Ni', verbose=False):
    '''Compute the diffracted intensity.
    
    This compute a number representing the diffracted intensity within the kinematical approximation. 
    This number includes many correction factors:
    
     * atomic factor
     * structural factor
     * polarisation factor
     * Debye-Waller factor
     * absorption factor
     * ...
    
    :param HklPlane hkl: the hkl Bragg reflection.
    :param float I0: intensity of the incident beam.
    :param str symbol: string representing the considered lattice.
    :param bool verbose: flag to activate verbose mode.
    :return: the diffracted intensity.
    :rtype: float.
    '''
    # TODO find a way to load the atom scattering data from the lattice... and not load it for every reflection!
    path = os.path.dirname(__file__)
    scat_data = np.genfromtxt(os.path.join(path, 'data', '%s_atom_scattering.txt' % symbol))
    # scale the intensity with the atomic scattering factor
    q = 1 / (2 * hkl.interplanar_spacing())
    try:
        index = (scat_data[:, 0] < q).tolist().index(False)
    except ValueError:
        index = -1
    fa = scat_data[index, 1] / scat_data[0, 1]  # should we normalize with respect to Z?

    I = I0 * fa

    if verbose:
        print('q=%f in nm^-1 -> fa=%.3f' % (q, fa))
        print('intensity:', I)
    return I


def compute_Laue_pattern(orientation, detector, hkl_planes=None, Xu=np.array([1., 0., 0.]), use_friedel_pair=False,
                         spectrum=None, spectrum_thr=0., r_spot=5, color_field='constant', inverted=False,
                         show_direct_beam=False, verbose=False):
    """
    Compute a transmission Laue pattern. The data array of the given
    `Detector2d` instance is initialized with the result.
    
    The incident beam is assumed to be along the X axis: (1, 0, 0) but can be changed to any direction. 
    The crystal can have any orientation using an instance of the `Orientation` class. 
    The `Detector2d` instance holds all the geometry (detector size and position).

    A parameter controls the meaning of the values in the diffraction spots in the image. It can be just a constant 
    value, the diffracted beam energy (in keV) or the intensity as computed by the :py:meth:`diffracted_intensity` 
    method.

    :param orientation: The crystal orientation.
    :param detector: An instance of the Detector2d class.
    :param list hkl_planes: A list of the lattice planes to include in the pattern.
    :param Xu: The unit vector of the incident X-ray beam (default along the X-axis).
    :param bool use_friedel_pair: also consider the Friedel pair of each lattice plane in the list as candidate for diffraction. 
    :param spectrum: A two columns array of the spectrum to use for the calculation.
    :param float spectrum_thr: The threshold to use to determine if a wave length is contributing or not.
    :param int r_spot: Size of the spots on the detector in pixel (5 by default)
    :param str color_field: a traing describing, must be 'constant', 'energy' or 'intensity'
    :param bool inverted: A flag to control if the pattern needs to be inverted.
    :param bool show_direct_beam: A flag to control if the direct beam is shown.
    :param bool verbose: activate verbose mode (False by default).
    :return: the computed pattern as a numpy array.
    """
    detector.data = np.zeros(detector.size, dtype=np.float32)
    # create a small square image for one spot
    spot = np.ones((2 * r_spot + 1, 2 * r_spot + 1), dtype=np.uint8)
    max_val = np.iinfo(np.uint8).max  # 255 here
    direct_beam_lab = detector.project_along_direction(Xu)
    direct_beam_pix = detector.lab_to_pixel(direct_beam_lab)[0]
    if show_direct_beam:
        add_to_image(detector.data, max_val * 3 * spot, (direct_beam_pix[0], direct_beam_pix[1]))
    if spectrum is not None:
        print('using spectrum')
        #indices = np.argwhere(spectrum[:, 1] > spectrum_thr)
        E_min = min(spectrum) #float(spectrum[indices[0], 0])
        E_max = max(spectrum) #float(spectrum[indices[-1], 0])
        lambda_min = lambda_keV_to_nm(E_max)
        lambda_max = lambda_keV_to_nm(E_min)
        #if verbose:
        print('energy bounds: [{0:.1f}, {1:.1f}] keV'.format(E_min, E_max))

    for hkl in hkl_planes:
        (the_energy, theta) = select_lambda(hkl, orientation, Xu=Xu, verbose=False)
        if the_energy < 0:
            if use_friedel_pair:
                if verbose:
                    print('switching to Friedel pair')
                hkl = hkl.friedel_pair()
                (the_energy, theta) = select_lambda(hkl, orientation, Xu=Xu, verbose=False)
            else:
                continue
        assert(the_energy >= 0)
        if spectrum is not None:
            if the_energy < E_min or the_energy > E_max:
                #print('skipping reflection {0:s} which would diffract at {1:.1f}'.format(hkl.miller_indices(), abs(the_energy)))
                continue
                #print('including reflection {0:s} which will diffract at {1:.1f}'.format(hkl.miller_indices(), abs(the_energy)))
        K = diffracted_vector(hkl, orientation, Xu=Xu, use_friedel_pair=False, verbose=verbose)
        if K is None or np.dot(Xu, K) == 0:
            continue  # skip diffraction // to the detector
        d = np.dot((detector.ref_pos - np.array([0., 0., 0.])), detector.w_dir) / np.dot(K, detector.w_dir)
        if d < 0:
            if verbose:
                print('skipping diffraction not towards the detector')
            continue

        R = detector.project_along_direction(K, origin=[0., 0., 0.])
        (u, v) = detector.lab_to_pixel(R)[0]
        if verbose and u >= 0 and u < detector.size[0] and v >= 0 and v < detector.size[1]:
            print('* %d%d%d reflexion' % hkl.miller_indices())
            print('diffracted beam will hit the detector at (%.3f, %.3f) mm or (%d, %d) pixels' % (R[1], R[2], u, v))
            print('diffracted beam energy is {0:.1f} keV'.format(abs(the_energy)))
            print('Bragg angle is {0:.2f} deg'.format(abs(theta * 180 / pi)))
        # mark corresponding pixels on the image detector
        if color_field == 'constant':
            add_to_image(detector.data, max_val * spot, (u, v))
        elif color_field == 'energy':
            add_to_image(detector.data, abs(the_energy) * spot.astype(float), (u, v))
        elif color_field == 'intensity':
            I = diffracted_intensity(hkl, I0=max_val, verbose=verbose)
            add_to_image(detector.data, I * spot, (u, v))
        else:
            raise ValueError('unsupported color_field: %s' % color_field)
    if inverted:
        # np.invert works only with integer types
        print('casting to uint8 and inverting image')
        # limit maximum to max_val (255) and convert to uint8
        over = detector.data > 255
        detector.data[over] = 255
        detector.data = detector.data.astype(np.uint8)
        detector.data = np.invert(detector.data)
    return detector.data


def gnomonic_projection_point(data, OC=None):
    """compute the gnomonic projection of a given point or series of points in the general case.

    This methods *does not* assumes the incident X-ray beam is along (1, 0, 0). This is accounted for with the 
    parameter OC which indicates the center of the projection (the incident beam intersection with the detector). 
    A conditional treatment is done as the projection is is faster to compute in the case of normal incidence.

    The points coordinates are passed along with a single array which must be of size (n, 3) where n is the number of 
    points. If a single point is used, the data can indifferently be of size (1, 3) or (3). 

    :param ndarray data: array of the point(s) coordinates in the laboratory frame, aka OR components.
    :param ndarray OC: coordinates of the center of the gnomonic projection in the laboratory frame.
    :return data_gp: array of the projected point(s) coordinates in the laboratory frame.
    """
    if data.ndim == 1:
        data = data.reshape((1, 3))
    if OC is None:
        # fall back on normal incidence case
        r = np.sqrt(data[:, 1] ** 2 + data[:, 2] ** 2)  # mm
        theta = 0.5 * np.arctan(r / data[:, 0])
        p = data[:, 0] * np.tan(pi / 2 - theta)  # distance from the incident beam to the gnomonic projection mm
        data_gp = np.zeros_like(data)  # mm
        data_gp[:, 0] = data[:, 0]
        data_gp[:, 1] = - data[:, 1] * p / r
        data_gp[:, 2] = - data[:, 2] * p / r
    else:
        data_gp = np.zeros_like(data)  # mm
        data_gp[:, 0] = data[:, 0]
        for i in range(data.shape[0]):
            R = data[i]  # diffraction spot position in the laboratory frame
            CR = R - OC
            alpha = atan2(
                np.linalg.norm(np.cross(OC / np.linalg.norm(OC), CR / np.linalg.norm(CR))),
                np.dot(OC / np.linalg.norm(OC), CR / np.linalg.norm(CR))) - pi / 2
            # the Bragg angle can be measured using the diffraction spot position
            theta = 0.5 * np.arccos(np.dot(OC / np.linalg.norm(OC), R / np.linalg.norm(R)))
            r = np.sqrt(CR[1] ** 2 + CR[2] ** 2)  # mm
            # distance from the incident beam to the gnomonic projection mm
            p = np.linalg.norm(OC) * cos(theta) / sin(theta - alpha)
            data_gp[i, 1] = OC[1] - CR[1] * p / r
            data_gp[i, 2] = OC[2] - CR[2] * p / r
    return data_gp


def gnomonic_projection(detector, pixel_size=None, OC=None, verbose=False):
    """This function carries out the gnomonic projection of the detector image.
    
    The data must be of uint8 type (between 0 and 255) with diffraction spots equals to 255.
    The function create a new detector instance (think of it as a virtual detector) located at the same position 
    as the given detector and with an inverse pixel size. The gnomonic projection is stored into this new detector data.
    The gnomonic projection of each white pixel (value at 255) is computed. The projection is carried out with respect 
    to the center detector (ucen, vcen) point.
    
    :param RegArrayDetector2d detector: the detector instance with the data from which to compute the projection.
    :param float pixel_size: pixel size to use in the virtual detector for the gnomonic projection.
    :param tuple OC: coordinates of the center of the gnomonic projection in the laboratory frame.
    :param bool verbose: flag to activate verbose mode.
    :returns RegArrayDetector2d gnom: A virtual detector with the gnomonic projection as its data.
    """
    assert detector.data.dtype == np.uint8
    dif = detector.data == 255  # boolean array used to select pixels with diffracted intensity
    dif_indices = np.where(dif)  # (ui, vi) tuple with 1D arrays of the coordinates u and v
    n = dif_indices[0].shape
    if verbose:
        print('%d points in the gnomonic projection' % n)
        print('center of the projection: %s' % OC)
    uv_mm = detector.pixel_to_lab(dif_indices[0], dif_indices[1])
    uvg_mm = gnomonic_projection_point(uv_mm, OC)  # mm
    print(uvg_mm.shape)

    # create the new virtual detector
    from pymicro.xray.detectors import RegArrayDetector2d
    gnom = RegArrayDetector2d(size=np.array(detector.size))
    gnom.ref_pos = detector.ref_pos  # same ref position as the actual detector
    if not pixel_size:
        gnom.pixel_size = 1. / detector.pixel_size  # mm
    else:
        gnom.pixel_size = pixel_size  # mm
    # TODO remove the next two lines
    gnom.ucen = detector.ucen
    gnom.vcen = detector.vcen

    # create the gnom.data array (zeros with pixels set to 1 for gnomonic projection points)
    gnom.data = np.zeros(gnom.size, dtype=np.uint8)
    # uvg_px = gnom.lab_to_pixel(uvg_mm)
    uvg_px = np.zeros((uvg_mm.shape[0], 2), dtype=np.int)
    for i in range(uvg_mm.shape[0]):
        uvg_px[i, :] = gnom.lab_to_pixel(uvg_mm[i, :])
    # filter out point outside the virtual detector
    detin = (uvg_px[:, 0] >= 0) & (uvg_px[:, 0] <= gnom.size[0] - 1) & \
            (uvg_px[:, 1] >= 0) & (uvg_px[:, 1] <= gnom.size[1] - 1)
    gnom.data[uvg_px[detin, 0], uvg_px[detin, 1]] = 1
    return gnom


def identify_hkl_from_list(hkl_list):
    if hkl_list[1] == hkl_list[2]:
        ids = [1, 3, 0]
    elif hkl_list[1] == hkl_list[3]:
        ids = [1, 2, 0]
    elif hkl_list[0] == hkl_list[2]:
        ids = [0, 3, 1]
    elif hkl_list[0] == hkl_list[3]:
        ids = [0, 2, 1]
    '''
    if hkl_list[1] == hkl_list[2]:
        if hkl_list[3] == hkl_list[4] and hkl_list[0] == hkl_list[5] or \
                                hkl_list[3] == hkl_list[5] and hkl_list[0] == hkl_list[4]:
            identified_list = [hkl_list[1], hkl_list[3], hkl_list[0]]

    elif hkl_list[1] == hkl_list[3]:
        if hkl_list[2] == hkl_list[4]:
            if hkl_list[0] == hkl_list[5]:
                identified_list = [hkl_list[1], hkl_list[2], hkl_list[0]]

        elif hkl_list[2] == hkl_list[5]:
            if hkl_list[0] == hkl_list[4]:
                identified_list = [hkl_list[1], hkl_list[2], hkl_list[0]]

    elif hkl_list[0] == hkl_list[2]:
        if hkl_list[3] == hkl_list[4]:
            if hkl_list[1] == hkl_list[5]:
                identified_list = [hkl_list[0], hkl_list[3], hkl_list[1]]

        elif hkl_list[3] == hkl_list[5]:
            if hkl_list[1] == hkl_list[4]:
                identified_list = [hkl_list[0], hkl_list[3], hkl_list[1]]

    elif hkl_list[0] == hkl_list[3]:
        if hkl_list[2] == hkl_list[4]:
            if hkl_list[1] == hkl_list[5]:
                identified_list = [hkl_list[0], hkl_list[2], hkl_list[1]]

        elif hkl_list[2] == hkl_list[5]:
            if hkl_list[1] == hkl_list[4]:
                identified_list = [hkl_list[0], hkl_list[2], hkl_list[1]]
    '''
    return [hkl_list[i] for i in ids]


def triplet_indexing(OP, angles_exp, angles_th, tol=1.0, verbose=False):
    '''
    evaluate each triplet composed by 3 diffracted points.
    the total number of triplet is given by the binomial coefficient (n, 3) = n*(n-1)*(n-2)/6
    '''
    nombre_triplet = 0
    OP_indexed = []
    for ti in range(len(OP) - 2):

        for tj in range(ti + 1, len(OP) - 1):

            triplet = np.zeros(3)
            for tk in range(tj + 1, len(OP)):
                nombre_triplet = nombre_triplet + 1
                print('\n\ntriplet number = {0:d}'.format(nombre_triplet))
                # look for possible matches for a given triplet
                triplet = [(ti, tj), (tj, tk), (ti, tk)]
                print('** testing triplet %s' % triplet)
                candidates = []  # will contain the 3 lists of candidates for a given triplet
                for (i, j) in triplet:
                    angle_to_match = angles_exp[i, j]
                    # find the index by looking to matching angles within the tolerance
                    cand = np.argwhere(
                        abs(angles_th['angle'] - angle_to_match) < tol).flatten()  # flatten since shape is (n, 1)
                    print('* couple (OP%d, OP%d) has angle %.2f -> %d couple candidate(s) in the angle_th list: %s' % (
                        i, j, angle_to_match, len(cand), cand))
                    candidates.append(cand)
                print('looking for real triplet in all %d candidates pairs:' % (
                    len(candidates[0]) * len(candidates[1]) * len(candidates[2])))
                for ci in candidates[0]:
                    for cj in candidates[1]:
                        for ck in candidates[2]:
                            # make a list of the candidate hkl plane indices, based on the angle match
                            hkl_list = [angles_th['hkl1'][ci], angles_th['hkl2'][ci],
                                        angles_th['hkl1'][cj], angles_th['hkl2'][cj],
                                        angles_th['hkl1'][ck], angles_th['hkl2'][ck]]
                            # look how many unique planes are in the list (should be 3 for a triplet)
                            if verbose:
                                print('- candidates with indices %s correspond to %s' % ([ci, cj, ck], hkl_list))
                            unique_indices = np.unique(hkl_list)
                            if len(unique_indices) != 3:
                                if verbose:
                                    print('not a real triplet, skipping this one')
                                continue
                            print('this is a triplet: %s' % unique_indices)
                            # the points can now be identified from the hkl_list ordering
                            identified_list = identify_hkl_from_list(hkl_list)
                            full_list = [tj, tk, ti]
                            full_list.extend(identified_list)
                            print(full_list)
                            OP_indexed.append(full_list)
                            print('*  (OP%d) -> %d ' % (tj, identified_list[0]))
                            print('*  (OP%d) -> %d ' % (tk, identified_list[1]))
                            print('*  (OP%d) -> %d ' % (ti, identified_list[2]))
    print('indexed list length is %d' % len(OP_indexed))
    print(OP_indexed)
    return OP_indexed


def transformation_matrix(hkl_plane_1, hkl_plane_2, xyz_normal_1, xyz_normal_2):
    # create the vectors representing this frame in the crystal coordinate system
    e1_hat_c = hkl_plane_1.normal()
    e2_hat_c = np.cross(hkl_plane_1.normal(), hkl_plane_2.normal()) / np.linalg.norm(
        np.cross(hkl_plane_1.normal(), hkl_plane_2.normal()))
    e3_hat_c = np.cross(e1_hat_c, e2_hat_c)
    e_hat_c = np.array([e1_hat_c, e2_hat_c, e3_hat_c])
    # create local frame attached to the indexed crystallographic features in XYZ
    e1_hat_star = xyz_normal_1
    e2_hat_star = np.cross(xyz_normal_1, xyz_normal_2) / np.linalg.norm(
        np.cross(xyz_normal_1, xyz_normal_2))
    e3_hat_star = np.cross(e1_hat_star, e2_hat_star)
    e_hat_star = np.array([e1_hat_star, e2_hat_star, e3_hat_star])
    # now build the orientation matrix
    orientation_matrix = np.dot(e_hat_c.T, e_hat_star)
    return orientation_matrix


def confidence_index(votes):
    n = np.sum(votes)  # total number of votes
    v1 = max(votes)
    votes[votes.index(v1)] = 0.
    v2 = max(votes)
    ci = (1. * (v1 - v2)) / n
    return ci


def poll_system(g_list, dis_tol=1.0, verbose=False):
    """
    Poll system to sort a series of orientation matrices determined by the indexation procedure.

    For each orientation matrix, check if it corresponds to an existing solution, if so: vote for it,
    if not add a new solution to the list
    :param list g_list: the list of orientation matrices (should be in the fz)
    :param float dis_tol: angular tolerance (degrees)
    :param bool verbose: activate verbose mode (False by default)
    :return: a tuple composed by the most popular orientation matrix, the corresponding vote number and the confidence index
    """
    solution_indices = [0]
    votes = [0]
    vote_index = np.zeros(len(g_list), dtype=int)
    dis_tol_rad = dis_tol * pi / 180
    from pymicro.crystal.microstructure import Orientation
    for i in range(len(g_list)):
        g = g_list[i]
        # rotations are already in the fundamental zone
        for j in range(len(solution_indices)):
            index = solution_indices[j]
            delta = np.dot(g, g_list[index].T)
            # compute misorientation angle in radians
            angle = Orientation.misorientation_angle_from_delta(delta)
            if verbose:
                print('j=%d -- angle=%f' % (j, angle))
            if angle <= dis_tol_rad:
                votes[j] += 1
                vote_index[i] = j
                if verbose:
                    print('angle (deg) is %.2f' % (180 / pi * angle))
                    print('vote list is now %s' % votes)
                    print('solution_indices list is now %s' % solution_indices)
                break
            elif j == len(solution_indices) - 1:
                solution_indices.append(i)
                votes.append(1)
                vote_index[i] = len(votes) - 1
                if verbose:
                    print('vote list is now %s' % votes)
                    print('solution_indices list is now %s' % solution_indices)
                break
    index_result = np.argwhere(votes == np.amax(votes)).flatten()
    if verbose:
        print('Max vote =', np.amax(votes))
        print('index result:', index_result)
        print('Number of equivalent solutions :', len(index_result))
        print(type(index_result))
        print(index_result.shape)
    final_orientation_matrix = []
    for n in range(len(index_result)):
        solutions = g_list[solution_indices[index_result[n]]]
        if verbose:
            print('Solution number {0:d} is'.format(n+1), solutions)
        final_orientation_matrix.append(solutions)
    result_vote = max(votes)
    ci = confidence_index(votes)
    vote_field = [votes[i] for i in vote_index]
    return final_orientation_matrix, result_vote, ci, vote_field


def index(hkl_normals, hkl_planes, tol_angle=0.5, tol_disorientation=1.0, symmetry=Symmetry.cubic, display=False):
    # angles between normal from the gnomonic projection
    angles_exp = np.zeros((len(hkl_normals), len(hkl_normals)), dtype=float)
    print('\nlist of angles between points on the detector')
    for i in range(len(hkl_normals)):
        for j in range(i + 1, len(hkl_normals)):
            angle = 180 / pi * np.arccos(np.dot(hkl_normals[i], hkl_normals[j]))
            angles_exp[i, j] = angle
            print('%.2f, OP%d, OP%d' % (angles_exp[i, j], i, j))
    # keep a list of the hkl values as string
    hkl_str = []
    for p in hkl_planes:
        (h, k, l) = p.miller_indices()
        hkl_str.append('(%d%d%d)' % (h, k, l))
    # compute theoretical angle between each plane normal, store the results using a structured array
    angles_th = np.empty(len(hkl_planes) * (len(hkl_planes) - 1) // 2,
                         dtype=[('angle', 'f4'), ('hkl1', 'i4'), ('hkl2', 'i4')])
    index = 0
    for i in range(len(hkl_planes)):
        for j in range(i + 1, len(hkl_planes)):
            angle = 180 / pi * np.arccos(np.dot(hkl_planes[i].normal(), hkl_planes[j].normal()))
            angles_th[index] = (angle, i, j)
            index += 1

    # sort the array by increasing angle
    angles_th.sort(order='angle')
    print('\nsorted list of angles between hkl plane normals')
    for i in range(len(angles_th)):
        print('%.3f, (%d, %d) -> %s, %s' % (
            angles_th['angle'][i], angles_th['hkl1'][i], angles_th['hkl2'][i], hkl_str[angles_th['hkl1'][i]],
            hkl_str[angles_th['hkl2'][i]]))

    # index by triplets
    normal_indexed = triplet_indexing(hkl_normals, angles_exp, angles_th, tol=tol_angle)
    print('indexed list length is %d' % len(normal_indexed))
    print(normal_indexed)

    # Compute the orientation matrix g for all different triplets
    g_indexation = []
    for i in range(len(normal_indexed)):
        # a given indexed triplet allow to construct 3 orientation matrices (which should be identical)
        pos = [[0, 1, 3, 4], [1, 2, 4, 5], [2, 0, 5, 3]]
        for j in range(3):
            orientation_matrix = transformation_matrix(
                hkl_planes[normal_indexed[i][pos[j][2]]], hkl_planes[normal_indexed[i][pos[j][3]]],
                hkl_normals[normal_indexed[i][pos[j][0]]], hkl_normals[normal_indexed[i][pos[j][1]]])
        # move to the fundamental zone
        om_fz = symmetry.move_rotation_to_FZ(orientation_matrix)  # we only add the third one
        g_indexation.append(om_fz)
    final_orientation_matrix, vote, ci, vote_field = poll_system(g_indexation, dis_tol=tol_disorientation)
    if final_orientation_matrix == 0:
        print('Troubles in the data set !')
    else:
        print('\n\n\n### FINAL SOLUTION(S) ###\n')
        for n in range(len(final_orientation_matrix)):
            print('- SOLUTION %d -' % (n + 1))
            final_orientation = Orientation(final_orientation_matrix[n])
            print(final_orientation.inFZ())
            print('- Cristal orientation in Fundamental Zone \n {} \n'.format(final_orientation.euler))
            print('- Rodrigues vector in the fundamental Zone \n {} \n'.format(final_orientation.rod))
            if display:
                from pymicro.crystal.texture import PoleFigure
                PoleFigure.plot(final_orientation, axis='Z')
        return final_orientation_matrix, ci

def zone_axis_list(angle, orientation, lattice,  max_miller=5,  Xu=np.array([1., 0., 0.]), verbose=False):
    """
    This function allows to get easily the Miller indices of zone axis present in a pattern.

    :param float list angle: the angle max or the zone axis angle range admissible around the detector center.
    :param orientation: The orientation of the crystal lattice.
    :param lattice: The corresponding crystal lattice, instance of Lattice.
    :param int max_miller: Maximal value allowed of Miller indices direction.
    :param array Xu: The unit vector of the incident X-ray beam (default along the X-axis).
    :param bool verbose: activate verbose mode (default False).
    :return: A list of HklDirection instance of all zone axis in the angle range.
    """
    ZA_list = []
    if len(angle) == 1:
        angle_max = angle[0] * np.pi / 180.
        angle_min = 0.
    if len(angle) == 2:
        angle_max = max(angle) * np.pi / 180.
        angle_min = min(angle) * np.pi / 180.
        print("Get the indices of directions between [%d, %d] degrees" %(min(angle), max(angle)))
    indices = range(-max_miller, max_miller+1)
    for h in indices:
        for k in indices:
            for l in indices:
                if h == k == l == 0:  # skip (0, 0, 0)
                    continue
                uvw = HklDirection(h, k, l, lattice)
                # compute the direction in the lab frame
                ZA = np.dot(orientation.orientation_matrix().transpose(), uvw.direction())
                psi = np.arccos(np.dot(ZA, Xu))
                if angle_min < psi < angle_max:
                    if verbose == True:
                        print('found zone axis [%d%d%d] at %.1f deg from incident beam' % (h,k,l,(psi*180/pi)))
                    ZA_list.append(uvw) #zones axis list which are inferior with max angle
    ZA_list = HklObject.skip_higher_order(ZA_list)
    return ZA_list

def get_gnomonic_edges(detector, gnom, OC=None, num_points=21):
    """
    This function allows to get the blind area of the gnomonic projection.

    :param RegArrayDetector2d detector: the detector instance with the data from which to compute the projection.
    :param RegArrayDetector2d gnom: A virtual detector with the gnomonic projection as its data.
    :param ndarray OC: coordinates of the center of the gnomonic projection in the laboratory frame.
    :param int num_points: number of points to describe an edge (minimum 2)
    :return: ndarray of gnomonic blind area edges coordinates.
    """
    # Get detector pixel edges
    uv_detector_edges = detector.get_edges(num_points=num_points, verbose=False)  # pixels
    # Compute edges position in the lab coordinates
    detector_edges_mm = detector.pixel_to_lab(uv_detector_edges[:, 0], uv_detector_edges[:, 1])
    # Apply the gnomonic projection  in the lab coordinates
    detector_edges_gp = gnomonic_projection_point(detector_edges_mm, OC=OC)  # mm
    # return the gnomonic projected point on the detector (pixel coordinates)
    return gnom.lab_to_pixel(detector_edges_gp)

def diffracting_normals_vector(gnom):
    """
    This function allows to get easily the diffracting normal vector from the gnomonic projection images.

    :param RegArrayDetector2d gnom: A virtual detector with the gnomonic projection as its data.
    :return: Normalized normal vector of diffracting plane
    """
    uv_g = np.argwhere(gnom.data == 1)  # points on the gnomonic projection
    OP = [gnom.pixel_to_lab(uv_g[i, 0], uv_g[i, 1]).tolist() for i in range(len(uv_g))]
    hkl_normals = [(n / np.linalg.norm(n)).tolist() for n in OP]  # normalized list of vectors
    print('%d normals found in the gnomonic projection' % len(hkl_normals))

    return hkl_normals

from pymicro.xray.experiment import ForwardSimulation, Experiment

class LaueForwardSimulation(ForwardSimulation):
    """Class to represent a Forward Simulation."""

    def __init__(self, verbose=False):
        super(LaueForwardSimulation, self).__init__('laue', verbose=verbose)
        self.hkl_planes = []
        self.max_miller = 5
        self.use_energy_limits = False
        self.exp = Experiment()

    def set_experiment(self, experiment):
        """Attach an X-ray experiment to this simulation."""
        self.exp = experiment

    def set_use_energy_limits(self, use_energy_limits):
        """Activate or deactivate the use of energy limits."""
        self.use_energy_limits = use_energy_limits

    def set_hkl_planes(self, hkl_planes):
        self.hkl_planes = hkl_planes

    def setup(self, include_grains=None):
        """Setup the forward simulation."""
        pass

    @staticmethod
    def fsim_laue(orientation, hkl_planes, positions, source_position):
        """Simulate Laue diffraction conditions based on a crystal orientation, a set of lattice planes and physical
        positions.

        This function is the work horse of the forward model. It uses a set of HklPlane instances and the voxel
        coordinates to compute the diffraction quantities.

        :param Orientation orientation: the crystal orientation.
        :param list hkl_planes: a list of `HklPlane` instances.
        :param list positions: a list of (x, y, z) positions.
        :param tuple source_position: a (x, y, z) tuple describing the source position.
        """
        n_hkl = len(hkl_planes)
        n_vox = len(positions)
        gt = orientation.orientation_matrix().transpose()

        # here we use list comprehension to avoid for loops
        d_spacings = [hkl.interplanar_spacing() for hkl in hkl_planes]  # size n_hkl
        G_vectors = [hkl.scattering_vector() for hkl in hkl_planes]  # size n_hkl, with 3 elements items
        Gs_vectors = [gt.dot(Gc) for Gc in G_vectors]  # size n_hkl, with 3 elements items
        Xu_vectors = [(pos - source_position) / np.linalg.norm(pos - source_position)
                      for pos in positions]  # size n_vox
        thetas = [np.arccos(np.dot(Xu, Gs / np.linalg.norm(Gs))) - np.pi / 2
                  for Xu in Xu_vectors
                  for Gs in Gs_vectors]  # size n_vox * n_hkl
        the_energies = [lambda_nm_to_keV(2 * d_spacings[i_hkl] * np.sin(thetas[i_Xu * n_hkl + i_hkl]))
                        for i_Xu in range(n_vox)
                        for i_hkl in range(n_hkl)]  # size n_vox * n_hkl
        X_vectors = [np.array(Xu_vectors[i_Xu]) / 1.2398 * the_energies[i_Xu * n_hkl + i_hkl]
                     for i_Xu in range(n_vox)
                     for i_hkl in range(n_hkl)]  # size n_vox * n_hkl
        K_vectors = [X_vectors[i_Xu * n_hkl + i_hkl] + Gs_vectors[i_hkl]
                     for i_Xu in range(n_vox)
                     for i_hkl in range(n_hkl)]  # size n_vox * n_hkl
        return Xu_vectors, thetas, the_energies, X_vectors, K_vectors

    def fsim_grain(self, gid=1):
        self.grain = self.exp.get_sample().get_microstructure().get_grain(gid)
        sample = self.exp.get_sample()
        lattice = sample.get_microstructure().get_lattice()
        source = self.exp.get_source()
        detector = self.exp.get_active_detector()
        data = np.zeros_like(detector.data)
        if self.verbose:
            print('Forward Simulation for grain %d' % self.grain.id)
        sample.geo.discretize_geometry(grain_id=self.grain.id)
        # we use either the hkl planes for this grain or the ones defined for the whole simulation
        if hasattr(self.grain, 'hkl_planes') and len(self.grain.hkl_planes) > 0:
            print('using hkl from the grain')
            hkl_planes = [HklPlane(h, k, l, lattice) for (h, k, l) in self.grain.hkl_planes]
        else:
            if len(self.hkl_planes) == 0:
                print('warning: no reflection defined for this simulation, using all planes with max miller=%d' % self.max_miller)
                self.set_hkl_planes(build_list(lattice=lattice, max_miller=self.max_miller))
            hkl_planes = self.hkl_planes
        n_hkl = len(hkl_planes)
        positions = sample.geo.get_positions()  # size n_vox, with 3 elements items
        n_vox = len(positions)
        Xu_vectors, thetas, the_energies, X_vectors, K_vectors = LaueForwardSimulation.fsim_laue(
            self.grain.orientation, hkl_planes, positions, source.position)
        OR_vectors = [detector.project_along_direction(origin=positions[i_vox],
                                                       direction=K_vectors[i_vox * n_hkl + i_hkl])
                      for i_vox in range(n_vox)
                      for i_hkl in range(n_hkl)]  # size nb_vox * n_hkl
        uv = [detector.lab_to_pixel(OR)[0].astype(np.int)
              for OR in OR_vectors]
        # now construct a boolean list to select the diffraction spots
        if source.min_energy is None and source.max_energy is None:
            # TODO use the use_energy_limits attribute
            energy_in = [True for k in range(len(the_energies))]
        else:
            energy_in = [source.min_energy < the_energies[k] < source.max_energy
                         for k in range(len(the_energies))]
        uv_in = [0 < uv[k][0] < detector.get_size_px()[0] and 0 < uv[k][1] < detector.get_size_px()[1]
                 for k in range(len(uv))]  # size n, diffraction located on the detector
        spot_in = [uv_in[k] and energy_in[k] for k in range(len(uv))]
        if self.verbose:
            print('%d diffraction events on the detector among %d' % (sum(spot_in), len(uv)))

        # now sum the counts on the detector individual pixels
        for k in range(len(uv)):
            if spot_in[k]:
                data[uv[k][0], uv[k][1]] += 1
        return data

    def fsim(self):
        """run the forward simulation.

        If the sample has a CAD type of geometry, a single grain (the first from the list) is assumed. In the other
        cases all the grains from the microstructure are used. In particular, if the microstructure has a grain map,
        it can be used to carry out an extended sample simulation.
        """
        full_data = np.zeros_like(self.exp.get_active_detector().data)
        micro = self.exp.get_sample().get_microstructure()
        # for cad geometry we assume only one grain (the first in the list)
        if self.exp.get_sample().geo.geo_type == 'cad':
            full_data += self.fsim_grain(gid=micro.grains[0].id)
        else:
            # in the other cases, we use all the grains defined in the microstructure
            for grain in micro.grains:
                full_data += self.fsim_grain(gid=grain.id)
        return full_data

