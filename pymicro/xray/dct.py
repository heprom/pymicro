import os
import numpy as np
from scipy import ndimage
from skimage.transform import radon
from matplotlib import pyplot as plt, cm
from pymicro.crystal.microstructure import Grain
from pymicro.crystal.lattice import HklPlane
from pymicro.xray.xray_utils import lambda_keV_to_nm, radiograph


def dct_projection(orientations, data, dif_grains, omega, lambda_keV, detector, lattice, include_direct_beam=True,
                   att=5, verbose=True):
    '''Work in progress, will replace function in the microstructure module.'''
    full_proj = np.zeros(detector.size, dtype=np.float)
    lambda_nm = lambda_keV_to_nm(lambda_keV)
    omegar = omega * np.pi / 180
    R = np.array([[np.cos(omegar), -np.sin(omegar), 0], [np.sin(omegar), np.cos(omegar), 0], [0, 0, 1]])

    if include_direct_beam:
        # add the direct beam part by computing the radiograph of the sample without the diffracting grains
        data_abs = np.where(data > 0, 1, 0)
        for (gid, (h, k, l)) in dif_grains:
            mask_dif = (data == gid)
            data_abs[mask_dif] = 0  # remove this grain from the absorption
        proj = radiograph(data_abs, omega)
        add_to_image(full_proj, proj[::-1, ::-1] / att, np.array(full_proj.shape) // 2)

    # add diffraction spots
    X = np.array([1., 0., 0.]) / lambda_nm
    for (gid, (h, k, l)) in dif_grains:
        grain_data = np.where(data == gid, 1, 0)
        if np.sum(grain_data) < 1:
            print('skipping grain %d' % gid)
            continue
        local_com = np.array(ndimage.measurements.center_of_mass(grain_data, data))
        print('local center of mass (voxel): {0}'.format(local_com))
        g_center_mm = detector.pixel_size * (local_com - 0.5 * np.array(data.shape))
        print('center of mass (voxel): {0}'.format(local_com - 0.5 * np.array(data.shape)))
        print('center of mass (mm): {0}'.format(g_center_mm))
        # compute scattering vector
        gt = orientations[gid].orientation_matrix().transpose()
        # gt = micro.get_grain(gid).orientation_matrix().transpose()
        p = HklPlane(h, k, l, lattice)
        G = np.dot(R, np.dot(gt, p.scattering_vector()))
        K = X + G
        # position of the grain at this rotation
        g_pos_rot = np.dot(R, g_center_mm)
        pg = detector.project_along_direction(g_pos_rot, K)
        (up, vp) = detector.lab_to_pixel(pg)
        if verbose:
            print('\n* gid=%d, (%d,%d,%d) plane, angle=%.1f' % (gid, h, k, l, omega))
            print('diffraction vector:', K)
            print('postion of the grain at omega=%.1f is ' % omega, g_pos_rot)
            print('up=%d, vp=%d for plane (%d,%d,%d)' % (up, vp, h, k, l))
        data_dif = grain_data[ndimage.find_objects(data == gid)[0]]
        proj_dif = radiograph(data_dif, omega)  # (Y, Z) coordinate system
        add_to_image(full_proj, proj_dif[::-1, ::-1], (up, vp), verbose)  # (u, v) axes correspond to (-Y, -Z)
    return full_proj


def add_to_image(image, inset, (u, v), verbose=False):
    """Add an image to another image at a specified position.

    The inset image may be of any size and may only overlap partly on the overall image depending on the location
    specified. In such a case, the inset image is cropped accordingly.

    :param np.array image: the master image taht will be modified.
    :param np.array inset: the inset to add to the image.
    :param tuple (u,v): the location (center) where to add the inset.
    :param bool verbose: activate verbose mode (False by default).
    """
    # round the center to the closest integer value
    u = int(u)
    v = int(v)
    spot_size = inset.shape
    u_start = 0
    u_end = spot_size[0]
    v_start = 0
    v_end = spot_size[1]
    # check bounds for spots that may be completely or partly outside the image
    if (u + spot_size[0] // 2 < 0) or (u - spot_size[0] // 2 > image.shape[0] - 1) or (
                    v + spot_size[1] // 2 < 0) or (v - spot_size[1] // 2 > image.shape[1] - 1):
        if verbose:
            print('skipping this spot which is outside the detector area')
        return None  # spot is completely outside the detector area
    if u - spot_size[0] // 2 < 0:
        u_start = int(spot_size[0] // 2 - u + 1)
    elif u - spot_size[0] // 2 + spot_size[0] > image.shape[0] - 1:
        u_end = int(image.shape[0] - (u - spot_size[0] // 2))
    if v - spot_size[1] // 2 < 0:
        v_start = int(spot_size[1] // 2 - v + 1)
    elif v - spot_size[1] // 2 + spot_size[1] > image.shape[1] - 1:
        v_end = int(image.shape[1] - (v - spot_size[1] // 2))
    # now add spot to the image
    image[u - spot_size[0] // 2 + u_start: u - spot_size[0] // 2 + u_end,
    v - spot_size[1] // 2 + v_start: v - spot_size[1] // 2 + v_end] \
        += inset[u_start:u_end, v_start:v_end]


def all_dif_spots(g_proj, g_uv, verbose=False):
    """Produce a 2D image placing all diffraction spots at their respective position on the detector.

    Spots outside the detector are are skipped while those only partially on the detector are cropped accordingly.

    :param g_proj: image stack of the diffraction spots. The first axis is so that g_proj[0] is the first spot \
    the second axis is the horizontal coordinate of the detector (u) and the third axis the vertical coordinate \
    of the detector (v).
    :param g_uv: list or array of the diffraction spot position.
    :param bool verbose: activate verbose mode (False by default).
    :returns: a 2D composite image of all the diffraction spots.
    """
    # TODO add a detector object to account for the image size and the position of the direct beam
    image = np.zeros((2048, 2048), dtype=g_proj.dtype)
    print(g_proj.shape[0], len(g_uv))
    assert g_proj.shape[0] == len(g_uv)
    for i in range(g_proj.shape[0]):
        spot = g_proj[i]
        if verbose:
            print('i={0}, size of spot: {1}'.format(i, spot.shape))
            print('placing diffraction spot at location {0}'.format(g_uv[i]))
        add_to_image(image, spot, g_uv[i], verbose)
    return image


def plot_all_dif_spots(gid, detector, hkl_miller=None, uv=None, lattice=None, lambda_keV=None,
                       spots=True, dif_spots_image=None, max_value=None, positions=True, debye_rings=True, suffix=''):
    plt.figure(figsize=(13, 8))
    # plt.figure()
    if spots and dif_spots_image is not None:
        if not max_value:
            max_value = dif_spots_image.max()
        plt.imshow(dif_spots_image.T, cmap=cm.gray, vmin=0, vmax=max_value)
    families = []
    indices = []
    colors = 'crbgmy'  # crbgmycrbgmycrbgmycrbgmy'  # use a cycler here
    t = np.linspace(0.0, 2 * np.pi, num=37)
    if positions or debye_rings:
        if hkl_miller is None:
            raise ValueError(
                'The list of miller indices of each reflection must be provided using variable g_hkl_miller')
        for i, (h, k, l) in enumerate(hkl_miller):
            l = [abs(h), abs(k), abs(l)]
            # l.sort()  # miller indices are now sorted, should use the lattice symmetry here
            family_name = '%d%d%d' % (l[0], l[1], l[2])
            if families.count(family_name) == 0:
                families.append(family_name)
                indices.append(i)
        indices.append(len(hkl_miller))
        print(families, indices)
        # now plot each family
        for i in range(len(families)):
            family = families[i]
            c = colors[i % len(colors)]
            if positions and uv is not None:
                plt.plot(uv[indices[i]:indices[i + 1], 0], uv[indices[i]:indices[i + 1], 1],
                         's', color=c, label=family)
            if debye_rings and lattice is not None and lambda_keV:
                theta = HklPlane(int(family[0]), int(family[1]), int(family[2]), lattice).bragg_angle(lambda_keV)
                L = detector.ref_pos[0] / detector.pixel_size * np.tan(2 * theta)  # 2 theta distance on the detector
                print('2theta = %g, L = %g' % (2 * theta * 180 / np.pi, L))
                plt.plot(0.5 * detector.size[0] + L * np.cos(t), 0.5 * detector.size[1] + L * np.sin(t), '--', color=c)
    plt.title('grain %d diffraction spot locations on the detector' % gid)
    # plt.legend(numpoints=1, loc='center')
    plt.legend(numpoints=1, ncol=2, bbox_to_anchor=(1.02, 1), loc=2)
    # plt.axis('equal')
    plt.axis([0, 2047, 2047, 0])
    plt.savefig('g%d%s_difspot_positions.pdf' % (gid, suffix))


def output_tikzpicture(proj_dif, omegas, gid=1, d_uv=[0, 0], suffix=''):
    axes = ['horizontal', 'vertical']
    for i, d in enumerate(d_uv):
        if d:
            # pad the corresponding axis if needed
            p = (d - proj_dif.shape[i + 1]) // 2
            if p > 0:
                print('padding %s axis with %d zeros' % (axes[i], p))
                if i == 0:
                    proj_dif_pad = np.pad(proj_dif, ((0, 0), (p, p), (0, 0)), 'constant')
                else:
                    proj_dif_pad = np.pad(proj_dif, ((0, 0), (0, 0), (p, p)), 'constant')
        else:
            d_uv[i] = proj_dif.shape[i + 1]
    print('output proj images will be {0}'.format(d_uv))
    N = proj_dif.shape[0]
    for i in range(N):
        proj_path = os.path.join('proj_dif', 'g%d_proj%s_%02d.png' % (gid, suffix, i))
        # proj_dif is in (u, v) form so as usual, we need to save the transposed array with imsave
        plt.imsave(proj_path,
                   proj_dif[i, (proj_dif.shape[1] - d_uv[0]) // 2:(proj_dif.shape[1] + d_uv[0]) // 2,
                   (proj_dif.shape[2] - d_uv[1]) // 2:(proj_dif.shape[2] + d_uv[1]) // 2].T,
                   cmap=cm.gray, origin='upper')
    # HST_write(proj_dif, 'g4_proj_stack.raw')

    # image ratio is 4/3 so we will use a n x m matrix with
    L = 10.0  # cm
    H = 3 * L / 4  # cm
    n = 0
    while (3. / 4 * n ** 2) < N:
        n += 1
    m = int(3. * n / 4)
    # size of a picture (assuming square)
    l = L / n
    tex_path = os.path.join('proj_dif', 'g%d_proj%s.tex' % (gid, suffix))
    print('building a tikzpicture with {0}x{1} miniatures, output file {2}'.format(n, m, tex_path))
    f = open(tex_path, 'w')
    f.write('\\documentclass{article}')
    f.write('\\usepackage[latin1]{inputenc}\n')
    f.write('\\usepackage{tikz}\n')
    f.write('\\usetikzlibrary{shapes,arrows}\n')
    f.write('\\usepackage{xcolor}\n')
    f.write('\\pagecolor{black}\n')
    f.write('\\usepackage[active,tightpage]{preview}\n')
    f.write('\\PreviewEnvironment{tikzpicture}\n')
    f.write('\\setlength\PreviewBorder{0pt}\n')
    f.write('\\begin{document}\n')
    f.write('\\pagestyle{empty}\n')
    f.write('\\sffamily\n')
    f.write('\\fontsize{5}{6}\n')
    f.write('\\begin{tikzpicture}[white]\n')
    # f.write(
    #    '\\filldraw[black] (%.3f,%.3f) rectangle (%.3f,%.3f);\n' % (-0.5 * l, 0.5 * l, L - 0.5 * l, -m * l + 0.5 * l))
    for j in range(m):
        Y = -j * l
        for i in range(n):
            X = i * l
            index = j * n + i
            if index >= N:
                continue  # skip last incomplete line
            f.write('\\node at (%.2f,%.2f) {\includegraphics[width=%.2fcm]{g%d_proj%s_%02d.png}};\n' % (
                X, Y, l, gid, suffix, index))
            # f.write('\\node at (%.2f,%.2f) {$\omega=%.1f^\circ$};\n' % (X, Y - 0.5 * l, omegas[index]))
            f.write('\\node at (%.2f,%.2f) {$%.1f^\circ$};\n' % (X, Y - 0.5 * l, omegas[index]))
    f.write('\\end{tikzpicture}\n')
    f.write('\\end{document}\n')
    f.close()
