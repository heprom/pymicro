from matplotlib import pyplot as plt, cm
import numpy as np

from pymicro.crystal.lattice import *
from pymicro.crystal.microstructure import *
from pymicro.xray.laue import *
from pymicro.xray.detectors import Varian2520

if __name__ == '__main__':
    det_distance = 150.
    detector = Varian2520()
    ni = Lattice.face_centered_cubic(0.3524)
    phi1 = 89.4  # deg
    phi = 92.0  # deg
    phi2 = 86.8  # deg
    orientation = Orientation.from_euler([phi1, phi, phi2])
    Bt = orientation.orientation_matrix().transpose()

    uvw = HklDirection(1, 0, 5, ni)
    hklplanes = uvw.find_planes_in_zone(max_miller=8)

    all_planes = build_list(lattice=ni, max_miller=8)
    compute_Laue_pattern(orientation, detector, det_distance, all_planes, inverted=True)

    plt.figure()  # new figure
    plt.imshow(detector.data.T, cmap=cm.gray)
    plt.axis('equal')
    plt.xlabel('u coordinate (pixel)')
    plt.ylabel('v coordinate (pixel)')
    plt.axis([1, detector.size[0], detector.size[1], 1])  # to display with (0, 0) in the top left corner
    plt.title("Laue pattern, detector distance = %g" % det_distance)
    plt.plot(detector.ucen, detector.vcen, 'ks', label='diffracted beams')
    uvw_miller = uvw.miller_indices()
    ellipse = compute_ellpisis(orientation, detector, det_distance, uvw, verbose=True)
    plt.plot(ellipse[0], ellipse[1], 'r--', label='ellipse zone [%d%d%d]' % uvw_miller)
    plt.legend(numpoints=1, loc='lower right')

    image_name = os.path.splitext(__file__)[0] + '.png'
    print('writting %s' % image_name)
    plt.savefig(image_name, format='png')

    from matplotlib import image

    image.thumbnail(image_name, 'thumb_' + image_name, 0.2)
