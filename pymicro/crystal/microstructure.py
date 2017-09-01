"""
The microstructure module provide elementary classes to describe a 
crystallographic granular microstructure such as mostly present in 
metallic materials.

It contains several classes which are used to describe a microstructure 
composed of several grains, each one having its own crystallographic 
orientation:

 * :py:class:`~pymicro.crystal.microstructure.Microstructure`
 * :py:class:`~pymicro.crystal.microstructure.Grain`
 * :py:class:`~pymicro.crystal.microstructure.Orientation`
"""
import numpy as np
import os, vtk
from matplotlib import pyplot as plt, colors, cm
from xml.dom.minidom import Document, parse


class Orientation:
    """Crystallographic orientation class.

    This follows the passive rotation definition which means that it brings
    the sample coordinate system into coincidence with the crystal coordinate
    system. Then one may express a vector :math:`V_c` in the crystal coordinate system
    from the vector in the sample coordinate system :math:`V_s` by:

    .. math::

      V_c = g.V_s

    and inversely (because :math:`g^{-1}=g^T`):

    .. math::

      V_s = g^T.V_c

    Most of the code to handle rotations has been written to comply with the conventions 
    laid in :cite:`Rowenhorst2015`.
    """

    def __init__(self, matrix):
        """Initialization from the 9 components of the orientation matrix."""
        g = np.array(matrix, dtype=np.float64).reshape((3, 3))
        self._matrix = g
        self.euler = Orientation.OrientationMatrix2Euler(g)
        self.rod = Orientation.OrientationMatrix2Rodrigues(g)

    def orientation_matrix(self):
        """Returns the orientation matrix in the form of a 3x3 numpy array."""
        return self._matrix

    def __repr__(self):
        """Provide a string representation of the class."""
        s = 'Crystal Orientation'
        s += '\norientation matrix = %s' % self._matrix.view()
        s += '\nEuler angles (degrees) = (%8.3f,%8.3f,%8.3f)' % (self.phi1(), self.Phi(), self.phi2())
        s += '\nRodrigues vector = %s' % self.OrientationMatrix2Rodrigues(self._matrix)
        return s

    @staticmethod
    def cube():
        """Create the particular crystal orientation called Cube and which
        corresponds to euler angle (0, 0, 0)."""
        return Orientation.from_euler((0., 0., 0.))

    @staticmethod
    def brass():
        """Create the particular crystal orientation called Brass and which
        corresponds to euler angle (35.264, 45, 0)."""
        return Orientation.from_euler((35.264, 45., 0.))

    @staticmethod
    def copper():
        """Create the particular crystal orientation called Copper and which
        corresponds to euler angle (90, 35.264, 45)."""
        return Orientation.from_euler((90., 35.264, 45.))

    @staticmethod
    def s3():
        """Create the particular crystal orientation called S3 and which
        corresponds to euler angle (59, 37, 63)."""
        return Orientation.from_euler((58.980, 36.699, 63.435))

    @staticmethod
    def goss():
        """Create the particular crystal orientation called Goss and which
        corresponds to euler angle (0, 45, 0)."""
        return Orientation.from_euler((0., 45., 0.))

    @staticmethod
    def shear():
        """Create the particular crystal orientation called shear and which
        corresponds to euler angle (45, 0, 0)."""
        return Orientation.from_euler((45., 0., 0.))

    def get_ipf_colour(self, axis=np.array([0., 0., 1.])):
        """Compute the IPF (inverse pole figure) colour for this orientation.

        Given a particular axis expressed in the laboratory coordinate system,
        one can compute the so called IPF colour based on that direction
        expressed in the crystal coordinate system as :math:`[x_c,y_c,z_c]`.
        There is only one tuple (u,v,w) such that:

        .. math::

          [x_c,y_c,z_c]=u.[0,0,1]+v.[0,1,1]+w.[1,1,1]

        and it is used to assign the RGB colour.

        .. warning::

           hard coded for the cubic crystal symmetry for now on, it should
           be rather straightforward to generalize this to any symmetry
           making use of the Lattice.symmetry() method.
        """
        axis /= np.linalg.norm(axis)
        from pymicro.crystal.lattice import Lattice
        # find the axis lying in the fundamental zone
        for sym in Lattice.symmetry(crystal_structure='cubic'):
            Osym = np.dot(sym, self.orientation_matrix())
            Vc = np.dot(Osym, axis)
            if Vc[2] < 0:
                Vc *= -1.  # using the upward direction
            uvw = np.array([Vc[2] - Vc[1], Vc[1] - Vc[0], Vc[0]])
            uvw /= np.linalg.norm(uvw)
            uvw /= max(uvw)
            if (uvw[0] >= 0. and uvw[0] <= 1.0) and (uvw[1] >= 0. and uvw[1] <= 1.0) and (
                            uvw[2] >= 0. and uvw[2] <= 1.0):
                #print('found sym for sst')
                break
        return uvw

    def inFZ(self, symmetry='cubic'):
        """Check if the given Orientation lies within the fundamental zone.
        
        For a given crystal symmetry, several rotations can describe the same 
        physcial crystllographic arangement. The Rodrigues fundamental zone 
        restrict the orientation space accordingly. 
        """
        r = self.rod
        if symmetry == 'cubic':
            inFZT23 = np.abs(r).sum() <= 1.0
            # in the cubic symmetry, each component must be < 2 ** 0.5 - 1
            inFZ = inFZT23 and np.abs(r).max() <= 2 ** 0.5 - 1
        else:
            raise(ValueError('unsupported crystal symmetry: %s' % symmetry))
        return inFZ

    def move_to_FZ(self, symmetry='cubic', verbose=False):
        """
        Compute the equivalent crystal orientation in the Fundamental Zone of a given symmetry.

        :param str symmetry: a string describing the crystal symmetry 
        :param verbose: flag for verbose mode
        :return: a new Orientation instance which lies in the fundamental zone.
        """
        from pymicro.crystal.lattice import Lattice
        om = Lattice.move_rotation_to_FZ(self.orientation_matrix(), crystal_structure=symmetry, verbose=verbose)
        return Orientation(om)

    @staticmethod
    def misorientation_MacKenzie(psi):
        """Return the fraction of the misorientations corresponding to the
        given :math:`\\psi` angle in the reference solution derived By MacKenzie in
        his 1958 paper :cite:`MacKenzie_1958`.

        :param psi: the misorientation angle in radians.
        :returns: the value in the cummulative distribution corresponding to psi.
        """
        from math import sqrt, sin, cos, tan, pi, acos
        psidg = 180 * psi / pi
        if psidg >= 0 and psidg <= 45:
            p = 2. / 15 * (1 - cos(psi))
        elif psidg > 45 and psidg <= 60:
            p = 2. / 15 * (3 * (sqrt(2) - 1) * sin(psi) - 2 * (1 - cos(psi)))
        elif psidg > 60 and psidg <= 60.72:
            p = 2. / 15 * ((3 * (sqrt(2) - 1) + 4. / sqrt(3)) * sin(psi) - 6. * (1 - cos(psi)))
        elif psidg > 60.72 and psidg <= 62.8:
            X = (sqrt(2) - 1) / (1 - (sqrt(2) - 1) ** 2 / tan(0.5 * psi) ** 2) ** 0.5
            Y = (sqrt(2) - 1) ** 2 / ((3 - 1 / tan(0.5 * psi) ** 2) ** 0.5)
            p = (2. / 15) * ((3 * (sqrt(2) - 1) + 4 / sqrt(3)) * sin(psi) - 6 * (1 - cos(psi))) \
                - 8. / (5 * pi) * (
                2 * (sqrt(2) - 1) * acos(X / tan(0.5 * psi)) + 1. / sqrt(3) * acos(Y / tan(0.5 * psi))) * sin(psi) \
                + 8. / (5 * pi) * (2 * acos((sqrt(2) + 1) * X / sqrt(2)) + acos((sqrt(2) + 1) * Y / sqrt(2))) * (
                1 - cos(psi))
        else:
            p = 0.
        return p

    @staticmethod
    def misorientation_axis_from_delta(delta):
        """Compute the misorientation axis from the misorientation matrix.
 
        :param delta: The 3x3 misorientation matrix.
        :returns: the misorientation axis (normalised vector).
        """
        n = np.array([delta[1, 2] - delta[2, 1], delta[2, 0] - delta[0, 2], delta[0, 1] - delta[1, 0]])
        n /= np.sqrt(
            (delta[1, 2] - delta[2, 1]) ** 2 + (delta[2, 0] - delta[0, 2]) ** 2 + (delta[0, 1] - delta[1, 0]) ** 2)
        return n

    def misorientation_axis(self, orientation):
        """Compute the misorientation axis with another crystal orientation.
        This vector is by definition common to both crystalline orientations.

        :param orientation: an instance of :py:class:`~pymicro.crystal.microstructure.Orientation` class.
        :returns: the misorientation axis (normalised vector).
        """
        delta = np.dot(self.orientation_matrix(), orientation.orientation_matrix().T)
        return Orientation.misorientation_axis_from_delta(delta)

    @staticmethod
    def misorientation_angle_from_delta(delta):
        """Compute the misorientation angle from the misorientation matrix.

        Compute the angle assocated with this misorientation matrix :math:`\\Delta g`.
        It is defined as :math:`\\omega = \\arccos(\\text{trace}(\\Delta g)/2-1)`.
        To avoid float rounding error, the argument is rounded to 1. if it is within 1 and 1 plus 32 bits floating 
        point precison.

        .. note::

          This does not account for the crystal symmetries. If you want to
          find the disorientation between two orientations, use the
          :py:meth:`~pymicro.crystal.microstructure.Orientation.disorientation`
          method.

        :param delta: The 3x3 misorientation matrix.
        :returns float: the misorientation angle in radians.
        """
        cw = 0.5 * (delta.trace() - 1)
        if cw > 1. and cw - 1. < np.finfo('float32').eps:
            print('cw=%.20f, rounding to 1.' % cw)
            cw = 1.
        omega = np.arccos(cw)
        return omega

    def disorientation(self, orientation, crystal_structure='cubic'):
        """Compute the disorientation another crystal orientation.

        Considering all the possible crystal symmetries, the disorientation
        is defined as the combination of the minimum misorientation angle
        and the misorientation axis lying in the fundamental zone, which
        can be used to bring the two lattices into coincidence.

        :param orientation: an instance of :py:class:`~pymicro.crystal.microstructure.Orientation` class desribing the other crystal orientation from which to compute the angle.
        :param str crystal_structure: a string describing the crystal structure, 'cubic' by default.
        :returns tuple: the misorientation angle in radians, the axis as a numpy vector (crystal coordinates), the axis as a numpy vector (sample coordinates).
        """
        the_angle = np.pi
        from pymicro.crystal.lattice import Lattice
        symmetries = Lattice.symmetry(crystal_structure)
        (gA, gB) = (self.orientation_matrix(), orientation.orientation_matrix())  # nicknames
        for (g1, g2) in [(gA, gB), (gB, gA)]:
            for j in range(symmetries.shape[0]):
                sym_j = symmetries[j]
                oj = np.dot(sym_j, g1)  # the crystal symmetry operator is left applied
                for i in range(symmetries.shape[0]):
                    sym_i = symmetries[i]
                    oi = np.dot(sym_i, g2)
                    delta = np.dot(oi, oj.T)
                    mis_angle = Orientation.misorientation_angle_from_delta(delta)
                    if mis_angle < the_angle:
                        # now compute the misorientation axis, should check if it lies in the fundamental zone
                        mis_axis = Orientation.misorientation_axis_from_delta(delta)
                        # here we have np.dot(oi.T, mis_axis) = np.dot(oj.T, mis_axis)
                        # print(mis_axis, mis_angle*180/np.pi, np.dot(oj.T, mis_axis))
                        the_angle = mis_angle
                        the_axis = mis_axis
                        the_axis_xyz = np.dot(oi.T, the_axis)
        return (the_angle, the_axis, the_axis_xyz)

    def phi1(self):
        """Convenience methode to expose the first Euler angle."""
        return self.euler[0]

    def Phi(self):
        """Convenience methode to expose the second Euler angle."""
        return self.euler[1]

    def phi2(self):
        """Convenience methode to expose the third Euler angle."""
        return self.euler[2]

    def compute_XG_angle(self, hkl, omega, verbose=False):
        """Compute the angle between the scattering vector :math:`\mathbf{G_{l}}`
        and :math:`\mathbf{-X}` the X-ray unit vector at a given angular position :math:`\\omega`.

        A given hkl plane defines the scattering vector :math:`\mathbf{G_{hkl}}` by
        the miller indices in the reciprocal space. It is expressed in the
        cartesian coordinate system by :math:`\mathbf{B}.\mathbf{G_{hkl}}` and in the
        laboratory coordinate system accounting for the crystal orientation
        by :math:`\mathbf{g}^{-1}.\mathbf{B}.\mathbf{G_{hkl}}`.

        The crystal is assumed to be placed on a rotation stage around the
        laboratory vertical axis. The scattering vector can finally be
        written as :math:`\mathbf{G_l}=\mathbf{\\Omega}.\mathbf{g}^{-1}.\mathbf{B}.\mathbf{G_{hkl}}`.
        The X-rays unit vector is :math:`\mathbf{X}=[1, 0, 0]`. So the computed angle
        is :math:`\\alpha=acos(-\mathbf{X}.\mathbf{G_l}/||\mathbf{G_l}||`

        The Bragg condition is fulfilled when :math:`\\alpha=\pi/2-\\theta_{Bragg}`

        :param hkl: the hkl plane, an instance of :py:class:`~pymicro.crystal.lattice.HklPlane`
        :param omega: the angle of rotation of the crystal around the laboratory vertical axis.
        :param bool verbose: activate verbose mode (False by default).
        :return float: the angle between :math:`-\mathbf{X}` and :math:`\mathbf{G_{l}}` in degrees.
        """
        X = np.array([1., 0., 0.])
        gt = self.orientation_matrix().transpose()
        Gc = hkl.scattering_vector()
        Gs = gt.dot(Gc)  # in the cartesian sample CS
        omegar = omega * np.pi / 180
        R = np.array([[np.cos(omegar), -np.sin(omegar), 0], [np.sin(omegar), np.cos(omegar), 0], [0, 0, 1]])
        Gl = R.dot(Gs)
        alpha = np.arccos(np.dot(-X, Gl) / np.linalg.norm(Gl)) * 180 / np.pi
        if verbose:
            print('scattering vector in the crystal CS', Gc)
            print('scattering vector in the sample CS', Gs)
            print('scattering vector in the laboratory CS (including Omega rotation)', Gl)
            print('angle (deg) between -X and G', alpha)
        return alpha

    def dct_omega_angles(self, hkl, lambda_keV, verbose=False):
        """Compute the two omega angles which satisfy the Bragg condition.

        For a given crystal orientation sitting on a vertical rotation axis,
        there is exactly two :math:`\omega` positions in :math:`[0, 2\pi]` for which
        a particular :math:`(hkl)` reflexion will fulfil Bragg's law.

        According to the Bragg's law, a crystallographic plane of a given
        grain will be in diffracting condition if:

        .. math::

           \sin\\theta=-[\mathbf{\Omega}.\mathbf{g}^{-1}\mathbf{G_c}]_1

        with :math:`\mathbf{\Omega}` the matrix associated with the rotation
        axis:

        .. math::

           \mathbf{\Omega}=\\begin{pmatrix}
                           \cos\omega & -\sin\omega & 0 \\\\
                           \sin\omega & \cos\omega  & 0 \\\\
                           0          & 0           & 1 \\\\
                           \end{pmatrix}

        This method solves the associated second order equation to return
        the two corresponding omega angles.

        :param hkl: The given cristallographic plane :py:class:`~pymicro.crystal.lattice.HklPlane`
        :param float lambda_keV: The X-rays energy expressed in keV
        :param bool verbose: Verbose mode (False by default)
        :returns tuple: :math:`(\omega_1, \omega_2)` the two values of the \
        rotation angle around the vertical axis (in degrees).
        """
        (h, k, l) = hkl.miller_indices()
        (a, b, c) = hkl._lattice._lengths
        theta = hkl.bragg_angle(lambda_keV, verbose=verbose)
        lambda_nm = 1.2398 / lambda_keV
        gt = self.orientation_matrix().T  # gt = g^{-1} in Poulsen 2004
        Gc = hkl.scattering_vector()
        A = np.dot(Gc, gt[0])
        B = - np.dot(Gc, gt[1])
        #A = h / a * gt[0, 0] + k / b * gt[0, 1] + l / c * gt[0, 2]
        #B = -h / a * gt[1, 0] - k / b * gt[1, 1] - l / c * gt[1, 2]
        C = -2 * np.sin(theta) ** 2 / lambda_nm  # the minus sign comes from the main equation
        Delta = 4 * (A ** 2 + B ** 2 - C ** 2)
        if Delta < 0:
            raise ValueError('Delta < 0 (%f) for reflexion (%d%d%d)' % (Delta, h, k, l))
        t1 = (B - 0.5 * np.sqrt(Delta)) / (A + C)
        t2 = (B + 0.5 * np.sqrt(Delta)) / (A + C)
        w1 = 2 * np.arctan(t1) * 180. / np.pi
        w2 = 2 * np.arctan(t2) * 180. / np.pi
        if w1 < 0: w1 += 360.
        if w2 < 0: w2 += 360.
        if verbose:
            print('A={0:.3f}, B={1:.3f}, C={2:.3f}, Delta={3:.1f}'.format(A, B, C, Delta))
            print('the two omega values in degrees fulfilling the Bragg condition are (%.1f, %.1f)' % (w1, w2))
        return w1, w2

    def rotating_crystal(self, hkl, lambda_keV, omega_step=0.5, display=True, verbose=False):
        from pymicro.xray.xray_utils import lambda_keV_to_nm
        lambda_nm = lambda_keV_to_nm(lambda_keV)
        X = np.array([1., 0., 0.]) / lambda_nm
        print 'magnitude of X', np.linalg.norm(X)
        gt = self.orientation_matrix().transpose()
        (h, k, l) = hkl.miller_indices()
        theta = hkl.bragg_angle(lambda_keV) * 180. / np.pi
        print('bragg angle for %d%d%d reflection is %.1f' % (h, k, l, theta))
        Gc = hkl.scattering_vector()
        Gs = gt.dot(Gc)
        alphas = []
        twothetas = []
        magnitude_K = []
        omegas = np.linspace(0.0, 360.0, num=360.0 / omega_step, endpoint=False)
        for omega in omegas:
            print('\n** COMPUTING AT OMEGA=%03.1f deg' % omega)
            # prepare rotation matrix
            omegar = omega * np.pi / 180
            R = np.array([[np.cos(omegar), -np.sin(omegar), 0], [np.sin(omegar), np.cos(omegar), 0], [0, 0, 1]])
            # R = R.dot(Rlt).dot(Rut) # with tilts
            Gl = R.dot(Gs)
            print('scattering vector in laboratory CS', Gl)
            n = R.dot(gt.dot(hkl.normal()))
            print 'plane normal:', hkl.normal()
            print R
            print 'rotated plane normal:', n, ' with a norm of', np.linalg.norm(n)
            G = n / hkl.interplanar_spacing()  # here G == N
            print 'G vector:', G, ' with a norm of', np.linalg.norm(G)
            K = X + G
            print 'X + G vector', K
            magnitude_K.append(np.linalg.norm(K))
            print 'magnitude of K', np.linalg.norm(K)
            alpha = np.arccos(np.dot(-X, G) / (np.linalg.norm(-X) * np.linalg.norm(G))) * 180 / np.pi
            print 'angle between -X and G', alpha
            alphas.append(alpha)
            twotheta = np.arccos(np.dot(K, X) / (np.linalg.norm(K) * np.linalg.norm(X))) * 180 / np.pi
            print 'angle (deg) between K and X', twotheta
            twothetas.append(twotheta)
        print 'min alpha angle is ', min(alphas)

        # compute omega_1 and omega_2 to verify graphically
        (w1, w2) = self.dct_omega_angles(hkl, lambda_keV, verbose=False)

        # gather the results in a single figure
        fig = plt.figure(figsize=(12, 10))
        fig.add_subplot(311)
        plt.title('Looking for (%d%d%d) Bragg reflexions' % (h, k, l))
        plt.plot(omegas, alphas, 'k-')
        plt.xlim(0, 360)
        plt.ylim(0, 180)
        plt.xticks(np.arange(0, 390, 30))
        # add bragg condition
        plt.axhline(90 - theta, xmin=0, xmax=360, linewidth=2)
        plt.annotate('$\pi/2-\\theta_{Bragg}$', xycoords='data', xy=(360, 90 - theta), horizontalalignment='left',
                     verticalalignment='center', fontsize=16)
        # add omega solutions
        plt.axvline(w1 + 180, ymin=0, ymax=180, linewidth=2, linestyle='dashed', color='gray')
        plt.axvline(w2 + 180, ymin=0, ymax=180, linewidth=2, linestyle='dashed', color='gray')
        plt.annotate('$\\omega_1$', xycoords='data', xy=(w1 + 180, 0), horizontalalignment='center',
                     verticalalignment='bottom', fontsize=16)
        plt.annotate('$\\omega_2$', xycoords='data', xy=(w2 + 180, 0), horizontalalignment='center',
                     verticalalignment='bottom', fontsize=16)
        plt.ylabel(r'Angle between $-X$ and $\mathbf{G}$')
        fig.add_subplot(312)
        plt.plot(omegas, twothetas, 'k-')
        plt.xlim(0, 360)
        # plt.ylim(0,180)
        plt.xticks(np.arange(0, 390, 30))
        plt.axhline(2 * theta, xmin=0, xmax=360, linewidth=2)
        plt.annotate('$2\\theta_{Bragg}$', xycoords='data', xy=(360, 2 * theta), horizontalalignment='left',
                     verticalalignment='center', fontsize=16)
        plt.axvline(w1 + 180, linewidth=2, linestyle='dashed', color='gray')
        plt.axvline(w2 + 180, linewidth=2, linestyle='dashed', color='gray')
        plt.ylabel('Angle between $X$ and $K$')
        fig.add_subplot(313)
        plt.plot(omegas, magnitude_K, 'k-')
        plt.xlim(0, 360)
        plt.axhline(np.linalg.norm(X), xmin=0, xmax=360, linewidth=2)
        plt.annotate('$1/\\lambda$', xycoords='data', xy=(360, 1 / lambda_nm), horizontalalignment='left',
                     verticalalignment='center', fontsize=16)
        plt.axvline(w1 + 180, linewidth=2, linestyle='dashed', color='gray')
        plt.axvline(w2 + 180, linewidth=2, linestyle='dashed', color='gray')
        plt.xlabel(r'Angle of rotation $\omega$')
        plt.ylabel(r'Magnitude of $X+G$ (nm$^{-1}$)')
        plt.subplots_adjust(top=0.925, bottom=0.05, left=0.1, right=0.9)
        if display:
            plt.show()
        else:
            plt.savefig('rotating_crystal_plot_%d%d%d.pdf' % (h, k, l))

    def topotomo_tilts(self, hkl, verbose=False):
        """Compute the tilts for topotomography alignment.

        :param hkl: the hkl plane, an instance of :py:class:`~pymicro.crystal.lattice.HklPlane`
        :param bool verbose: activate verbose mode (False by default).
        :returns tuple: (ut, lt) the two values of tilts to apply (in radians).
        """
        gt = self.orientation_matrix().transpose()
        Gc = hkl.scattering_vector()
        Gs = gt.dot(Gc)  # in the cartesian sample CS
        # find topotomo tilts
        ut = np.arctan(-Gs[0] / Gs[2])
        lt = np.arctan(Gs[1] / (-Gs[0] * np.sin(ut) + Gs[2] * np.cos(ut)))
        if verbose:
            print('up tilt (samry) should be %.3f' % (ut * 180 / np.pi))
            print('low tilt (samrx) should be %.3f' % (lt * 180 / np.pi))
        return (ut, lt)

    def to_xml(self, doc):
        """
        Returns an XML representation of the Orientation instance.
        """
        orientation = doc.createElement('Orientation')
        orientation_phi1 = doc.createElement('phi1')
        orientation_phi1_text = doc.createTextNode('%f' % self.phi1())
        orientation_phi1.appendChild(orientation_phi1_text)
        orientation.appendChild(orientation_phi1)
        orientation_Phi = doc.createElement('Phi')
        orientation_Phi_text = doc.createTextNode('%f' % self.Phi())
        orientation_Phi.appendChild(orientation_Phi_text)
        orientation.appendChild(orientation_Phi)
        orientation_phi2 = doc.createElement('phi2')
        orientation_phi2_text = doc.createTextNode('%f' % self.phi2())
        orientation_phi2.appendChild(orientation_phi2_text)
        orientation.appendChild(orientation_phi2)
        return orientation

    @staticmethod
    def from_xml(orientation_node):
        orientation_phi1 = orientation_node.childNodes[0]
        orientation_Phi = orientation_node.childNodes[1]
        orientation_phi2 = orientation_node.childNodes[2]
        phi1 = float(orientation_phi1.childNodes[0].nodeValue)
        Phi = float(orientation_Phi.childNodes[0].nodeValue)
        phi2 = float(orientation_phi2.childNodes[0].nodeValue)
        orientation = Orientation.from_euler(np.array([phi1, Phi, phi2]))
        return orientation

    @staticmethod
    def from_euler(euler):
        g = Orientation.Euler2OrientationMatrix(euler)
        o = Orientation(g)
        return o

    @staticmethod
    def from_rodrigues(rod):
        g = Orientation.Rodrigues2OrientationMatrix(rod)
        o = Orientation(g)
        return o

    @staticmethod
    def Euler2OrientationMatrix(euler):
        """
        Compute the orientation matrix :math:`\mathbf{g}` associated with the 3 Euler angles 
        :math:`(\phi_1, \Phi, \phi_2)`. The matrix is calculated via (see the `euler_angles` recipe in the cookbook 
        for a detailed example):

        .. math::
        
           \mathbf{g}=\\begin{pmatrix}
           \cos\phi_1\cos\phi_2 - \sin\phi_1\sin\phi_2\cos\Phi & \sin\phi_1\cos\phi_2 + \cos\phi_1\sin\phi_2\cos\Phi & \sin\phi_2\sin\Phi \\\\
           -\cos\phi_1\sin\phi_2 - \sin\phi_1\cos\phi_2\cos\Phi & -\sin\phi_1\sin\phi_2 + \cos\phi_1\cos\phi_2\cos\Phi & \cos\phi_2\sin\Phi \\\\
           \sin\phi_1\sin\Phi & -\cos\phi_1\sin\Phi & \cos\Phi \\\\
           \end{pmatrix}
        
        :param euler: The triplet of the Euler angles (in degrees).
        :returns g: The 3x3 orientation matrix.
        """
        (rphi1, rPhi, rphi2) = np.radians(euler)
        c1 = np.cos(rphi1)
        s1 = np.sin(rphi1)
        c = np.cos(rPhi)
        s = np.sin(rPhi)
        c2 = np.cos(rphi2)
        s2 = np.sin(rphi2)

        # rotation matrix g
        g11 = c1 * c2 - s1 * s2 * c
        g12 = s1 * c2 + c1 * s2 * c
        g13 = s2 * s
        g21 = -c1 * s2 - s1 * c2 * c
        g22 = -s1 * s2 + c1 * c2 * c
        g23 = c2 * s
        g31 = s1 * s
        g32 = -c1 * s
        g33 = c
        g = np.array([[g11, g12, g13], [g21, g22, g23], [g31, g32, g33]])
        return g

    @staticmethod
    def Zrot2OrientationMatrix(x1=None, x2=None, x3=None):
        """Compute the orientation matrix  from the rotated coordinates given in the
           .inp file for Zebulon's computations
           Need at least two vectors to compute cross product

           Still need some tests to validate this function
        """

        if (x1 is None and x2 is None):
            raise NameError('Need at least two vectors to compute the matrix')
        elif (x1 == None and x3 == None):
            raise NameError('Need at least two vectors to compute the matrix')
        elif (x3 == None and x2 == None):
            raise NameError('Need at least two vectors to compute the matrix')

        if x1 == None:
            x1 = np.cross(x2, x3)
        elif x2 == None:
            x2 = np.cross(x3, x1)
        elif x3 == None:
            x3 = np.cross(x1, x2)

        x1 = x1 / np.linalg.norm(x1)
        x2 = x2 / np.linalg.norm(x2)
        x3 = x3 / np.linalg.norm(x3)

        g = np.array([x1, x2, x3]).transpose()
        return g

    @staticmethod
    def OrientationMatrix2EulerSF(g):
        """
        Compute the Euler angles (in degrees) from the orientation matrix
        in a similar way as done in Mandel_crystal.c
        """
        tol = 0.1
        r = np.zeros(9, dtype=np.float64)  # double precision here
        # Z-set order for tensor is 11 22 33 12 23 13 21 32 31
        r[0] = g[0, 0]
        r[1] = g[1, 1]
        r[2] = g[2, 2]
        r[3] = g[0, 1]
        r[4] = g[1, 2]
        r[5] = g[0, 2]
        r[6] = g[1, 0]
        r[7] = g[2, 1]
        r[8] = g[2, 0]
        phi = np.arccos(r[2])
        if phi == 0.:
            phi2 = 0.
            phi1 = np.arcsin(r[6])
            if abs(np.cos(phi1) - r[0]) > tol:
                phi1 = np.pi - phi1
        else:
            x2 = r[5] / np.sin(phi)
            x1 = r[8] / np.sin(phi);
            if x1 > 1.:
                x1 = 1.
            if x2 > 1.:
                x2 = 1.
            if x1 < -1.:
                x1 = -1.
            if x2 < -1.:
                x2 = -1.
            phi2 = np.arcsin(x2)
            phi1 = np.arcsin(x1)
            if abs(np.cos(phi2) * np.sin(phi) - r[7]) > tol:
                phi2 = np.pi - phi2
            if abs(np.cos(phi1) * np.sin(phi) + r[4]) > tol:
                phi1 = np.pi - phi1
        return np.degrees(np.array([phi1, phi, phi2]))

    @staticmethod
    def OrientationMatrix2Euler(g):
        """
        Compute the Euler angles from the orientation matrix.
        
        This conversion follows the paper of Rowenhorst et al. :cite:`Rowenhorst2015`.
        In particular when :math:`g_{33} = 1` within the machine precision, 
        there is no way to determine the values of :math:`\phi_1` and :math:`\phi_2` 
        (only their sum is defined). The convention is to attribute 
        the entire angle to :math:`\phi_1` and set :math:`\phi_2` to zero.         

        :param g: The 3x3 orientation matrix
        :return: The 3 euler angles in degrees. 
        """
        eps = np.finfo('float').eps
        (phi1, Phi, phi2) = (0.0, 0.0, 0.0)
        # treat special case where g[2, 2] = 1
        if np.abs(g[2, 2]) >= 1 - eps:
            if g[2, 2] > 0.0:
                phi1 = np.arctan2(g[0][1], g[0][0])
            else:
                phi1 = -np.arctan2(-g[0][1], g[0][0])
                Phi = np.pi
        else:
            Phi = np.arccos(g[2][2])
            zeta = 1.0 / np.sqrt(1.0 - g[2][2] ** 2)
            phi1 = np.arctan2(g[2][0] * zeta, -g[2][1] * zeta)
            phi2 = np.arctan2(g[0][2] * zeta, g[1][2] * zeta)
        # ensure angles are in the range [0, 2*pi]
        if phi1 < 0.0:
            phi1 += 2 * np.pi
        if Phi < 0.0:
            Phi += 2 * np.pi
        if phi2 < 0.0:
            phi2 += 2 * np.pi
        return np.degrees([phi1, Phi, phi2])

    @staticmethod
    def OrientationMatrix2Rodrigues(g):
        """
        Compute the rodrigues vector from the orientation matrix.
        
        :param g: The 3x3 orientation matrix representing the rotation.
        :returns: The Rodrigues vector as a 3 components array. 
        """
        t = g.trace() + 1
        if np.abs(t) < np.finfo(g.dtype).eps:
            return np.zeros(3)
        else:
            r1 = (g[1, 2] - g[2, 1]) / t
            r2 = (g[2, 0] - g[0, 2]) / t
            r3 = (g[0, 1] - g[1, 0]) / t
        return np.array([r1, r2, r3])

    @staticmethod
    def Rodrigues2OrientationMatrix(rod):
        """
        Compute the orientation matrix from the Rodrigues vector.

        :param rod: The Rodrigues vector as a 3 components array.
        :returns: The 3x3 orientation matrix representing the rotation.
        """
        r = np.linalg.norm(rod)
        I = np.diagflat(np.ones(3))
        if r < np.finfo(r.dtype).eps:
            return I
        else:
            theta = 2 * np.arctan(r)
            n = rod / r
            omega = np.array([[0.0, n[2], -n[1]], [-n[2], 0.0, n[0]], [n[1], -n[0], 0.0]])
            return I + np.sin(theta) * omega + (1 - np.cos(theta)) * omega.dot(omega)

    @staticmethod
    def Axis2OrientationMatrix(axis, angle):
        """
        Compute the (passive) orientation matrix associated the rotation defined by the given (axis, angle) pair.

        :param axis: the rotation axis.
        :param angle: the rotation angle (degrees).
        :returns: the 3x3 orientation matrix.
        """
        omega = np.radians(angle)
        c = np.cos(omega)
        s = np.sin(omega)
        g = np.array([[c + (1 - c) * axis[0]**2, (1 - c) * axis[0] * axis[1] + s * axis[2], (1 - c) * axis[0] * axis[2] - s * axis[1]],
                      [(1 - c) * axis[0] * axis[1] - s * axis[2], c + (1 - c) * axis[1]**2, (1 - c) * axis[1] * axis[2] + s * axis[0]],
                      [(1 - c) * axis[0] * axis[2] + s * axis[1], (1 - c) * axis[1] * axis[2] - s * axis[0], c + (1 - c) * axis[2]**2]])
        return g

    @staticmethod
    def Euler2Axis(euler):
        """
        Compute the (axis, angle) representation associated to this (passive) rotation expressed by the Euler angles.

        :param euler: 3 euler angles (in degrees)
        :returns: a tuple containing the axis (a vector) and the angle (in radians).
        """
        (phi1, Phi, phi2) = np.radians(euler)
        t = np.tan(0.5 * Phi)
        s = 0.5 * (phi1 + phi2)
        d = 0.5 * (phi1 - phi2)
        tau = np.sqrt(t**2 + np.sin(s)**2)
        alpha = 2 * np.arctan2(tau, np.cos(s))
        if alpha > np.pi:
            axis = np.array([-t / tau * np.cos(d), -t / tau * np.sin(d), -1 / tau * np.sin(s)])
            angle = 2 * pi - alpha
        else:
            axis = np.array([t / tau * np.cos(d), t / tau * np.sin(d), 1 / tau * np.sin(s)])
            angle = alpha
        return axis, angle

    @staticmethod
    def Euler2Quaternion(euler):
        """
        Compute the quaternion from the 3 euler angles (in degrees)
        """
        (phi1, Phi, phi2) = np.radians(euler)
        q0 = np.cos(0.5 * (phi1 + phi2)) * np.cos(0.5 * Phi)
        q1 = np.cos(0.5 * (phi1 - phi2)) * np.sin(0.5 * Phi)
        q2 = np.sin(0.5 * (phi1 - phi2)) * np.sin(0.5 * Phi)
        q3 = np.sin(0.5 * (phi1 + phi2)) * np.cos(0.5 * Phi)
        return np.array([q0, q1, q2, q3])

    @staticmethod
    def Euler2Rodrigues(euler):
        """
        Compute the rodrigues vector from the 3 euler angles (in degrees)
        """
        (phi1, Phi, phi2) = np.radians(euler)
        a = 0.5 * (phi1 - phi2)
        b = 0.5 * (phi1 + phi2)
        r1 = np.tan(0.5 * Phi) * np.cos(a) / np.cos(b)
        r2 = np.tan(0.5 * Phi) * np.sin(a) / np.cos(b)
        r3 = np.tan(b)
        return np.array([r1, r2, r3])

    @staticmethod
    def read_euler_txt(txt_path):
        """
        Read a set of euler angles from an ascii file.

        :param str txt_path: path to the text file containing the euler angles.
        :returns dict: a dictionary with the line number and the corresponding orientation.
        """
        return Orientation.read_orientations(txt_path)

    @staticmethod
    def read_orientations(txt_path, data_type='euler', **kwargs):
        """
        Read a set of grain orientations from a text file.

        The text file must be organised in 3 columns (the other are ignored), corresponding to either the three euler
        angles or the three rodrigues veotor components, depending on the data_type). Internally the ascii file is read
        by the genfromtxt function of numpy, additional keyworks (such as the delimiter) can be passed to via the
        kwargs dictionnary.

        :param str txt_path: path to the text file containing the orientations.
        :param str data_type: 'euler' (default) or 'rodrigues'.
        :param dict kwargs: additional parameters passed to genfromtxt.
        :returns dict: a dictionary with the line number and the corresponding orientation.
        """
        data = np.genfromtxt(txt_path, **kwargs)
        size = len(data)
        orientations = []
        for i in range(size):
            angles = np.array([float(data[i, 0]), float(data[i, 1]), float(data[i, 2])])
            if data_type == 'euler':
                orientations.append([i + 1, Orientation.from_euler(angles)])
            elif data_type == 'rodrigues':
                orientations.append([i + 1, Orientation.from_rodrigues(angles)])
        return dict(orientations)

    @staticmethod
    def read_euler_from_zset_inp(inp_path):
        """Read a set of grain orientations from a z-set input file.

        In z-set input files, the orientation data may be specified
        either using the rotation of two vector, euler angles or
        rodrigues components directly. For instance the following
        lines are extracted from a polycrystalline calculation file
        using the rotation keyword:

        ::

         **elset elset1 *file au.mat *integration theta_method_a 1.0 1.e-9 150 *rotation x1 0.438886 -1.028805 0.197933 x3 1.038339 0.893172 1.003888
         **elset elset2 *file au.mat *integration theta_method_a 1.0 1.e-9 150 *rotation x1 0.178825 -0.716937 1.043300 x3 0.954345 0.879145 1.153101
         **elset elset3 *file au.mat *integration theta_method_a 1.0 1.e-9 150 *rotation x1 -0.540479 -0.827319 1.534062 x3 1.261700 1.284318 1.004174
         **elset elset4 *file au.mat *integration theta_method_a 1.0 1.e-9 150 *rotation x1 -0.941278 0.700996 0.034552 x3 1.000816 1.006824 0.885212
         **elset elset5 *file au.mat *integration theta_method_a 1.0 1.e-9 150 *rotation x1 -2.383786 0.479058 -0.488336 x3 0.899545 0.806075 0.984268

        :param str inp_path: the path to the ascii file to read.
        :returns dict: a dictionary of the orientations associated with the elset names.
        """
        inp = open(inp_path)
        lines = inp.readlines()
        for i, line in enumerate(lines):
            if line.lstrip().startswith('***material'):
                break
        euler_lines = []
        for j, line in enumerate(lines[i + 1:]):
            # read until next *** block
            if line.lstrip().startswith('***'):
                break
            if (not line.lstrip().startswith('%') and line.find('**elset') >= 0):
                euler_lines.append(line)
        euler = []
        for l in euler_lines:
            tokens = l.split()
            elset = tokens[tokens.index('**elset') + 1]
            irot = tokens.index('*rotation')
            if tokens[irot + 1] == 'x1':
                x1 = np.empty(3, dtype=float)
                x1[0] = float(tokens[irot + 2])
                x1[1] = float(tokens[irot + 3])
                x1[2] = float(tokens[irot + 4])
                x3 = np.empty(3, dtype=float)
                x3[0] = float(tokens[irot + 6])
                x3[1] = float(tokens[irot + 7])
                x3[2] = float(tokens[irot + 8])
                euler.append([elset, Orientation.Zrot2OrientationMatrix(x1=x1, x3=x3)])
            else:  # euler angles
                phi1 = tokens[irot + 1]
                Phi = tokens[irot + 2]
                phi2 = tokens[irot + 3]
                angles = np.array([float(phi1), float(Phi), float(phi2)])
                euler.append([elset, Orientation.from_euler(angles)])
        return dict(euler)

    def slip_system_orientation_tensor(self, s):
        """Compute the orientation strain tensor m^s for this :py:class:`~pymicro.crystal.microstructure.Orientation`
        and the given slip system.

        :param s: an instance of :py:class:`~pymicro.crystal.lattice.SlipSystem`

        .. math::

          M^s_{ij} = \left(l^s_i.n^s_j)
        """
        gt = self.orientation_matrix().transpose()
        plane = s.get_slip_plane()
        n_rot = np.dot(gt, plane.normal())
        slip = s.get_slip_direction()
        l_rot = np.dot(gt, slip.direction())
        return np.outer(l_rot, n_rot)

    def slip_system_orientation_strain_tensor(self, s):
        """Compute the orientation strain tensor m^s for this :py:class:`~pymicro.crystal.microstructure.Orientation`
        and the given slip system.

        :param s: an instance of :py:class:`~pymicro.crystal.lattice.SlipSystem`

        .. math::

          m^s_{ij} = \\frac{1}{2}\left(l^s_i.n^s_j + l^s_j.n^s_i)
        """
        gt = self.orientation_matrix().transpose()
        plane = s.get_slip_plane()
        n_rot = np.dot(gt, plane.normal())
        slip = s.get_slip_direction()
        l_rot = np.dot(gt, slip.direction())
        m = 0.5 * (np.outer(l_rot, n_rot) + np.outer(n_rot, l_rot))
        return m

    def slip_system_orientation_rotation_tensor(self, s):
        """Compute the orientation rotation tensor q^s for this :py:class:`~pymicro.crystal.microstructure.Orientation`
        and the given slip system.

        :param s: an instance of :py:class:`~pymicro.crystal.lattice.SlipSystem`

        .. math::

          q^s_{ij} = \\frac{1}{2}\left(l^s_i.n^s_j - l^s_j.n^s_i)
        """
        gt = self.orientation_matrix().transpose()
        plane = s.get_slip_plane()
        n_rot = np.dot(gt, plane.normal())
        slip = s.get_slip_direction()
        l_rot = np.dot(gt, slip.direction())
        q = 0.5 * (np.outer(l_rot, n_rot) - np.outer(n_rot, l_rot))
        return q

    def schmid_factor(self, slip_system, load_direction=[0., 0., 1]):
        """Compute the Schmid factor for this crystal orientation and the
        given slip system.

        :param slip_system: a slip system instance.
        :param load_direction: a unit vector describing the loading direction (default: vertical axis [0, 0, 1]).
        :returns float: a number between 0 ad 0.5.
        """
        plane = slip_system.get_slip_plane()
        gt = self.orientation_matrix().transpose()
        n_rot = np.dot(gt, plane.normal())  # plane.normal() is a unit vector
        slip = slip_system.get_slip_direction().direction()
        slip_rot = np.dot(gt, slip)
        # print n_rot, load_direction, slip_rot
        SF = np.abs(np.dot(n_rot, load_direction) * np.dot(slip_rot, load_direction))
        return SF

    def compute_all_schmid_factors(self, slip_systems, load_direction=[0., 0., 1], verbose=False):
        """Compute all Schmid factors for this crystal orientation and the
        given list of slip systems.

        :param slip_systems: a list of the slip system from which to compute the Schmid factor values.
        :param load_direction: a unit vector describing the loading direction (default: vertical axis [0, 0, 1]).
        :param bool verbose: activate verbose mode.
        :returns list: a list of the schmid factors.
        """
        SF_list = []
        for ss in slip_systems:
            sf = self.schmid_factor(ss, load_direction)
            if verbose:
                print('Slip system: %s, Schmid factor is %.3f' % (ss, sf))
            SF_list.append(sf)
        return SF_list


class Grain:
    """
    Class defining a crystallographic grain.

    A grain has its own crystallographic orientation.
    An optional id for the grain may be specified.
    The position field is the center of mass of the grain in world coordinates.
    The volume of the grain is expressed in pixel/voxel unit.
    """

    def __init__(self, grain_id, grain_orientation):
        self.id = grain_id
        self.orientation = grain_orientation
        self.position = np.array([0., 0., 0.])
        self.volume = 0  # warning not implemented
        self.vtkmesh = None

    def __repr__(self):
        """Provide a string representation of the class."""
        s = '%s\n * id = %d\n' % (self.__class__.__name__, self.id)
        s += ' * %s\n' % (self.orientation)
        s += ' * position %s\n' % np.array_str(self.position)
        s += ' * has vtk mesh ? %s\n' % (self.vtkmesh != None)
        return s

    def schmid_factor(self, slip_system, load_direction=[0., 0., 1]):
        """Compute the Schmid factor of this grain for the given slip system.

        **Parameters**:

        *slip_system*: a slip system instance.

        *load_direction*: a unit vector describing the loading direction.

        **Returns**

        The Schmid factor of this grain for the given slip system.
        """
        plane = slip_system.get_slip_plane()
        gt = self.orientation_matrix().transpose()
        n_rot = np.dot(gt, plane.normal())  # plane.normal() is a unit vector
        slip = slip_system.get_slip_direction().direction()
        slip_rot = np.dot(gt, slip)
        SF = np.abs(np.dot(n_rot, load_direction) * np.dot(slip_rot, load_direction))
        return self.orientation.schmid_factor(slip_system, load_direction)

    def SetVtkMesh(self, mesh):
        """Set the VTK mesh of this grain.

        **Parameters:**

        *mesh* The grain mesh in VTK format (typically vtkunstructuredgrid)
        """
        self.vtkmesh = mesh

    def add_vtk_mesh(self, array, contour=True, verbose=False):
        label = self.id  # we use the grain id here...
        # create vtk structure
        from scipy import ndimage
        from vtk.util import numpy_support
        grain_size = np.shape(array)
        array_bin = (array == label).astype(np.uint8)
        if verbose: print np.unique(array_bin)
        local_com = ndimage.measurements.center_of_mass(array_bin, array)
        vtk_data_array = numpy_support.numpy_to_vtk(np.ravel(array_bin, order='F'), deep=1)
        grid = vtk.vtkUniformGrid()
        grid.SetOrigin(-local_com[0], -local_com[1], -local_com[2])
        grid.SetSpacing(1, 1, 1)
        if vtk.vtkVersion().GetVTKMajorVersion() > 5:
            grid.SetScalarType(vtk.VTK_UNSIGNED_CHAR, vtk.vtkInformation())
        else:
            grid.SetScalarType(vtk.VTK_UNSIGNED_CHAR)
        if contour:
            grid.SetExtent(0, grain_size[0] - 1, 0, grain_size[1] - 1, 0, grain_size[2] - 1)
            grid.GetPointData().SetScalars(vtk_data_array)
            # contouring selected grain
            contour = vtk.vtkContourFilter()
            if vtk.vtkVersion().GetVTKMajorVersion() > 5:
                contour.SetInputData(grid)
            else:
                contour.SetInput(grid)
            contour.SetValue(0, 0.5)
            contour.Update()
            if verbose: print contour.GetOutput()
            self.SetVtkMesh(contour.GetOutput())
        else:
            grid.SetExtent(0, grain_size[0], 0, grain_size[1], 0, grain_size[2])
            grid.GetCellData().SetScalars(vtk_data_array)
            # threshold selected grain
            if verbose: print 'thresholding label', label
            thresh = vtk.vtkThreshold()
            thresh.ThresholdBetween(0.5, 1.5)
            # thresh.ThresholdBetween(label-0.5, label+0.5)
            if vtk.vtkVersion().GetVTKMajorVersion() > 5:
                thresh.SetInputData(grid)
            else:
                thresh.SetInput(grid)
            thresh.Update()
            if verbose: print thresh.GetOutput()
            self.SetVtkMesh(thresh.GetOutput())

    def to_xml(self, doc, file_name=None):
        """
        Returns an XML representation of the Grain instance.
        """
        grain = doc.createElement('Grain')
        grain_id = doc.createElement('Id')
        grain_id_text = doc.createTextNode('%s' % self.id)
        grain_id.appendChild(grain_id_text)
        grain.appendChild(grain_id)
        grain.appendChild(self.orientation.to_xml(doc))
        grain_position = doc.createElement('Position')
        grain_position_x = doc.createElement('X')
        grain_position.appendChild(grain_position_x)
        grain_position_x_text = doc.createTextNode('%f' % self.position[0])
        grain_position_x.appendChild(grain_position_x_text)
        grain_position_y = doc.createElement('Y')
        grain_position.appendChild(grain_position_y)
        grain_position_y_text = doc.createTextNode('%f' % self.position[1])
        grain_position_y.appendChild(grain_position_y_text)
        grain_position_z = doc.createElement('Z')
        grain_position.appendChild(grain_position_z)
        grain_position_z_text = doc.createTextNode('%f' % self.position[2])
        grain_position_z.appendChild(grain_position_z_text)
        grain.appendChild(grain_position)
        grain_mesh = doc.createElement('Mesh')
        if not file_name:
            file_name = self.vtk_file_name()
        grain_mesh_text = doc.createTextNode('%s' % file_name)
        grain_mesh.appendChild(grain_mesh_text)
        grain.appendChild(grain_mesh)
        return grain

    @staticmethod
    def from_xml(grain_node, verbose=False):
        grain_id = grain_node.childNodes[0]
        grain_orientation = grain_node.childNodes[1]
        orientation = Orientation.from_xml(grain_orientation)
        id = int(grain_id.childNodes[0].nodeValue)
        grain = Grain(id, orientation)
        grain_position = grain_node.childNodes[2]
        xg = float(grain_position.childNodes[0].childNodes[0].nodeValue)
        yg = float(grain_position.childNodes[1].childNodes[0].nodeValue)
        zg = float(grain_position.childNodes[2].childNodes[0].nodeValue)
        grain.position = np.array([xg, yg, zg])
        grain_mesh = grain_node.childNodes[3]
        grain_mesh_file = grain_mesh.childNodes[0].nodeValue
        if verbose: print grain_mesh_file
        grain.load_vtk_repr(grain_mesh_file, verbose)
        return grain

    def vtk_file_name(self):
        return 'grain_%d.vtu' % self.id

    def save_vtk_repr(self, file_name=None):
        import vtk
        if not file_name:
            file_name = self.vtk_file_name()
        print 'writting ' + file_name
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(file_name)
        writer.SetInput(self.vtkmesh)
        writer.Write()

    def load_vtk_repr(self, file_name, verbose=False):
        import vtk
        if verbose: print 'reading ' + file_name
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(file_name)
        reader.Update()
        self.vtkmesh = reader.GetOutput()

    def orientation_matrix(self):
        """Returns the grain orientation matrix."""
        return self.orientation.orientation_matrix()

    def dct_omega_angles(self, hkl, lambda_keV, verbose=False):
        """Compute the two omega angles which satisfy the Bragg condition.

        For a grain with a given crystal orientation sitting on a vertical
        rotation axis, there is exactly two omega positions in [0, 2pi] for
        which a particular hkl reflexion will fulfil Bragg's law.
        See :py:func:`~pymicro.crystal.microstructure.Orientation.dct_omega_angles`
        of the :py:class:`~pymicro.crystal.microstructure.Orientation` class.

        :param hkl: The given cristallographic :py:class:`~pymicro.crystal.lattice.HklPlane`
        :param float lambda_keV: The X-rays energy expressed in keV
        :param bool verbose: Verbose mode (False by default)
        :returns tuple: (w1, w2) the two values of the omega angle.
        """
        return self.orientation.dct_omega_angles(hkl, lambda_keV, verbose)


class Microstructure:
    """
    Class used to manipulate a full microstructure.

    It is typically defined as a list of grains objects.
    """

    def __init__(self, name='empty'):
        self.name = name
        self.grains = []
        self.vtkmesh = None

    @staticmethod
    def random_texture(n=100):
        """Generate a random texture microstructure.

        **parameters:**

        *n* The number of grain orientations in the microstructure.
        """
        from random import random
        from math import acos
        m = Microstructure(name='random_texture')
        for i in range(n):
            phi1 = random() * 360.
            Phi = 180. * acos(2 * random() - 1) / np.pi
            phi2 = random() * 360.
            m.grains.append(Grain(i + 1, Orientation.from_euler([phi1, Phi, phi2])))
        return m

    @staticmethod
    def rand_cmap(N=4096, first_is_black=False):
        """Creates a random color map.

           The first color can be enforced to black and usually figure out the background.
           The random seed is fixed to consistently produce the same colormap.
        """
        np.random.seed(13)
        rand_colors = np.random.rand(N, 3)
        if first_is_black:
            rand_colors[0] = [0., 0., 0.]  # enforce black background (value 0)
        return colors.ListedColormap(rand_colors)

    def ipf_cmap(self):
        """
        Return a colormap with ipf colors.
        """

        N = len(self.grains)
        ipf_colors = np.zeros((4096, 3))
        for g in self.grains:
            ipf_colors[g.id,:] = g.orientation.get_ipf_colour()
        return colors.ListedColormap(ipf_colors)

    @staticmethod
    def from_xml(xml_file_name, grain_ids=None, verbose=False):
        """Load a Microstructure object from an xml file.

        It is possible to restrict the grains which are loaded by providing
        the list of ids of the grains of interest.
        """
        if verbose and grain_ids: print 'loading only grain ids %s' % grain_ids
        micro = Microstructure()
        dom = parse(xml_file_name)
        root = dom.childNodes[0]
        name = root.childNodes[0]
        micro.name = name.childNodes[0].nodeValue
        grains = root.childNodes[1]
        for node in grains.childNodes:
            if grain_ids and not (int(node.childNodes[0].childNodes[0].nodeValue) in grain_ids): continue
            if verbose: print node
            micro.grains.append(Grain.from_xml(node, verbose))
        return micro

    def get_grain(self, gid):
        """Get a particular grain given its id.

        This method browses the microstructure and return the grain
        corresponding to the given id. If the grain is not found, the
        method raises a `ValueError`.

        *Parameters*

        **gid**: the grain id.

        *Returns*

        The method return a `Grain` with the corresponding id.
        """
        for grain in self.grains:
            if grain.id == gid:
                return grain
        raise ValueError('grain %d not found in the microstructure' % gid)

    def __repr__(self):
        """Provide a string representation of the class."""
        s = '%s\n' % self.__class__.__name__
        s += '* name: %s\n' % self.name
        for g in self.grains:
            s += '* %s' % g.__repr__
        return s

    def SetVtkMesh(self, mesh):
        self.vtkmesh = mesh

    def print_zset_material_block(self, mat_file, grain_prefix='_ELSET'):
        """
        Outputs the material block corresponding to this microstructure for
        a finite element calculation with z-set.

        :param str mat_file: The name of the file where the material behaviour is located
        :param str grain_prefix: The grain prefix used to name the elsets corresponding to the different grains
        """
        f = open('elset_list.txt', 'w')
        for g in self.grains:
            o = g.orientation
            f.write(
                '  **elset %s%d *file %s *integration theta_method_a 1.0 1.e-9 150 *rotation %7.3f %7.3f %7.3f\n' % (
                    grain_prefix, g.id, mat_file, o.phi1(), o.Phi(), o.phi2()))
        f.close()

    def to_xml(self, doc):
        """
        Returns an XML representation of the Microstructure instance.
        """
        root = doc.createElement('Microstructure')
        doc.appendChild(root)
        name = doc.createElement('Name')
        root.appendChild(name)
        name_text = doc.createTextNode(self.name)
        name.appendChild(name_text)
        grains = doc.createElement('Grains')
        root.appendChild(grains)
        for i, grain in enumerate(self.grains):
            file_name = os.path.join(self.name, '%s_%d.vtu' % (self.name, i))
            grains.appendChild(grain.to_xml(doc, file_name))

    def save(self):
        """Saving the microstructure to the disk.

        Save the metadata as a XML file and when available, also save the
        vtk representation of the grains.
        """
        # save the microstructure instance as xml
        doc = Document()
        self.to_xml(doc)
        xml_file_name = '%s.xml' % self.name
        print 'writting ' + xml_file_name
        f = open(xml_file_name, 'wb')
        doc.writexml(f, encoding='utf-8')
        f.close()
        # now save the vtk representation
        if self.vtkmesh != None:
            import vtk
            vtk_file_name = '%s.vtm' % self.name
            print 'writting ' + vtk_file_name
            writer = vtk.vtkXMLMultiBlockDataWriter()
            writer.SetFileName(vtk_file_name)
            if vtk.vtkVersion().GetVTKMajorVersion() > 5:
                writer.SetInputData(self.vtkmesh)
            else:
                writer.SetInput(self.vtkmesh)
            writer.Write()

    def dct_projection(self, data, lattice, omega, dif_grains, lambda_keV, d, ps, det_npx=np.array([2048, 2048]), ds=1,
                       display=False, verbose=False):
        """Compute the detector image in dct configuration.

        :params np.ndarray data: The 3d data set from which to compute the projection.
        :params lattice: The crystal lattice of the material.
        :params float omega: The rotation angle at which the projection is computed.
        """
        lambda_nm = 1.2398 / lambda_keV
        # prepare rotation matrix
        omegar = omega * np.pi / 180
        R = np.array([[np.cos(omegar), -np.sin(omegar), 0], [np.sin(omegar), np.cos(omegar), 0], [0, 0, 1]])
        data_abs = np.where(data > 0, 1, 0)
        x_max = np.ceil(max(data_abs.shape[0], data_abs.shape[1]) * 2 ** 0.5)
        proj = np.zeros((np.shape(data_abs)[2], x_max), dtype=np.float)
        if verbose:
            print 'diffracting grains', dif_grains
            print 'proj size is ', np.shape(proj)
        # handle each grain in Bragg condition
        for (gid, (h, k, l)) in dif_grains:
            mask_dif = (data == gid)
            data_abs[mask_dif] = 0  # remove this grain from the absorption
        from skimage.transform import radon
        for i in range(np.shape(data_abs)[2]):
            proj[i, :] = radon(data_abs[:, :, i], [omega])[:, 0]
        # create the detector image (larger than the FOV) by padding the transmission image with zeros
        full_proj = np.zeros(det_npx / ds, dtype=np.float)
        if verbose:
            print 'full proj size is ', np.shape(full_proj)
            print 'max proj', proj.max()
            # here we could use np.pad with numpy version > 1.7
            print int(0.5 * det_npx[0] / ds - proj.shape[0] / 2.)
            print int(0.5 * det_npx[0] / ds + proj.shape[0] / 2.)
            print int(0.5 * det_npx[1] / ds - proj.shape[1] / 2.)
            print int(0.5 * det_npx[1] / ds + proj.shape[1] / 2.)
        # let's moderate the direct beam so we see nicely the spots with a 8 bits scale
        att = 6.0 / ds  # 1.0
        full_proj[int(0.5 * det_npx[0] / ds - proj.shape[0] / 2.):int(0.5 * det_npx[0] / ds + proj.shape[0] / 2.), \
        int(0.5 * det_npx[1] / ds - proj.shape[1] / 2.):int(0.5 * det_npx[1] / ds + proj.shape[1] / 2.)] += proj / att
        # add diffraction spots
        from pymicro.crystal.lattice import HklPlane
        from scipy import ndimage
        for (gid, (h, k, l)) in dif_grains:
            # compute scattering vector
            gt = self.get_grain(gid).orientation_matrix().transpose()
            p = HklPlane(h, k, l, lattice)
            X = np.array([1., 0., 0.]) / lambda_nm
            n = R.dot(gt.dot(p.normal()))
            G = n / p.interplanar_spacing()  # also G = R.dot(gt.dot(h*astar + k*bstar + l*cstar))
            K = X + G
            # TODO explain the - signs, account for grain position in the rotated sample
            (u, v) = (d * K[1] / K[0], d * K[2] / K[0])  # unit is mm
            (u_mic, v_mic) = (1000 * u, 1000 * v)  # unit is micron
            (up, vp) = (0.5 * det_npx[0] / ds + u_mic / (ps * ds),
                        0.5 * det_npx[1] / ds + v_mic / (ps * ds))  # unit is pixel on the detector
            if verbose:
                print 'plane normal:', p.normal()
                print R
                print 'rotated plane normal:', n
                print 'scattering vector:', G
                print 'K = X + G vector', K
                print 'lenght X', np.linalg.norm(X)
                print 'lenght K', np.linalg.norm(K)
                print 'angle between X and K', np.arccos(
                    np.dot(K, X) / (np.linalg.norm(K) * np.linalg.norm(X))) * 180 / np.pi
                print 'diffracted beam will hit the detector at (%.3f,%.3f) mm or (%d,%d) pixels' % (u, v, up, vp)
            grain_data = np.where(data == gid, 1, 0)
            data_dif = grain_data[ndimage.find_objects(data == gid)[0]]
            x_max = np.ceil(max(data_dif.shape[0], data_dif.shape[1]) * 2 ** 0.5)
            proj_dif = np.zeros((np.shape(data_dif)[2], x_max), dtype=np.float)
            for i in range(np.shape(data_dif)[2]):
                a = radon(data_dif[:, :, i], [omega])
                proj_dif[i, :] = a[:, 0]
            if verbose:
                print '* proj_dif size is ', np.shape(proj_dif)
                print int(up - proj_dif.shape[0] / 2.)
                print int(up + proj_dif.shape[0] / 2.)
                print int(vp - proj_dif.shape[1] / 2.)
                print int(vp + proj_dif.shape[1] / 2.)
                print 'max proj_dif', proj_dif.max()
            # add diffraction spot to the image detector
            try:
                # warning full_proj image is transposed (we could fix that and plot with .T since pyplot plots images like (y,x))
                full_proj[int(vp - proj_dif.shape[0] / 2.):int(vp + proj_dif.shape[0] / 2.), \
                int(up - proj_dif.shape[1] / 2.):int(up + proj_dif.shape[1] / 2.)] += proj_dif
                # full_proj[int(up - proj_dif.shape[0]/2.):int(up + proj_dif.shape[0]/2.), \
                #        int(vp - proj_dif.shape[1]/2.):int(vp + proj_dif.shape[1]/2.)] += proj_dif
            except:
                print 'error occured'  # grain diffracts outside the detector
                pass
            plt.imsave('proj_dif/proj_dif_grain%d_omega=%05.1f.png' % (gid, omega), proj_dif, cmap=cm.gray,
                       origin='lower')
        if display:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            ax.imshow(full_proj[:, ::-1], cmap=cm.gray, vmin=0, vmax=255, origin='lower')  # check origin
            for (h, k, l) in [(1, 1, 0), (2, 0, 0), (2, 1, 1), (2, 2, 0), (2, 2, 2), (3, 1, 0), (3, 2, 1), (3, 3, 0),
                              (3, 3, 2)]:
                hkl = HklPlane(h, k, l, lattice)
                theta = hkl.bragg_angle(lambda_keV)
                print 'bragg angle for %s reflection is %.2f deg' % (hkl.miller_indices(), theta * 180. / np.pi)
                t = np.linspace(0.0, 2 * np.pi, num=37)
                L = d * 1000 / ps / ds * np.tan(2 * theta)  # 2 theta distance on the detector
                ax.plot(0.5 * det_npx[0] / ds + L * np.cos(t), 0.5 * det_npx[1] / ds + L * np.sin(t), 'g--')
                ax.annotate(str(h) + str(k) + str(l), xy=(0.5 * det_npx[0] / ds, 0.5 * det_npx[1] / ds + L),
                            xycoords='data', color='green', horizontalalignment='center', verticalalignment='bottom',
                            fontsize=16)
            plt.xlim(0, det_npx[0] / ds)
            plt.ylim(0, det_npx[1] / ds)
            plt.show()
        else:
            # save projection image with origin = lower since Z-axis is upwards
            plt.imsave('proj/proj_omega=%05.1f.png' % omega, full_proj, cmap=cm.gray, vmin=0, vmax=100, origin='lower')
