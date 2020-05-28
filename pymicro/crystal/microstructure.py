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
import os
import vtk
import h5py
from scipy import ndimage
from matplotlib import pyplot as plt, colors, cm
from xml.dom.minidom import Document, parse
from pymicro.crystal.lattice import Lattice, Symmetry
from pymicro.crystal.quaternion import Quaternion
from math import atan2, pi


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
        self.quat = Orientation.OrientationMatrix2Quaternion(g, P=1)


    def orientation_matrix(self):
        """Returns the orientation matrix in the form of a 3x3 numpy array."""
        return self._matrix

    def __repr__(self):
        """Provide a string representation of the class."""
        s = 'Crystal Orientation'
        s += '\norientation matrix = %s' % self._matrix.view()
        s += '\nEuler angles (degrees) = (%8.3f,%8.3f,%8.3f)' % (self.phi1(), self.Phi(), self.phi2())
        s += '\nRodrigues vector = %s' % self.OrientationMatrix2Rodrigues(self._matrix)
        s += '\nQuaternion = %s' % self.OrientationMatrix2Quaternion(self._matrix, P=1)
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

    def get_ipf_colour(self, axis=np.array([0., 0., 1.]), symmetry=Symmetry.cubic):
        """Compute the IPF (inverse pole figure) colour for this orientation.

        Given a particular axis expressed in the laboratory coordinate system,
        one can compute the so called IPF colour based on that direction
        expressed in the crystal coordinate system as :math:`[x_c,y_c,z_c]`.
        There is only one tuple (u,v,w) such that:

        .. math::

          [x_c,y_c,z_c]=u.[0,0,1]+v.[0,1,1]+w.[1,1,1]

        and it is used to assign the RGB colour.
        """
        axis /= np.linalg.norm(axis)
        # find the axis lying in the fundamental zone
        for sym in symmetry.symmetry_operators():
            Osym = np.dot(sym, self.orientation_matrix())
            Vc = np.dot(Osym, axis)
            if Vc[2] < 0:
                Vc *= -1.  # using the upward direction
            uvw = np.array([Vc[2] - Vc[1], Vc[1] - Vc[0], Vc[0]])
            uvw /= np.linalg.norm(uvw)
            uvw /= max(uvw)
            if (uvw[0] >= 0. and uvw[0] <= 1.0) and (uvw[1] >= 0. and uvw[1] <= 1.0) and (
                            uvw[2] >= 0. and uvw[2] <= 1.0):
                # print('found sym for sst')
                break
        return uvw

    def fzDihedral(rod, n):
        """check if the given Rodrigues vector is in the fundamental zone.
        
        After book from Morawiecz.
        """
        # top and bottom face at +/-tan(pi/2n)
        t = np.tan(np.pi / (2 * n))
        if abs(rod[2]) > t:
            return False

        # 2n faces distance 1 from origin
        # y <= ((2+sqrt(2))*t - (1+sqrt(2))) * x + (1+sqrt(2))*(1-t)
        y, x = sorted([abs(ro[0]), abs(ro[1])])
        if x > 1:
            return False

        return {
            2: True,
            3: y / (1 + math.sqrt(2)) + (1 - math.sqrt(2 / 3)) * x < 1 - 1 / math.sqrt(3),
            4: y + x < math.sqrt(2),
            6: y / (1 + math.sqrt(2)) + (1 - 2 * math.sqrt(2) + math.sqrt(6)) * x < math.sqrt(3) - 1
        }[n]

    def inFZ(self, symmetry=Symmetry.cubic):
        """Check if the given Orientation lies within the fundamental zone.
        
        For a given crystal symmetry, several rotations can describe the same 
        physcial crystllographic arangement. The Rodrigues fundamental zone 
        restrict the orientation space accordingly. 
        """
        r = self.rod
        if symmetry == Symmetry.cubic:
            inFZT23 = np.abs(r).sum() <= 1.0
            # in the cubic symmetry, each component must be < 2 ** 0.5 - 1
            inFZ = inFZT23 and np.abs(r).max() <= 2 ** 0.5 - 1
        else:
            raise (ValueError('unsupported crystal symmetry: %s' % symmetry))
        return inFZ

    def move_to_FZ(self, symmetry=Symmetry.cubic, verbose=False):
        """
        Compute the equivalent crystal orientation in the Fundamental Zone of a given symmetry.

        :param Symmetry symmetry: an instance of the `Symmetry` class 
        :param verbose: flag for verbose mode
        :return: a new Orientation instance which lies in the fundamental zone.
        """
        om = symmetry.move_rotation_to_FZ(self.orientation_matrix(), verbose=verbose)
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
        if 0 <= psidg <= 45:
            p = 2. / 15 * (1 - cos(psi))
        elif 45 < psidg <= 60:
            p = 2. / 15 * (3 * (sqrt(2) - 1) * sin(psi) - 2 * (1 - cos(psi)))
        elif 60 < psidg <= 60.72:
            p = 2. / 15 * ((3 * (sqrt(2) - 1) + 4. / sqrt(3)) * sin(psi) - 6. * (1 - cos(psi)))
        elif 60.72 < psidg <= 62.8:
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
        if cw > 1. and cw - 1. < 10 * np.finfo('float32').eps:
            #print('cw=%.20f, rounding to 1.' % cw)
            cw = 1.
        omega = np.arccos(cw)
        return omega

    def disorientation(self, orientation, crystal_structure=Symmetry.triclinic):
        """Compute the disorientation another crystal orientation.

        Considering all the possible crystal symmetries, the disorientation
        is defined as the combination of the minimum misorientation angle
        and the misorientation axis lying in the fundamental zone, which
        can be used to bring the two lattices into coincidence.
        
        .. note::
        
         Both orientations are supposed to have the same symmetry. This is not necessarily the case in multi-phase 
         materials.

        :param orientation: an instance of :py:class:`~pymicro.crystal.microstructure.Orientation` class desribing the other crystal orientation from which to compute the angle.
        :param crystal_structure: an instance of the `Symmetry` class describing the crystal symmetry, triclinic (no symmetry) by default.
        :returns tuple: the misorientation angle in radians, the axis as a numpy vector (crystal coordinates), the axis as a numpy vector (sample coordinates).
        """
        the_angle = np.pi
        symmetries = crystal_structure.symmetry_operators()
        (gA, gB) = (self.orientation_matrix(), orientation.orientation_matrix())  # nicknames
        for (g1, g2) in [(gA, gB), (gB, gA)]:
            for j in range(symmetries.shape[0]):
                sym_j = symmetries[j]
                oj = np.dot(sym_j, g1)  # the crystal symmetry operator is left applied
                for i in range(symmetries.shape[0]):
                    sym_i = symmetries[i]
                    oi = np.dot(sym_i, g2)
                    delta = np.dot(oi, oj.T)
                    #print('delta={}'.format(delta))
                    mis_angle = Orientation.misorientation_angle_from_delta(delta)
                    #print(np.degrees(mis_angle))
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

    @staticmethod
    def solve_trig_equation(A, B, C, verbose=False):
        """Solve the trigonometric equation in the form of:
        
        .. math::

           A\cos\\theta + B\sin\\theta = C

        :param float A: the A constant in the equation.
        :param float B: the B constant in the equation.
        :param float C: the C constant in the equation.
        :return tuple: the two solutions angular values in degrees.
        """
        Delta = 4 * (A ** 2 + B ** 2 - C ** 2)
        if Delta < 0:
            raise ValueError('Delta < 0 (%f)' % Delta)
        if verbose:
            print('A={0:.3f}, B={1:.3f}, C={2:.3f}, Delta={3:.1f}'.format(A, B, C, Delta))
        theta_1 = 2 * np.arctan2(B - 0.5 * np.sqrt(Delta), A + C) * 180. / np.pi % 360
        theta_2 = 2 * np.arctan2(B + 0.5 * np.sqrt(Delta), A + C) * 180. / np.pi % 360
        return theta_1, theta_2

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
        theta = hkl.bragg_angle(lambda_keV, verbose=verbose)
        lambda_nm = 1.2398 / lambda_keV
        gt = self.orientation_matrix().T  # gt = g^{-1} in Poulsen 2004
        Gc = hkl.scattering_vector()
        A = np.dot(Gc, gt[0])
        B = - np.dot(Gc, gt[1])
        # A = h / a * gt[0, 0] + k / b * gt[0, 1] + l / c * gt[0, 2]
        # B = -h / a * gt[1, 0] - k / b * gt[1, 1] - l / c * gt[1, 2]
        C = -2 * np.sin(theta) ** 2 / lambda_nm  # the minus sign comes from the main equation
        omega_1, omega_2 = Orientation.solve_trig_equation(A, B, C, verbose=verbose)
        if verbose:
            print('the two omega values in degrees fulfilling the Bragg condition are (%.1f, %.1f)' % (omega_1, omega_2))
        return omega_1, omega_2

    def rotating_crystal(self, hkl, lambda_keV, omega_step=0.5, display=True, verbose=False):
        from pymicro.xray.xray_utils import lambda_keV_to_nm
        lambda_nm = lambda_keV_to_nm(lambda_keV)
        X = np.array([1., 0., 0.]) / lambda_nm
        print('magnitude of X', np.linalg.norm(X))
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
            print('plane normal:', hkl.normal())
            print(R)
            print('rotated plane normal:', n, ' with a norm of', np.linalg.norm(n))
            G = n / hkl.interplanar_spacing()  # here G == N
            print('G vector:', G, ' with a norm of', np.linalg.norm(G))
            K = X + G
            print('X + G vector', K)
            magnitude_K.append(np.linalg.norm(K))
            print('magnitude of K', np.linalg.norm(K))
            alpha = np.arccos(np.dot(-X, G) / (np.linalg.norm(-X) * np.linalg.norm(G))) * 180 / np.pi
            print('angle between -X and G', alpha)
            alphas.append(alpha)
            twotheta = np.arccos(np.dot(K, X) / (np.linalg.norm(K) * np.linalg.norm(X))) * 180 / np.pi
            print('angle (deg) between K and X', twotheta)
            twothetas.append(twotheta)
        print('min alpha angle is ', min(alphas))

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

    @staticmethod
    def compute_instrument_transformation_matrix(rx_offset, ry_offset, rz_offset):
        """ Compute the instrument transformation matrix for given rotation offset.

        This function compute a 3x3 rotation matrix (passive convention) that transform the sample coordinate system
        by rotating around the 3 cartesian axes in this order: rotation around X is applied first, then around Y and
        finally around Z.

        A sample vector :math:`V_s` is consequently transformed into :math:`V'_s` as:

        .. math::

          V'_s = T^T.V_s

        :param double rx_offset: value to apply for the rotation around X.
        :param double ry_offset: value to apply for the rotation around Y.
        :param double rz_offset: value to apply for the rotation around Z.
        :return: a 3x3 rotation matrix describing the transformation applied by the diffractometer.
        """
        angle_zr = np.radians(rz_offset)
        angle_yr = np.radians(ry_offset)
        angle_xr = np.radians(rx_offset)
        Rz = np.array([[np.cos(angle_zr), -np.sin(angle_zr), 0], [np.sin(angle_zr), np.cos(angle_zr), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(angle_yr), 0, np.sin(angle_yr)], [0, 1, 0], [-np.sin(angle_yr), 0, np.cos(angle_yr)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(angle_xr), -np.sin(angle_xr)], [0, np.sin(angle_xr), np.cos(angle_xr)]])
        T = Rz.dot(np.dot(Ry, Rx))
        return T

    def topotomo_tilts(self, hkl, T=None, verbose=False):
        """Compute the tilts for topotomography alignment.

        :param hkl: the hkl plane, an instance of :py:class:`~pymicro.crystal.lattice.HklPlane`
        :param ndarray T: transformation matrix representing the diffractometer direction at omega=0.
        :param bool verbose: activate verbose mode (False by default).
        :returns tuple: (ut, lt) the two values of tilts to apply (in radians).
        """
        if T is None:
            T = np.eye(3)  # identity be default
        gt = self.orientation_matrix().transpose()
        Gc = hkl.scattering_vector()
        Gs = gt.dot(Gc)  # in the cartesian sample CS
        # apply instrument specific settings
        Gs = np.dot(T.T, Gs)
        # find topotomo tilts
        ut = np.arctan(Gs[1] / Gs[2])
        lt = np.arctan(-Gs[0] / (Gs[1] * np.sin(ut) + Gs[2] * np.cos(ut)))
        if verbose:
            print('up tilt (samrx) should be %.3f' % (ut * 180 / np.pi))
            print('low tilt (samry) should be %.3f' % (lt * 180 / np.pi))
        return ut, lt

    def to_xml(self, doc):
        """
        Returns an XML representation of the Orientation instance.
        """
        print('deprecated as we are moving to hdf5 format')
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
    def from_euler(euler, convention='Bunge'):
        """Rotation matrix from Euler angles.
        
        This is the classical method to obtain an orientation matrix by 3 successive rotations. The result depends on 
        the convention used (how the successive rotation axes are chosen). In the Bunge convention, the first rotation 
        is around Z, the second around the new X and the third one around the new Z. In the Roe convention, the second 
        one is around Y.
        """
        if convention == 'Roe':
            (phi1, phi, phi2) = (euler[0] + 90, euler[1], euler[2] - 90)
        else:
            (phi1, phi, phi2) = euler
        g = Orientation.Euler2OrientationMatrix((phi1, phi, phi2))
        o = Orientation(g)
        return o

    @staticmethod
    def from_rodrigues(rod):
        g = Orientation.Rodrigues2OrientationMatrix(rod)
        o = Orientation(g)
        return o

    @staticmethod
    def from_Quaternion(q):
        g = Orientation.Quaternion2OrientationMatrix(q)
        o = Orientation(g)
        return o

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
            print('warning, returning [0., 0., 0.], consider using axis, angle representation instead')
            return np.zeros(3)
        else:
            r1 = (g[1, 2] - g[2, 1]) / t
            r2 = (g[2, 0] - g[0, 2]) / t
            r3 = (g[0, 1] - g[1, 0]) / t
        return np.array([r1, r2, r3])

    @staticmethod
    def OrientationMatrix2Quaternion(g, P=1):
        q0 = 0.5 * np.sqrt(1 + g[0, 0] + g[1, 1] + g[2, 2])
        q1 = P * 0.5 * np.sqrt(1 + g[0, 0] - g[1, 1] - g[2, 2])
        q2 = P * 0.5 * np.sqrt(1 - g[0, 0] + g[1, 1] - g[2, 2])
        q3 = P * 0.5 * np.sqrt(1 - g[0, 0] - g[1, 1] + g[2, 2])

        if g[2, 1] < g[1, 2]:
            q1 = q1 * -1
        elif g[0, 2] < g[2, 0]:
            q2 = q2 * -1
        elif g[1, 0] < g[0, 1]:
            q3 = q3 * -1

        q = Quaternion(np.array([q0, q1, q2, q3]), convention=P)
        return q.quat

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
    def Rodrigues2Axis(rod):
        """
        Compute the axis/angle representation from the Rodrigues vector.

        :param rod: The Rodrigues vector as a 3 components array.
        :returns: A tuple in the (axis, angle) form.
        """
        r = np.linalg.norm(rod)
        axis = rod / r
        angle = 2 * np.arctan(r)
        return axis, angle

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
        g = np.array([[c + (1 - c) * axis[0] ** 2, (1 - c) * axis[0] * axis[1] + s * axis[2],
                       (1 - c) * axis[0] * axis[2] - s * axis[1]],
                      [(1 - c) * axis[0] * axis[1] - s * axis[2], c + (1 - c) * axis[1] ** 2,
                       (1 - c) * axis[1] * axis[2] + s * axis[0]],
                      [(1 - c) * axis[0] * axis[2] + s * axis[1], (1 - c) * axis[1] * axis[2] - s * axis[0],
                       c + (1 - c) * axis[2] ** 2]])
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
        tau = np.sqrt(t ** 2 + np.sin(s) ** 2)
        alpha = 2 * np.arctan2(tau, np.cos(s))
        if alpha > np.pi:
            axis = np.array([-t / tau * np.cos(d), -t / tau * np.sin(d), -1 / tau * np.sin(s)])
            angle = 2 * np.pi - alpha
        else:
            axis = np.array([t / tau * np.cos(d), t / tau * np.sin(d), 1 / tau * np.sin(s)])
            angle = alpha
        return axis, angle

    @staticmethod
    def Euler2Quaternion(euler, P=1):
        """
        Compute the quaternion from the 3 euler angles (in degrees).
        @param tuple euler: the 3 euler angles in degrees.
        @param int P: +1 to compute an active quaternion (default), -1 for a passive quaternion.
        @return: a `Quaternion` instance representing the rotation.
        """
        (phi1, Phi, phi2) = np.radians(euler)
        q0 = np.cos(0.5 * (phi1 + phi2)) * np.cos(0.5 * Phi)
        q1 = np.cos(0.5 * (phi1 - phi2)) * np.sin(0.5 * Phi)
        q2 = np.sin(0.5 * (phi1 - phi2)) * np.sin(0.5 * Phi)
        q3 = np.sin(0.5 * (phi1 + phi2)) * np.cos(0.5 * Phi)
        q = Quaternion(np.array([q0, -P * q1, -P * q2, -P * q3]), convention=P)
        return q

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
    def Quaternion2Euler(q):
        """
        Compute Euler angles from a Quaternion
        :param q: Quaternion
        :return: Euler angles (in degrees, Bunge convention)
        """
        P = q.convention
        (q0, q1, q2, q3) = q.quat
        q03 = q0 ** 2 + q3 ** 2
        q12 = q1 ** 2 + q2 ** 2
        chi = np.sqrt(q03 * q12)
        if chi == 0.:
            if q12 == 0.:
                phi_1 = atan2(-2 * P * q0 * q3, q0 ** 2 - q3 ** 2)
                Phi = 0.
            else:
                phi_1 = atan2(-2 * q1 * q2, q1 ** 2 - q2 ** 2)
                Phi = pi
            phi_2 = 0.
        else:
            phi_1 = atan2((q1 * q3 - P * q0 * q2) / chi, (-P * q0 * q1 - q2 * q3) / chi)
            Phi = atan2(2 * chi, q03 - q12)
            phi_2 = atan2((P * q0 * q2 + q1 * q3) / chi, (q2 * q3 - P * q0 * q1) / chi)
        return np.degrees([phi_1, Phi, phi_2])

    @staticmethod
    def Quaternion2OrientationMatrix(q):
        P = q.convention
        (q0, q1, q2, q3) = q.quat
        qbar = q0 ** 2 - q1 ** 2 - q2 ** 2 - q3 ** 2
        g = np.array([[qbar + 2 * q1 ** 2, 2 * (q1 * q2 - P * q0 * q3), 2 * (q1 * q3 + P * q0 * q2)],
                      [2 * (q1 * q2 + P * q0 * q3), qbar + 2 * q2 ** 2, 2 * (q2 * q3 - P * q0 * q1)],
                      [2 * (q1 * q3 - P * q0 * q2), 2 * (q2 * q3 + P * q0 * q1), qbar + 2 * q3 ** 2]])
        return g

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
    The center attribute is the center of mass of the grain in world coordinates.
    The volume of the grain is expressed in pixel/voxel unit.
    """

    def __init__(self, grain_id, grain_orientation):
        self.id = grain_id
        self.orientation = grain_orientation
        self.center = np.array([0., 0., 0.])
        self.volume = 0  # warning not implemented
        self.vtkmesh = None
        self.hkl_planes = []

    def __repr__(self):
        """Provide a string representation of the class."""
        s = '%s\n * id = %d\n' % (self.__class__.__name__, self.id)
        s += ' * %s\n' % (self.orientation)
        s += ' * center %s\n' % np.array_str(self.center)
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
        """Add a mesh to this grain.

        This method process a labeled array to extract the geometry of the grain. The grain shape is defined by 
        the pixels with a value of the grain id. A vtkUniformGrid object is created and thresholded or contoured 
        depending on the value of the flag `contour`. 
        The resulting mesh is returned, centered on the center of mass of the grain.

        :param ndarray array: a numpy array from which to extract the grain shape.
        :param bool contour: a flag to use contour mode for the shape.
        :param bool verbose: activate verbose mode.
        """
        label = self.id  # we use the grain id here...
        # create vtk structure
        from scipy import ndimage
        from vtk.util import numpy_support
        grain_size = np.shape(array)
        array_bin = (array == label).astype(np.uint8)
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
            if verbose:
                print(contour.GetOutput())
            self.SetVtkMesh(contour.GetOutput())
        else:
            grid.SetExtent(0, grain_size[0], 0, grain_size[1], 0, grain_size[2])
            grid.GetCellData().SetScalars(vtk_data_array)
            # threshold selected grain
            thresh = vtk.vtkThreshold()
            thresh.ThresholdBetween(0.5, 1.5)
            # thresh.ThresholdBetween(label-0.5, label+0.5)
            if vtk.vtkVersion().GetVTKMajorVersion() > 5:
                thresh.SetInputData(grid)
            else:
                thresh.SetInput(grid)
            thresh.Update()
            if verbose:
                print('thresholding label: %d' % label)
                print(thresh.GetOutput())
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
        grain_position_x_text = doc.createTextNode('%f' % self.center[0])
        grain_position_x.appendChild(grain_position_x_text)
        grain_position_y = doc.createElement('Y')
        grain_position.appendChild(grain_position_y)
        grain_position_y_text = doc.createTextNode('%f' % self.center[1])
        grain_position_y.appendChild(grain_position_y_text)
        grain_position_z = doc.createElement('Z')
        grain_position.appendChild(grain_position_z)
        grain_position_z_text = doc.createTextNode('%f' % self.center[2])
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
        grain.center = np.array([xg, yg, zg])
        grain_mesh = grain_node.childNodes[3]
        grain_mesh_file = grain_mesh.childNodes[0].nodeValue
        if verbose:
            print(grain_mesh_file)
        grain.load_vtk_repr(grain_mesh_file, verbose)
        return grain

    def vtk_file_name(self):
        return 'grain_%d.vtu' % self.id

    def save_vtk_repr(self, file_name=None):
        import vtk
        if not file_name:
            file_name = self.vtk_file_name()
        print('writting ' + file_name)
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(file_name)
        if vtk.vtkVersion().GetVTKMajorVersion() > 5:
            writer.SetInputData(self.vtkmesh)
        else:
            writer.SetInput(self.vtkmesh)
        writer.Write()

    def load_vtk_repr(self, file_name, verbose=False):
        import vtk
        if verbose:
            print('reading ' + file_name)
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

    @staticmethod
    def from_dct(label=1, data_dir='.'):
        """Create a `Grain` instance from a DCT grain file.

        :param int label: the grain id.
        :param str data_dir: the data root from where to fetch data files.
        :return: A new grain instance.
        """
        grain_path = os.path.join(data_dir, '4_grains', 'phase_01', 'grain_%04d.mat' % label)
        grain_info = h5py.File(grain_path)
        g = Grain(label, Orientation.from_rodrigues(grain_info['R_vector'].value))
        g.center = grain_info['center'].value
        # add spatial representation of the grain if reconstruction is available
        grain_map_path = os.path.join(data_dir, '5_reconstruction', 'phase_01_vol.mat')
        if os.path.exists(grain_map_path):
            with h5py.File(grain_map_path, 'r') as f:
                # because how matlab writes the data, we need to swap X and Z axes in the DCT volume
                vol = f['vol'].value.transpose(2, 1, 0)
                from scipy import ndimage
                grain_data = vol[ndimage.find_objects(vol == label)[0]]
                g.volume = ndimage.measurements.sum(vol == label)
                # create the vtk representation of the grain
            g.add_vtk_mesh(grain_data, contour=False)
        return g


class Microstructure:
    """
    Class used to manipulate a full microstructure.

    It is typically defined as a list of grains objects, has an associated crystal `Lattice` instance.
    A grain map and a mask can be added to the microstructure instance. For simplicity a simple field `voxel_size`
    describe the spatial resolution of teses maps.
    """

    def __init__(self, name='empty', lattice=None):
        self.name = name
        if lattice is None:
            lattice = Lattice.cubic(1.0)
        self._lattice = lattice
        self.grains = []
        self.grain_map = None
        self.mask = None
        self.voxel_size = 1.0  # unit is voxel by default
        self.vtkmesh = None

    def get_number_of_phases(self):
        """Return the number of phases in this microstructure.

        For the moment only one phase is supported, so this function simply returns 1."""
        return 1

    def get_number_of_grains(self):
        """Return the number of grains in this microstructure."""
        return len(self.grains)

    def set_lattice(self, lattice):
        """Set the crystallographic lattice associated with this microstructure.

        :param Lattice lattice: an instance of the `Lattice class`.
        """
        self._lattice = lattice

    def get_lattice(self):
        """Get the crystallographic lattice associated with this microstructure.

        :return: an instance of the `Lattice class`.
        """
        return self._lattice

    def set_grain_map(self, grain_map, voxel_size):
        """Set the grain map for this microstructure.

        :param ndarray grain_map: a 2D or 3D numpy array.
        :param float voxel_size: the size of the voxels in mm unit.
        """
        self.grain_map = grain_map
        self.voxel_size = voxel_size

    def set_mask(self, mask, voxel_size):
        """Set the mask for this microstructure.

        :param ndarray mask: a 2D or 3D numpy array.
        :param float voxel_size: the size of the voxels in mm unit.
        """
        self.mask = mask
        self.voxel_size = voxel_size

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
            ipf_colors[g.id, :] = g.orientation.get_ipf_colour()
        return colors.ListedColormap(ipf_colors)

    @staticmethod
    def from_xml(xml_file_name, grain_ids=None, verbose=False):
        """Load a Microstructure object from an xml file.

        It is possible to restrict the grains which are loaded by providing
        the list of ids of the grains of interest.
        """
        if verbose and grain_ids:
            print('loading only grain ids %s' % grain_ids)
        micro = Microstructure()
        dom = parse(xml_file_name)
        root = dom.childNodes[0]
        name = root.childNodes[0]
        micro.name = name.childNodes[0].nodeValue
        grains = root.childNodes[1]
        for node in grains.childNodes:
            if grain_ids and not (int(node.childNodes[0].childNodes[0].nodeValue) in grain_ids): continue
            if verbose:
                print(node)
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

    @staticmethod
    def match_grains(micro1, micro2, use_grain_ids=None, verbose=False):
        return micro1.match_grains(micro2, use_grain_ids=use_grain_ids, verbose=verbose)

    def match_grains(self, micro2, mis_tol=1, use_grain_ids=None, verbose=False):
        """Match grains from a second microstructure to this microstructure.

        This function try to find pair of grains based on their orientations.

        .. warning::

          This function works only for microstructures with the same symmetry.

        :param micro2: the second instance of `Microstructure` from which to match grains.
        :param float mis_tol: the tolerance is misorientation to use to detect matches (in degrees).
        :param bool use_grain_ids: a list of ids to restrict the grains in which to search for matches.
        :param bool verbose: activate verbose mode.
        :raise ValueError: if the microstructures do not have the same symmetry.
        :returns tuple: A tuple of three lists holding respectively the matches, the candidates for each match and
        the grains that were unmatched.
        """
        if not self.get_lattice().get_symmetry() == micro2.get_lattice().get_symmetry():
            raise ValueError('warning, microstructure should have the same symmetry, got: {} and {}'.
                             format(self.get_lattice().get_symmetry(), micro2.get_lattice().get_symmetry()))
        candidates = []
        matched = []
        unmatched = []  # grain that were not matched within the given tolerance
        # restrict the grain ids to match if needed
        if use_grain_ids:
            grains_to_match = [self.get_grain(gid) for gid in use_grain_ids]
        else:
            grains_to_match = self.grains
        # look at each grain
        for i, g1 in enumerate(grains_to_match):
            cands_for_g1 = []
            best_mis = mis_tol
            best_match = -1
            for g2 in micro2.grains:
                # compute disorientation
                mis, _, _ = g1.orientation.disorientation(g2.orientation, crystal_structure=self.get_lattice().get_symmetry())
                misd = np.degrees(mis)
                if misd < mis_tol:
                    if verbose:
                        print('grain %3d -- candidate: %3d, misorientation: %.2f deg' % (g1.id, g2.id, misd))
                    # add this grain to the list of candidates
                    cands_for_g1.append(g2.id)
                    if misd < best_mis:
                        best_mis = misd
                        best_match = g2.id
            # add our best match or mark this grain as unmatched
            if best_match > 0:
                matched.append([g1.id, best_match])
            else:
                unmatched.append(g1.id)
            candidates.append(cands_for_g1)
        if verbose:
            print('done with matching')
            print('%d/%d grains were matched ' % (len(matched), len(grains_to_match)))
        return matched, candidates, unmatched

    def dilate_grains(self, dilation_steps=1):
        """Dilate grains to fill the gap beween them.

        This code is based on the gtDilateGrains function from the DCT code. It has been extended to handle both 2D
        and 3D cases.

        :param int dilation_steps: the umber of dilation steps to apply.
        """
        if not hasattr(self, 'grain_map'):
            raise ValueError('microstructure %s must have an associated grain_map attribute' % self.name)
            return

        grain_map = self.grain_map.copy()
        # get rid of overlap regions flaged by -1
        grain_map[grain_map == -1] = 0

        # carry out dilation in iterative steps
        for step in range(dilation_steps):
            grains = (grain_map > 0).astype(np.uint8)
            from scipy import ndimage
            grains_dil = ndimage.morphology.binary_dilation(grain_map).astype(np.uint8)
            if hasattr(self, 'mask'):
                # only dilate within the mask
                grains_dil *= self.mask.astype(np.uint8)
            todo = (grains_dil - grains)
            # get the list of voxel for this dilation step
            X, Y, Z = np.where(todo)

            xstart = X - 1
            xend = X + 1
            ystart = Y - 1
            yend = Y + 1
            zstart = Z - 1
            zend = Z + 1

            # check bounds
            xstart[xstart < 0] = 0
            ystart[ystart < 0] = 0
            zstart[zstart < 0] = 0
            xend[xend > grain_map.shape[0] - 1] = grain_map.shape[0] - 1
            yend[yend > grain_map.shape[1] - 1] = grain_map.shape[1] - 1
            zend[zend > grain_map.shape[2] - 1] = grain_map.shape[2] - 1

            dilation = np.zeros_like(X).astype(np.int16)
            print('%d voxels to replace' % len(X))
            for i in range(len(X)):
                neighbours = grain_map[xstart[i]:xend[i] + 1, ystart[i]:yend[i] + 1, zstart[i]:zend[i] + 1]
                if np.any(neighbours):
                    # at least one neighboring voxel in non zero
                    dilation[i] = min(neighbours[neighbours > 0])
            grain_map[X, Y, Z] = dilation
            print('dilation step %d done' % (step + 1))
        # finally assign the dilated grain map to the microstructure
        self.grain_map = grain_map

    def compute_grain_center(self, gid):
        """Compute the center of masses of a grain given its id.

        :param int gid: the grain id to consider.
        :return: a tuple with the center of mass in mm units (or voxel if the voxel_size is not specified).
        """
        # isolate the grain within the complete grain map
        slices = ndimage.find_objects(self.grain_map == gid)
        if not len(slices) > 0:
            raise ValueError('warning grain %d not found in grain map' % gid)
        sl = slices[0]
        offset = np.array([sl[0].start, sl[1].start, sl[2].start])
        grain_data_bin = (self.grain_map[sl] == gid).astype(np.uint8)
        local_com = ndimage.measurements.center_of_mass(grain_data_bin)
        com = self.voxel_size * (offset + local_com - 0.5 * np.array(self.grain_map.shape))
        return com

    def recompute_grain_centers(self, verbose=False):
        """Compute and assign the center of all grains in the microstructure using the grain map.

        Each grain center is computed using its center of mass. The value is assigned to the grain.center attribute.
        If the voxel size is specified, the grain centers will be in mm unit, if not in voxel unit.

        .. note::

          A grain map need to be associated with this microstructure instance for the method to run.

        :param bool verbose: flag for verbose mode.
        """
        if not hasattr(self, 'grain_map'):
            print('warning: need a grain map to recompute the center of mass of the grains')
            return
        for g in self.grains:
            try:
                com = self.compute_grain_center(g.id)
            except ValueError:
                print('skipping grain %d' % g.id)
                continue
            if verbose:
                print('grain %d center: %.3f, %.3f, %.3f' % (g.id, com[0], com[1], com[2]))
            g.center = com


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

    def to_h5(self):
        """Write the microstructure as a hdf5 file."""
        import time
        from pymicro import __version__ as pymicro_version

        print('opening file %s.h5 for writing' % self.name)
        f = h5py.File('%s.h5' % self.name, 'w')
        f.attrs['Pymicro_Version'] = np.string_(pymicro_version)
        f.attrs['HDF5_Version'] = h5py.version.hdf5_version
        f.attrs['h5py_version'] = h5py.version.version
        f.attrs['file_time'] = time.time()
        f.attrs['microstructure_name'] = self.name
        if hasattr(self, 'data_dir'):
            f.attrs['data_dir'] = self.data_dir
        # ensemble data
        ed = f.create_group('EnsembleData')
        cs = ed.create_group('CrystalStructure')
        sym = self.get_lattice().get_symmetry()
        cs.attrs['symmetry'] = sym.to_string()
        lp = cs.create_dataset('LatticeParameters',
                               data=np.array(self.get_lattice().get_lattice_parameters(), dtype=np.float32))
        # feature data
        fd = f.create_group('FeatureData')
        grain_ids = fd.create_dataset('grain_ids',
                                      data=np.array([g.id for g in self.grains], dtype=np.int))
        avg_rods = fd.create_dataset('R_vectors',
                                     data=np.array([g.orientation.rod for g in self.grains], dtype=np.float32))
        centers = fd.create_dataset('centers',
                                    data=np.array([g.center for g in self.grains], dtype=np.float32))
        # cell data
        cd = f.create_group('CellData')
        if hasattr(self, 'grain_map') and self.grain_map is not None:
            gm = cd.create_dataset('grain_ids', data=self.grain_map, compression='gzip', compression_opts=9)
            gm.attrs['voxel_size'] = self.voxel_size
        if hasattr(self, 'mask') and self.mask is not None:
            ma = cd.create_dataset('mask', data=self.mask, compression='gzip', compression_opts=9)
            ma.attrs['voxel_size'] = self.voxel_size
        print('done writing')
        f.close()

    def from_h5(file_path):
        """read a microstructure object from a HDF5 file.

        :param str file_path: the path to the file to read.
        :return: the new `Microstructure` instance created from the file.
        """
        with h5py.File(file_path, 'r') as f:
            micro = Microstructure(name=f.attrs['microstructure_name'])
            if 'symmetry' in f['EnsembleData/CrystalStructure'].attrs:
                sym = f['EnsembleData/CrystalStructure'].attrs['symmetry']
                parameters = f['EnsembleData/CrystalStructure/LatticeParameters'][()]
                micro.set_lattice(Lattice.from_symmetry(Symmetry.from_string(sym), parameters))
            if 'data_dir' in f.attrs:
                micro.data_dir = f.attrs['data_dir']
            # load feature data
            if 'R_vectors' in f['FeatureData']:
                print('some grains')
                avg_rods = f['FeatureData/R_vectors'][()]
                print(avg_rods.shape)
                if 'grain_ids' in f['FeatureData']:
                    grain_ids = f['FeatureData/grain_ids'][()]
                else:
                    grain_ids = range(1, 1 + avg_rods.shape[0])
                if 'centers' in f['FeatureData']:
                    centers = f['FeatureData/centers'][()]
                else:
                    centers = np.zeros_like(avg_rods)
                for i in range(avg_rods.shape[0]):
                    g = Grain(grain_ids[i], Orientation.from_rodrigues(avg_rods[i, :]))
                    g.center = centers[i]
                    micro.grains.append(g)
            # load cell data
            if 'grain_ids' in f['CellData']:
                micro.grain_map = f['CellData/grain_ids'][()]
                if 'voxel_size' in f['CellData/grain_ids'].attrs:
                    micro.voxel_size = f['CellData/grain_ids'].attrs['voxel_size']
            if 'mask' in f['CellData']:
                micro.mask = f['CellData/mask'][()]
                if 'voxel_size' in f['CellData/mask'].attrs:
                    micro.voxel_size = f['CellData/mask'].attrs['voxel_size']
            return micro

    def to_dream3d(self):
        """Write the microstructure as a hdf5 file compatible with DREAM3D."""
        import time
        f = h5py.File('%s.h5' % self.name, 'w')
        f.attrs['FileVersion'] = np.string_('7.0')
        f.attrs['DREAM3D Version'] = np.string_('6.1.77.d28a796')
        f.attrs['HDF5_Version'] = h5py.version.hdf5_version
        f.attrs['h5py_version'] = h5py.version.version
        f.attrs['file_time'] = time.time()
        # pipeline group (empty here)
        pipeline = f.create_group('Pipeline')
        pipeline.attrs['Number_Filters'] = np.int32(0)
        # create the data container group
        data_containers = f.create_group('DataContainers')
        m = data_containers.create_group('DataContainer')
        # ensemble data
        ed = m.create_group('EnsembleData')
        ed.attrs['AttributeMatrixType'] = np.uint32(11)
        ed.attrs['TupleDimensions'] = np.uint64(2)
        cryst_structure = ed.create_dataset('CrystalStructures', data=np.array([[999], [1]], dtype=np.uint32))
        cryst_structure.attrs['ComponentDimensions'] = np.uint64(1)
        cryst_structure.attrs['DataArrayVersion'] = np.int32(2)
        cryst_structure.attrs['ObjectType'] = np.string_('DataArray<uint32_t>')
        cryst_structure.attrs['Tuple Axis Dimensions'] = np.string_('x=2')
        cryst_structure.attrs['TupleDimensions'] = np.uint64(2)
        mat_name = ed.create_dataset('MaterialName', data=[a.encode('utf8') for a in ['Invalid Phase', 'Unknown']])
        mat_name.attrs['ComponentDimensions'] = np.uint64(1)
        mat_name.attrs['DataArrayVersion'] = np.int32(2)
        mat_name.attrs['ObjectType'] = np.string_('StringDataArray')
        mat_name.attrs['Tuple Axis Dimensions'] = np.string_('x=2')
        mat_name.attrs['TupleDimensions'] = np.uint64(2)
        # feature data
        fd = m.create_group('FeatureData')
        fd.attrs['AttributeMatrixType'] = np.uint32(7)
        fd.attrs['TupleDimensions'] = np.uint64(len(self.grains))
        avg_euler = fd.create_dataset('AvgEulerAngles',
                                      data=np.array([g.orientation.euler for g in self.grains], dtype=np.float32))
        avg_euler.attrs['ComponentDimensions'] = np.uint64(3)
        avg_euler.attrs['DataArrayVersion'] = np.int32(2)
        avg_euler.attrs['ObjectType'] = np.string_('DataArray<float>')
        avg_euler.attrs['Tuple Axis Dimensions'] = np.string_('x=%d' % len(self.grains))
        avg_euler.attrs['TupleDimensions'] = np.uint64(len(self.grains))
        # geometry
        geom = m.create_group('_SIMPL_GEOMETRY')
        geom.attrs['GeometryType'] = np.uint32(999)
        geom.attrs['GeometryTypeName'] = np.string_('UnkownGeometry')
        # create the data container bundles group
        f.create_group('DataContainerBundles')
        f.close()

    @staticmethod
    def from_dream3d(file_path, main_key='DataContainers', data_container='DataContainer', grain_data='FeatureData',
                grain_orientations='AvgEulerAngles', orientation_type='euler', grain_centroid='Centroids'):
        """Read a microstructure from a hdf5 file.
        
        :param str file_path: the path to the hdf5 file to read.
        :param str main_key: the string describing the root key.
        :param str data_container: the string describing the data container group in the hdf5 file.
        :param str grain_data: the string describing the grain data group in the hdf5 file.
        :param str grain_orientations: the string describing the average grain orientations in the hdf5 file.
        :param str orientation_type: the string describing the descriptor used for orientation data.
        :param str grain_centroid: the string describing the grain centroid in the hdf5 file.
        :return: a `Microstructure` instance created from the hdf5 file.
        """
        micro = Microstructure()
        with h5py.File(file_path, 'r') as f:
            grain_data_path = '%s/%s/%s' % (main_key, data_container, grain_data)
            orientations = f[grain_data_path][grain_orientations].value
            if grain_centroid:
                centroids = f[grain_data_path][grain_centroid].value
                offset = 0
                if len(centroids) < len(orientations):
                    offset = 1  # if grain 0 has not a centroid
            for i in range(len(orientations)):
                if orientations[i, 0] == 0. and orientations[i, 1] == 0. and orientations[i, 2] == 0.:
                    # skip grain 0 which is always (0., 0., 0.)
                    print('skipping (0., 0., 0.)')
                    continue
                if orientation_type == 'euler':
                    g = Grain(i, Orientation.from_euler(orientations[i] * 180 / np.pi))
                elif orientation_type == 'rodrigues':
                    g = Grain(i, Orientation.from_rodrigues(orientations[i]))
                if grain_centroid:
                    g.center = centroids[i - offset]
                micro.grains.append(g)
        return micro

    @staticmethod
    def from_dct(data_dir='.', grain_file='index.mat', vol_file='phase_01_vol.mat', mask_file='volume_mask.mat',
                 use_dct_path=True, verbose=True):
        """Create a microstructure from a DCT reconstruction.

        DCT reconstructions are stored in several files. The indexed grain inforamtions are stored in a matlab file in
        the '4_grains/phase_01' folder. Then, the reconstructed volume file (labeled image) is stored
        in the '5_reconstruction' folder as an hdf5 file, possibly stored alongside a mask file coming from the
        absorption reconstruction.
        
        :param str data_dir: the path to the folder containing the reconstruction data.
        :param str grain_file: the name of the file containing grains info.
        :param str vol_file: the name of the volume file.
        :param str mask_file: the name of the mask file.
        :param bool use_dct_path: if True, the grain_file should be located in 4_grains/phase_01 folder and the
        vol_file and mask_file in the 5_reconstruction folder.
        :param bool verbose: activate verbose mode.
        :return: a `Microstructure` instance created from the DCT reconstruction.
        """
        if data_dir == '.':
            data_dir = os.getcwd()
        scan = data_dir.split(os.sep)[-1]
        print('creating microstructure for DCT scan %s' % scan)
        micro = Microstructure(name=scan)
        micro.data_dir = data_dir
        if use_dct_path:
            index_path = os.path.join(data_dir, '4_grains', 'phase_01', grain_file)
        else:
            index_path = os.path.join(data_dir, grain_file)
        print(index_path)
        if not os.path.exists(index_path):
            raise ValueError('%s not found, please specify a valid path to the grain file.' % index_path)
            return None
        from scipy.io import loadmat
        index = loadmat(index_path)
        micro.voxel_size = index['cryst'][0][0][25][0][0]
        # grab the crystal lattice
        lattice_params = index['cryst'][0][0][3][0]
        sym = Symmetry.from_string(index['cryst'][0][0][7][0])
        print('creating crystal lattice {} ({}) with parameters {}'.format(index['cryst'][0][0][0][0], sym, lattice_params))
        lattice_params[:3] /= 10  # angstrom to nm
        lattice = Lattice.from_parameters(*lattice_params, symmetry=sym)
        micro.set_lattice(lattice)
        # add all grains to the microstructure
        for i in range(len(index['grain'][0])):
            gid = index['grain'][0][i][0][0][0][0][0]
            rod = index['grain'][0][i][0][0][3][0]
            g = Grain(gid, Orientation.from_rodrigues(rod))
            g.center = index['grain'][0][i][0][0][15][0]
            micro.grains.append(g)

        # load the grain map if available
        if use_dct_path:
            grain_map_path = os.path.join(data_dir, '5_reconstruction', vol_file)
        else:
            grain_map_path = os.path.join(data_dir, vol_file)
        if os.path.exists(grain_map_path):
            with h5py.File(grain_map_path, 'r') as f:
                # because how matlab writes the data, we need to swap X and Z axes in the DCT volume
                micro.grain_map = f['vol'].value.transpose(2, 1, 0)
                if verbose:
                    print('loaded grain ids volume with shape: {}'.format(micro.grain_map.shape))
        # load the mask if available
        if use_dct_path:
            mask_path = os.path.join(data_dir, '5_reconstruction', mask_file)
        else:
            mask_path = os.path.join(data_dir, mask_file)
        if os.path.exists(mask_path):
            with h5py.File(mask_path, 'r') as f:
                micro.mask = f['vol'].value.transpose(2, 1, 0).astype(np.uint8)
                if verbose:
                    print('loaded mask volume with shape: {}'.format(micro.mask.shape))
        return micro

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
        print('writing ' + xml_file_name)
        f = open(xml_file_name, 'wb')
        doc.writexml(f, encoding='utf-8')
        f.close()
        # now save the vtk representation
        if self.vtkmesh != None:
            import vtk
            vtk_file_name = '%s.vtm' % self.name
            print('writing ' + vtk_file_name)
            writer = vtk.vtkXMLMultiBlockDataWriter()
            writer.SetFileName(vtk_file_name)
            if vtk.vtkVersion().GetVTKMajorVersion() > 5:
                writer.SetInputData(self.vtkmesh)
            else:
                writer.SetInput(self.vtkmesh)
            writer.Write()

    @staticmethod
    def merge_microstructures(micros, overlap, plot=False):
        """Merge two `Microstructure` instances together.

        The function works for two microstructures with grain maps and an overlap between them. Temporarily
        `Microstructures` restricted to the overlap regions are created and grains are matched between the two based
        on a disorientation tolerance.

        .. note::

          The two microstructure must have the same crystal lattice and the same voxel_size for this method to run.

        :param list micros: a list containing the two microstructures to merge.
        :param int overlap: the overlap to use.
        :param bool plot: a flag to plot some results.
        :return: a new `Microstructure`instance containing the merged microstructure.
        """
        from scipy import ndimage

        # perform some sanity checks
        for i in range(2):
            if not hasattr(micros[i], 'grain_map'):
                raise ValueError('microstructure instance %s must have an associated grain_map attribute' % micros[i].name)
        if micros[0].get_lattice() != micros[1].get_lattice():
            raise ValueError('both microstructure must have the same crystal lattice')
        lattice = micros[0].get_lattice()
        if micros[0].voxel_size != micros[1].voxel_size:
            raise ValueError('both microstructure must have the same voxel size')
        voxel_size = micros[0].voxel_size

        # create two microstructure of the overlapping regions: end slices in first scan and first slices in second scan
        grain_ids_ol1 = micros[0].grain_map[:, :, micros[0].grain_map.shape[2] - overlap:]
        grain_ids_ol2 = micros[1].grain_map[:, :, :overlap]
        dims_ol1 = np.array(grain_ids_ol1.shape)
        print(dims_ol1)
        dims_ol2 = np.array(grain_ids_ol2.shape)
        print(dims_ol2)

        # build a microstructure for the overlap region in each volumes
        grain_ids_ols = [grain_ids_ol1, grain_ids_ol2]
        micros_ol = []
        for i in range(2):
            grain_ids_ol = grain_ids_ols[i]
            ids_ol = np.unique(grain_ids_ol)
            print(ids_ol)

            # difference due to the crop (restricting the grain map to the overlap region)
            #offset_mm =  (2 * i - 1) * voxel_size * np.array([0., 0., grain_ids_ol.shape[2] - 0.5 * micros[i].grain_map.shape[2]])
            # here we use an ad-hoc offset to voxel (0, 0, 0) in the full volume: offset is zero for the second volume
            offset_px = (i - 1) * np.array([0., 0., grain_ids_ol.shape[2] - micros[i].grain_map.shape[2]])
            offset_mm = voxel_size * offset_px
            print('offset [px] is {}'.format(offset_px))
            print('offset [mm] is {}'.format(offset_mm))

            # make the microstructure
            micro_ol = Microstructure(name='%sol_' % micros[i].name)
            print('* building overlap microstructure %s' % micro_ol.name)
            micro_ol.set_lattice(lattice)
            micro_ol.grain_map = grain_ids_ol
            for gid in ids_ol:
                if gid < 1:
                    print('skipping %d' % gid)
                    continue
                g = Grain(gid, micros[i].get_grain(gid).orientation)

                array_bin = (grain_ids_ol == gid).astype(np.uint8)
                local_com = ndimage.measurements.center_of_mass(array_bin, grain_ids_ol)
                #print('local_com = {}'.format(local_com))
                com_px = (local_com + offset_px - 0.5 * np.array(micros[i].grain_map.shape))
                #print('com [px] = {}'.format(com_px))
                com_mm = voxel_size * com_px
                print('grain %2d center: %6.3f, %6.3f, %6.3f' % (gid, com_mm[0], com_mm[1], com_mm[2]))

                #array_bin = (grain_ids_ol == gid).astype(np.uint8)
                #local_com = ndimage.measurements.center_of_mass(array_bin, grain_ids_ol)
                #com_mm = voxel_size * (local_com - 0.5 * np.array(grain_ids_ol.shape)) + offset
                #print('grain %2d position: %6.3f, %6.3f, %6.3f' % (gid, com_mm[0], com_mm[1], com_mm[2]))
                g.center = com_mm

                micro_ol.grains.append(g)
            #TODO recalculate position as we look at a truncated volume
            '''
            micro_ol.recompute_grain_centers(verbose=True)
            for g in micro_ol.grains:
                g.center += offset_mm
            '''
            # add the overlap microstructure to the list
            micros_ol.append(micro_ol)

        # match grain from micros_ol[1] to micros_ol[0] (the reference)
        matched, _, unmatched = micros_ol[0].match_grains(micros_ol[1], verbose=True)

        from pymicro.view.vol_utils import compute_affine_transform

        # compute the affine transform
        n_points = len(matched)
        fixed = np.zeros((n_points, 3))
        moving = np.zeros((n_points, 3))
        moved = np.zeros_like(moving)

        # markers in ref grain map
        for i in range(n_points):
            fixed[i] = micros_ol[0].get_grain(matched[i][0]).center
            moving[i] = micros_ol[1].get_grain(matched[i][1]).center

        # call the registration method
        translation, transformation = compute_affine_transform(fixed, moving)
        invt = np.linalg.inv(transformation)

        # check what are now the points after transformation
        fixed_centroid = np.average(fixed, axis=0)
        moving_centroid = np.average(moving, axis=0)
        print('fixed centroid: {}'.format(fixed_centroid))
        print('moving centroid: {}'.format(moving_centroid))

        for j in range(n_points):
            moved[j] = fixed_centroid + np.dot(transformation, moving[j] - moving_centroid)
            print('point %d will move to (%6.3f, %6.3f, %6.3f) to be compared with (%6.3f, %6.3f, %6.3f)' % (
                j, moved[j, 0], moved[j, 1], moved[j, 2], fixed[j, 0], fixed[j, 1], fixed[j, 2]))
        print('transformation is:')
        print(invt)

        # offset and translation, here we only look for rigid body translation
        offset = -np.dot(invt, translation)
        print(translation, offset)
        translation_voxel = (translation / voxel_size).astype(int)
        print(translation_voxel)

        # look at ids in the reference volume
        ids_ref = np.unique(micros[0].grain_map)
        ids_ref_list = ids_ref.tolist()
        if -1 in ids_ref_list:
            ids_ref_list.remove(-1)  # grain overlap
        if 0 in ids_ref_list:
            ids_ref_list.remove(0)  # background
        print(ids_ref_list)
        id_offset = max(ids_ref_list)
        print('grain ids in volume %s will be offset by %d' % (micros[1].name, id_offset))

        # gather ids in the merging volume (will be modified)
        ids_mrg = np.unique(micros[1].grain_map)
        ids_mrg_list = ids_mrg.tolist()
        if -1 in ids_mrg_list:
            ids_mrg_list.remove(-1)  # grain overlap
        if 0 in ids_mrg_list:
            ids_mrg_list.remove(0)  # background
        print(ids_mrg_list)

        # prepare a volume with the same size as the second grain map, with grain ids renumbered and (X, Y) translations applied.
        grain_map_translated = micros[1].grain_map.copy()
        print('renumbering grains in the overlap region of volume %s' % micros[1].name)
        for match in matched:
            ref_id, other_id = match
            print('replacing %d by %d' % (other_id, ref_id))
            ids_mrg_list.remove(other_id)
            grain_map_translated[micros[1].grain_map == other_id] = ref_id
            #TODO should flag those grains so their center can be recomputed
        # also renumber the rest using the offset
        renumbered_grains = []
        for i, other_id in enumerate(ids_mrg_list):
            new_id = id_offset + i + 1
            grain_map_translated[micros[1].grain_map == other_id] = new_id
            print('replacing %d by %d' % (other_id, new_id))
            renumbered_grains.append([other_id, new_id])

        # apply translation along the (X, Y) axes
        grain_map_translated = np.roll(grain_map_translated, translation_voxel[:2], (0, 1))

        check = overlap // 2
        print(grain_map_translated.shape)
        print(overlap)
        print(translation_voxel[2] + check)
        if plot:
            fig = plt.figure(figsize=(15, 7))
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(micros[0].grain_map[:, :, translation_voxel[2] + check].T, vmin=0)
            plt.axis('off')
            plt.title('micros[0].grain_map (ref)')
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(grain_map_translated[:, :, check].T, vmin=0)
            plt.axis('off')
            plt.title('micros[1].grain_map (renumbered)')
            ax3 = fig.add_subplot(1, 3, 3)
            same = micros[0].grain_map[:, :, translation_voxel[2] + check] == grain_map_translated[:, :, check]
            ax3.imshow(same.T, vmin=0, vmax=2)
            plt.axis('off')
            plt.title('voxels that are identicals')
            plt.savefig('merging_check1.pdf')

        # start the merging: the first volume is the reference
        overlap = micros[0].grain_map.shape[2] - translation_voxel[2]
        print('overlap is %d voxels' % overlap)
        z_shape = micros[0].grain_map.shape[2] + micros[1].grain_map.shape[2] - overlap
        print('vertical size will be: %d + %d + %d = %d' % (
            micros[0].grain_map.shape[2] - overlap, overlap, micros[1].grain_map.shape[2] - overlap, z_shape))
        shape_merged = np.array(micros[0].grain_map.shape) + [0, 0, micros[1].grain_map.shape[2] - overlap]
        print('initializing volume with shape {}'.format(shape_merged))
        grain_ids_merged = np.zeros(shape_merged, dtype=np.int16)
        print(micros[0].grain_map.shape)
        print(micros[1].grain_map.shape)

        # add the non-overlapping part of the 2 volumes as is
        grain_ids_merged[:, :, :micros[0].grain_map.shape[2] - overlap] = micros[0].grain_map[:, :, :-overlap]
        grain_ids_merged[:, :, micros[0].grain_map.shape[2]:] = grain_map_translated[:, :, overlap:]

        # look at vertices with the same label
        print(micros[0].grain_map[:, :, translation_voxel[2]:].shape)
        print(grain_map_translated[:, :, :overlap].shape)
        print('translation_voxel[2] = %d' % translation_voxel[2])
        print('micros[0].grain_map.shape[2] - overlap = %d' % (micros[0].grain_map.shape[2] - overlap))
        same_voxel = micros[0].grain_map[:, :, translation_voxel[2]:] == grain_map_translated[:, :, :overlap]
        print(same_voxel.shape)
        grain_ids_merged[:, :, translation_voxel[2]:micros[0].grain_map.shape[2]] = grain_map_translated[:, :, :overlap] * same_voxel

        # look at vertices with a single label
        single_voxels_0 = (micros[0].grain_map[:, :, translation_voxel[2]:] > 0) & (grain_map_translated[:, :, :overlap] == 0)
        print(single_voxels_0.shape)
        grain_ids_merged[:, :, translation_voxel[2]:micros[0].grain_map.shape[2]] += micros[0].grain_map[:, :, translation_voxel[2]:] * single_voxels_0
        single_voxels_1 = (grain_map_translated[:, :, :overlap] > 0) & (micros[0].grain_map[:, :, translation_voxel[2]:] == 0)
        print(single_voxels_1.shape)
        grain_ids_merged[:, :, translation_voxel[2]:micros[0].grain_map.shape[2]] += grain_map_translated[:, :,
                                                                                     :overlap] * single_voxels_1

        if plot:
            fig = plt.figure(figsize=(14, 10))
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.imshow(grain_ids_merged[:, 320, :].T)
            plt.axis('off')
            plt.title('XZ slice')
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.imshow(grain_ids_merged[320, :, :].T)
            plt.axis('off')
            plt.title('YZ slice')
            plt.savefig('merging_check2.pdf')

        if hasattr(micros[0], 'mask') and hasattr(micros[1], 'mask'):
            mask_translated = np.roll(micros[1].mask, translation_voxel[:2], (0, 1))

            # merging the masks
            mask_merged = np.zeros(shape_merged, dtype=np.uint8)
            # add the non-overlapping part of the 2 volumes as is
            mask_merged[:, :, :micros[0].mask.shape[2] - overlap] = micros[0].mask[:, :, :-overlap]
            mask_merged[:, :, micros[0].grain_map.shape[2]:] = mask_translated[:, :, overlap:]

            # look at vertices with the same label
            same_voxel = micros[0].mask[:, :, translation_voxel[2]:] == mask_translated[:, :, :overlap]
            print(same_voxel.shape)
            mask_merged[:, :, translation_voxel[2]:micros[0].mask.shape[2]] = mask_translated[:, :, :overlap] * same_voxel

            # look at vertices with a single label
            single_voxels_0 = (micros[0].mask[:, :, translation_voxel[2]:] > 0) & (mask_translated[:, :, :overlap] == 0)
            mask_merged[:, :, translation_voxel[2]:micros[0].mask.shape[2]] += (
                        micros[0].mask[:, :, translation_voxel[2]:] * single_voxels_0).astype(np.uint8)
            single_voxels_1 = (mask_translated[:, :, :overlap] > 0) & (micros[0].mask[:, :, translation_voxel[2]:] == 0)
            mask_merged[:, :, translation_voxel[2]:micros[0].mask.shape[2]] += (
                        mask_translated[:, :, :overlap] * single_voxels_1).astype(np.uint8)

            if plot:
                fig = plt.figure(figsize=(14, 10))
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.imshow(mask_merged[:, 320, :].T)
                plt.axis('off')
                plt.title('XZ slice')
                ax2 = fig.add_subplot(1, 2, 2)
                ax2.imshow(mask_merged[320, :, :].T)
                plt.axis('off')
                plt.title('YZ slice')
                plt.savefig('merging_check3.pdf')

        # merging finished, build the new microstructure instance
        merged_micro = Microstructure(name='%s-%s' % (micros[0].name, micros[1].name))
        merged_micro.set_lattice(lattice)
        # add all grains from the reference volume
        merged_micro.grains = micros[0].grains
        #TODO recompute center of masses of grains in the overlap region
        print(renumbered_grains)
        # add all new grains from the merged volume
        for i in range(len(renumbered_grains)):
            other_id, new_id = renumbered_grains[i]
            g = micros[1].get_grain(other_id)
            new_g = Grain(new_id, Orientation.from_rodrigues(g.orientation.rod))
            new_g.center = g.center
            print('adding grain with new id %d (was %d)' % (new_id, other_id))
            merged_micro.grains.append(new_g)
        print('%d grains in merged microstructure' % merged_micro.get_number_of_grains())
        # add the full grain map
        merged_micro.grain_map = grain_ids_merged
        if hasattr(micros[0], 'mask') and hasattr(micros[1], 'mask'):
            merged_micro.mask = mask_merged
        return merged_micro
