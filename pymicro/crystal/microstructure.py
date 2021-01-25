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
import math
from scipy import ndimage
from matplotlib import pyplot as plt, colors
from pymicro.crystal.lattice import Lattice, Symmetry
from pymicro.crystal.quaternion import Quaternion
from pymicro.core.samples import SampleData
from pymicro.core.images import ImageObject
import tables
from math import atan2, pi


class Orientation:
    """Crystallographic orientation class.

    This follows the passive rotation definition which means that it brings
    the sample coordinate system into coincidence with the crystal coordinate
    system. Then one may express a vector :math:`V_c` in the crystal coordinate
    system from the vector in the sample coordinate system :math:`V_s` by:

    .. math::

      V_c = g.V_s

    and inversely (because :math:`g^{-1}=g^T`):

    .. math::

      V_s = g^T.V_c

    Most of the code to handle rotations has been written to comply with the
    conventions laid in :cite:`Rowenhorst2015`.
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

    def to_crystal(self, v):
        """Transform a vector or a matrix from the sample frame to the crystal
        frame.

        :param ndarray v: a 3 component vector or a 3x3 array expressed in
        the sample frame.
        :return: the vector or matrix expressed in the crystal frame.
        """
        if v.size not in [3, 9]:
            raise ValueError('input arg must be a 3 components vector '
                             'or a 3x3 matrix, got %d vlaues' % v.size)
        g = self.orientation_matrix()
        if v.size == 3:
            # input is vector
            return np.dot(g, v)
        else:
            # input is 3x3 matrix
            return np.dot(g, np.got(v, g.T))

    def to_sample(self, v):
        """Transform a vector or a matrix from the crystal frame to the sample
        frame.

        :param ndarray v: a 3 component vector or a 3x3 array expressed in
        the crystal frame.
        :return: the vector or matrix expressed in the sample frame.
        """
        if v.size not in [3, 9]:
            raise ValueError('input arg must be a 3 components vector '
                             'or a 3x3 matrix, got %d vlaues' % v.size)
        g = self.orientation_matrix()
        if v.size == 3:
            # input is vector
            return np.dot(g.T, v)
        else:
            # input is 3x3 matrix
            return np.dot(g.T, np.got(v, g))

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

    @staticmethod
    def random():
        """Create  a random crystal orientation."""
        from random import random
        from math import acos
        phi1 = random() * 360.
        Phi = 180. * acos(2 * random() - 1) / np.pi
        phi2 = random() * 360.
        return Orientation.from_euler([phi1, Phi, phi2])

    def get_ipf_colour(self, axis=np.array([0., 0., 1.]), symmetry=Symmetry.cubic):
        """Compute the IPF (inverse pole figure) colour for this orientation.

        Given a particular axis expressed in the laboratory coordinate system,
        one can compute the so called IPF colour based on that direction
        expressed in the crystal coordinate system as :math:`[x_c,y_c,z_c]`.
        There is only one tuple (u,v,w) such that:

        .. math::

          [x_c,y_c,z_c]=u.[0,0,1]+v.[0,1,1]+w.[1,1,1]

        and it is used to assign the RGB colour.

        :param ndarray axis: the direction to use to compute the IPF colour.
        :param Symmetry symmetry: the symmetry operator to use.
        :return tuple: a tuple contining the RGB values.
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
        y, x = sorted([abs(rod[0]), abs(rod[1])])
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
        Compute the equivalent crystal orientation in the Fundamental Zone of
        a given symmetry.

        :param Symmetry symmetry: an instance of the `Symmetry` class.
        :param verbose: flag for verbose mode.
        :return: a new Orientation instance which lies in the fundamental zone.
        """
        om = symmetry.move_rotation_to_FZ(self.orientation_matrix(), verbose=verbose)
        return Orientation(om)

    @staticmethod
    def misorientation_MacKenzie(psi):
        """Return the fraction of the misorientations corresponding to the
        given :math:`\\psi` angle in the reference solution derived By MacKenzie
        in his 1958 paper :cite:`MacKenzie_1958`.

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
                - 8. / (5 * pi) * (2 * (sqrt(2) - 1) * acos(X / tan(0.5 * psi))
                                   + 1. / sqrt(3) * acos(Y / tan(0.5 * psi))) \
                * sin(psi) + 8. / (5 * pi) * (2 *acos((sqrt(2) + 1) * X / sqrt(2))
                                              + acos((sqrt(2) + 1) * Y / sqrt(2))) * (1 - cos(psi))
        else:
            p = 0.
        return p

    @staticmethod
    def misorientation_axis_from_delta(delta):
        """Compute the misorientation axis from the misorientation matrix.

        :param delta: The 3x3 misorientation matrix.
        :returns: the misorientation axis (normalised vector).
        """
        n = np.array([delta[1, 2] - delta[2, 1], delta[2, 0] -
                      delta[0, 2], delta[0, 1] - delta[1, 0]])
        n /= np.sqrt((delta[1, 2] - delta[2, 1]) ** 2 +
                     (delta[2, 0] - delta[0, 2]) ** 2 +
                     (delta[0, 1] - delta[1, 0]) ** 2)
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

        Compute the angle associated with this misorientation matrix :math:`\\Delta g`.
        It is defined as :math:`\\omega = \\arccos(\\text{trace}(\\Delta g)/2-1)`.
        To avoid float rounding error, the argument is rounded to 1.0 if it is
        within 1 and 1 plus 32 bits floating point precison.

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
            # print('cw=%.20f, rounding to 1.' % cw)
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

         Both orientations are supposed to have the same symmetry. This is not
         necessarily the case in multi-phase materials.

        :param orientation: an instance of :py:class:`~pymicro.crystal.microstructure.Orientation` class describing
        the other crystal orientation from which to compute the angle.
        :param crystal_structure: an instance of the `Symmetry` class describing
        the crystal symmetry, triclinic (no symmetry) by default.
        :returns tuple: the misorientation angle in radians, the axis as a numpy
        vector (crystal coordinates), the axis as a numpy vector (sample coordinates).
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
                    mis_angle = Orientation.misorientation_angle_from_delta(delta)
                    if mis_angle < the_angle:
                        # now compute the misorientation axis, should check if it lies in the fundamental zone
                        mis_axis = Orientation.misorientation_axis_from_delta(delta)
                        # here we have np.dot(oi.T, mis_axis) = np.dot(oj.T, mis_axis)
                        # print(mis_axis, mis_angle*180/np.pi, np.dot(oj.T, mis_axis))
                        the_angle = mis_angle
                        the_axis = mis_axis
                        the_axis_xyz = np.dot(oi.T, the_axis)
        return the_angle, the_axis, the_axis_xyz

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
        R = np.array([[np.cos(omegar), -np.sin(omegar), 0],
                      [np.sin(omegar), np.cos(omegar), 0],
                      [0, 0, 1]])
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
            print('the two omega values in degrees fulfilling the Bragg condition are (%.1f, %.1f)' % (
            omega_1, omega_2))
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
            R = np.array([[np.cos(omegar), -np.sin(omegar), 0],
                          [np.sin(omegar), np.cos(omegar), 0],
                          [0, 0, 1]])
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
        """Compute the instrument transformation matrix for given rotation offset.

        This function compute a 3x3 rotation matrix (passive convention) that
        transforms the sample coordinate system by rotating around the 3
        cartesian axes in this order: rotation around X is applied first,
        then around Y and finally around Z.

        A sample vector :math:`V_s` is consequently transformed into :math:`V'_s` as:

        .. math::

          V'_s = T^T.V_s

        :param double rx_offset: value to apply for the rotation around X.
        :param double ry_offset: value to apply for the rotation around Y.
        :param double rz_offset: value to apply for the rotation around Z.
        :return: a 3x3 rotation matrix describing the transformation applied
        by the diffractometer.
        """
        angle_zr = np.radians(rz_offset)
        angle_yr = np.radians(ry_offset)
        angle_xr = np.radians(rx_offset)
        Rz = np.array([[np.cos(angle_zr), -np.sin(angle_zr), 0],
                       [np.sin(angle_zr), np.cos(angle_zr), 0],
                       [0, 0, 1]])
        Ry = np.array([[np.cos(angle_yr), 0, np.sin(angle_yr)],
                       [0, 1, 0],
                       [-np.sin(angle_yr), 0, np.cos(angle_yr)]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angle_xr), -np.sin(angle_xr)],
                       [0, np.sin(angle_xr), np.cos(angle_xr)]])
        T = Rz.dot(np.dot(Ry, Rx))
        return T

    def topotomo_tilts(self, hkl, T=None, verbose=False):
        """Compute the tilts for topotomography alignment.

        :param hkl: the hkl plane, an instance of :py:class:`~pymicro.crystal.lattice.HklPlane`
        :param ndarray T: transformation matrix representing the diffractometer
        direction at omega=0.
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

    @staticmethod
    def from_euler(euler, convention='Bunge'):
        """Rotation matrix from Euler angles.

        This is the classical method to obtain an orientation matrix by 3
        successive rotations. The result depends on the convention used
        (how the successive rotation axes are chosen). In the Bunge convention,
        the first rotation is around Z, the second around the new X and the
        third one around the new Z. In the Roe convention, the second one
        is around Y.
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
        """Compute the orientation matrix from the rotated coordinates given
        in the .inp file for Zebulon's computations.

        The function needs two of the three base vectors, the third one is
        computed using a cross product.

        .. note::

            Still need some tests to validate this function.

        :param x1: the first basis vector.
        :param x2: the second basis vector.
        :param x3: the third basis vector.
        :return: the corresponding 3x3 orientation matrix.
        """

        if x1 is None and x2 is None:
            raise NameError('Need at least two vectors to compute the matrix')
        elif x1 is None and x3 is None:
            raise NameError('Need at least two vectors to compute the matrix')
        elif x3 is None and x2 is None:
            raise NameError('Need at least two vectors to compute the matrix')

        if x1 is None:
            x1 = np.cross(x2, x3)
        elif x2 is None:
            x2 = np.cross(x3, x1)
        elif x3 is None:
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
            print('warning, returning [0., 0., 0.], consider using axis, angle '
                  'representation instead')
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
            # the rodrigues vector is zero, return the identity matrix
            return I
        theta = 2 * np.arctan(r)
        n = rod / r
        omega = np.array([[0.0, n[2], -n[1]],
                          [-n[2], 0.0, n[0]],
                          [n[1], -n[0], 0.0]])
        g = I + np.sin(theta) * omega + (1 - np.cos(theta)) * omega.dot(omega)
        return g

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
        g = np.array([[c + (1 - c) * axis[0] ** 2,
                       (1 - c) * axis[0] * axis[1] + s * axis[2],
                       (1 - c) * axis[0] * axis[2] - s * axis[1]],
                      [(1 - c) * axis[0] * axis[1] - s * axis[2],
                       c + (1 - c) * axis[1] ** 2,
                       (1 - c) * axis[1] * axis[2] + s * axis[0]],
                      [(1 - c) * axis[0] * axis[2] + s * axis[1],
                       (1 - c) * axis[1] * axis[2] - s * axis[0],
                       c + (1 - c) * axis[2] ** 2]])
        return g

    @staticmethod
    def Euler2Axis(euler):
        """Compute the (axis, angle) representation associated to this (passive)
        rotation expressed by the Euler angles.

        :param euler: 3 euler angles (in degrees).
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
        """Compute the quaternion from the 3 euler angles (in degrees).

        :param tuple euler: the 3 euler angles in degrees.
        :param int P: +1 to compute an active quaternion (default), -1 for a passive quaternion.
        :return: a `Quaternion` instance representing the rotation.
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
        """Compute the rodrigues vector from the 3 euler angles (in degrees).

        :param euler: the 3 Euler angles (in degrees).
        :return: the roodrigues vector as a 3 components numpy array.
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
        """Compute the orientation matrix :math:`\mathbf{g}` associated with
        the 3 Euler angles :math:`(\phi_1, \Phi, \phi_2)`.

        The matrix is calculated via (see the `euler_angles` recipe in the
        cookbook for a detailed example):

        .. math::

           \mathbf{g}=\\begin{pmatrix}
           \cos\phi_1\cos\phi_2 - \sin\phi_1\sin\phi_2\cos\Phi &
           \sin\phi_1\cos\phi_2 + \cos\phi_1\sin\phi_2\cos\Phi &
           \sin\phi_2\sin\Phi \\\\
           -\cos\phi_1\sin\phi_2 - \sin\phi_1\cos\phi_2\cos\Phi &
           -\sin\phi_1\sin\phi_2 + \cos\phi_1\cos\phi_2\cos\Phi &
           \cos\phi_2\sin\Phi \\\\
           \sin\phi_1\sin\Phi & -\cos\phi_1\sin\Phi & \cos\Phi \\\\
           \end{pmatrix}

        :param euler: The triplet of the Euler angles (in degrees).
        :return g: The 3x3 orientation matrix.
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
            phi_1 = atan2((q1 * q3 - P * q0 * q2) / chi,
                          (-P * q0 * q1 - q2 * q3) / chi)
            Phi = atan2(2 * chi, q03 - q12)
            phi_2 = atan2((P * q0 * q2 + q1 * q3) / chi,
                          (q2 * q3 - P * q0 * q1) / chi)
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

        This method is deprecated, please use `read_orientations`.

        :param str txt_path: path to the text file containing the euler angles.
        :returns dict: a dictionary with the line number and the corresponding
        orientation.
        """
        return Orientation.read_orientations(txt_path)

    @staticmethod
    def read_orientations(txt_path, data_type='euler', **kwargs):
        """
        Read a set of grain orientations from a text file.

        The text file must be organised in 3 columns (the other are ignored),
        corresponding to either the three euler angles or the three rodrigues
        vector components, depending on the data_type). Internally the ascii
        file is read by the genfromtxt function of numpy, to which additional
        keyworks (such as the delimiter) can be passed to via the kwargs
        dictionnary.

        :param str txt_path: path to the text file containing the orientations.
        :param str data_type: 'euler' (default) or 'rodrigues'.
        :param dict kwargs: additional parameters passed to genfromtxt.
        :returns dict: a dictionary with the line number and the corresponding
        orientation.
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

         **elset elset1 *file au.mat *integration theta_method_a 1.0 1.e-9 150
          *rotation x1 0.438886 -1.028805 0.197933 x3 1.038339 0.893172 1.003888
         **elset elset2 *file au.mat *integration theta_method_a 1.0 1.e-9 150
          *rotation x1 0.178825 -0.716937 1.043300 x3 0.954345 0.879145 1.153101
         **elset elset3 *file au.mat *integration theta_method_a 1.0 1.e-9 150
          *rotation x1 -0.540479 -0.827319 1.534062 x3 1.261700 1.284318 1.004174
         **elset elset4 *file au.mat *integration theta_method_a 1.0 1.e-9 150
          *rotation x1 -0.941278 0.700996 0.034552 x3 1.000816 1.006824 0.885212
         **elset elset5 *file au.mat *integration theta_method_a 1.0 1.e-9 150
          *rotation x1 -2.383786 0.479058 -0.488336 x3 0.899545 0.806075 0.984268

        :param str inp_path: the path to the ascii file to read.
        :returns dict: a dictionary of the orientations associated with the
        elset names.
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
            if not line.lstrip().startswith('%') and line.find('**elset') >= 0:
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
                euler.append([elset,
                              Orientation.Zrot2OrientationMatrix(x1=x1, x3=x3)])
            else:  # euler angles
                phi1 = tokens[irot + 1]
                Phi = tokens[irot + 2]
                phi2 = tokens[irot + 3]
                angles = np.array([float(phi1), float(Phi), float(phi2)])
                euler.append([elset, Orientation.from_euler(angles)])
        return dict(euler)

    def slip_system_orientation_tensor(self, s):
        """Compute the orientation strain tensor m^s for this
        :py:class:`~pymicro.crystal.microstructure.Orientation` and the given
        slip system.

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
        """Compute the orientation strain tensor m^s for this
        :py:class:`~pymicro.crystal.microstructure.Orientation` and the given
        slip system.

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
        """Compute the orientation rotation tensor q^s for this
        :py:class:`~pymicro.crystal.microstructure.Orientation and the given
        slip system.

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

        :param slip_system: a `SlipSystem` instance.
        :param load_direction: a unit vector describing the loading direction
        (default: vertical axis [0, 0, 1]).
        :return float: a number between 0 ad 0.5.
        """
        plane = slip_system.get_slip_plane()
        gt = self.orientation_matrix().transpose()
        n_rot = np.dot(gt, plane.normal())  # plane.normal() is a unit vector
        slip = slip_system.get_slip_direction().direction()
        slip_rot = np.dot(gt, slip)
        schmid_factor = np.abs(np.dot(n_rot, load_direction) *
                               np.dot(slip_rot, load_direction))
        return schmid_factor

    def compute_all_schmid_factors(self, slip_systems,
                                   load_direction=[0., 0., 1], verbose=False):
        """Compute all Schmid factors for this crystal orientation and the
        given list of slip systems.

        :param slip_systems: a list of the slip systems from which to compute
        the Schmid factor values.
        :param load_direction: a unit vector describing the loading direction
        (default: vertical axis [0, 0, 1]).
        :param bool verbose: activate verbose mode.
        :return list: a list of the schmid factors.
        """
        schmid_factor_list = []
        for ss in slip_systems:
            sf = self.schmid_factor(ss, load_direction)
            if verbose:
                print('Slip system: %s, Schmid factor is %.3f' % (ss, sf))
            schmid_factor_list.append(sf)
        return schmid_factor_list


class Grain:
    """
    Class defining a crystallographic grain.

    A grain has a constant crystallographic `Orientation` and a grain id. The
    center attribute is the center of mass of the grain in world coordinates.
    The volume of the grain is expressed in pixel/voxel unit.
    """

    def __init__(self, grain_id, grain_orientation):
        self.id = grain_id
        self.orientation = grain_orientation
        self.center = np.array([0., 0., 0.])
        self.volume = 0
        self.vtkmesh = None
        self.hkl_planes = []

    def __repr__(self):
        """Provide a string representation of the class."""
        s = '%s\n * id = %d\n' % (self.__class__.__name__, self.id)
        s += ' * %s\n' % (self.orientation)
        s += ' * center %s\n' % np.array_str(self.center)
        s += ' * has vtk mesh ? %s\n' % (self.vtkmesh != None)
        return s

    def get_volume(self):
        return self.volume

    def get_volume_fraction(self, total_volume=None):
        """Compute the grain volume fraction.

        :param float total_volume: the total volume value to use.
        :return float: the grain volume fraction as a number in the range [0, 1].
        """
        if not total_volume:
            return 1.
        else:
            return self.volume / total_volume

    def schmid_factor(self, slip_system, load_direction=[0., 0., 1]):
        """Compute the Schmid factor of this grain for the given slip system
        and loading direction.

        :param slip_system: a `SlipSystem` instance.
        :param load_direction: a unit vector describing the loading direction
        (default: vertical axis [0, 0, 1]).
        :return float: a number between 0 ad 0.5.
        """
        return self.orientation.schmid_factor(slip_system, load_direction)

    def SetVtkMesh(self, mesh):
        """Set the VTK mesh of this grain.

        :param mesh: the grain mesh in VTK format.
        """
        self.vtkmesh = mesh

    def add_vtk_mesh(self, array, contour=True, verbose=False):
        """Add a mesh to this grain.

        This method process a labeled array to extract the geometry of the
        grain. The grain shape is defined by the pixels with a value of the
        grain id. A vtkUniformGrid object is created and thresholded or
        contoured depending on the value of the flag `contour`. The resulting
        mesh is returned, centered on the center of mass of the grain.

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
            grid.SetExtent(0, grain_size[0] - 1, 0,
                           grain_size[1] - 1, 0, grain_size[2] - 1)
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
        """A method to access the grain orientation matrix.

        :return: the grain 3x3 orientation matrix.
        """
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
        :return tuple: (w1, w2) the two values of the omega angle.
        """
        return self.orientation.dct_omega_angles(hkl, lambda_keV, verbose)

    @staticmethod
    def from_dct(label=1, data_dir='.'):
        """Create a `Grain` instance from a DCT grain file.

        :param int label: the grain id.
        :param str data_dir: the data root from where to fetch data files.
        :return: a new grain instance.
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


class GrainData(tables.IsDescription):
    """
       Description class specifying structured storage of grain data in
       Microstructure Class, in HDF5 node /GrainData/GrainDataTable
    """
    # grain identity number
    idnumber = tables.Int32Col()  # Signed 64-bit integer
    # grain volume
    volume = tables.Float32Col()  # float
    # grain center of mass coordinates
    center = tables.Float32Col(shape=(3,))  # float  (double-precision)
    # Rodriguez vector defining grain orientation
    orientation = tables.Float32Col(shape=(3,))  # float  (double-precision)
    # Grain Bounding box
    bounding_box = tables.Int32Col(shape=(3, 2))  # Signed 64-bit integer


class Microstructure(SampleData):
    """
    Class used to manipulate a full microstructure derived from the
    `SampleData` class.

    As SampleData, this class is a data container for a mechanical sample and
    its microstructure, synchronized with a HDF5 file and a XML file
    Microstructure implements a hdf5 data model specific to polycrystalline
    sample data.

    The dataset maintains a `GrainData` instance which inherits from
    tables.IsDescription and acts as a structured array containing the grain
    attributes such as id, orientations (in form of rodrigues vectors), volume
    and bounding box.

    A crystal `Lattice` is also associated to the microstructure and used in
    all crystallography calculations.
    """

    def __init__(self,
                 filename=None, name='micro', description='empty',
                 verbose=False, overwrite_hdf5=False, lattice=None,
                 autodelete=False, **keywords):
        if filename is None:
            # only add '_' if not present at the end of name
            filename = name + (not name.endswith('_')) * '_' + 'data'

        SampleData.__init__(self, filename, name, description, verbose,
                            overwrite_hdf5, autodelete, **keywords)
        # TODO find a way not to overwrite the sample name when an existing file is read
        #self.set_sample_name(name)
        self.grains = self.get_node('GrainDataTable')
        self._init_lattice(lattice)
        self.vtkmesh = None
        self.sync()
        return

    def __repr__(self):
        """Provide a string representation of the class."""
        s = '%s\n' % self.__class__.__name__
        s += '* name: %s\n' % self.get_sample_name()
        s += '* lattice: %s\n' % self.get_lattice()
        s += '\n'
        if self._verbose:
            for g in self.grains:
                s += '* %s' % g.__repr__
        s += SampleData.__repr__(self)
        return s

    def minimal_data_model(self):
        """Data model for a polycrystalline microstructure.

        Specify the minimal contents of the hdf5 (Group names, paths and group
        types) in the form of a dictionary {content: location}. This extends
        `~pymicro.core.SampleData.minimal_data_model` method.

        :return: a tuple containing the two dictionnaries.
        """
        minimal_content_index_dic = {'Image_data': '/CellData',
                                     'grain_map': '/CellData/grain_map',
                                     'mask': '/CellData/mask',
                                     'Mesh_data': '/MeshData',
                                     'Grain_data': '/GrainData',
                                     'GrainDataTable': ('/GrainData/'
                                                        'GrainDataTable'),
                                     'Crystal_data': '/CrystalStructure',
                                     'lattice_params': ('/CrystalStructure'
                                                        '/LatticeParameters'), }
        minimal_content_type_dic = {'Image_data': '3DImage',
                                    'grain_map': 'Array',
                                    'mask': 'Array',
                                    'Mesh_data': 'Mesh',
                                    'Grain_data': 'Group',
                                    'GrainDataTable': GrainData,
                                    'Crystal_data': 'Group',
                                    'lattice_params': 'Array', }
        return minimal_content_index_dic, minimal_content_type_dic

    def _init_lattice(self, lattice):
        if lattice is None:
            lattice = Lattice.cubic(1.0)
        if not (self._file_exist):
            self._lattice = lattice
            sym = {'symmetry': self._lattice.get_symmetry().to_string()}
            params = np.array(self.get_lattice().get_lattice_parameters(),
                              dtype=np.float32)
            self.add_attributes(sym, 'Crystal_data')
            self.add_data_array('Crystal_data', name='LatticeParameters',
                                array=params, indexname='lattice_params',
                                replace=True)
        else:
            sym = Symmetry.from_string(self.get_attribute('symmetry',
                                                          'Crystal_data'))
            params = self.get_node('LatticeParameters', as_numpy=True)
            lattice = Lattice.from_symmetry(sym, params)
            self._lattice = lattice
        return

    def get_number_of_phases(self):
        """Return the number of phases in this microstructure.

        For the moment only one phase is supported, so this function simply
        returns 1.

        :return int: the number of phases in the microstructure.
        """
        return 1

    def get_number_of_grains(self):
        """Return the number of grains in this microstructure.

        :return: the number of grains in the microstructure.
        """
        return self.grains.nrows

    def get_lattice(self):
        """Get the crystallographic lattice associated with this microstructure.

        :return: an instance of the `Lattice class`.
        """
        return self._lattice

    def get_grain_map(self, as_numpy=False):
        grain_map = self.get_node(name='grain_map', as_numpy=as_numpy)
        if self._is_empty('grain_map'):
            grain_map = None
        elif grain_map.ndim == 2:
            # reshape to 3D
            grain_map = grain_map.reshape((grain_map.shape[0],
                                           grain_map.shape[1], 1))
        return grain_map

    def get_mask(self, as_numpy=False):
        mask = self.get_node(name='mask', as_numpy=as_numpy)
        if self._is_empty('mask'):
            mask = None
        elif mask.ndim == 2:
            # reshape to 3D
            mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
        return mask

    def get_ids_from_grain_map(self):
        """Return the list of grain ids found in the grain map.

        By convention, only positive values are taken into account, 0 is
        reserved for the background and -1 for overlap regions.

        :return: a 1D numpy array containing the grain ids.
        """
        grain_map = self.get_node('grain_map')
        grains_id = np.unique(grain_map)
        grains_id = grains_id[grains_id > 0]
        return grains_id

    def get_grain_ids(self):
        """Return the grain ids found in the GrainDataTable.

        :return: a 1D numpy array containing the grain ids.
        """
        return self.get_tablecol('GrainDataTable', 'idnumber')

    def get_grain_volumes(self, id_list=None):
        if id_list is None:
            return self.get_tablecol('GrainDataTable', 'volume')
        else:
            vol = []
            for g in self.grains:
                vol.append(g['volume'])
            return vol

    def get_grain_centers(self, id_list=None):
        if id_list is None:
            return self.get_tablecol('GrainDataTable', 'center')
        else:
            com = []
            for g in self.grains:
                com.append(g['center'])
            return com

    def get_grain_rodrigues(self, id_list=None):
        if id_list is None:
            return self.get_tablecol('GrainDataTable', 'orientation')
        else:
            o = []
            for g in self.grains:
                o.append(g['orientation'])
            return o

    def get_grain_orientations(self, id_list=None):
        orientations = []
        for gr in self.grains:
            if id_list is not None:
                if gr['idnumber'] not in id_list:
                    continue
            o = Orientation.from_rodrigues(gr['orientation'])
            orientations.append(o)
        return orientations

    def get_grain_bounding_boxes(self, id_list=None):
        if id_list is None:
            return self.get_tablecol('GrainDataTable', 'bounding_box')
        else:
            Bbox = []
            for g in self.grains:
                Bbox.append(g['bounding_box'])
            return Bbox

    def get_voxel_size(self):
        """Get the voxel size for image data of the microstructure.

        If this instance of `Microstructure` has no image data, None is returned.
        """
        try:
            return self.get_attribute(attrname='spacing',
                                      nodename='/CellData')[0]
        except:
            return None

    def get_grain(self, gid):
        """Get a particular grain given its id.

        This method browses the microstructure and return the grain
        corresponding to the given id. If the grain is not found, the
        method raises a `ValueError`.

        :param int gid: the grain id.
        :return: The method return a new `Grain` instance with the corresponding id.
        """
        try:
            gr = self.grains.read_where('(idnumber == gid)')[0]
        except:
            raise ValueError('grain %d not found in the microstructure' % gid)
        grain = Grain(gr['idnumber'],
                      Orientation.from_rodrigues(gr['orientation']))
        grain.center = gr['center']
        grain.volume = gr['volume']
        return grain

    def get_all_grains(self):
        """Build a list of `Grain` instances for all grains in this `Microstructure`.

        :return: a list of the grains.
        """
        grains_list = [self.get_grain(gid)
                       for gid in self.get_tablecol('GrainDataTable', 'idnumber')]
        return grains_list

    def get_grain_positions(self):
        """Return all the grain positions as a numpy array of shape (n, 3)
        where n is the number of grains.

        :return: a numpy array of shape (n, 3) of the grain positions.
        """
        return self.grains[:]['center']

    def get_grain_volume_fractions(self):
        """Compute all grains volume fractions.

        :return: a 1D numpy array with all grain volume fractions.
        """
        total_volume = np.sum(self.grains[:]['volume'])
        return self.grains[:]['volume'] / total_volume

    def get_grain_volume_fraction(self, gid, use_total_volume_value=None):
        """Compute the volume fraction of this grain.

        :param int gid: the grain id.
        :param float use_total_volume_value: the total volume value to use.
        :return float: the grain volume fraction as a number in the range [0, 1].
        """
        # compute the total volume
        if use_total_volume_value:
            total_volume = use_total_volume_value
        else:
            # sum all the grain volume to compute the total volume
            total_volume = 0.
            for grain in self.grains:
                total_volume += grain['volume']
        volume_fraction = [grain['volume'] / total_volume for grain
                           in self.grains.where('(idnumber == gid)')]
        return volume_fraction[0]

    def set_orientations(self, orientations):
        """ Store grain orientations array in GrainDataTable

            orientation : (Ngrains, 3) array of rodrigues orientation vectors
        """
        self.set_tablecol('GrainDataTable', 'orientation', column=orientations)
        return

    def set_centers(self, centers):
        """ Store grain centers array in GrainDataTable

            centers : (Ngrains, 3) array of grain centers of mass
        """
        self.set_tablecol('GrainDataTable', 'center', column=centers)
        return

    def set_bounding_boxes(self, bounding_boxes):
        """ Store grain bounding boxes array in GrainDataTable
        """
        self.set_tablecol('GrainDataTable', 'bounding_box', column=bounding_boxes)
        return

    def set_volumes(self, volumes):
        """ Store grain volumes array in GrainDataTable
        """
        self.set_tablecol('GrainDataTable', 'volume', column=volumes)
        return

    def set_lattice(self, lattice):
        """Set the crystallographic lattice associated with this microstructure.

        :param Lattice lattice: an instance of the `Lattice class`.
        """
        self._lattice = lattice

    def set_grain_map(self, grain_map, voxel_size=None, **keywords):
        """Set the grain map for this microstructure.

        :param ndarray grain_map: a 2D or 3D numpy array.
        :param float voxel_size: the size of the voxels in mm unit.
        """
        create_image = True
        print('VOXELSIZE', voxel_size)
        if self.__contains__('CellData'):
            empty = self.get_attribute(attrname='empty', nodename='CellData')
            if not empty:
                create_image = False
        if create_image:
            if voxel_size is None:
                msg = '(set_grain_map) Please specify voxel size for CellData image'
                raise ValueError(msg)
            image_object = ImageObject()
            image_object.dimension = grain_map.shape
            image_object.spacing = np.array([voxel_size, voxel_size,
                                             voxel_size])
            image_object.add_field(grain_map, 'grain_map')
            self.add_image(image_object, imagename='CellData', location='/',
                           replace=True, **keywords)
        else:
            im_vox_size = self.get_attribute('spacing', 'CellData')
            mismatch = im_vox_size[0] != voxel_size or \
                       im_vox_size[1] != voxel_size or \
                       im_vox_size[2] != voxel_size
            if (voxel_size is not None) and mismatch:
                msg = 'Voxel size mismatch between input and CellData node ' \
                      '`spacing` attribute'
                raise ValueError(msg)
            self.add_data_array(location='CellData', name='grain_map',
                                array=grain_map, replace=True, **keywords)
        return

    def set_mask(self, mask, voxel_size=None, **keywords):
        """Set the mask for this microstructure.

        :param ndarray mask: a 2D or 3D numpy array.
        :param float voxel_size: the size of the voxels in mm unit.
        """
        create_image = True
        if self.__contains__('CellData'):
            empty = self.get_attribute(attrname='empty', nodename='CellData')
            if not (empty):
                create_image = False
        if create_image:
            if (voxel_size is None):
                msg = '(set_mask) Please specify voxel size for CellData image'
                raise ValueError(msg)
            image_object = ImageObject()
            image_object.dimension = mask.shape
            image_object.spacing = np.array([voxel_size, voxel_size,
                                             voxel_size])
            image_object.add_field(mask, 'mask')
            self.add_image(image_object, imagename='CellData', location='/',
                           replace=True, **keywords)
        else:
            im_vox_size = self.get_attribute('spacing', 'CellData')[0]
            if (voxel_size is not None) and (im_vox_size != voxel_size):
                msg = ('Voxel size mismatch between input and CellData node'
                       '`spacing` attribute')
                raise ValueError(msg)
            self.add_data_array(location='CellData', name='mask',
                                array=mask, replace=True, **keywords)
        return

    def set_random_orientations(self):
        """ Set random orientations for all grains in GrainDataTable """
        for grain in self.grains:
            o = Orientation.random()
            grain['orientation'] = o.rod
            grain.update()
        self.grains.flush()
        return

    def add_grains(self, euler_list, grain_ids=None):
        """A a list of grains to this microstructure.

        This function adds a list of grains represented by a list of Euler
        angles triplets, to the microstructure. If provided, the `grain_ids`
        list will be used for the grain ids.

        :param list euler_list: the list of euler angles (Bunge passive convention).
        :param list grain_ids: an optional list for the ids of the new grains.
        """
        grain = self.grains.row
        # build a list of grain ids if it is not given
        if not grain_ids:
            if self.get_number_of_grains() > 0:
                min_id = max(self.get_grain_ids())
            else:
                min_id = 0
            grain_ids = range(min_id, min_id + len(euler_list))
        print('adding %d grains to the microstructure' % len(grain_ids))
        for gid, euler in zip(grain_ids, euler_list):
            grain['idnumber'] = gid
            grain['orientation'] = Orientation.Euler2Rodrigues(euler)
            grain.append()
        self.grains.flush()

    @staticmethod
    def random_texture(n=100):
        """Generate a random texture microstructure.

        **parameters:**

        *n* The number of grain orientations in the microstructure.
        """
        m = Microstructure(name='random_texture', overwrite_hdf5=True)
        grain = m.grains.row
        for i in range(n):
            grain['idnumber'] = i + 1
            o = Orientation.random()
            grain['orientation'] = o.rod
            grain.append()
        m.grains.flush()
        return m

    def SetVtkMesh(self, mesh):
        # TODO : create a dedicated node in h5_dataset
        self.vtkmesh = mesh

    def view_slice(self, slice=None, color='random', show_mask=True):
        """A simple utility method to show one microstructure slice.

        :param int slice: the slice number
        :param str color: a string to chose the colormap from ('random', 'ipf')
        :param bool show_mask: a flag to show the mask by transparency.
        """
        if self._is_empty('grain_map'):
            print('Microstructure instance mush have a grain_map field to use '
                  'this method')
            return
        grain_map = self.get_grain_map()
        if slice is None or slice > grain_map.shape[2] - 1 or slice < 0:
            slice = grain_map.shape[2] // 2
            print('using slice value %d' % slice)
        if color == 'random':
            grain_cmap = Microstructure.rand_cmap(first_is_black=True)
        elif color == 'ipf':
            grain_cmap = self.ipf_cmap()
        else:
            grain_cmap = 'viridis'
        fig, ax = plt.subplots()
        ax.imshow(grain_map[:, :, slice].T, cmap=grain_cmap, vmin=0)
        ax.xaxis.set_label_position('top')
        plt.xlabel('X')
        plt.ylabel('Y')
        if not self._is_empty('mask') and show_mask:
            from pymicro.view.vol_utils import alpha_cmap
            mask = self.get_mask()
            plt.imshow(mask[:, :, slice].T, cmap=alpha_cmap(opacity=0.3))
        plt.show()

    @staticmethod
    def rand_cmap(n=4096, first_is_black=False, seed=13):
        """Creates a random color map to color the grains.

        The first color can be enforced to black and usually figure out the
        background. The random seed is fixed to consistently produce the same
        colormap.

        :param int n: the number of colors in the list.
        :param bool first_is_black: set black as the first color of the list.
        :param int seed: the random seed.
        :return: a matplotlib colormap.
        """
        np.random.seed(seed)
        rand_colors = np.random.rand(n, 3)
        if first_is_black:
            rand_colors[0] = [0., 0., 0.]  # enforce black background (value 0)
        return colors.ListedColormap(rand_colors)

    def ipf_cmap(self):
        """
        Return a colormap with ipf colors.

        .. warning::

          This function works only for a microstructure with the cubic symmetry
          due to current limitation in the `Orientation` get_ipf_colour method.

        :return: a color map that can be directly used in pyplot.
        """
        ipf_colors = np.zeros((4096, 3))
        for grain in self.grains:
            o = Orientation.from_rodrigues(grain['orientation'])
            ipf_colors[grain['idnumber'], :] = o.get_ipf_colour()
        return colors.ListedColormap(ipf_colors)

    @staticmethod
    def from_grain_file(grain_file_path, col_id=0, col_phi1=1, col_phi=2,
                        col_phi2=3, col_x=4, col_y=5, col_z=None,
                        col_volume=None, autodelete=True):
        """Create a `Microstructure` reading grain infos from a file.

        This file is typically created using EBSD. the usual pattern is:
            grain_id, phi1, phi, phi2, x, y, volume.
        The column number are tunable using the function arguments.
        """
        # get the file name without extension
        name = os.path.splitext(os.path.basename(grain_file_path))[0]
        print('creating microstructure %s' % name)
        micro = Microstructure(name=name, overwrite_hdf5=True,
                               autodelete=autodelete)
        grain = micro.grains.row
        # read grain infos from the grain file
        grains_EBSD = np.genfromtxt(grain_file_path)
        for i in range(len(grains_EBSD)):
            o = Orientation.from_euler([grains_EBSD[i, col_phi1],
                                        grains_EBSD[i, col_phi],
                                        grains_EBSD[i, col_phi2]])
            grain['idnumber'] = int(grains_EBSD[i, col_id])
            grain['orientation'] = o.rod
            z = grains_EBSD[i, col_z] if col_z else 0.
            grain['center'] = np.array([grains_EBSD[i, col_x],
                                        grains_EBSD[i, col_y], z])
            if col_volume:
                grain['volume'] = grains_EBSD[i, col_volume]
            grain.append()
        micro.grains.flush()
        return micro

    def print_grains_info(self, grain_list=None, as_string=False):
        """ Print informations on the grains in the microstructure"""
        s = ''
        if grain_list is None:
            grain_list = self.get_tablecol(tablename='GrainDataTable',
                                           colname='idnumber')
        for row in self.grains:
            if row['idnumber'] in grain_list:
                o = Orientation.from_rodrigues(row['orientation'])
                s = 'Grain %d\n' % (row['idnumber'])
                s += ' * %s\n' % (o)
                s += ' * center %s\n' % np.array_str(row['center'])
                s += ' * volume %f\n' % (row['volume'])
                # s += ' * has vtk mesh ? %s\n' % (self.vtkmesh != None)
            if not (as_string):
                print(s)
        return s

    @staticmethod
    def match_grains(micro1, micro2, use_grain_ids=None, verbose=False):
        return micro1.match_grains(micro2, use_grain_ids=use_grain_ids,
                                   verbose=verbose)

    def match_grains(self, micro2, mis_tol=1, use_grain_ids=None, verbose=False):
        """Match grains from a second microstructure to this microstructure.

        This function try to find pair of grains based on their orientations.

        .. warning::

          This function works only for microstructures with the same symmetry.

        :param micro2: the second instance of `Microstructure` from which
        to match the grains.
        :param float mis_tol: the tolerance is misorientation to use
        to detect matches (in degrees).
        :param list use_grain_ids: a list of ids to restrict the grains
        in which to search for matches.
        :param bool verbose: activate verbose mode.
        :raise ValueError: if the microstructures do not have the same symmetry.
        :return tuple: a tuple of three lists holding respectively the matches,
        the candidates for each match and the grains that were unmatched.
        """
        # TODO : Test
        if not (self.get_lattice().get_symmetry()
                == micro2.get_lattice().get_symmetry()):
            raise ValueError('warning, microstructure should have the same '
                             'symmetry, got: {} and {}'.format(
                self.get_lattice().get_symmetry(),
                micro2.get_lattice().get_symmetry()))
        candidates = []
        matched = []
        unmatched = []  # grain that were not matched within the given tolerance
        # restrict the grain ids to match if needed
        sym = self.get_lattice().get_symmetry()
        if use_grain_ids is None:
            grains_to_match = self.get_tablecol(tablename='GrainDataTable',
                                                colname='idnumber')
        else:
            grains_to_match = use_grain_ids
        # look at each grain
        for i, g1 in enumerate(self.grains):
            if not (g1['idnumber'] in grains_to_match):
                continue
            cands_for_g1 = []
            best_mis = mis_tol
            best_match = -1
            o1 = Orientation.from_rodrigues(g1['orientation'])
            for g2 in micro2.grains:
                o2 = Orientation.from_rodrigues(g2['orientation'])
                # compute disorientation
                mis, _, _ = o1.disorientation(o2, crystal_structure=sym)
                misd = np.degrees(mis)
                if misd < mis_tol:
                    if verbose:
                        print('grain %3d -- candidate: %3d, misorientation:'
                              ' %.2f deg' % (g1['idnumber'], g2['idnumber'],
                                             misd))
                    # add this grain to the list of candidates
                    cands_for_g1.append(g2['idnumber'])
                    if misd < best_mis:
                        best_mis = misd
                        best_match = g2['idnumber']
            # add our best match or mark this grain as unmatched
            if best_match > 0:
                matched.append([g1['idnumber'], best_match])
            else:
                unmatched.append(g1['idnumber'])
            candidates.append(cands_for_g1)
        if verbose:
            print('done with matching')
            print('%d/%d grains were matched ' % (len(matched),
                                                  len(grains_to_match)))
        return matched, candidates, unmatched

    def find_neighbors(self, grain_id, distance=1):
        """Find the neighbor ids of a given grain.

        This function find the ids of the neighboring grains. A mask is
        constructed by dilating the grain to encompass the immediate
        neighborhood of the grain. The ids can then be determined using numpy
        unique function.

        :param int grain_id: the grain id from which the neighbors need
        to be determined.
        :param int distance: the distance to use for the dilation (default
        is 1 voxel).
        :return: a list (possibly empty) of the neighboring grain ids.
        """
        grain_map = self.get_grain_map(as_numpy=True)
        if grain_map is None:
            return []
        grain_data = (grain_map == grain_id)
        grain_data_dil = ndimage.binary_dilation(grain_data,
                                                 iterations=distance).astype(
            np.uint8)
        neighbor_ids = np.unique(grain_map[grain_data_dil - grain_data == 1])
        return neighbor_ids.tolist()

    def dilate_grain(self, grain_id, dilation_steps=1, use_mask=False):
        """Dilate a single grain overwriting the neighbors.

        :param int grain_id: the grain id to dilate.
        :param int dilation_steps: the number of dilation steps to apply.
        :param bool use_mask: if True and that this microstructure has a mask,
               the dilation will be limite by it.
        """
        # TODO : test
        grain_map = self.get_grain_map(as_numpy=True)
        grain_volume_init = (grain_map == grain_id).sum()
        grain_data = grain_map == grain_id
        grain_data = ndimage.binary_dilation(grain_data,
                                             iterations=dilation_steps).astype(np.uint8)
        if use_mask and self.__contains__('mask'):
            grain_data *= self.get_node('mask')
        grain_map[grain_data == 1] = grain_id
        grain_volume_final = (grain_map == grain_id).sum()
        print('grain %s was dilated by %d voxels' % (grain_id,
                                                     grain_volume_final - grain_volume_init))
        self.set_grain_map(grain_map, self.get_voxel_size())
        self.sync()

    @staticmethod
    def dilate_labels(array, dilation_steps=1, mask=None, dilation_ids=None,
                      struct=None):
        """Dilate labels isotropically to fill the gap between them.

        This code is based on the gtDilateGrains function from the DCT code.
        It has been extended to handle both 2D and 3D cases.

        :param ndarray array: the numpy array to dilate.
        :param int dilation_steps: the number of dilation steps to apply.
        :param ndarray mask: a msk to constrain the dilation (None by default).
        :param list dilation_ids: a list to restrict the dilation to the given ids.
        :param ndarray struct: the structuring element to use (strong
        connectivity by default).
        :return: the dilated array.
        """
        from scipy import ndimage
        if struct is None:
            struct = ndimage.morphology.generate_binary_structure(array.ndim, 1)
        assert struct.ndim == array.ndim
        # carry out dilation in iterative steps
        for step in range(dilation_steps):
            if dilation_ids:
                grains = np.isin(array, dilation_ids)
            else:
                grains = (array > 0).astype(np.uint8)
            grains_dil = ndimage.morphology.binary_dilation(grains,
                                                            structure=struct).astype(np.uint8)
            if mask is not None:
                # only dilate within the mask
                grains_dil *= mask.astype(np.uint8)
            todo = (grains_dil - grains)
            # get the list of voxel for this dilation step
            if array.ndim == 2:
                X, Y = np.where(todo)
            else:
                X, Y, Z = np.where(todo)

            xstart = X - 1
            xend = X + 1
            ystart = Y - 1
            yend = Y + 1

            # check bounds
            xstart[xstart < 0] = 0
            ystart[ystart < 0] = 0
            xend[xend > array.shape[0] - 1] = array.shape[0] - 1
            yend[yend > array.shape[1] - 1] = array.shape[1] - 1
            if array.ndim == 3:
                zstart = Z - 1
                zend = Z + 1
                zstart[zstart < 0] = 0
                zend[zend > array.shape[2] - 1] = array.shape[2] - 1

            dilation = np.zeros_like(X).astype(np.int16)
            print('%d voxels to replace' % len(X))
            for i in range(len(X)):
                if array.ndim == 2:
                    neighbours = array[xstart[i]:xend[i] + 1,
                                       ystart[i]:yend[i] + 1]
                else:
                    neighbours = array[xstart[i]:xend[i] + 1,
                                       ystart[i]:yend[i] + 1,
                                       zstart[i]:zend[i] + 1]
                if np.any(neighbours):
                    # at least one neighboring voxel in non zero
                    dilation[i] = min(neighbours[neighbours > 0])
            if array.ndim == 2:
                array[X, Y] = dilation
            else:
                array[X, Y, Z] = dilation
            print('dilation step %d done' % (step + 1))
        return array

    def dilate_grains(self, dilation_steps=1, dilation_ids=None):
        """Dilate grains to fill the gap between them.

        This function calls `dilate_labels` with the grain map of the
        microstructure.

        :param int dilation_steps: the number of dilation steps to apply.
        :param list dilation_ids: a list to restrict the dilation to the given ids.
        """
        # TODO : Test
        if not self.__contains__('grain_map'):
            raise ValueError('microstructure %s must have an associated '
                             'grain_map ' % self.get_sample_name())
            return
        grain_map = self.get_grain_map(as_numpy=True).copy()
        # get rid of overlap regions flaged by -1
        grain_map[grain_map == -1] = 0

        if self.__contains__('mask'):
            grain_map = Microstructure.dilate_labels(grain_map,
                                                     dilation_steps=dilation_steps,
                                                     mask=self.get_node('mask', as_numpy=True),
                                                     dilation_ids=dilation_ids)
        else:
            grain_map = Microstructure.dilate_labels(grain_map,
                                                     dilation_steps=dilation_steps,
                                                     dilation_ids=dilation_ids)
        # finally assign the dilated grain map to the microstructure
        self.set_grain_map(grain_map)

    def crop(self, x_start=None, x_end=None, y_start=None, y_end=None,
             z_start=None, z_end=None, crop_name=None, autodelete=False):
        """Crop the microstructure to create a new one.

        :param int x_start: start value for slicing the first axis.
        :param int x_end: end value for slicing the first axis.
        :param int y_start: start value for slicing the second axis.
        :param int y_end: end value for slicing the second axis.
        :param int z_start: start value for slicing the third axis.
        :param int z_end: end value for slicing the third axis.
        :param str crop name: the name for the cropped microstructure
        (the default is to append '_crop' to the initial name).
        :param bool autodelete: a flag to delete the microstructure files
        on the disk when it is not needed anymore.
        :return: a new `Microstructure` instance with the cropped grain map.
        """
        if self._is_empty('grain_map'):
            print('warning: needs a grain map to crop the microstructure')
            return
        # input default values for bounds if not specified
        if not x_start:
            x_start = 0
        if not y_start:
            y_start = 0
        if not z_start:
            z_start = 0
        if not x_end:
            x_end = self.get_grain_map().shape[0]
        if not y_end:
            y_end = self.get_grain_map().shape[1]
        if not z_end:
            z_end = self.get_grain_map().shape[2]
        if not crop_name:
            crop_name = self.get_sample_name() + \
                    (not self.get_sample_name().endswith('_')) * '_' + 'crop'
        micro_crop = Microstructure(name=crop_name, overwrite_hdf5=True,
                                    autodelete=autodelete)
        micro_crop.set_lattice(self.get_lattice())
        print('cropping microstructure to %s' % micro_crop.h5_file)
        grain_map_crop = self.get_grain_map()[x_start:x_end,
                         y_start:y_end, z_start:z_end]
        print(micro_crop.h5_dataset)
        micro_crop.set_grain_map(grain_map_crop,
                                 voxel_size=self.get_voxel_size())
        if self.get_mask():
            mask_crop = self.get_mask()[x_start:x_end, y_start:y_end,
                                z_start:z_end]
            micro_crop.set_mask(mask_crop)
        grain_ids = np.unique(grain_map_crop)
        for gid in grain_ids:
            if not gid > 0:
                continue
            grain = self.grains.read_where('idnumber == gid')
            micro_crop.grains.append(grain)
        print('%d grains in cropped microstructure' % micro_crop.grains.nrows)
        micro_crop.grains.flush()
        # recompute the grain geometry
        print('updating grain geometry')
        micro_crop.recompute_grain_bounding_boxes()
        micro_crop.recompute_grain_centers()
        micro_crop.recompute_grain_volumes()
        return micro_crop

    def renumber_grains(self, sort_by_size=False):
        """Renumber the grains in the microstructure.

        Renumber the grains from 1 to n, with n the total number of grains
        so that the numbering is consecutive. Only positive grain ids are taken
        into account (the id 0 is reserved for the background).

        :param bool sort_by_size: use the grain volume to sort the grain ids
        (the larger grain will become grain 1, etc).
        """
        if self._is_empty('grain_map'):
            print('warning: a grain map is needed to renumber the grains')
            return
        if sort_by_size:
            print('sorting ids by grain size')
            sizes = self.get_grain_volumes()
            new_ids = self.get_grain_ids()[np.argsort(sizes)][::-1]
        else:
            new_ids = range(1, self.get_number_of_grains() + 1)
        grain_map = self.get_grain_map(as_numpy=True)
        grain_map_renum = grain_map.copy()
        for i, g in enumerate(self.grains):
            gid = g['idnumber']
            if not gid > 0:
                # only renumber positive grain ids
                continue
            new_id = new_ids[i]
            grain_map_renum[grain_map == gid] = new_id
            g['idnumber'] = new_id
            g.update()
        print('maxium grain id is now %d' % max(new_ids))
        # assign the renumbered grain_map to the microstructure
        self.set_grain_map(grain_map_renum, self.get_voxel_size())

    def compute_grain_volume(self, gid):
        """Compute the volume of the grain given its id.

        The total number of voxels with the given id is computed. The value is
        converted to mm unit using the `voxel_size`. The unit will be mm² for a
        2D grain map or mm³ for a 3D grain map.

        .. warning::

          This function assume the grain bounding box is correct, call
          `recompute_grain_bounding_boxes()` if this is not the case.

        :param int gid: the grain id to consider.
        :return: the volume of the grain.
        """
        bb = self.grains.read_where('idnumber == %d' % gid)['bounding_box'][0]
        grain_map = self.get_grain_map()[bb[0][0]:bb[0][1],
                                         bb[1][0]:bb[1][1],
                                         bb[2][0]:bb[2][1]]
        voxel_size = self.get_attribute('spacing', 'CellData')
        volume_vx = np.sum(grain_map == np.array(gid))
        return volume_vx * np.prod(voxel_size)

    def compute_grain_center(self, gid):
        """Compute the center of masses of a grain given its id.

        .. warning::

          This function assume the grain bounding box is correct, call
          `recompute_grain_bounding_boxes()` if this is not the case.

        :param int gid: the grain id to consider.
        :return: a tuple with the center of mass in mm units
                 (or voxel if the voxel_size is not specified).
        """
        # isolate the grain within the complete grain map
        bb = self.grains.read_where('idnumber == %d' % gid)['bounding_box'][0]
        grain_map = self.get_grain_map()[bb[0][0]:bb[0][1],
                                         bb[1][0]:bb[1][1],
                                         bb[2][0]:bb[2][1]]
        voxel_size = self.get_attribute('spacing', 'CellData')
        offset = bb[:, 0]
        grain_data_bin = (grain_map == gid).astype(np.uint8)
        local_com = ndimage.measurements.center_of_mass(grain_data_bin) \
                    + np.array([0.5, 0.5, 0.5])  # account for first voxel coordinates
        com = voxel_size * (offset + local_com
                            - 0.5 * np.array(self.get_grain_map().shape))
        return com

    def compute_grain_bounding_box(self, gid, as_slice=False):
        """Compute the grain bounding box indices in the grain map.

        :param int gid: the id of the grain.
        :param bool as_slice: a flag to return the grain bounding box as a slice.
        :return: the bounding box coordinates.
        """
        slices = ndimage.find_objects(self.get_grain_map() == np.array(gid))[0]
        if as_slice:
            return slices
        x_indices = (slices[0].start, slices[0].stop)
        y_indices = (slices[1].start, slices[1].stop)
        z_indices = (slices[2].start, slices[2].stop)
        return x_indices, y_indices, z_indices

    def recompute_grain_volumes(self, verbose=False):
        """Compute the volume of all grains in the microstructure.

        Each grain volume is computed using the grain map. The value is
        assigned to the volume column of the GrainDataTable node.
        If the voxel size is specified, the grain centers will be in mm unit,
        if not in voxel unit.

        .. note::

          A grain map need to be associated with this microstructure instance
          for the method to run.

        :param bool verbose: flag for verbose mode.
        :return: a 1D array with all grain volumes.
        """
        if self._is_empty('grain_map'):
            print('warning: needs a grain map to recompute the volumes '
                  'of the grains')
            return
        for g in self.grains:
            try:
                vol = self.compute_grain_volume(g['idnumber'])
            except ValueError:
                print('skipping grain %d' % g['idnumber'])
                continue
            if verbose:
                print('grain {}, computed volume is {}'.format(g['idnumber'],
                                                               vol))
            g['volume'] = vol
            g.update()
        self.grains.flush()
        return self.get_grain_volumes()

    def recompute_grain_centers(self, verbose=False):
        """Compute and assign the center of all grains in the microstructure.

        Each grain center is computed using its center of mass. The value is
        assigned to the grain.center attribute. If the voxel size is specified,
        the grain centers will be in mm unit, if not in voxel unit.

        .. note::

          A grain map need to be associated with this microstructure instance
          for the method to run.

        :param bool verbose: flag for verbose mode.
        :return: a 1D array with all grain centers.
        """
        if self._is_empty('grain_map'):
            print('warning: need a grain map to recompute the center of mass'
                  ' of the grains')
            return
        for g in self.grains:
            try:
                com = self.compute_grain_center(g['idnumber'])
            except ValueError:
                print('skipping grain %d' % g['idnumber'])
                continue
            if verbose:
                print('grain %d center: %.3f, %.3f, %.3f'
                      % (g['idnumber'], com[0], com[1], com[2]))
            g['center'] = com
            g.update()
        self.grains.flush()
        return self.get_grain_centers()

    def recompute_grain_bounding_boxes(self, verbose=False):
        """Compute and assign the center of all grains in the microstructure.

        Each grain center is computed using its center of mass. The value is
        assigned to the grain.center attribute. If the voxel size is specified,
        the grain centers will be in mm unit, if not in voxel unit.

        .. note::

          A grain map need to be associated with this microstructure instance
          for the method to run.

        :param bool verbose: flag for verbose mode.
        """
        if self._is_empty('grain_map'):
            print('warning: need a grain map to recompute the bounding boxes'
                  ' of the grains')
            return
        # find_objects will return a list of N slices with N being the max grain id
        slices = ndimage.find_objects(self.get_grain_map(as_numpy=True))
        for g in self.grains:
            try:
                g_slice = slices[g['idnumber'] - 1]
                x_indices = (g_slice[0].start, g_slice[0].stop)
                y_indices = (g_slice[1].start, g_slice[1].stop)
                z_indices = (g_slice[2].start, g_slice[2].stop)
                bbox = x_indices, y_indices, z_indices
            except (ValueError, TypeError):
                print('skipping grain %d' % g['idnumber'])
                continue
            if verbose:
                print('grain %d bounding box: [%d:%d, %d:%d, %d:%d]'
                      % (g['idnumber'], bbox[0][0], bbox[0][1], bbox[1][0],
                         bbox[1][1], bbox[2][0], bbox[2][1]))
            g['bounding_box'] = bbox
            g.update()
        self.grains.flush()
        return self.get_grain_bounding_boxes()

    def compute_grains_geometry(self, overwrite_table=False):
        """ Compute grain centers, volume and bounding box from grain_map """
        #TODO revisit this method as we now rely on the grain bounding boxes to compute the geometry
        grains_id = self.get_ids_from_grain_map()
        if self.grains.nrows > 0 and overwrite_table:
            self.grains.remove_rows(start=0)
        for i in grains_id:
            gidx = self.grains.get_where_list('(idnumber == i)')
            if len(gidx) > 0:
                gr = self.grains[gidx]
            else:
                gr = np.zeros((1,), dtype=self.grains.dtype)
            gr['bounding_box'] = self.compute_grain_bounding_box(i)
            gr['center'] = self.compute_grain_center(i)
            gr['volume'] = self.compute_grain_volume(i)
            if len(gidx) > 0:
                self.grains[gidx] = gr
            else:
                self.grains.append(gr)
        self.grains.flush()
        return

    def to_amitex_fftp(self, binary=True,
                       add_grips=False, grip_size=10,
                       add_exterior=False, exterior_size=10):
        """Write orientation data to ascii files to prepare for FFT computation.

        AMITEX_FFTP can be used to compute the elastoplastic response of
        polycrystalline microstructures. The calculation needs orientation data
        for each grain written in the form of the coordinates of the first two
        basis vectors expressed in the crystal local frame which is given by
        the first two rows of the orientation matrix. The values are written
        in 6 files N1X.txt, N1Y.txt, N1Z.txt, N2X.txt, N2Y.txt, N2Z.txt, each
        containing n values with n the number of grains. The data is written
        either in BINARY or in ASCII form.

        Additional options exist to pad the grain map with two constant regions.
        One region called grips can be added on the top and bottom (third axis).
        The second region is around the sample (first and second axes).

        :param bool binary: flag to write the files in binary or ascii format.
        :param bool add_grips: add a constant region at the beginning and the
        end of the third axis.
        :param int grip_size: thickness of the region.
        :param bool add_exterior: add a constant region around the sample at
        the beginning and the end of the first two axes.
        :param int exterior_size: thickness of the exterior region.
        """
        ext = 'bin' if binary else 'txt'
        n1x = open('N1X.%s' % ext, 'w')
        n1y = open('N1Y.%s' % ext, 'w')
        n1z = open('N1Z.%s' % ext, 'w')
        n2x = open('N2X.%s' % ext, 'w')
        n2y = open('N2Y.%s' % ext, 'w')
        n2z = open('N2Z.%s' % ext, 'w')
        files = [n1x, n1y, n1z, n2x, n2y, n2z]
        if binary:
            import struct
            for f in files:
                f.write('%d \ndouble \n' % self.get_number_of_grains())
                f.close()
            n1x = open('N1X.%s' % ext, 'ab')
            n1y = open('N1Y.%s' % ext, 'ab')
            n1z = open('N1Z.%s' % ext, 'ab')
            n2x = open('N2X.%s' % ext, 'ab')
            n2y = open('N2Y.%s' % ext, 'ab')
            n2z = open('N2Z.%s' % ext, 'ab')
            for g in self.grains:
                o = Orientation.from_rodrigues(g['orientation'])
                g = o.orientation_matrix()
                n1 = g[0]  # first row
                n2 = g[1]  # second row
                n1x.write(struct.pack('>d', n1[0]))
                n1y.write(struct.pack('>d', n1[1]))
                n1z.write(struct.pack('>d', n1[2]))
                n2x.write(struct.pack('>d', n2[0]))
                n2y.write(struct.pack('>d', n2[1]))
                n2z.write(struct.pack('>d', n2[2]))
        else:
            for g in self.grains:
                o = Orientation.from_rodrigues(g['orientation'])
                g = o.orientation_matrix()
                n1 = g[0]  # first row
                n2 = g[1]  # second row
                n1x.write('%f\n' % n1[0])
                n1y.write('%f\n' % n1[1])
                n1z.write('%f\n' % n1[2])
                n2x.write('%f\n' % n2[0])
                n2y.write('%f\n' % n2[1])
                n2z.write('%f\n' % n2[2])
        n1x.close()
        n1y.close()
        n1z.close()
        n2x.close()
        n2y.close()
        n2z.close()
        print('orientation data written for AMITEX_FFTP')

        # if possible, write the vtk file to run the computation
        if self.__contains__('grain_map'):
            # convert the grain map to vtk file
            from vtk.util import numpy_support
            #TODO build a continuous grain map for amitex
            grain_ids = self.get_grain_map(as_numpy=True)
            material_ids = np.zeros_like(grain_ids)
            new_id = 1
            if add_grips:
                # add a layer of new_id (the value must actually be the first
                # grain id) above and below the sample.
                grain_ids = np.pad(grain_ids, ((0, 0),
                                               (0, 0),
                                               (grip_size, grip_size)),
                                   mode='constant', constant_values=1)
                material_ids = np.pad(material_ids, ((0, 0),
                                                     (0, 0),
                                                     (grip_size, grip_size)),
                                      mode='constant', constant_values=new_id)
                new_id += 1
            if add_exterior:
                # add a layer of new_id around the first two dimensions
                grain_ids = np.pad(grain_ids, ((exterior_size, exterior_size),
                                               (exterior_size, exterior_size),
                                               (0, 0)),
                                   mode='constant', constant_values=1)
                material_ids = np.pad(material_ids,
                                      ((exterior_size, exterior_size),
                                       (exterior_size, exterior_size),
                                       (0, 0)),
                                      mode='constant', constant_values=new_id)
            # write both arrays as VTK files for amitex
            voxel_size = self.get_voxel_size()
            for array, array_name in zip([grain_ids, material_ids],
                                         ['grain_ids', 'material_ids']):
                vtk_data_array = numpy_support.numpy_to_vtk(np.ravel(array,
                                                                     order='F'),
                                                            deep=1)
                vtk_data_array.SetName(array_name)
                grid = vtk.vtkImageData()
                size = array.shape
                grid.SetExtent(0, size[0], 0, size[1], 0, size[2])
                grid.GetCellData().SetScalars(vtk_data_array)
                grid.SetSpacing(voxel_size, voxel_size, voxel_size)
                writer = vtk.vtkStructuredPointsWriter()
                writer.SetFileName('%s_%s.vtk' % (self.get_sample_name(),
                                                  array_name))
                if binary:
                    writer.SetFileTypeToBinary()
                writer.SetInputData(grid)
                writer.Write()
                print('%s array written in legacy vtk form for AMITEX_FFTP' %
                      array_name)

    def print_zset_material_block(self, mat_file, grain_prefix='_ELSET'):
        """
        Outputs the material block corresponding to this microstructure for
        a finite element calculation with z-set.

        :param str mat_file: The name of the file where the material behaviour
        is located
        :param str grain_prefix: The grain prefix used to name the elsets
        corresponding to the different grains
        """
        f = open('elset_list.txt', 'w')
        # TODO : test
        for g in self.grains:
            o = Orientation.from_rodrigues(g['orientation'])
            f.write('  **elset %s%d *file %s *integration '
                    'theta_method_a 1.0 1.e-9 150 *rotation '
                    '%7.3f %7.3f %7.3f\n' % (grain_prefix, g['idnumber'],
                                             mat_file,
                                             o.phi1(), o.Phi(), o.phi2()))
        f.close()
        return

    def to_dream3d(self):
        """Write the microstructure as a hdf5 file compatible with DREAM3D."""
        # TODO: test
        import time
        f = h5py.File('%s.h5' % self.get_sample_name(), 'w')
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
        cryst_structure = ed.create_dataset('CrystalStructures',
                                            data=np.array([[999], [1]],
                                                          dtype=np.uint32))
        cryst_structure.attrs['ComponentDimensions'] = np.uint64(1)
        cryst_structure.attrs['DataArrayVersion'] = np.int32(2)
        cryst_structure.attrs['ObjectType'] = np.string_('DataArray<uint32_t>')
        cryst_structure.attrs['Tuple Axis Dimensions'] = np.string_('x=2')
        cryst_structure.attrs['TupleDimensions'] = np.uint64(2)
        mat_name = ed.create_dataset('MaterialName',
                                     data=[a.encode('utf8')
                                           for a in ['Invalid Phase', 'Unknown']])
        mat_name.attrs['ComponentDimensions'] = np.uint64(1)
        mat_name.attrs['DataArrayVersion'] = np.int32(2)
        mat_name.attrs['ObjectType'] = np.string_('StringDataArray')
        mat_name.attrs['Tuple Axis Dimensions'] = np.string_('x=2')
        mat_name.attrs['TupleDimensions'] = np.uint64(2)
        # feature data
        fd = m.create_group('FeatureData')
        fd.attrs['AttributeMatrixType'] = np.uint32(7)
        fd.attrs['TupleDimensions'] = np.uint64(self.grains.nrows)
        Euler = np.array([Orientation.from_rodrigues(g['orientation'])
                          for g in self.grains], dtype=np.float32)
        avg_euler = fd.create_dataset('AvgEulerAngles', data=Euler)
        avg_euler.attrs['ComponentDimensions'] = np.uint64(3)
        avg_euler.attrs['DataArrayVersion'] = np.int32(2)
        avg_euler.attrs['ObjectType'] = np.string_('DataArray<float>')
        avg_euler.attrs['Tuple Axis Dimensions'] = np.string_('x=%d' %
                                                              self.grains.nrows)
        avg_euler.attrs['TupleDimensions'] = np.uint64(self.grains.nrows)
        # geometry
        geom = m.create_group('_SIMPL_GEOMETRY')
        geom.attrs['GeometryType'] = np.uint32(999)
        geom.attrs['GeometryTypeName'] = np.string_('UnkownGeometry')
        # create the data container bundles group
        f.create_group('DataContainerBundles')
        f.close()

    @staticmethod
    def from_dream3d(file_path, main_key='DataContainers',
                     data_container='DataContainer', grain_data='FeatureData',
                     grain_orientations='AvgEulerAngles',
                     orientation_type='euler', grain_centroid='Centroids'):
        """Read a microstructure from a hdf5 file.

        :param str file_path: the path to the hdf5 file to read.
        :param str main_key: the string describing the root key.
        :param str data_container: the string describing the data container
        group in the hdf5 file.
        :param str grain_data: the string describing the grain data group in the
        hdf5 file.
        :param str grain_orientations: the string describing the average grain
        orientations in the hdf5 file.
        :param str orientation_type: the string describing the descriptor used
        for orientation data.
        :param str grain_centroid: the string describing the grain centroid in
        the hdf5 file.
        :return: a `Microstructure` instance created from the hdf5 file.
        """
        # TODO: test
        head, tail = os.path.split(file_path)
        micro = Microstructure(name=tail, file_path=head, overwrite_hdf5=True)
        with h5py.File(file_path, 'r') as f:
            grain_data_path = '%s/%s/%s' % (main_key, data_container, grain_data)
            orientations = f[grain_data_path][grain_orientations].value
            if grain_centroid:
                centroids = f[grain_data_path][grain_centroid].value
                offset = 0
                if len(centroids) < len(orientations):
                    offset = 1  # if grain 0 has not a centroid
            grain = micro.grains.row
            for i in range(len(orientations)):
                grain['idnumber'] = i
                if orientations[i, 0] == 0. and orientations[i, 1] == 0. and \
                        orientations[i, 2] == 0.:
                    # skip grain 0 which is always (0., 0., 0.)
                    print('skipping (0., 0., 0.)')
                    continue
                if orientation_type == 'euler':
                    grain['orientation'] = Orientation.from_euler(
                        orientations[i] * 180 / np.pi).rod
                elif orientation_type == 'rodrigues':
                    grain['orientation'] = Orientation.from_rodrigues(
                        orientations[i]).rod
                if grain_centroid:
                    grain['center'] = centroids[i - offset]
                grain.append()
            micro.grains.flush()
        return micro

    @staticmethod
    def copy_sample(src_micro_file, dst_micro_file, overwrite=False,
                    get_object=False, dst_name=None, autodelete=False):
        """ Initiate a new SampleData object and files from existing one"""
        SampleData.copy_sample(src_micro_file, dst_micro_file, overwrite,
                               new_sample_name=dst_name)
        if get_object:
            return Microstructure(filename=dst_micro_file, autodelete=autodelete)
        else:
            return

    @staticmethod
    def from_neper(neper_file_path):
        """Create a microstructure from a neper tesselation.

        Neper is an open source program to generate polycristalline
        microstructure using voronoi tesselations. It is available at
        https://neper.info

        :param str neper_file_path: the path to the tesselation file generated
        by Neper.
        :return: a pymicro `Microstructure` instance.
        """
        neper_file = neper_file_path.split(os.sep)[-1]
        neper_dir = os.path.dirname(neper_file_path)
        print('creating microstructure from Neper tesselation %s' % neper_file)
        name, ext = os.path.splitext(neper_file)
        print(name, ext)
        assert ext == '.tesr'  # assuming raster tesselation
        micro = Microstructure(name=name, file_path=neper_dir, overwrite_hdf5=True)
        with open(neper_file_path, 'r', encoding='latin-1') as f:
            line = f.readline()  # ***tesr
            # look for **general
            while True:
                line = f.readline().strip()  # get rid of unnecessary spaces
                if line.startswith('**general'):
                    break
            dim = f.readline().strip()
            print(dim)
            dims = np.array(f.readline().split()).astype(int).tolist()
            print(dims)
            voxel_size = np.array(f.readline().split()).astype(float).tolist()
            print(voxel_size)
            # look for **cell
            while True:
                line = f.readline().strip()
                if line.startswith('**cell'):
                    break
            n = int(f.readline().strip())
            print('microstructure contains %d grains' % n)
            f.readline()  # *id
            grain_ids = []
            # look for *ori
            while True:
                line = f.readline().strip()
                if line.startswith('*ori'):
                    break
                else:
                    grain_ids.extend(np.array(line.split()).astype(int).tolist())
            print('grain ids are:', grain_ids)
            oridescriptor = f.readline().strip()  # must be euler-bunge:passive
            if oridescriptor != 'euler-bunge:passive':
                print('Wrong orientation descriptor: %s, must be '
                      'euler-bunge:passive' % oridescriptor)
            grain = micro.grains.row
            for i in range(n):
                euler_angles = np.array(f.readline().split()).astype(float).tolist()
                print('adding grain %d' % grain_ids[i])
                grain['idnumber'] = grain_ids[i]
                grain['orientation'] = Orientation.from_euler(euler_angles).rod
                grain.append()
            micro.grains.flush()
            # look for **data
            while True:
                line = f.readline().strip()
                if line.startswith('**data'):
                    break
            print(f.tell())
            print('reading data from byte %d' % f.tell())
            data = np.fromfile(f, dtype=np.uint16)[:-4]  # leave out the last 4 values
            print(data.shape)
            assert np.prod(dims) == data.shape[0]
            micro.set_grain_map(data.reshape(dims[::-1]).transpose(2, 1, 0),
                                voxel_size=voxel_size[0])  # swap X/Z axes
            print('updating grain geometry')
            micro.recompute_grain_bounding_boxes()
            micro.recompute_grain_centers()
            micro.recompute_grain_volumes()
        print('done')
        return micro

    @staticmethod
    def from_dct(data_dir='.', grain_file='index.mat',
                 vol_file='phase_01_vol.mat', mask_file='volume_mask.mat',
                 use_dct_path=True, verbose=True):
        """Create a microstructure from a DCT reconstruction.

        DCT reconstructions are stored in several files. The indexed grain
        informations are stored in a matlab file in the '4_grains/phase_01'
        folder. Then, the reconstructed volume file (labeled image) is stored
        in the '5_reconstruction' folder as an hdf5 file, possibly stored
        alongside a mask file coming from the absorption reconstruction.

        :param str data_dir: the path to the folder containing the
                              reconstruction data.
        :param str grain_file: the name of the file containing grains info.
        :param str vol_file: the name of the volume file.
        :param str mask_file: the name of the mask file.
        :param bool use_dct_path: if True, the grain_file should be located in
                                  4_grains/phase_01 folder and the
        vol_file and mask_file in the 5_reconstruction folder.
        :param bool verbose: activate verbose mode.
        :return: a `Microstructure` instance created from the DCT reconstruction.
        """
        if data_dir == '.':
            data_dir = os.getcwd()
        if data_dir.endswith(os.sep):
            data_dir = data_dir[:-1]
        scan = data_dir.split(os.sep)[-1]
        print('creating microstructure for DCT scan %s' % scan)
        micro = Microstructure(name=scan, path=data_dir, overwrite_hdf5=True)
        micro.data_dir = data_dir
        if use_dct_path:
            index_path = os.path.join(data_dir, '4_grains', 'phase_01',
                                      grain_file)
        else:
            index_path = os.path.join(data_dir, grain_file)
        print(index_path)
        if not os.path.exists(index_path):
            raise ValueError('%s not found, please specify a valid path to the'
                             ' grain file.' % index_path)
            return None
        from scipy.io import loadmat
        index = loadmat(index_path)
        voxel_size = index['cryst'][0][0][25][0][0]
        # grab the crystal lattice
        lattice_params = index['cryst'][0][0][3][0]
        sym = Symmetry.from_string(index['cryst'][0][0][7][0])
        print('creating crystal lattice {} ({}) with parameters {}'
              ''.format(index['cryst'][0][0][0][0], sym, lattice_params))
        lattice_params[:3] /= 10  # angstrom to nm
        lattice = Lattice.from_parameters(*lattice_params, symmetry=sym)
        micro.set_lattice(lattice)
        # add all grains to the microstructure
        grain = micro.grains.row
        for i in range(len(index['grain'][0])):
            grain['idnumber'] = index['grain'][0][i][0][0][0][0][0]
            grain['orientation'] = index['grain'][0][i][0][0][3][0]
            grain['center'] = index['grain'][0][i][0][0][15][0]
            grain.append()
        micro.grains.flush()

        # load the grain map if available
        if use_dct_path:
            grain_map_path = os.path.join(data_dir, '5_reconstruction',
                                          vol_file)
        else:
            grain_map_path = os.path.join(data_dir, vol_file)
        if os.path.exists(grain_map_path):
            with h5py.File(grain_map_path, 'r') as f:
                # because how matlab writes the data, we need to swap X and Z
                # axes in the DCT volume
                micro.set_grain_map(f['vol'][()].transpose(2, 1, 0), voxel_size)
                if verbose:
                    print('loaded grain ids volume with shape: {}'
                          ''.format(micro.get_grain_map().shape))
        # load the mask if available
        if use_dct_path:
            mask_path = os.path.join(data_dir, '5_reconstruction', mask_file)
        else:
            mask_path = os.path.join(data_dir, mask_file)
        if os.path.exists(mask_path):
            try:
                with h5py.File(mask_path, 'r') as f:
                    mask = f['vol'][()].transpose(2, 1, 0).astype(np.uint8)
                    # check if mask shape needs to be zero padded
                    if not mask.shape == micro.get_grain_map().shape:
                        offset = np.array(micro.get_grain_map().shape) - np.array(mask.shape)
                        padding = [(o // 2, o // 2) for o in offset]
                        print('mask padding is {}'.format(padding))
                        mask = np.pad(mask, padding, mode='constant')
                    print('now mask shape is {}'.format(mask.shape))
                    micro.set_mask(mask, voxel_size)
            except:
                # fallback on matlab format
                micro.set_mask(loadmat(mask_path)['vol'], voxel_size)
            if verbose:
                print('loaded mask volume with shape: {}'.format(micro.get_mask().shape))
        return micro

    @staticmethod
    def from_legacy_h5(file_path):
        """read a microstructure object from a HDF5 file created by pymicro
        until version 0.4.5.

        :param str file_path: the path to the file to read.
        :return: the new `Microstructure` instance created from the file.
        """
        with h5py.File(file_path, 'r') as f:
            micro = Microstructure(name=f.attrs['microstructure_name'],
                                   overwrite_hdf5=True)
            if 'symmetry' in f['EnsembleData/CrystalStructure'].attrs:
                sym = f['EnsembleData/CrystalStructure'].attrs['symmetry']
                parameters = f['EnsembleData/CrystalStructure/LatticeParameters'][()]
                micro.set_lattice(Lattice.from_symmetry(Symmetry.from_string(sym),
                                                        parameters))
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
                # add all grains to the microstructure
                grain = micro.grains.row
                for i in range(avg_rods.shape[0]):
                    grain['idnumber'] = grain_ids[i]
                    grain['orientation'] = avg_rods[i, :]
                    grain['center'] = centers[i]
                    grain.append()
                micro.grains.flush()
            # load cell data
            if 'grain_ids' in f['CellData']:
                micro.set_grain_map(f['CellData/grain_ids'][()],
                                    f['CellData/grain_ids'].attrs['voxel_size'])
            if 'mask' in f['CellData']:
                micro.set_mask(f['CellData/mask'][()],
                               f['CellData/mask'].attrs['voxel_size'])
            return micro

    @staticmethod
    def from_ebsd(file_path, roi=None):
        """"Create a microstructure from an EBSD scan.

        :param str file_path: the path to the file to read.
        :param list roi: a list of 4 integers to crop the EBSD scan.
        :return: a new instance of `Microstructure`.
        """
        name = os.path.splitext(os.path.basename(file_path))[0]
        micro = Microstructure(name=name, autodelete=False, overwrite_hdf5=True)
        from pymicro.crystal.ebsd import OimScan
        scan = OimScan.from_file(file_path)
        if roi:
            print('importing data from region {}'.format(roi))
            scan.cols = roi[1] - roi[0]
            scan.rows = roi[3] - roi[2]
            scan.iq = scan.iq[roi[0]:roi[1], roi[2]:roi[3]]
            scan.ci = scan.ci[roi[0]:roi[1], roi[2]:roi[3]]
            scan.euler = scan.euler[roi[0]:roi[1], roi[2]:roi[3], :]
        iq = scan.iq
        ci = scan.ci
        euler = scan.euler
        mask = np.ones_like(iq)
        # segment the grains
        grain_ids = scan.segment_grains()
        micro.set_grain_map(grain_ids, scan.xStep)

        # add each array to the data file
        mask_array = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
        iq_array = np.reshape(iq, (iq.shape[0], iq.shape[1], 1))
        ci_array = np.reshape(ci, (ci.shape[0], ci.shape[1], 1))
        micro.add_data_array(location='CellData', name='mask', array=mask_array, replace=True)
        micro.add_data_array(location='CellData', name='iq', array=iq_array, replace=True)
        micro.add_data_array(location='CellData', name='ci', array=ci_array, replace=True)
        from pymicro.core.images import ImageObject
        image_euler = ImageObject()
        image_euler.dimension = euler.shape
        image_euler.spacing = np.array([scan.xStep, scan.xStep, scan.xStep])
        image_euler.add_field(euler, 'euler')
        micro.add_image(image_euler, imagename='OrientationData', location='/', replace=True)


        grains = micro.grains.row
        for gid in np.unique(grain_ids):
            if gid == 0:
                continue
            print('adding grain %d' % gid)
            # use the first pixel of the grain
            pixel = np.where(grain_ids == gid)[0][0], np.where(grain_ids == gid)[1][0]
            # TODO compute mean grain orientation
            o_tsl = Orientation.from_euler(np.degrees(micro.get_node('euler')[pixel]))
            # TODO link OimScan lattice to pymicro
            # o_fz = o_tsl.move_to_FZ(symmetry=Ti7Al._symmetry)
            grains['idnumber'] = gid
            grains['orientation'] = o_tsl.rod
            grains.append()
        micro.grains.flush()
        micro.recompute_grain_centers(verbose=False)
        micro.recompute_grain_volumes(verbose=False)
        micro.sync()
        return micro

    @staticmethod
    def merge_microstructures(micros, overlap, translation_offset=[0, 0, 0],
                              plot=False):
        """Merge two `Microstructure` instances together.

        The function works for two microstructures with grain maps and an
        overlap between them. Temporarily `Microstructures` restricted to the
        overlap regions are created and grains are matched between the two based
        on a disorientation tolerance.

        .. note::

          The two microstructure must have the same crystal lattice and the
          same voxel_size for this method to run.

        :param list micros: a list containing the two microstructures to merge.
        :param int overlap: the overlap to use.
        :param list translation_offset: a manual translation (in voxels) offset
        to add to the result.
        :param bool plot: a flag to plot some results.
        :return: a new `Microstructure`instance containing the merged
                 microstructure.
        """
        # TODO: test
        from scipy import ndimage

        # perform some sanity checks
        for i in range(2):
            if micros[i]._is_empty('grain_map'):
                raise ValueError('microstructure instance %s must have an '
                                 'associated grain_map attribute'
                                 % micros[i].get_sample_name())
        if micros[0].get_lattice() != micros[1].get_lattice():
            raise ValueError('both microstructure must have the same crystal '
                             'lattice')
        lattice = micros[0].get_lattice()
        if micros[0].get_voxel_size() != micros[1].get_voxel_size():
            raise ValueError('both microstructure must have the same'
                             ' voxel size')
        voxel_size = micros[0].get_voxel_size()

        # create two microstructures for the two overlapping regions:
        # end slices in first scan and first slices in second scan
        micro1_ol = micros[0].crop(z_start=micros[0].get_grain_map().shape[2] -
                                           overlap, autodelete=True)
        micro2_ol = micros[1].crop(z_end=overlap, autodelete=True)
        micros_ol = [micro1_ol, micro2_ol]

        # match grain from micros_ol[1] to micros_ol[0] (the reference)
        matched, _, unmatched = micros_ol[0].match_grains(micros_ol[1],
                                                          verbose=True)

        # to find the translation, we compute the differences in coordinates of
        # the center of mass of the matched grains between the two microstructures
        translation_mm = np.zeros(3)
        for i in range(len(matched)):
            # look at the pair of grains
            match = matched[i]
            delta = (micros_ol[0].get_grain(match[0]).center
                     - micros_ol[1].get_grain(match[1]).center)
            translation_mm += delta
        translation_mm /= len(matched)
        # account for the origin of the overlap region
        translation_mm[2] += (micros[0].get_grain_map().shape[2] -
                              overlap) * voxel_size
        print('average shift (voxels): {}'.format(translation_mm / voxel_size))
        translation_voxel = (translation_mm / voxel_size).astype(int)
        print('translation is in mm: {}'.format(translation_mm))
        print('translation is in voxels {}'.format(translation_voxel))
        # manually correct the result if necessary
        translation_voxel += translation_offset

        # now delete overlapping microstructures
        del micro1_ol, micro2_ol

        # look at ids in the reference volume
        ids_ref = np.unique(micros[0].get_grain_map())
        ids_ref_list = ids_ref.tolist()
        if -1 in ids_ref_list:
            ids_ref_list.remove(-1)  # grain overlap
        if 0 in ids_ref_list:
            ids_ref_list.remove(0)  # background
        print(ids_ref_list)
        id_offset = max(ids_ref_list)
        print('grain ids in volume %s will be offset by %d'
              % (micros[1].get_sample_name(), id_offset))

        # gather ids in the merging volume (will be modified)
        ids_mrg = np.unique(micros[1].get_grain_map())
        ids_mrg_list = ids_mrg.tolist()
        if -1 in ids_mrg_list:
            ids_mrg_list.remove(-1)  # grain overlap
        if 0 in ids_mrg_list:
            ids_mrg_list.remove(0)  # background
        print(ids_mrg_list)

        # prepare a volume with the same size as the second grain map,
        # with grain ids renumbered and (X, Y) translations applied.
        grain_map = micros[1].get_grain_map(as_numpy=True)
        grain_map_translated = grain_map.copy()
        print('renumbering grains in the overlap region of volume %s'
              % micros[1].get_sample_name())
        for match in matched:
            ref_id, other_id = match
            print('replacing %d by %d' % (other_id, ref_id))
            grain_map_translated[grain_map == other_id] = ref_id
            try:
                ids_mrg_list.remove(other_id)
            except ValueError:
                # this can happend if a reference grain was matched to more than 1 grain
                print('%d was not in list anymore' % other_id)
        # also renumber the rest using the offset
        renumbered_grains = []
        for i, other_id in enumerate(ids_mrg_list):
            new_id = id_offset + i + 1
            grain_map_translated[grain_map == other_id] = new_id
            print('replacing %d by %d' % (other_id, new_id))
            renumbered_grains.append([other_id, new_id])

        # apply translation along the (X, Y) axes
        grain_map_translated = np.roll(grain_map_translated,
                                       translation_voxel[:2], (0, 1))

        check = overlap // 2
        print(grain_map_translated.shape)
        print(overlap)
        print(translation_voxel[2] + check)
        if plot:
            slice_ref = micros[0].get_grain_map()[:, :,
                                                  translation_voxel[2] + check]
            slice_renum = grain_map_translated[:, :, check]
            id_max = max(slice_ref.max(), slice_renum.max())
            fig = plt.figure(figsize=(15, 7))
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(micros[0].get_grain_map()[:, :, translation_voxel[2]
                                                 + check].T, vmin=0, vmax=id_max)
            plt.axis('off')
            plt.title('micros[0].grain_map (ref)')
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(grain_map_translated[:, :, check].T, vmin=0, vmax=id_max)
            plt.axis('off')
            plt.title('micros[1].grain_map (renumbered)')
            ax3 = fig.add_subplot(1, 3, 3)
            same_voxel = (micros[0].get_grain_map()[:, :,
                          translation_voxel[2] + check]
                          == grain_map_translated[:, :, check])
            ax3.imshow(same_voxel.T, vmin=0, vmax=2)
            plt.axis('off')
            plt.title('voxels that are identicals')
            plt.savefig('merging_check1.pdf')

        # start the merging: the first volume is the reference
        gmap_shape = [micros[0].get_grain_map().shape,
                      micros[1].get_grain_map().shape]
        overlap = gmap_shape[0][2] - translation_voxel[2]
        print('overlap is %d voxels' % overlap)
        z_shape = gmap_shape[0][2] + gmap_shape[1][2] - overlap
        print('vertical size will be: %d + %d + %d = %d'
              % (gmap_shape[0][2] - overlap, overlap,
                 gmap_shape[1][2] - overlap, z_shape))
        shape_merged = (np.array(gmap_shape[0])
                        + [0, 0, gmap_shape[1][2] - overlap])
        print('initializing volume with shape {}'.format(shape_merged))
        grain_ids_merged = np.zeros(shape_merged, dtype=np.int16)
        print(gmap_shape[0])
        print(gmap_shape[1])

        # add the non-overlapping part of the 2 volumes as is
        grain_ids_merged[:, :, :gmap_shape[0][2] - overlap] = (
            micros[0].get_grain_map()[:, :, :-overlap])
        grain_ids_merged[:, :, gmap_shape[0][2]:] = grain_map_translated[:, :,
                                                    overlap:]

        # look at vertices with the same label
        print(micros[0].get_grain_map()[:, :, translation_voxel[2]:].shape)
        print(grain_map_translated[:, :, :overlap].shape)
        print('translation_voxel[2] = %d' % translation_voxel[2])
        print('micros[0].grain_map.shape[2] - overlap = %d'
              % (gmap_shape[0][2] - overlap))
        same_voxel = (micros[0].get_grain_map()[:, :, translation_voxel[2]:]
                      == grain_map_translated[:, :, :overlap])
        print(same_voxel.shape)
        grain_ids_merged[:, :, translation_voxel[2]:gmap_shape[0][2]] = (
                grain_map_translated[:, :, :overlap] * same_voxel)

        # look at vertices with a single label
        single_voxels_0 = ((micros[0].get_grain_map()[:, :,
                            translation_voxel[2]:] > 0)
                           & (grain_map_translated[:, :, :overlap] == 0))
        print(single_voxels_0.shape)
        grain_ids_merged[:, :, translation_voxel[2]:gmap_shape[0][2]] += (
            (micros[0].get_grain_map()[:, :, translation_voxel[2]:]
             * single_voxels_0))
        single_voxels_1 = ((grain_map_translated[:, :, :overlap] > 0)
                           & (micros[0].get_grain_map()[:, :,
                              translation_voxel[2]:] == 0))
        print(single_voxels_1.shape)
        grain_ids_merged[:, :, translation_voxel[2]:gmap_shape[0][2]] += (
                grain_map_translated[:, :, :overlap] * single_voxels_1)

        if plot:
            fig = plt.figure(figsize=(14, 10))
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.imshow(grain_ids_merged[:, grain_ids_merged.shape[1] // 2, :].T)
            plt.axis('off')
            plt.title('XZ slice')
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.imshow(grain_ids_merged[grain_ids_merged.shape[0] // 2, :, :].T)
            plt.axis('off')
            plt.title('YZ slice')
            plt.savefig('merging_check2.pdf')

        if not micros[0]._is_empty('mask') and not micros[1]._is_empty('mask'):
            mask_translated = np.roll(micros[1].get_mask(),
                                      translation_voxel[:2], (0, 1))

            # merging the masks
            mask_merged = np.zeros(shape_merged, dtype=np.uint8)
            # add the non-overlapping part of the 2 volumes as is
            mask_merged[:, :, :gmap_shape[0][2] - overlap] = (
                micros[0].get_mask()[:, :, :-overlap])
            mask_merged[:, :, gmap_shape[0][2]:] = (
                mask_translated[:, :, overlap:])

            # look at vertices with the same label
            same_voxel = (micros[0].get_mask()[:, :, translation_voxel[2]:] ==
                          mask_translated[:, :, :overlap])
            print(same_voxel.shape)
            mask_merged[:, :,
            translation_voxel[2]:micros[0].get_mask().shape[2]] = (
                    mask_translated[:, :, :overlap] * same_voxel)

            # look at vertices with a single label
            single_voxels_0 = (micros[0].get_mask()[:, :, translation_voxel[2]:]
                               > 0) & (mask_translated[:, :, :overlap] == 0)
            mask_merged[:, :,
            translation_voxel[2]:micros[0].get_mask().shape[2]] += (
                (micros[0].get_mask()[:, :, translation_voxel[2]:]
                 * single_voxels_0).astype(np.uint8))
            single_voxels_1 = ((mask_translated[:, :, :overlap] > 0)
                               & (micros[0].get_mask()[:, :,
                                  translation_voxel[2]:] == 0))
            mask_merged[:, :,
            translation_voxel[2]:micros[0].get_mask().shape[2]] += (
                (mask_translated[:, :, :overlap]
                 * single_voxels_1).astype(np.uint8))

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
        name_part1 = micros[0].get_sample_name()
        if name_part1.endswith('_data'):
            name_part1 = name_part1[:-5]
        name_part2 = micros[1].get_sample_name()
        if name_part2.endswith('_data'):
            name_part2 = name_part2[:-5]
        merged_name = '%s-%s' % (name_part1, name_part2)
        merged_path = os.path.dirname(micros[0].h5_file)
        desc = 'merged microstructure from %s and %s' % \
               (micros[0].get_sample_name(), micros[1].get_sample_name())
        merged_micro = Microstructure(name=merged_name, description=desc,
                                      overwrite_hdf5=True)
        merged_micro.set_lattice(lattice)
        # add all grains from the reference volume
        grains_0 = micros[0].grains.read()
        merged_micro.grains.append(grains_0)
        merged_micro.grains.flush()
        print(renumbered_grains)
        # add all new grains from the merged volume
        for i in range(len(renumbered_grains)):
            other_id, new_id = renumbered_grains[i]
            g = micros[1].grains.read_where('idnumber == other_id')
            g['idnumber'] = new_id
            merged_micro.grains.append(g)
            print('adding grain with new id %d (was %d)' % (new_id, other_id))
        print('%d grains in merged microstructure'
              % merged_micro.get_number_of_grains())
        merged_micro.grains.flush()

        # add the full grain map
        print('assigning merged grain map')
        merged_micro.set_grain_map(grain_ids_merged, voxel_size)
        # recompute the geometry of the grains
        print('updating grain geometry')
        merged_micro.recompute_grain_bounding_boxes()
        merged_micro.recompute_grain_centers()
        merged_micro.recompute_grain_volumes()
        if not micros[0]._is_empty('mask') and not micros[1]._is_empty('mask'):
            print('assigning merged mask')
            merged_micro.set_mask(mask_merged, voxel_size)
        merged_micro.sync()
        return merged_micro
