"""
The microstructure module provide elementary classes to describe a
crystallographic granular microstructure such as mostly present in
metallic materials.

It contains several classes which are used to describe a microstructure
composed of multiple grains, each one having its own crystallographic
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
from pathlib import Path
from scipy import ndimage
from matplotlib import pyplot as plt, colors
from pymicro.crystal.lattice import Lattice, Symmetry, CrystallinePhase, Crystal
from pymicro.crystal.rotation import om2ro, ro2qu, qu2om
from pymicro.crystal.quaternion import Quaternion
from pymicro.core.samples import SampleData
import tables
from math import atan2, pi
from tqdm import tqdm
import time


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
        self.rod = om2ro(g)
        self.quat = Quaternion(ro2qu(self.rod))

    def orientation_matrix(self):
        """Returns the orientation matrix in the form of a 3x3 numpy array."""
        return self._matrix

    def __eq__(self, value: object) -> bool:
        return np.allclose(self.orientation_matrix(), value.orientation_matrix())

    def __repr__(self):
        """Provide a string representation of the class."""
        s = 'Crystal Orientation \n-------------------'
        s += '\norientation matrix = \n %s' % self._matrix.view()
        s += '\nEuler angles (degrees) = (%8.3f,%8.3f,%8.3f)' % (self.phi1(), self.Phi(), self.phi2())
        s += '\nRodrigues vector = %s' % self.rod
        s += '\nQuaternion = %s' % self.quat
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
                             'or a 3x3 matrix, got %d values' % v.size)
        g = self.orientation_matrix()
        if v.size == 3:
            # input is vector
            return np.dot(g, v)
        else:
            # input is 3x3 matrix
            return np.dot(g, np.dot(v, g.T))

    def to_sample(self, v):
        """Transform a vector or a matrix from the crystal frame to the sample
        frame.

        :param ndarray v: a 3 component vector or a 3x3 array expressed in
            the crystal frame.
        :return: the vector or matrix expressed in the sample frame.
        """
        if v.size not in [3, 9]:
            raise ValueError('input arg must be a 3 components vector '
                             'or a 3x3 matrix, got %d values' % v.size)
        g = self.orientation_matrix()
        if v.size == 3:
            # input is vector
            return np.dot(g.T, v)
        else:
            # input is 3x3 matrix
            return np.dot(g.T, np.dot(v, g))

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

    def ipf_color(self, axis=np.array([0., 0., 1.]), symmetry=Symmetry.cubic, saturate=True):
        """Compute the IPF (inverse pole figure) colour for this orientation.

        This method has bee adapted from the DCT code and works for most of 
        the crystal symmetries.

        :param ndarray axis: the direction to use to compute the IPF colour.
        :param Symmetry symmetry: the symmetry operator to use.
        :param bool saturate: a flag to saturate the RGB values.
        :return: the IPF colour in RGB form.
        """
        axis /= np.linalg.norm(axis)
        Vc = np.dot(self.orientation_matrix(), axis)
        # get the symmetry operators
        syms = symmetry.symmetry_operators()
        syms = np.concatenate((syms, -syms))
        Vc_syms = np.dot(syms, Vc)
        # phi: rotation around 001 axis, from 100 axis to Vc vector, projected on (100,010) plane
        Vc_phi = np.arctan2(Vc_syms[:, 1], Vc_syms[:, 0]) * 180 / pi
        # chi: rotation around 010 axis, from 001 axis to Vc vector, projected on (100,001) plane
        Vc_chi = np.arctan2(Vc_syms[:, 0], Vc_syms[:, 2]) * 180 / pi
        # psi : angle from 001 axis to Vc vector
        Vc_psi = np.arccos(Vc_syms[:, 2]) * 180 / pi
        if symmetry is Symmetry.cubic:
            angleR = 45 - Vc_chi  # red color proportional to (45 - chi)
            minAngleR = 0
            maxAngleR = 45
            angleB = Vc_phi  # blue color proportional to phi
            minAngleB = 0
            maxAngleB = 45
        elif symmetry is Symmetry.hexagonal:
            angleR = 90 - Vc_psi  # red color proportional to (90 - psi)
            minAngleR = 0
            maxAngleR = 90
            angleB = Vc_phi  # blue color proportional to phi
            minAngleB = 0
            maxAngleB = 30
        elif symmetry is Symmetry.tetragonal:
            angleR = 90 - Vc_psi  # red color proportional to (90 - psi)
            minAngleR = 0
            maxAngleR = 90
            angleB = Vc_phi  # blue color proportional to phi
            minAngleB = 0
            maxAngleB = 45
        elif symmetry is Symmetry.orthorhombic:
            angleR = 90 - Vc_psi  # red color proportional to (90 - psi)
            minAngleR = 0
            maxAngleR = 90
            angleB = Vc_phi  # blue color proportional to phi
            minAngleB = 0
            maxAngleB = 90
        else:
            raise(ValueError('unsupported crystal symmetry to compute IPF color'))
        # find the axis lying in the fundamental zone
        fz_list = ((angleR >= minAngleR) & (angleR < maxAngleR) &
                   (angleB >= minAngleB) & (angleB < maxAngleB)).tolist()
        if not fz_list.count(True) == 1:
            raise(ValueError('problem moving to the fundamental zone'))
            return None
        i_SST = fz_list.index(True)
        r = angleR[i_SST] / maxAngleR
        g = (maxAngleR - angleR[i_SST]) / maxAngleR * (maxAngleB - angleB[i_SST]) / maxAngleB
        b = (maxAngleR - angleR[i_SST]) / maxAngleR * angleB[i_SST] / maxAngleB
        rgb = np.array([r, g, b])
        if saturate:
            rgb = rgb / rgb.max()
        return rgb

    @staticmethod
    def compute_mean_orientation(rods, symmetry=Symmetry.cubic):
        """Compute the mean orientation.

        This function computes a mean orientation from several data points
        representing orientations by averaging the corresponding Rodrigues
        vector. One caveat with this averaging method is if the vectors belong
        to different asymmetric domains, the mean orientation will be wrong.

        To avoid this, we compute all equivalent rotation for all data points
        and then use kmeans clustering to separate all points. We do this using
        quaternions to avoid arbitrarily large rodrigues vectors when the
        rotation angle approaches pi (this would make the clustering algorithm
        fail). The number of clusters is known to be equal to the number of
        symmetry operators.

        A mean orientation is then computed for each cluster and the one which
        belongs to the fundamental zone is returned.

        :param ndarray rods: a (n, 3) shaped array containing the Rodrigues
        vectors of the orientations.
        :param `Symmetry` symmetry: the symmetry used to move orientations
        to their fundamental zone (cubic by default)
        :returns: the mean orientation as an `Orientation` instance.
        """
        rod_sym_mean = Orientation.compute_mean_rodrigues(rods, symmetry=symmetry)
        return Orientation.from_rodrigues(rod_sym_mean)

    @staticmethod
    def compute_mean_rodrigues(rods, symmetry=Symmetry.cubic):
        """Compute the mean orientation as a rodrigues vector.

        This function computes a mean orientation from several data points
        representing orientations by averaging the corresponding Rodrigues
        vector. One caveat with this averaging method is if the vectors belong
        to different asymmetric domains, the mean orientation will be wrong.

        To avoid this, we compute all equivalent rotation for all data points
        and then use kmeans clustering to separate all points. We do this using
        quaternions to avoid arbitrarily large rodrigues vectors when the
        rotation angle approaches pi (this would make the clustering algorithm
        fail). The number of clusters is known to be equal to the number of
        symmetry operators.

        A mean orientation is then computed for each cluster and the one which
        belongs to the fundamental zone is returned.

        :param ndarray rods: a (n, 3) shaped array containing the Rodrigues
        vectors of the orientations.
        :param `Symmetry` symmetry: the symmetry used to move orientations
        to their fundamental zone (cubic by default)
        :returns: the mean orientation as a rodrigues vector.
        """
        rods = np.atleast_2d(rods)
        # omit nan values from the calculation
        rods = rods[~np.prod(np.isnan(rods), 1).astype(bool)]
        syms = symmetry.symmetry_operators()

        # apply all symmetries to compute Rodrigues vectors
        rods_syms = np.zeros((len(syms), len(rods), 3), dtype=float)
        for i in range(len(rods)):
            g = Orientation.from_rodrigues(rods[i]).orientation_matrix()
            # apply all symmetries
            g_syms = np.dot(syms, g)
            for j in range(len(syms)):
                # compute the rodrigues vector for each symmetry
                rods_syms[j, i] = Orientation(g_syms[j]).rod

        # we now apply kmeans clustering in quaternion space
        X = rods_syms.reshape((len(syms) * len(rods), 3))
        Q = np.empty((len(X), 4), dtype=float)
        for i in range(len(X)):
            Q[i] = ro2qu(X[i])
        # we need to initialize the centroid with one point in each zone
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=len(syms),  # number of clusters = nb of symmetry operators
                        init=Q[::len(rods)],  # explicit centroids
                        n_init=1).fit(Q)

        # now compute the mean orientation for each cluster
        mean_rods_syms = np.empty((len(syms), 3), dtype=float)
        for j in range(len(syms)):
            #mean_rods_syms[j] = np.mean(X[kmeans.labels_ == j], axis=0)
            mean_quat_syms_j = np.mean(Q[kmeans.labels_ == j], axis=0)
            mean_quat_syms_j /= np.sqrt(np.sum(mean_quat_syms_j ** 2))
            mean_rods_syms[j] = om2ro(qu2om(mean_quat_syms_j))
        # find which orientation belongs to the FZ
        if symmetry == Symmetry.cubic:
             index_fz = np.argmax([(np.abs(r).sum() <= 1.0) and
                                   (np.abs(r).max() <= 2 ** 0.5 - 1)
                                   for r in mean_rods_syms])
        elif symmetry == Symmetry.hexagonal:
            index_fz = np.argmax([Orientation.fzDihedral(r, 6)
                                  for r in mean_rods_syms])
        else:
            raise (ValueError('unsupported crystal symmetry: %s' % symmetry))

        return mean_rods_syms[index_fz]

    @staticmethod
    def fzDihedral(rod, n):
        """check if the given Rodrigues vector is in the fundamental zone.

        After book from Morawiec :cite`Morawiec_2004`:

        .. pull_quote::

          The asymmetric domain is a prism with 2n-sided polygons (at the
          distance $h_n$ from 0) as prism bases, and $2n$ square prism faces at
          the distance $h_2 = 1$. The bases are perpendicular to the n-fold axis
          and the faces are perpendicular to the twofold axes.
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
        (also called the asymmetric domain) restricts the orientation space
        accordingly.

        :param symmetry: the `Symmetry` to use.
        :return bool: True if this orientation is in the fundamental zone,
        False otherwise.
        """
        r = self.rod
        if symmetry == Symmetry.cubic:
            inFZT23 = np.abs(r).sum() <= 1.0
            # in the cubic symmetry, each component must be < 2 ** 0.5 - 1
            inFZ = inFZT23 and np.abs(r).max() <= 2 ** 0.5 - 1
        elif symmetry == Symmetry.hexagonal:
            inFZ = Orientation.fzDihedral(r, 6)
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

    def rotate_orientation(self, rotation_xyz):
        """Rotate this grain orientation given the rotation matrix expressed
        in the laboratory coordinate system.

        :param rotation_xyz: a 3x3 array representing the rotation matrix to apply.
        :return: a new Orientation instance rotated.
        """
        # express the rotation in the crystal frame
        g = self.orientation_matrix()
        rotation_cry = np.dot(g, np.dot(rotation_xyz, g.T)).T
        # apply the rotation and create a new orientation
        g_rot = np.dot(rotation_cry, g)
        return Orientation(g_rot)

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
        To avoid float rounding point error, the value of :math:`\\cos\\omega`
        is clipped to [-1.0, 1.0].

        .. note::

          This does not account for the crystal symmetries. If you want to
          find the disorientation between two orientations, use the
          :py:meth:`~pymicro.crystal.microstructure.Orientation.disorientation`
          method.

        :param delta: The 3x3 misorientation matrix.
        :returns float: the misorientation angle in radians.
        """
        cw = np.clip(0.5 * (delta.trace() - 1), -1., 1.)
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

        :param orientation: an instance of
            :py:class:`~pymicro.crystal.microstructure.Orientation` class
            describing the other crystal orientation from which to compute the
            angle.
        :param crystal_structure: an instance of the `Symmetry` class
            describing the crystal symmetry, triclinic (no symmetry) by
            default.
        :returns tuple: the misorientation angle in radians, the axis as a
            numpy vector (crystal coordinates), the axis as a numpy vector
            (sample coordinates).
        """
        the_angle = np.pi
        the_axis = np.array([0., 0., 1.])
        the_axis_xyz = np.array([0., 0., 1.])
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
        """Convenience method to expose the first Euler angle."""
        return self.euler[0]

    def Phi(self):
        """Convenience method to expose the second Euler angle."""
        return self.euler[1]

    def phi2(self):
        """Convenience method to expose the third Euler angle."""
        return self.euler[2]

    def plot(self, lattice=None):
        """create a figure to plot the orientation.

        Create a 3D representation of a crystal lattice rotated 
        by this orientation in the (X, Y, Z) local frame.

        :param Lattice lattice: an optional crystal lattice can be specified.
        """
        if lattice is None:
            lattice = Lattice.cubic(0.5)
        coords, edges, faces = lattice.get_points(origin='mid')
        print(coords.shape, len(edges), len(faces))
        # now apply the crystal orientation
        g = self.orientation_matrix()
        coords_rot = np.empty_like(coords)
        for i in range(len(coords_rot)):
            coords_rot[i] = np.dot(g.T, coords[i])
        # plot the rotated lattice
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(coords_rot[:, 0], coords_rot[:, 1], coords_rot[:, 2])
        for i, face in enumerate(faces):
            face_coords = coords_rot[face]
            ax.plot(face_coords[:, 0], face_coords[:, 1], face_coords[:, 2], 'k-')
        #for i in range(len(edge_point_ids)):
        #    ax.plot(coords_rot[edge_point_ids[i, :], 0],
        #            coords_rot[edge_point_ids[i, :], 1],
        #            coords_rot[edge_point_ids[i, :], 2], 'k-')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=60., azim=45)
        plt.show()

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
        """Compute instrument transformation matrix for given rotation offset.

        This function compute a 3x3 rotation matrix (passive convention) that
        transforms the sample coordinate system by rotating around the 3
        cartesian axes in this order: rotation around X is applied first,
        then around Y and finally around Z.

        A sample vector :math:`V_s` is consequently transformed into
        :math:`V'_s` as:

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

        :param hkl: the hkl plane, an instance of
            :py:class:`~pymicro.crystal.lattice.HklPlane`
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
    def from_n1n2(n1, n2):
        """Create an orientation instance from the two vectors used in
        Amitex_FFTP which are the first two rows of the orientation matrix.

        :param np.array n1: the first vector describing the orientation.
        :param np.array n2: the second vector describing the orientation.
        :return: an instance of the `Orientation` class.
        """
        n3 = np.cross(n1, n2)  # third row
        g = np.array([n1, n2, n3])  # orientation matrix
        o = Orientation(g)
        return o

    @staticmethod
    def from_amitex(data_dir='.', binary=True, prefix=''):
        """This methods reads the orientations from Amitex files.

        The orientation data is read from 6 files: N1X.bin, N1Y.bin, N1Z.bin,
        N2X.bin, N2Y.bin and N2Z.bin (in binary format). The bin extension is
        replaced by txt for the ascii format.

        :param str data_dir: the folder where the orientation files are located.
        :param bool binary: use binary format (True by default).
        :param str prefix: a prefix for the file names.
        :return: a list containing instances of the `Orientation` class.
        """
        orientations = []
        n_components = []
        if binary:
            for file_name in ['%sN1X.bin' % prefix, '%sN1Y.bin' % prefix, '%sN1Z.bin' % prefix,
                              '%sN2X.bin' % prefix, '%sN2Y.bin' % prefix, '%sN2Z.bin' % prefix]:
                with open(os.path.join(data_dir, file_name), 'rb') as f:
                    line = f.readline()
                    n = int(line.decode('utf-8').split('\n')[0])
                    print('%d entries in file %s' % (n, file_name))
                    line = f.readline()
                    n_components.append(np.frombuffer(f.read(), dtype='>d'))
                    assert (len(n_components[-1]) == n)
        else:
            # using ascii file format
            for file_name in ['%sN1X.txt' % prefix, '%sN1Y.txt' % prefix, '%sN1Z.txt' % prefix,
                              '%sN2X.txt' % prefix, '%sN2Y.txt' % prefix, '%sN2Z.txt' % prefix]:
                with open(os.path.join(data_dir, file_name), 'r') as f:
                    n_components.append(np.atleast_1d(np.genfromtxt(file_name)))
                n = len(n_components[-1])

        # now we have the data, construct the orientations
        for i in range(n):
            n1 = n_components[0][i], n_components[1][i], n_components[2][i]
            n2 = n_components[3][i], n_components[4][i], n_components[5][i]
            orientations.append(Orientation.from_n1n2(n1, n2))
        return orientations

    @staticmethod
    def transformation_matrix(hkl_1, hkl_2, n_1, n_2):
        """Compute the orientation matrix from the two known hkl plane 
        normals.
        
        The function build two orthonormal basis, one in the crystal 
        frame and the other in the sample frame. the orientation matrix 
        brings the second one into coincidence with first one.

        :param hkl_1: the first `HklPlane` instance.
        :param hkl_2: the second `HklPlane` instance.
        :param n_1: a vector normal to the first lattice plane.
        :param n_2: a vector normal to the second lattice plane.
        :return: the corresponding 3x3 orientation matrix.
        """
        # create the vectors representing this frame in the crystal coordinate system
        e1_hat_c = hkl_1.normal()
        e2_hat_c = np.cross(hkl_1.normal(), hkl_2.normal()) / np.linalg.norm(
            np.cross(hkl_1.normal(), hkl_2.normal()))
        e3_hat_c = np.cross(e1_hat_c, e2_hat_c)
        e_hat_c = np.array([e1_hat_c, e2_hat_c, e3_hat_c])
        # create local frame attached to the indexed crystallographic features in XYZ
        e1_hat_s = n_1
        e2_hat_s = np.cross(n_1, n_2) / np.linalg.norm(
            np.cross(n_1, n_2))
        e3_hat_s = np.cross(e1_hat_s, e2_hat_s)
        e_hat_s = np.array([e1_hat_s, e2_hat_s, e3_hat_s])
        # now build the orientation matrix
        orientation_matrix = np.dot(e_hat_c.T, e_hat_s)
        return orientation_matrix

    @staticmethod
    def from_two_hkl_normals(hkl_1, hkl_2, xyz_normal_1, xyz_normal_2):
        g = Orientation.transformation_matrix(hkl_1, hkl_2, 
                                              xyz_normal_1, xyz_normal_2)
        return Orientation(g)

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
        Compute the (passive) orientation matrix associated the rotation
        defined by the given (axis, angle) pair.

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
    def Axis2Quaternion(axis, angle, P=1):
        """
        Compute the quaternion associated the rotation defined by the given
        (axis, angle) pair.

        :param axis: the rotation axis.
        :param angle: the rotation angle (degrees).
        :param int P: convention (1 for active, -1 for passive)
        :return: the corresponding Quaternion.
        """
        omega = np.radians(angle)
        axis /= np.linalg.norm(axis)
        q = Quaternion([np.cos(0.5 * omega), *(-P * np.sin(0.5 * omega) * axis)], convention=P)
        return q

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
        if q0 < 0:
            # the scalar part must be positive
            q.quat = q.quat * -1
        return q

    @staticmethod
    def Euler2Rodrigues(euler):
        """Compute the rodrigues vector from the 3 euler angles (in degrees).

        :param euler: the 3 Euler angles (in degrees).
        :return: the rodrigues vector as a 3 components numpy array.
        """
        (phi1, Phi, phi2) = np.radians(euler)
        a = 0.5 * (phi1 - phi2)
        b = 0.5 * (phi1 + phi2)
        r1 = np.tan(0.5 * Phi) * np.cos(a) / np.cos(b)
        r2 = np.tan(0.5 * Phi) * np.sin(a) / np.cos(b)
        r3 = np.tan(b)
        return np.array([r1, r2, r3])

    @staticmethod
    def eu2ro(euler):
        """Transform a series of euler angles into Rodrigues vectors.

        :param ndarray euler: the (n, 3) shaped array of Euler angles (radians).
        :returns: a (n, 3) array with the Rodrigues vectors.
        """
        if euler.ndim != 2 or euler.shape[1] != 3:
            raise ValueError('Wrong shape for the euler array: %s -> should be (n, 3)' % euler.shape)
        phi1, Phi, phi2 = np.squeeze(np.split(euler, 3, axis=1))
        a = 0.5 * (phi1 - phi2)
        b = 0.5 * (phi1 + phi2)
        r1 = np.tan(0.5 * Phi) * np.cos(a) / np.cos(b)
        r2 = np.tan(0.5 * Phi) * np.sin(a) / np.cos(b)
        r3 = np.tan(b)
        return np.array([r1, r2, r3]).T

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
        phi1, Phi, phi2 = np.radians(euler)
        c1 = np.cos(phi1)
        s1 = np.sin(phi1)
        c = np.cos(Phi)
        s = np.sin(Phi)
        c2 = np.cos(phi2)
        s2 = np.sin(phi2)

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
        :py:class:`~pymicro.crystal.microstructure.Orientation` and the given
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

    def schmid_factor(self, slip_system, load_direction=[0., 0., 1],
                      unidirectional=False):
        """Compute the Schmid factor for this crystal orientation and the
        given slip system.

        .. note::

          The Schmid factor is usually regarded as positive (dislocation slip
          can happend in both shear directions), because of this, the absolute
          value of the dot product is returned. In some case, is can be useful
          to compute the unidirectional value of the Schmid factor, in this
          case, set the unidirectional parameter to True.

        :param slip_system: a `SlipSystem` instance.
        :param load_direction: a unit vector describing the loading direction
            (default: vertical axis [0, 0, 1]).
        :param bool unidirectional: if True, a positive value is not enforced.
        :return float: a number between 0 ad 0.5 (or -0.5 and 0.5 with the
        unidirectional option).
        """
        t = load_direction / np.linalg.norm(load_direction)
        plane = slip_system.get_slip_plane()
        gt = self.orientation_matrix().transpose()
        n_rot = np.dot(gt, plane.normal())  # plane.normal() is a unit vector
        slip = slip_system.get_slip_direction().direction()
        slip_rot = np.dot(gt, slip)
        schmid_factor = np.dot(n_rot, t) * np.dot(slip_rot, t)
        if not unidirectional:
            schmid_factor = abs(schmid_factor)
        return schmid_factor

    def compute_all_schmid_factors(self, slip_systems, unidirectional=False,
                                   load_direction=[0., 0., 1], verbose=False):
        """Compute all Schmid factors for this crystal orientation and the
        given list of slip systems.

        :param slip_systems: a list of the slip systems from which to compute
            the Schmid factor values.
        :param load_direction: a unit vector describing the loading direction
            (default: vertical axis [0, 0, 1]).
        :param bool unidirectional: if True, a positive value is not enforced.
        :param bool verbose: activate verbose mode.
        :return list: a list of the schmid factors.
        """
        schmid_factor_list = []
        for ss in slip_systems:
            sf = self.schmid_factor(ss, load_direction, unidirectional)
            if verbose:
                print('Slip system: %s, Schmid factor is %.3f' % (ss, sf))
            schmid_factor_list.append(sf)
        return schmid_factor_list

    @staticmethod
    def compute_m_factor(o1, ss1, o2, ss2):
        """Compute the m factor with another slip system.

        :param Orientation o1: the orientation the first grain.
        :param SlipSystem ss1: the slip system in the first grain.
        :param Orientation o2: the orientation the second grain.
        :param SlipSystem ss2: the slip system in the second grain.
        :returns: the m factor as a float number < 1
        """
        # orientation matrices
        gt1 = o1.orientation_matrix().T
        gt2 = o2.orientation_matrix().T
        # slip plane normal in sample local frame
        n1 = np.dot(gt1, ss1.get_slip_plane().normal())
        n2 = np.dot(gt2, ss2.get_slip_plane().normal())
        # slip direction in sample local frame
        l1 = np.dot(gt1, ss1.get_slip_direction().direction())
        l2 = np.dot(gt2, ss2.get_slip_direction().direction())
        # m factor calculation
        m = abs(np.dot(n1, n2) * np.dot(l1, l2))
        return m


class Grain:
    """
    Class defining a crystallographic grain.

    A grain has a constant crystallographic `Orientation` and a grain id. The
    center attribute is the center of mass of the grain in world coordinates.
    The volume of the grain is normally expressed in mm unit especially when
    working in relation with a `Microstructure` instance; if the unit has not
    been set, the volume is then given in pixel/voxel unit.
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
            if verbose:
                print('thresholding label: %d' % label)
                print(thresh.GetOutput())
            #edges = vtk.vtkExtractEdges()
            #edges.SetInputConnection(thresh.GetOutputPort())
            #edges.Update()
            #self.SetVtkMesh(edges.GetOutput())
            thresh.Update()
            self.SetVtkMesh(thresh.GetOutput())

    def vtk_file_name(self):
        return 'grain_%d.vtu' % self.id

    def save_vtk_repr(self, file_name=None):
        import vtk
        if not file_name:
            file_name = self.vtk_file_name()
        print('writting ' + file_name)
        #writer = vtk.vtkXMLPolyDataWriter()
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
        #reader = vtk.vtkXMLPolyDataReader()
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
    idnumber = tables.Int32Col()  # Signed 32-bit integer
    # grain volume
    volume = tables.Float32Col()  # float
    # grain center of mass coordinates
    center = tables.Float32Col(shape=(3,))  # float  (double-precision)
    # Rodrigues vector defining grain orientation
    orientation = tables.Float32Col(shape=(3,))  # float  (double-precision)
    # Grain Bounding box
    bounding_box = tables.Int32Col(shape=(3, 2))  # Signed 64-bit integer
    # grain phase id
    phase = tables.UInt8Col(dflt=1)  # Unsigned 8-bit integer


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

    A list of `CrystallinePhase` instances (each on with its `Lattice`) is also
    associated to the microstructure and used for all crystallography calculations.
    """

#TODO: include length unit specification
#TODO: specific Pymicro file extension ?
#TODO: refactor class
#TODO: open in protected mode --> no modification of datasets allowed
    def __init__(self,
                 filename=None, name='micro', description='empty',
                 verbose=False, overwrite_hdf5=False,
                 phase=None, autodelete=False):
        if filename is None:
            # only add '_' if not present at the end of name
            filename = name + (not name.endswith('_')) * '_' + 'data'
        # prepare arguments for after file open
        if phase is None:
            phase = CrystallinePhase()
        if type(phase) is not list:
            phase_list = [phase]
        else:
            phase_list = phase
        after_file_open_args = {'phase_list': phase_list}
        # call SampleData constructor
        SampleData.__init__(self, filename=filename, sample_name=name,
                            sample_description=description, verbose=verbose,
                            overwrite_hdf5=overwrite_hdf5,
                            autodelete=autodelete,
                            after_file_open_args=after_file_open_args)
        return

    def _after_file_open(self, phase_list=None, **kwargs):
        """Initialization code to run after opening a Sample Data file."""
        self.grains = self.get_node('GrainDataTable')
        self.default_compression_options = {'complib': 'zlib', 'complevel': 5}
        if self._file_exist:
            self.active_grain_map = self.get_attribute('active_grain_map',
                                                       'CellData')
            self.active_phase_map = self.get_attribute('active_phase_map',
                                                       'CellData')
            if self.active_grain_map is None:
                # set active grain map to 'grain_map' if none exist
                self.set_active_grain_map()
            if self.active_phase_map is None:
                # set active phase map to 'phase_map' if none exist
                self.set_active_phase_map()
            self._init_phase(phase_list)
        else:
            self.set_active_grain_map()
            self.set_active_phase_map()
            self._init_phase(phase_list)

    def __repr__(self):
        """Provide a string representation of the class."""
        # TODO: print number of grains if available
        # TODO: print extent of CellData group if available
        s = '%s' % self.__class__.__name__
        s += ' "%s"\t' % self.get_sample_name()
        s += ' "File : {%s}"\n' % self.h5_path
        s += '------------------------------------------------------------\n'
        s += '* DESCRIPTION: %s\n' % self.get_description()
        phase_id_list = self.get_phase_ids_list()
        s += '\n* MATERIAL PHASES: \n'
        for id in phase_id_list:
            s += f'\t{self.get_phase(phase_id=id).__repr__()}'
            s += ' \n'
        s += '\n* CONTENT: \n'
        s += self.print_dataset_content(as_string=True, short=True)
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
                                     'phase_map': '/CellData/phase_map',
                                     'mask': '/CellData/mask',
                                     'Mesh_data': '/MeshData',
                                     'Grain_data': '/GrainData',
                                     'GrainDataTable': ('/GrainData/'
                                                        'GrainDataTable'),
                                     'Phase_data': '/PhaseData'}
        minimal_content_type_dic = {'Image_data': '3DImage',
                                    'grain_map': 'field_array',
                                    'phase_map': 'field_array',
                                    'mask': 'field_array',
                                    'Mesh_data': 'Mesh',
                                    'Grain_data': 'Group',
                                    'GrainDataTable': GrainData,
                                    'Phase_data': 'Group'}
        return minimal_content_index_dic, minimal_content_type_dic

    def _init_phase(self, phase_list) -> None:
        self._phases = []
        # if the h5 file does not exist yet, store the phase as metadata
        if not self._file_exist:
            self.set_phases(phase_list)
        else:
            self.sync_phases(verbose=False)
            # if no phase is there, create one
            if len(self.get_phase_ids_list()) == 0:
                print('no phase was found in this dataset, adding a default one')
                self.add_phase(phase_list[0])
        # for microstructure created without the phase column, it is initialize at 0, fix this
        if np.array_equal(np.unique(self.grains[:]['phase']), [0]):
            print('end of _init_phase, fixing phase array')
            self.set_grain_phases(np.ones_like(self.get_grain_ids()))
            print(self.grains[:]['phase'])

    def sync_phases(self, verbose=True) -> None:
        """This method sync the _phases attribute with the content of the hdf5
        file.
        """
        self._phases = []
        # loop on the phases present in the group /PhaseData
        phase_group = self.get_node('/PhaseData')
        for child in phase_group._v_children:
            d = self.get_dic_from_attributes('/PhaseData/%s' % child)
            phase = CrystallinePhase.from_dict(d)
            self._phases.append(phase)
        if verbose:
            print('%d phases found in the data set' % len(self._phases))

    def set_phase(self, phase, id_number=None) -> None:
        """Set a phase for the given `phase_id`.

        If the phase id does not correspond to one of the existing phase,
        nothing is done.

        :param CrystallinePhase phase: the phase to use.
        :param int phase_id:
        """
        if id_number is None:
            id_number = phase.phase_id
        if id_number > self.get_number_of_phases():
            print('the phase_id given (%d) does not correspond to any existing '
                  'phase, the phase list has not been modified.')
            return
        d = phase.to_dict()
        print('setting phase %d with %s' % (id_number, phase.name))
        self.add_attributes(d, '/PhaseData/phase_%02d' % id_number)
        self.sync_phases()

    def set_phases(self, phase_list) -> None:
        """Set a list of phases for this microstructure.

        The different phases in the list are added in that order.

        :param list phase_list: the list of phases to use.
        """
        # delete all node in the phase_group
        self.remove_node('/PhaseData', recursive=True)
        self.add_group('PhaseData', location='/', indexname='Phase_data')
        self.sync_phases()
        # add each phase
        for phase in phase_list:
            self.add_phase(phase)

    def set_phase_elastic_constants(self, elastic_constants, phase_id=1) -> None:
        """Set the elastic constants of the given phase.

        :param list elastic_constants: the list of elastic constants to use,
        this has to be in agreement with the symmetry of the phase.
        :param int phase_id: the id of the phase to use.
        :raise ValueError: if the elastic constants do not match the phase
        symmetry.
        """
        phase = self.get_phase(phase_id)
        print('assigning elastic constants %s to phase %s' %
              (elastic_constants, phase.name))
        try:
            phase.set_elastic_constants(elastic_constants)
            self.set_phase(phase)
        except ValueError as e:
            print(e)

    def get_number_of_phases(self) -> int:
        """Return the number of phases in this microstructure.

        Each crystal phase is stored in a list attribute: `_phases`. Note that
        it may be different (although it should not) from the different
        phase ids in the phase_map array.

        :return int: the number of phases in the microstructure.
        """
        return len(self._phases)

    def get_number_of_grains(self, from_grain_map=False) -> int:
        """Return the number of grains in this microstructure.

        :param bool from_grain_map: controls if the retrurned number of grains
        comes from the grain data table or from the grain map.
        :return: the number of grains in the microstructure.
        """
        if from_grain_map:
            return len(np.unique(self.get_grain_map()))
        else:
            return self.grains.nrows

    def add_phase(self, phase) -> None:
        """Add a new phase to this microstructure.

        Before adding this phase, the phase id is set to the corresponding id.

        :param CrystallinePhase phase: the phase to add.
        """
        # this phase should have id self.get_number_of_phases() + 1
        new_phase_id = self.get_number_of_phases() + 1
        if not phase.phase_id == new_phase_id:
            print('warning, adding phase with phase_id = %d (was %d)' %
                  (new_phase_id, phase.phase_id))
        phase.phase_id = new_phase_id
        self._phases.append(phase)
        self.add_group('phase_%02d' % new_phase_id, location='/PhaseData',
                       indexname='phase_%02d' % new_phase_id, replace=True)
        d = phase.to_dict()
        self.add_attributes(d, '/PhaseData/phase_%02d' % new_phase_id)
        print('new phase added: %s' % phase.name)

    def get_phase_list(self) -> list:
        """Return the list of the phases in this microstructure."""
        return self._phases

    def get_phase_ids_list(self) -> list:
        """Return the list of the phase ids."""
        return [phase.phase_id for phase in self._phases]

    def get_phase(self, phase_id=1) -> CrystallinePhase:
        """Get a crystalline phase.

        If no phase_id is given, the first phase is returned.

        :param int phase_id: the id of the phase to return.
        :return: the `CrystallinePhase` corresponding to the id.
        """
        index = self.get_phase_ids_list().index(phase_id)
        return self._phases[index]

    def get_lattice(self, phase_id=1) -> Lattice:
        """Get the crystallographic lattice associated with this microstructure.

        If no phase_id is given, the `Lattice` of the active phase is returned.

        :return: an instance of the `Lattice class`.
        """
        return self.get_phase(phase_id).get_lattice()

    def get_grain_map(self) -> np.array:
        """Get the active grain map as a numpy array.

        The grain map is the image constituted by the grain ids or labels.
        Label zero is reserved for the background or unattributed voxels.

        :return: the grain map as a numpy array.
        """
        grain_map = self.get_field(self.active_grain_map)
        if self._is_empty(self.active_grain_map):
            grain_map = None
        elif grain_map.ndim == 2:
            # reshape to 3D
            new_dim = self.get_attribute('dimension', 'CellData')
            if len(new_dim) == 3:
                grain_map = grain_map.reshape(new_dim)
            else:
                grain_map = grain_map.reshape(
                    (grain_map.shape[0], grain_map.shape[1], 1))
        return grain_map

    def get_phase_map(self):
        """Get the active phase map as a numpy array.

        The phase map is an array of int where each voxel value tells you what
        is the local material phase with respect to the `phase_list` attribute.

        :return: the phase map as a numpy array.
        """
        phase_map = self.get_field(self.active_phase_map)
        if self._is_empty('phase_map'):
            phase_map = None
        elif phase_map.ndim == 2:
            # reshape to 3D
            new_dim = self.get_attribute('dimension', 'CellData')
            if len(new_dim) == 3:
                phase_map = phase_map.reshape(new_dim)
            else:
                phase_map = phase_map.reshape(
                    (phase_map.shape[0], phase_map.shape[1], 1))
        return phase_map

    def get_orientation_map(self):
        """Get the orientation map as a numpy array.

        The orientation map is an array of triplets representing orientation
        data for each voxel in the forme of rodrigues vectors.

        :return: the orientation map as a numpy array.
        """
        orientation_map = self.get_field('orientation_map')
        if self._is_empty('orientation_map'):
            orientation_map = None
        elif orientation_map.ndim == 3:
            # case (nx, ny, 3)
            new_dim = self.get_attribute('dimension', 'CellData')
            if len(new_dim) == 3:
                orientation_map = orientation_map.reshape(new_dim)
            else:
                orientation_map = orientation_map.reshape(
                    (orientation_map.shape[0], orientation_map.shape[1], 1, 3))
        return orientation_map

    def get_mask(self):
        """Get the mask as a numpy array.

        The mask represent the sample outline. The value 1 means we are inside
        the sample, the value 0 means we are outside the sample.

        :return: the mask as a numpy array.
        """
        mask = self.get_field('mask')
        if self._is_empty('mask'):
            mask = None
        elif mask.ndim == 2:
            # reshape to 3D
            new_dim = self.get_attribute('dimension', 'CellData')
            if len(new_dim) == 3:
                mask = mask.reshape(new_dim)
            else:
                mask = mask.reshape(
                    (mask.shape[0], mask.shape[1], 1))
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

    @staticmethod
    def id_list_to_condition(id_list):
        """Convert a list of id to a condition to filter the grain table.

        The condition will be interpreted using Numexpr typically using
        a `read_where` call on the grain data table.

        :param list id_list: a non empty list of the grain ids.
        :return: the condition as a string .
        """
        if not len(id_list) > 0:
            raise ValueError('the list of grain ids must not be empty')
        condition = "\'(idnumber == %d)" % id_list[0]
        for grain_id in id_list[1:]:
            condition += " | (idnumber == %d)" % grain_id
        condition += "\'"
        return condition

    def get_grain_volumes(self, id_list=None):
        """Get the grain volumes.

        The grain data table is queried and the volumes of the grains are
        returned in a single array. An optional list of grain ids can be used
        to restrict the grains, by default all the grain volumes are returned.

        :param list id_list: a non empty list of the grain ids.
        :return: a numpy array containing the grain volumes.
        """
        if id_list is not None:
            condition = Microstructure.id_list_to_condition(id_list)
            return self.grains.read_where(eval(condition))['volume']
        else:
            return self.get_tablecol('GrainDataTable', 'volume')

    def get_grain_centers(self, id_list=None):
        """Get the grain centers.

        The grain data table is queried and the centers of the grains are
        returned in a single array. An optional list of grain ids can be used
        to restrict the grains, by default all the grain centers are returned.

        :param list id_list: a non empty list of the grain ids.
        :return: a numpy array containing the grain centers.
        """
        if id_list is not None:
            condition = Microstructure.id_list_to_condition(id_list)
            return self.grains.read_where(eval(condition))['center']
        else:
            return self.get_tablecol('GrainDataTable', 'center')

    def get_grain_rodrigues(self, id_list=None):
        """Get the grain rodrigues vectors.

        The grain data table is queried and the rodrigues vectors of the grains
        are returned in a single array. An optional list of grain ids can be
        used to restrict the grains, by default all the grain rodrigues vectors
        are returned.

        :param list id_list: a non empty list of the grain ids.
        :return: a numpy array containing the grain rodrigues vectors.
        """
        if id_list is not None:
            condition = Microstructure.id_list_to_condition(id_list)
            return self.grains.read_where(eval(condition))['orientation']
        else:
            return self.get_tablecol('GrainDataTable', 'orientation')

    def get_grain_orientations(self, id_list=None):
        """Get a list of the grain orientations.

        The grain data table is queried to retreiv the rodrigues vectors.
        An optional list of grain ids can be used to restrict the grains.
        A list of `Orientation` instances is then created and returned.

        :param list id_list: a non empty list of the grain ids.
        :return: a list of the grain orientations.
        """
        rods = self.get_grain_rodrigues(id_list)
        orientations = [Orientation.from_rodrigues(rod) for rod in rods]
        return orientations

    def get_grain_bounding_boxes(self, id_list=None):
        """Get the grain bounding boxes.

        The grain data table is queried and the bounding boxes of the grains
        are returned in a single array. An optional list of grain ids can be
        used to restrict the grains, by default all the grain bounding boxes
        are returned.

        .. note::

          The bounding boxes are returned in ascending order of the grain ids
          (not necessary the same order than the list if it is not ordered).
          The maximum length of the ids list is 256.

        :param list id_list: a non empty (preferably ordered) list of the
        selected grain ids (with a maximum number of ids of 256).
        :return: a numpy array containing the grain bounding boxes.
        :raise: a ValueError if the length of the id list is larger than 256.
        """
        if id_list is not None:
            if len(id_list) > 256:
                raise(ValueError("the id_list can only have 256 values"))
            condition = Microstructure.id_list_to_condition(id_list)
            return self.grains.read_where(eval(condition))['bounding_box']
        else:
            return self.get_tablecol('GrainDataTable', 'bounding_box')

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
            total_volume = np.sum(self.get_grain_volumes())
        volume_fraction = self.get_grain_volumes(id_list=[gid])[0] / total_volume
        return volume_fraction

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

    def set_grain_phases(self, phases):
        """ Store grain phases array in GrainDataTable

            phases : 1D array of the phase id for each grain.
        """
        self.set_tablecol('GrainDataTable', 'phase', column=phases)
        return

    def set_lattice(self, lattice, phase_id=1):
        """Set the crystallographic lattice associated with this microstructure.

        If no `phase_id` is specified, the lattice will be set for the first
        phase of the list.

        :param Lattice lattice: an instance of the `Lattice class`.
        :param int phase_id: the id of the phase to set the lattice.
        """
        self.get_phase(phase_id)._lattice = lattice

    def set_active_grain_map(self, map_name='grain_map'):
        """Set the active grain map name.

        The `active_grain_map` string attribute is used to locate the array
        when the `get_grain_map` method is called. This allows to have multiple
        grain maps within a data set.
        """
        self.active_grain_map = map_name
        self.add_attributes({'active_grain_map': map_name}, 'CellData')
        return

    def set_active_phase_map(self, map_name='phase_map'):
        """Set the active phase map name.

        The `active_phase_map` string attribute is used to locate the array
        when the `get_phase_map` method is called. This allows to have multiple
        grain maps within a data set.
        """
        self.active_phase_map = map_name
        self.add_attributes({'active_phase_map': map_name}, 'CellData')
        return

    def set_grain_map(self, grain_map, voxel_size=None,
                      map_name='grain_map', compression=None):
        """Set the grain map for this microstructure.

        :param ndarray grain_map: a 2D or 3D numpy array.
        :param float voxel_size: the size of the voxels in mm unit. Used only
            if the CellData image Node must be created.
        """
        if compression is None:
            compression = self.default_compression_options
        create_image = True
        if self.__contains__('CellData'):
            empty = self.get_attribute(attrname='empty', nodename='CellData')
            if not empty:
                create_image = False
        if create_image:
            if voxel_size is None:
                msg = 'Please specify voxel size for CellData image'
                raise ValueError(msg)
            if np.isscalar(voxel_size):
                dim = len(grain_map.shape)
                spacing_array = voxel_size*np.ones((dim,))
            else:
                if len(voxel_size) != len(grain_map.shape):
                    raise ValueError('voxel_size array must have a length '
                                     'equal to grain_map shape')
                spacing_array = voxel_size
            self.add_image_from_field(field_array=grain_map,
                                      fieldname=map_name,
                                      imagename='CellData', location='/',
                                      spacing=spacing_array,
                                      replace=True,
                                      compression_options=compression)
        else:
            # Handle case of a 2D Microstrucutre: squeeze grain map to
            # ensure (Nx,Ny,1) array will be stored as (Nx,Ny)
            if self._get_group_type('CellData') == '2DImage':
                grain_map = grain_map.squeeze()
            self.add_field(gridname='CellData', fieldname=map_name,
                           array=grain_map, replace=True,
                           compression_options=compression)
        self.set_active_grain_map(map_name)
        return

    def set_phase_map(self, phase_map, voxel_size=None, map_name='phase_map',
                      compression=None):
        """Set the phase map for this microstructure.

        :param ndarray phase_map: a 2D or 3D numpy array.
        :param float voxel_size: the size of the voxels in mm unit. Used only
            if the CellData image Node must be created.
        """
        if compression is None:
            compression = self.default_compression_options
        create_image = True
        if self.__contains__('CellData'):
            empty = self.get_attribute(attrname='empty', nodename='CellData')
            if not empty:
                create_image = False
        if create_image:
            if voxel_size is None:
                msg = 'Please specify voxel size for CellData image'
                raise ValueError(msg)
            if np.isscalar(voxel_size):
                dim = len(phase_map.shape)
                spacing_array = voxel_size*np.ones((dim,))
            else:
                if len(voxel_size) != len(phase_map.shape):
                    raise ValueError('voxel_size array must have a length '
                                     'equal to grain_map shape')
                spacing_array = voxel_size
            self.add_image_from_field(field_array=phase_map,
                                      fieldname=map_name,
                                      imagename='CellData', location='/',
                                      spacing=spacing_array,
                                      replace=True,
                                      compression_options=compression)
        else:
            self.add_field(gridname='CellData', fieldname=map_name,
                           array=phase_map, replace=True,
                           indexname='phase_map',
                           compression_options=compression)
        self.set_active_phase_map(map_name)
        return

    def update_phase_map_from_grains(self, grain_ids=None):
        """Update the phase map from the grain map.

        This method update the phase map by setting the phase of all
        voxels of the grains to the appropriate value.

        :param list grain_ids: the list of the grains ids that are
            concerned by the update (all grain by default).
        :param int phase_id: phase Id to set for the list of grains concerned
            by the update.
        """
        # TODO: check existence of grain map
        phase_map = self.get_phase_map()
        # handle case of empty phase map
        if phase_map is None:
            map_shape = self.get_attribute('dimension', 'CellData')
            phase_map = np.zeros(shape=map_shape, dtype=np.uint8)
        if not grain_ids:
            grain_ids = self.get_ids_from_grain_map()
        time.sleep(0.2)
        for gid in tqdm(grain_ids, desc='updating phase map'):
            g = self.grains.read_where('idnumber == %d' % gid)[0]
            bb = g['bounding_box']
            phase_id = g['phase']
            this_grain_map = self.get_grain_map()[bb[0][0]:bb[0][1],
                                                  bb[1][0]:bb[1][1],
                                                  bb[2][0]:bb[2][1]]
            phase_map[bb[0][0]:bb[0][1], bb[1][0]:bb[1][1],
                      bb[2][0]:bb[2][1]][this_grain_map == gid] = phase_id
        self.set_phase_map(phase_map)

    def set_orientation_map(self, orientation_map, compression=None):
        """Set the orientation_map map for this microstructure.

        The orientation map is an array containing the voxel wise orientation
        data in form of Rodigues vectors.

        :param ndarray orientation_map: a 2D or 3D numpy array with rodrigues
        vectors at each pixels. The size of the array must be compatible with
        the `CellData` node image dimensions.
        """
        if compression is None:
            compression = self.default_compression_options
        dims = orientation_map.shape
        cell_data_dims = self.get_attribute('dimension', 'CellData')
        if not np.all(np.equal(dims[:-1], cell_data_dims)):
            print('warning: the orientation map shape %s is not compatible with '
                  'CellData node shape %s' % (dims, cell_data_dims))
            return None
        self.add_field(gridname='CellData', fieldname='orientation_map',
                       array=orientation_map, replace=True,
                       indexname='orientation_map',
                       compression_options=compression)

    def set_mask(self, mask, voxel_size=None, compression=None):
        """Set the mask for this microstructure.

        :param ndarray mask: a 2D or 3D numpy array.
        :param float voxel_size: the size of the voxels in mm unit. Used only
            if the CellData image Node must be created.
        """
        if compression is None:
            compression = self.default_compression_options
        create_image = True
        if self.__contains__('CellData'):
            empty = self.get_attribute(attrname='empty', nodename='CellData')
            if not empty:
                create_image = False
        if mask.dtype == 'bool':
            # use uint8 encoding
            mask = mask.astype(np.uint8)
        if create_image:
            if voxel_size is None:
                msg = 'Please specify voxel size for CellData image'
                raise ValueError(msg)
            if np.isscalar(voxel_size):
                dim = len(mask.shape)
                spacing_array = voxel_size * np.ones((dim, ))
            else:
                if len(voxel_size) != len(mask.shape):
                    raise ValueError('voxel_size array must have a length '
                                     'equal to grain_map shape')
                spacing_array = voxel_size
            self.add_image_from_field(mask, 'mask',
                                      imagename='CellData', location='/',
                                      spacing=spacing_array,
                                      replace=True,
                                      compression_options=compression)
        else:
            self.add_field(gridname='CellData', fieldname='mask',
                           array=mask, replace=True, indexname='mask',
                           compression_options=compression)
        return

    def set_random_orientations(self):
        """ Set random orientations for all grains in GrainDataTable """
        for grain in self.grains:
            o = Orientation.random()
            grain['orientation'] = o.rod
            grain.update()
        self.grains.flush()
        return

    def remove_grains_not_in_map(self):
        """Remove from GrainDataTable grains that are not in the grain map."""
        _, not_in_map, _ = self.compute_grains_map_table_intersection()
        self.remove_grains_from_table(not_in_map)
        return

    def remove_small_grains(self, min_volume=1.0, sync_table=False,
                            new_grain_map_name=None):
        """Remove from grain_map and grain data table small volume grains.

        Removed grains in grain map will be replaced by background ID (0).
        To be sure that the method acts consistently with the current grain
        map, activate sync_table options.

        :param float min_volume: Grains whose volume is under or equal to this
            value willl be suppressed from grain_map and grain data table.
        :param bool sync_table: If `True`, synchronize gran data table with
            grain map before removing grains.
        :param str new_grain_map_name: If provided, store the new grain map
            with removed grain with this new name. If not, overright  the
            current active grain map
        """
        if sync_table and not self._is_empty('grain_map'):
            self.sync_grain_table_with_grain_map(sync_geometry=True)
        condition = f"(volume <= {min_volume})"
        id_list = self.grains.read_where(condition)['idnumber']
        if not self._is_empty('grain_map'):
            # Remove grains from grain map
            grain_map = self.get_grain_map()
            grain_map[np.where(np.isin(grain_map, id_list))] = 0
            if new_grain_map_name is not None:
                map_name = new_grain_map_name
            else:
                map_name = self.active_grain_map
            self.set_grain_map(grain_map.squeeze(), map_name=map_name)
        # Remove grains from table
        self.remove_grains_from_table(id_list)
        return

    def remove_grains_from_table(self, ids):
        """Remove from GrainDataTable the grains with given ids.

        :param ids: Array of grain ids to remove from GrainDataTable
        :type ids: list
        """
        for Id in ids:
            where = self.grains.get_where_list('idnumber == Id')[:]
            self.grains.remove_row(int(where))
        return

    def add_grains(self, orientation_list, orientation_type='euler',
                   grain_ids=None):
        """Add a list of grains to this microstructure.

        This function adds a list of grains represented by their orientation
        (either a list of Euler angles or Rodrigues vectors) to the
        microstructure. If provided, the `grain_ids` list will be used for
        the grain ids.

        :param list orientation_list: a list of values representing the orientations.
        :param str orientation_type: euler or rod for Euler angles (Bunge
        passive convention and degrees) or Rodrigues vectors.
        :param list grain_ids: an optional list for the ids of the new grains.
        """
        grain = self.grains.row
        # build a list of grain ids if it is not given
        if grain_ids is None:
            if self.get_number_of_grains() > 0:
                min_id = max(self.get_grain_ids()) + 1
            else:
                min_id = 0
            grain_ids = range(min_id, min_id + len(orientation_list))
        if len(grain_ids) > 0:
            s = 's' if len(grain_ids) > 1 else ''
            print(f'adding {len(grain_ids)} grain{s} to the microstructure')
        for gid, orientation in zip(grain_ids, orientation_list):
            grain['idnumber'] = gid
            if orientation_type == 'euler':
                grain['orientation'] = Orientation.Euler2Rodrigues(orientation)
            elif orientation_type in ['rod', 'rodrigues']:
                grain['orientation'] = orientation
            else:
                raise ValueError('unknown type of orientation: %s' % orientation_type)
            grain.append()
        self.grains.flush()

    def add_grains_in_map(self):
        """Add to GrainDataTable the grains in grain map missing in table.

        The grains are added with a random orientation by convention.
        """
        _, _, not_in_table = self.compute_grains_map_table_intersection()
        # remove ID <0 from list (reserved to background)
        not_in_table = np.delete(not_in_table, np.where(not_in_table <= 0))
        # generate random euler angles
        phi1 = np.random.rand(len(not_in_table), 1) * 360.
        Phi = 180. * np.arccos(2 * np.random.rand(len(not_in_table), 1) - 1) / np.pi
        phi2 = np.random.rand(len(not_in_table), 1) * 360.
        euler_list = np.concatenate((phi1, Phi, phi2), axis=1)
        self.add_grains(euler_list, orientation_type='euler', grain_ids=not_in_table)
        return

    @staticmethod
    def random_texture(n=100, phase=None):
        """Generate a random texture microstructure.

        :param int n: the number of grain orientations in the microstructure.
        :param CrystallinePhase phase: the phase to use for this microstructure.
        :return: a `Microstructure` instance with n randomly oriented grains.
        """
        m = Microstructure(name='random_texture', phase=phase,
                           overwrite_hdf5=True)
        grain = m.grains.row
        for i in range(n):
            grain['idnumber'] = i + 1
            o = Orientation.random()
            grain['orientation'] = o.rod
            grain.append()
        m.grains.flush()
        return m

    def set_mesh(self, mesh_object=None, file=None, meshname='micro_mesh'):
        """Add a mesh of the microstructure to the dataset.

        Mesh can be input as a BasicTools mesh object or as a mesh file.
        Handled file format for now only include .geof (Zset software format).
        """
        self.add_mesh(mesh_object=mesh_object, meshname=meshname,
                      location='/MeshData', replace=True, file=file)
        return

    def create_grain_ids_field(self, meshname=None, elset_prefix='grain_',
                               store=True):
        """Create a grain Id field of grain orientations on the input mesh.

        Creates a element wise field from the microsctructure mesh provided,
        adding to each element the value of the grain id of the local grain
        element set, as it is and if it is referenced in the `GrainDataTable`
        node.

        .. note::

          The grain elsets must be named with a prefix directly followed by the
          grain number. This means that if the elsets are named `grain_1`,
          `grain_2`, etc, the variable `elset_prefix' must be set to `grain_`.

        :param str meshname: Name, Path or index name of the mesh on which an
            orientation map element field must be constructed
        :param str elset_prefix: prefix to define the element sets representing
            the grains.
        :param bool store: If `True`, store the grain ids field in `MeshData`
            group, with name `grain_ids`
        """
        if meshname is None:
            raise ValueError('mesh_name do not refer to an existing mesh')
        if not self._is_mesh(meshname) or self._is_empty(meshname):
            raise ValueError('mesh_name do not refer to a non empty mesh group')
        # create empty element vector field
        n_elements = int(self.get_attribute('Number_of_elements', meshname))
        mesh = self.get_node(meshname)
        el_tag_path = '%s/Geometry/ElementsTags' % mesh._v_pathname
        grain_id_field = np.zeros((n_elements, 1), dtype=int)
        grain_ids = self.get_grain_ids()
        # if mesh is provided
        for i in range(len(grain_ids)):
            set_name = '%s%d' % (elset_prefix, grain_ids[i])
            print('using elset name %s' % set_name)
            elset_path = '%s/%s' % (el_tag_path, set_name)
            element_ids = self.get_node(elset_path, as_numpy=True).astype(int)
            grain_id_field[element_ids == 1] = grain_ids[i]
        if store:
            self.add_field(gridname=meshname, fieldname='grain_ids',
                           array=grain_id_field, replace=True)
        return grain_id_field

    def create_orientation_field(self, mesh_name=None, elset_prefix='grain_',
                                 store=True):
        """Create a vector field of grain orientations on the input mesh.

        This method creates a element wise field on the microstructure mesh
        indicated, adding to each element the value of the Rodrigues vector of
        this grain as referenced in the `GrainDataTable` node.

        :param str mesh_name: Name, Path or index name of the mesh on which an
            orientation field must be constructed
        :param str elset_prefix: prefix to define the element sets representing
            the grains.
        :param bool store: If `True`, store the orientation field in corresponding
            mesh group, with name `orientation_field`.
        """
        if mesh_name is None:
            raise ValueError('mesh_name do not refer to an existing mesh')
        if not(self._is_mesh(mesh_name)) or self._is_empty(mesh_name):
            raise ValueError('mesh_name do not refer to a non empty mesh group')
        # create empty element vector field
        n_elements = int(self.get_attribute('Number_of_elements', mesh_name))
        mesh = self.get_node(mesh_name)
        el_tag_path = '%s/Geometry/ElementsTags' % mesh._v_pathname
        orientation_field = np.zeros((n_elements, 3), dtype=float)
        grain_ids = self.get_grain_ids()
        grain_orientations = self.get_grain_rodrigues()
        print(grain_orientations)
        # if mesh is provided
        for i in range(len(grain_ids)):
            set_name = '%s%d' % (elset_prefix, grain_ids[i])
            print('using elset name %s' % set_name, grain_orientations[i, :])
            elset_path = '%s/%s' % (el_tag_path, set_name)
            element_ids = self.get_node(elset_path, as_numpy=True).astype(int)
            orientation_field[np.squeeze(element_ids) == 1, :] = grain_orientations[i, :]
        if store:
            self.add_field(gridname=mesh_name, fieldname='orientation_field',
                           array=orientation_field, replace=True)
        return orientation_field

    def create_orientation_map(self, store=True):
        """Create a vector field in CellData of grain orientations.

        Creates a (Nx, Ny, Nz, 3) or (Nx, Ny, 3) field from the microstructure
        `grain_map`, adding to each voxel the value of the Rodrigues vector
        of the local grain Id, as it is and if it is referenced in the
        `GrainDataTable` node.

        :param bool store: If `True`, store the orientation map in `CellData`
            image group, with name `orientation_map`
        """
        # safety check
        if self._is_empty(self.active_grain_map):
            msg = 'The microstructure instance has no associated grain_map. ' \
                  'Cannot create orientation map.'
            raise RuntimeError(msg)
        grain_map = self.get_grain_map()
        grain_ids = self.get_grain_ids()
        grain_orientations = self.get_grain_rodrigues()
        # safety check 2
        grain_list = np.unique(grain_map)
        # remove -1 and 0 from the list of grains in grain map (Ids reserved
        # for background and overlaps in non-dilated reconstructed grain maps)
        grain_list = np.delete(grain_list, np.isin(grain_list, [-1, 0]))
        if not np.all(np.isin(grain_list, grain_ids)):
            msg = 'Some grain ids in the grain_map are not referenced in the ' \
                  '`GrainDataTable` array. Cannot create orientation map.'
            raise ValueError(msg)
        # create empty orientation map with right dimensions
        im_dim = tuple(self.get_attribute('dimension', 'CellData'))
        print(im_dim)
        shape_orientation_map = im_dim + (3,)
        print(shape_orientation_map)
        orientation_map = np.zeros(shape=shape_orientation_map, dtype=float)
        for i in range(len(grain_ids)):
            #TODO use grain bounding box here
            #slc = np.where(grain_map == grain_ids[i])
            orientation_map[grain_map == grain_ids[i], 0] = grain_orientations[i, 0]
            orientation_map[grain_map == grain_ids[i], 1] = grain_orientations[i, 1]
            orientation_map[grain_map == grain_ids[i], 2] = grain_orientations[i, 2]
            #orientation_map[slc, 1] = grain_orientations[i, 1]
            #orientation_map[slc, 2] = grain_orientations[i, 2]
        if store:
            self.add_field(gridname='CellData', fieldname='orientation_map',
                           array=orientation_map, replace=True,
                           location='CellData')
        return orientation_map

    def add_grain_lattices_representation(self):
        """Add a mesh with one cell per grain representing the crystal lattices.

        This function create a mesh with one cell per grain. Each cell is a hexahedron
        shaped as the crystal lattice and sclaed by the grain size. A scalar field 
        of the grain ids is also attached to the mesh.

        :note: The case of hexagonal lattice is handled by putting together 3 hexahedrons 
        to build a hexagonal prism since the VTK_HEXAGONAL_PRISM topology is not presently
        supported by the XDMF format.
        """
        lattice = self.get_phase().get_lattice()
        coords, edges, faces = lattice.get_points(origin='mid', handle_hexagonal=False)
        grain_ids = self.get_grain_ids()
        grain_sizes = self.compute_grain_equivalent_diameters()  #self.get_grain_volumes()
        centers = self.get_grain_centers()
        n = self.get_number_of_grains()

        # local function to insert a new cell representing one lattice
        def insert_grain_lattice_cell(grid, id_offset, om, coords, center, size):
            coords_rot = np.empty_like(coords)
            assert len(coords) == 8
            Ids = vtk.vtkIdList()
            hexahedron_order = [0, 1, 3, 2, 4, 5, 7, 6]
            for k, coord in enumerate(coords):
                # scale coordinates with the grain size and center on the grain
                coords_rot[k] = center + size * np.dot(om.T, coord)
                points.InsertNextPoint(coords_rot[k])
                Ids.InsertNextId(id_offset + hexahedron_order[k])
            grid.InsertNextCell(vtk.VTK_HEXAHEDRON, Ids)

        # vtkPoints instance for all the vertices
        points = vtk.vtkPoints()
        # vtkUnstructuredGrid instance for all the cells
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(points)
        if lattice.get_symmetry() is Symmetry.hexagonal:
            grid.Allocate(3 * n, 1)
        else:
            grid.Allocate(n, 1)
        id_offset = 0

        for g in tqdm(self.grains, desc='creating lattice cell for all grains'):
            gid = g['idnumber']
            center = g['center']
            size = self.compute_grain_equivalent_diameters(id_list=[gid])[0]  #g['volume']
            om = self.get_grain(gid).orientation.orientation_matrix()
            insert_grain_lattice_cell(grid, id_offset, om, coords, center, size)
            id_offset += len(coords)
            if lattice.get_symmetry() is Symmetry.hexagonal:
                # handle hexagonal case by creating 3 hexahedrons
                euler = self.get_grain(gid).orientation.euler
                om = Orientation.from_euler(euler + np.array([0., 0. , 120.])).orientation_matrix()
                insert_grain_lattice_cell(grid, id_offset, om, coords, center, size)
                id_offset += len(coords)
                om = Orientation.from_euler(euler + np.array([0., 0. , 240.])).orientation_matrix()
                insert_grain_lattice_cell(grid, id_offset, om, coords, center, size)
                id_offset += len(coords)

        from vtk.util import numpy_support
        grain_ids_array = numpy_support.numpy_to_vtk(grain_ids)
        grain_sizes_array = numpy_support.numpy_to_vtk(grain_sizes)
        if lattice.get_symmetry() is Symmetry.hexagonal:
            grain_ids_array = numpy_support.numpy_to_vtk(np.repeat(grain_ids, 3))
            grain_sizes_array = numpy_support.numpy_to_vtk(np.repeat(grain_sizes, 3))
        grain_ids_array.SetName('grain_ids')
        grain_sizes_array.SetName('grain_sizes')
        grid.GetCellData().AddArray(grain_ids_array)
        grid.GetCellData().AddArray(grain_sizes_array)

        # now add the created mesh to the microstructure
        from BasicTools.Containers import vtkBridge
        mesh = vtkBridge.VtkToMesh(grid)
        self.add_mesh(mesh_object=mesh, location='/MeshData', meshname='grain_lattices',
                      indexname='mesh_grain_lattices', replace=True)
        
    def fz_grain_orientation_data(self, grain_id, plot=True, move_to_fz=True):
        """Plot the orientation data for this grain.

        This function extracts orientation data for a given grain and creates
        a three dimensional plot in Rodrigues space.

        :param int grain_id: the grain id to retrieve orientation data.
        :param bool plot: flag to create a 3D plot. If False, the Rodrigues
        orientation data is simply returned.
        :param bool plot: flag to move the orientation data to the fundamental
        zone (the crystal lattice of the microstructure is used for that).
        :return: a numpy array of size (n, 3) with n being the number of data
        points for this grain.
        """
        orientation_map = self.get_orientation_map()
        rods_gid = orientation_map[np.where(self.get_grain_map() == grain_id)]
        sym = self.get_lattice().get_symmetry()

        if move_to_fz:
            # move to the fundamental zone
            for i in range(len(rods_gid)):
                g = Orientation.from_rodrigues(rods_gid[i]).orientation_matrix()
                g_fz = sym.move_rotation_to_FZ(g, verbose=False)
                o_fz = Orientation(g_fz)
                rods_gid[i] = o_fz.rod

        if plot:
            # plot orientation data in Rodrigues space
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(rods_gid[:, 0], rods_gid[:, 1], rods_gid[:, 2])
            plt.show()
        return rods_gid

    def compute_god_map(self, id_list=None, store=True,
                        recompute_mean_orientation=False):
        """Create a GOD (grain orientation deviation) map.

        This method computes the grain orientation deviation map. For each
        grain in the list (all grain by default), the orientation of each
        voxel belonging to this grain is compared to the mean orientation in
        the grain and the resulting misorientation is assigned to the pixel.

        A grain ids list can be used to restrict the grains where to compute
        the orientation deviation (the number of grains in the list must be 256
        maximum). By default, this method uses the mean orientation in the
        GrainDataTable but the mean orientation can also be recomputed from the
        orientation map and grain map by activating the flag
        `recompute_mean_orientation`.

        .. note::

          This method needs both a grain map and an orientation map, a message
          will be displayed if this is not the case.

        :param list id_list: the list of the grain ids to include (compute
        for all grains by default).
        :param bool recompute_mean_orientation: if `True` the mean grain
        orientation is recalculated from the orientatin map instead of using
        the value in the `GrainDataTable`.
        :param bool store: If `True`, store the grain orientation deviation map
        in the `CellData` group, with name `grain_orientation_deviation`.
        """
        if self._is_empty('grain_map'):
            print('no grain map found, please add a grain map to your data set')
            return None
        elif self._is_empty('orientation_map'):
            print('no orientation map found, please add an orientation map to your data set')
            return None
        grain_ids = self.get_grain_map()
        print('grain ids shape', grain_ids.shape)
        orientation_map = self.get_orientation_map()
        print('orientation map shape', orientation_map.shape)
        # assume only one phase
        if self.get_number_of_phases() > 1:
            print('error, multiple phases not yet supported')
            return None
        sym = self.get_phase().get_symmetry()
        god = np.zeros_like(grain_ids, dtype=float)
        if not id_list:
            id_list = np.unique(self.get_ids_from_grain_map())
        if not recompute_mean_orientation:
            # verify that all required grain ids are present in the GrainDataTable
            all_grain_in_table = np.all([gid in self.get_grain_ids() for gid in id_list])
            if not all_grain_in_table:
                print('warning not all grains present in the grain map have an '
                      'entry in the grain data table, the GOD map cannot be '
                      'computed for the requested grains. Consider using the '
                      'option `recompute_mean_orientation=True` or restrict the '
                      'list of grains using the argument `id_list`.')
            else:
                print('all grains are present in the GrainDataTable')
        for index, gid in enumerate(id_list):
            if gid < 1:
                continue
            progress = 100 * index / len(id_list)
            print('GOD computation progress: {:.2f} % (grain {:d})'.format(progress, gid), end='\r')
            # we use grain bounding boxes to speed up calculations
            bb = self.get_grain_bounding_boxes(id_list=[gid])[0]
            grain_map = self.get_grain_map()[bb[0][0]:bb[0][1],
                                             bb[1][0]:bb[1][1],
                                             bb[2][0]:bb[2][1]]
            indices = np.where(grain_map == gid)
            if recompute_mean_orientation:
                # compute the mean orientation of this grain
                rods_gid = np.squeeze(orientation_map[bb[0][0]:bb[0][1],
                                                      bb[1][0]:bb[1][1],
                                                      bb[2][0]:bb[2][1]][grain_map == gid])
                #print('\nrods_gid shape', rods_gid.shape)
                o = Orientation.compute_mean_orientation(rods_gid, symmetry=sym)
            else:
                o = self.get_grain(gid).orientation
                #print('mean grain orientation:', o.rod)
            # now compute the orientation deviation for each pixel of the grain
            for i, j, k in zip(indices[0], indices[1], indices[2]):
                ii, jj, kk = bb[0][0] + i, bb[1][0] + j, bb[2][0] + k
                rod_ijk = orientation_map[ii, jj, kk, :]
                o_ijk = Orientation.from_rodrigues(rod_ijk)
                god[ii, jj, kk] = np.degrees(o.disorientation(o_ijk, crystal_structure=sym)[0])
        print('GOD computation progress: 100.00 %')
        if store:
            # pick the location of the grain map to add the new field
            location = self._get_parent_name(self.active_grain_map)
            self.add_field(gridname=location, array=god, replace=True,
                           fieldname='grain_orientation_deviation')
        return god

    def add_IPF_maps(self):
        """Add IPF maps to the data set.

        IPF colors are computed for the 3 cartesian directions and stored into
        the h5 file in the `CellData` image group, with names `ipf_map_100`,
        `ipf_map_010`, `ipf_map_001`.
        """
        ipf100 = self.create_IPF_map(axis=np.array([1., 0., 0.]))
        self.add_field(gridname='CellData', fieldname='ipf_map_100',
                       array=ipf100, replace=True,
                       compression_options=self.default_compression_options)
        ipf010 = self.create_IPF_map(axis=np.array([0., 1., 0.]))
        self.add_field(gridname='CellData', fieldname='ipf_map_010',
                       array=ipf010, replace=True,
                       compression_options=self.default_compression_options)
        ipf001 = self.create_IPF_map(axis=np.array([0., 0., 1.]))
        self.add_field(gridname='CellData', fieldname='ipf_map_001',
                       array=ipf001, replace=True,
                       compression_options=self.default_compression_options)
        del ipf100, ipf010, ipf001

    def create_IPF_map(self, axis=np.array([0., 0., 1.])):
        """Create a vector field in CellData to store the IPF colors.

        Creates a (Nx, Ny, Nz, 3) field with the IPF color for each voxel.
        Note that this function assumes a single orientation per grain.

        :param axis: the unit vector for the load direction to compute IPF
            colors.
        """
        grain_map = self.get_grain_map()
        dims = list(grain_map.shape)
        shape_ipf_map = list(dims) + [3]
        ipf_map = np.zeros(shape=shape_ipf_map, dtype=np.float32)
        for g in tqdm(self.grains, desc='computing IPF map'):
            gid = g['idnumber']
            # use the bounding box for this grain
            bb = g['bounding_box']
            this_grain_map = grain_map[bb[0][0]:bb[0][1],
                                       bb[1][0]:bb[1][1],
                                       bb[2][0]:bb[2][1]]
            o = Orientation.from_rodrigues(g['orientation'])
            sym = self.get_phase(g['phase']).get_symmetry()
            rgb = o.ipf_color(axis, symmetry=sym, saturate=True)
            ipf_map[bb[0][0]:bb[0][1],
                    bb[1][0]:bb[1][1],
                    bb[2][0]:bb[2][1]][this_grain_map == gid] = rgb
        return ipf_map.squeeze()

    def view_slice(self, **kwargs):
        """A simple utility method to show one microstructure slice.

        Refer to the view_map_slice methode in `pymicro.crystal.view`module 
        for the kwargs definition. 
        """
        from pymicro.crystal.view import View_slice
        view = View_slice(self)
        if 'unit' in kwargs:
            view.set_unit(kwargs['unit'])
            kwargs.pop('unit')
        if 'plane' in kwargs:
            view.set_plane(kwargs['plane'])
            kwargs.pop('plane')
        fig, ax = view.view_map_slice(**kwargs)
        return fig, ax

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
        Return a colormap with ipf colors for each grain.

        :return: a color map that can be directly used in pyplot.
        """
        ipf_colors = np.zeros((self.get_number_of_grains(), 3))
        for grain in self.grains:
            o = Orientation.from_rodrigues(grain['orientation'])
            sym = self.get_phase(phase_id=grain['phase']).get_symmetry()
            ipf_colors[grain['idnumber'], :] = o.ipf_color(symmetry=sym)
        return colors.ListedColormap(ipf_colors)

    @staticmethod
    def from_grain_file(grain_file_path, col_id=0, col_phi1=1, col_phi=2,
                        col_phi2=3, col_x=4, col_y=5, col_z=None,
                        col_volume=None, autodelete=True):
        """Create a `Microstructure` reading grain infos from a file.

        This file is typically created using EBSD. the usual pattern is:
        grain_id, phi1, phi, phi2, x, y, volume. The column number are tunable
        using the function arguments.
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
            if not (as_string):
                print(s)
        return s

    @staticmethod
    def match_grains(m1, m2, **kwargs):
        return micro1.match_grains(micro2, **kwargs)
        
    def match_grains(self, m2, mis_tol=1, 
                     grains_to_match=None, grains_to_search=None, 
                     use_centers=False, center_tol=0.1, scale_m2=1., 
                     offset_m2=None, center_merit=10., verbose=False):
        """Match grains from a second microstructure to this microstructure.

        This function try to find pair of grains based on a function of
        merit based on their orientations. This function can optionnally 
        use the center of the grains. In this case several paramaters 
        can be used to adjust the origin of the microstructure, their 
        scale and the relative merit of the center proximity with 
        respect to the orientation differences.

        .. warning::

        This function works only for microstructures with the same symmetry.

        :param m2: the second instance of `Microstructure` from which
            to match the grains.
        :param float mis_tol: the tolerance is misorientation to use
            to detect matches (in degrees).
        :param list use_grain_ids: a list of ids to restrict the grains
            of the first microstructure in which to search for matches.
        :param use_centers verbose: use the grain centers to build the 
            function of merit.
        :param float center_tol: the tolerance for 2 grains to be match 
            together (in mm).
        :param bool verbose: activate verbose mode.
        :raise ValueError: if the microstructures do not have the same symmetry.
        :return tuple: a tuple of three lists holding respectively the matches,
        the candidates for each match and the grains that were unmatched.
        """
        # TODO : Test
        if not (self.get_lattice().get_symmetry()
                == m2.get_lattice().get_symmetry()):
            raise ValueError('warning, microstructure should have the same '
                            'symmetry, got: {} and {}'.format(
                self.get_lattice().get_symmetry(),
                m2.get_lattice().get_symmetry()))
        candidates = []
        matched = []
        unmatched = []  # grain that were not matched within the given tolerance
        # restrict the grain ids to match and to search if needed
        sym = self.get_lattice().get_symmetry()
        if grains_to_match is None:
            grains_to_match = self.get_tablecol(tablename='GrainDataTable',
                                                colname='idnumber')
        if grains_to_search is None:
            grains_to_search = m2.get_tablecol(tablename='GrainDataTable',
                                               colname='idnumber')
        # look at each grain and compute a figure of merits
        for i, g1 in enumerate(self.grains):
            if not (g1['idnumber'] in grains_to_match):
                continue
            c1 = g1['center']
            cands_for_g1 = []
            best_merit = mis_tol
            if use_centers:
                best_merit *= (center_tol * center_merit)
            best_match = -1
            o1 = Orientation.from_rodrigues(g1['orientation'])
            for g2 in m2.grains:
                if not (g2['idnumber'] in grains_to_search):
                    continue
                if use_centers:
                    c2 = g2['center'] * scale_m2 + offset_m2
                    if np.linalg.norm(c2 - c1) > center_tol:
                        continue
                o2 = Orientation.from_rodrigues(g2['orientation'])
                # compute disorientation
                mis, _, _ = o1.disorientation(o2, crystal_structure=sym)
                misd = np.degrees(mis)
                if misd < mis_tol:
                    if use_centers:
                        center_dif = np.linalg.norm(c2 - c1)
                    if verbose:
                        print('grain %3d -- candidate: %3d, misorientation:'
                            ' %.2f deg' % (g1['idnumber'], g2['idnumber'],
                                            misd))
                        if use_centers:
                            print('center difference: %.2f' % center_dif)
                    # add this grain to the list of candidates
                    cands_for_g1.append(g2['idnumber'])
                    merit = misd
                    if use_centers:
                        merit *= (center_dif * center_merit)
                    if merit < best_merit:
                        best_merit = merit
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

    def match_orientation(self, orientation, use_grain_ids=None):
        """Find the best match between an orientation and the grains from this
        microstructure.

        :param orientation: an instance of `Orientation` to match the grains.
        :param list use_grain_ids: a list of ids to restrict the grains
            in which to search for matches.
        :return tuple: the grain id of the best match and the misorientation.
        """
        sym = self.get_lattice().get_symmetry()
        if use_grain_ids is None:
            grains_to_match = self.get_tablecol(tablename='GrainDataTable',
                                                colname='idnumber')
        else:
            grains_to_match = use_grain_ids
        best_mis = 180.
        for g in self.grains:
            if not (g['idnumber'] in grains_to_match):
                continue
            o = Orientation.from_rodrigues(g['orientation'])
            # compute disorientation
            mis, _, _ = o.disorientation(orientation, crystal_structure=sym)
            mis_deg = np.degrees(mis)
            if mis_deg < best_mis:
                best_mis = mis_deg
                best_match = g['idnumber']
        return best_match, best_mis

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
        # get the bounding box around the grain
        bb = self.grains.read_where('idnumber == %d' % grain_id)['bounding_box'][0]
        grain_map = self.get_grain_map()[bb[0][0]:bb[0][1],
                                         bb[1][0]:bb[1][1],
                                         bb[2][0]:bb[2][1]]
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
            the dilation will be limited by it.
        """
        grain_map = self.get_grain_map()
        grain_volume_init = (grain_map == grain_id).sum()
        grain_data = grain_map == grain_id
        grain_data = ndimage.binary_dilation(grain_data,
                                             iterations=dilation_steps).astype(np.uint8)
        if use_mask and not self._is_empty('mask'):
            grain_data *= self.get_mask()
        grain_map[grain_data == 1] = grain_id
        grain_volume_final = (grain_map == grain_id).sum()
        print('grain %s was dilated by %d voxels' % (grain_id,
                                                     grain_volume_final - grain_volume_init))
        self.set_grain_map(grain_map, self.get_voxel_size(),
                           map_name=self.active_grain_map)
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
        :param list dilation_ids: a list to restrict the dilation to the given
            ids.
        :param ndarray struct: the structuring element to use (strong
            connectivity by default).
        :return: the dilated array.
        """
        if struct is None:
            struct = ndimage.morphology.generate_binary_structure(array.ndim, 1)
        assert struct.ndim == array.ndim
        # carry out dilation in iterative steps
        step = 0
        while True:
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

            dilation = np.zeros_like(X).astype(array.dtype)#np.int16)
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
                    # at least one neighboring voxel is non zero
                    counts = np.bincount(neighbours.flatten())[1:]  # do not consider zero
                    # find the most frequent value
                    dilation[i] = np.argmax(counts) + 1
                    #dilation[i] = min(neighbours[neighbours > 0])
            if array.ndim == 2:
                array[X, Y] = dilation
            else:
                array[X, Y, Z] = dilation
            print('dilation step %d done' % (step + 1))
            step = step + 1
            if step == dilation_steps:
                break
            if dilation_steps == -1:
                if not np.any(array == 0):
                    break
        return array

    def dilate_grains(self, dilation_steps=1, dilation_ids=None,
                      new_map_name='dilated_grain_map',
                      update_microstructure_properties=False):
        """Dilate grains to fill the gap between them.

        This function calls `dilate_labels` with the grain map of the
        microstructure. The grain properties and the phase map can be updated
        after the dilation by setting the `update_microstructure_properties`
        parameter to True.

        :param int dilation_steps: the number of dilation steps to
            apply to the grain map.
        :param list dilation_ids: a list to restrict the dilation to
            the given ids.
        :param str new_map_name: the name to use for the dilated grain map.
        :param bool update_microstructure_properties: a flag to update all
            grains properties and update the microstructure phase map.
        """
        if not self.__contains__('grain_map'):
            raise ValueError('microstructure %s must have an associated '
                             'grain_map ' % self.get_sample_name())
            return
        grain_map = self.get_grain_map().copy()
        # get rid of overlap regions flaged by -1
        grain_map[grain_map == -1] = 0

        if not self._is_empty('mask'):
            grain_map = Microstructure.dilate_labels(grain_map,
                                                     dilation_steps=dilation_steps,
                                                     mask=self.get_mask(),
                                                     dilation_ids=dilation_ids)
        else:
            grain_map = Microstructure.dilate_labels(grain_map,
                                                     dilation_steps=dilation_steps,
                                                     dilation_ids=dilation_ids)
        # finally assign the dilated grain map to the microstructure
        self.set_grain_map(grain_map, map_name=new_map_name)

        if update_microstructure_properties:
            self.recompute_grain_bounding_boxes()
            self.recompute_grain_centers()
            self.recompute_grain_volumes()
            # and update the phase map if necessary
            if not self._is_empty('phase_map'):
                self.update_phase_map_from_grains()

    def clean_grain_map(self,  new_map_name='grain_map_clean'):
        """Apply a morphological cleaning treatment to the active grain map.


        A Matlab morphological cleaner is called to smooth the morphology of
        the different IDs in the grain map.

        This cleaning treatment is typically used to improve the quality of a
        mesh produced from the grain_map, or improved image based
        mechanical modelisation techniques results, such as FFT-based
        computational homogenization of the polycrystalline microstructure.

          ..Warning::

              This method relies on the code of the `core.utils` and on Matlab
              code developed by F. Nguyen at the 'Centre des Matériaux, Mines
              Paris'. These tools and codes must be installed and referenced
              in the PATH of your workstation for this method to work. For
              more details, see the `utils` package.
        """
        from pymicro.core.utils.SDZsetUtils.SDmeshers import SDImageMesher
        Mesher = SDImageMesher(data=self)
        Mesher.morphological_image_cleaner(
            target_image_field=self.active_grain_map,
            clean_fieldname=new_map_name, replace=True)
        del Mesher
        self.set_active_grain_map(new_map_name)
        return

    def mesh_grain_map(self, mesher_opts=dict(), print_output=False):
        """ Create a 2D or 3D conformal mesh from the grain map.

        A Matlab multiphase_image mesher is called to create a conformal mesh
        of the grain map that is stored as a SampleData Mesh group in the
        MeshData Group of the Microstructure dataset. The mesh data will
        contain an element set per grain in the grain map.

          ..Warning::

              This method relies on the code of the `core.utils`, on Matlab
              code developed by F. Nguyen at the 'Centre des Matériaux, Mines
              Paris', on the Z-set software and the Mesh GEMS software.
              These tools and codes must be installed and referenced
              in the PATH of your workstation for this method to work. For
              more details, see the `utils` package.
        """
        from pymicro.core.utils.SDZsetUtils.SDmeshers import SDImageMesher
        Mesher = SDImageMesher(data=self)
        Mesher.multi_phase_mesher(
            multiphase_image_name=self.active_grain_map,
            meshname='MeshData', location='/', replace=True,
            bin_fields_from_sets=False, mesher_opts=mesher_opts,
            elset_id_field=True, print_output=print_output)
        del Mesher
        return

    def crop(self, x_start=None, x_end=None, y_start=None, y_end=None,
             z_start=None, z_end=None, crop_name=None, autodelete=False,
             recompute_geometry=True, verbose=False):
        """Crop the microstructure to create a new one.

        This method crops the CellData image group to a new microstructure,
        and adapts the GrainDataTable to the crop.

        :param int x_start: start value for slicing the first axis.
        :param int x_end: end value for slicing the first axis.
        :param int y_start: start value for slicing the second axis.
        :param int y_end: end value for slicing the second axis.
        :param int z_start: start value for slicing the third axis.
        :param int z_end: end value for slicing the third axis.
        :param str crop_name: the name for the cropped microstructure
            (the default is to append '_crop' to the initial name).
        :param bool autodelete: a flag to delete the microstructure files
            on the disk when it is not needed anymore.
        :param bool recompute_geometry: if `True` (default), recompute the
            grain centers, volumes, and bounding boxes in the cropped
            microstructure. Use `False` when using a crop that do not cut
            grains, for instance when cropping a microstructure within the
            mask, to avoid the heavy computational cost of the grain geometry
            data update.
        :param bool verbose: activate verbose mode.
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
        print('CROP: %s' % crop_name)
        # create new microstructure dataset
        micro_crop = Microstructure(name=crop_name, overwrite_hdf5=True,
                                    phase=self.get_phase(),
                                    autodelete=autodelete)
        if self.get_number_of_phases() > 1:
            for i in range(2, self.get_number_of_phases()):
                micro_crop.add_phase(self.get_phase(phase_id=i))
        micro_crop.default_compression_options = self.default_compression_options
        print('cropping microstructure to %s' % micro_crop.h5_file)
        # crop all CellData fields
        image_group = self.get_node('CellData')
        spacing = self.get_attribute('spacing', 'CellData')
        FIndex_path = '%s/Field_index' % image_group._v_pathname
        field_list = self.get_node(FIndex_path)
        for name in field_list:
            field_name = name.decode('utf-8')
            print('cropping field %s' % field_name)
            field = self.get_field(field_name)
            if not self._is_empty(field_name):
                if self._get_group_type('CellData') == '2DImage':
                    field_crop = field[x_start:x_end, y_start:y_end, ...]
                else:
                    field_crop = field[x_start:x_end, y_start:y_end,
                                       z_start:z_end, ...]
                empty = micro_crop.get_attribute(attrname='empty',
                                                 nodename='CellData')
                if empty:
                    micro_crop.add_image_from_field(
                        field_array=field_crop, fieldname=field_name,
                        imagename='CellData', location='/',
                        spacing=spacing, replace=True)
                else:
                    micro_crop.add_field(gridname='CellData',
                                         fieldname=field_name,
                                         array=field_crop, replace=True)
        # update the origin of the image group according to the crop
        origin = self.get_attribute('origin', 'CellData')
        origin += spacing * np.array([x_start, y_start, z_start])
        print('origin will be set to', origin)
        micro_crop.set_origin('CellData', origin)
        if verbose:
            print('cropped dataset:')
            print(micro_crop)
        micro_crop.set_active_grain_map(self.active_grain_map)
        grain_ids = np.unique(micro_crop.get_grain_map())
        for gid in grain_ids:
            if not gid > 0:
                continue
            grain = self.grains.read_where('idnumber == gid')
            micro_crop.grains.append(grain)
        print('%d grains in cropped microstructure' % micro_crop.grains.nrows)
        micro_crop.grains.flush()
        # recompute the grain geometry
        if recompute_geometry:
            print('updating grain geometry')
            micro_crop.recompute_grain_bounding_boxes(verbose)
            micro_crop.recompute_grain_centers(verbose)
            micro_crop.recompute_grain_volumes()
        return micro_crop

    def sync_grain_table_with_grain_map(self, sync_geometry=False):
        """Update GrainDataTable with only grain IDs from active grain map.

        :param bool sync_geometry: If `True`, recomputes the geometrical
            parameters of the grains in the GrainDataTable from active grain
            map.
        """
        # Remove grains that are not in grain map from GrainDataTable
        self.remove_grains_not_in_map()
        # Add grains that are in grain map but not in GrainDataTable
        self.add_grains_in_map()
        if sync_geometry:
            self.recompute_grain_bounding_boxes()
            self.recompute_grain_centers()
            self.recompute_grain_volumes()
        return

    def renumber_grains(self, sort_by_size=False, new_map_name=None,
                        only_grain_map=False):
        """Renumber the grains in the microstructure.

        Renumber the grains from 1 to n, with n the total number of grains
        that are found in the active grain map array, so that the numbering is
        consecutive. Only positive grain ids are taken into account (the id 0
        is reserved for the background).

        :param bool sort_by_size: use the grain volume to sort the grain ids
            (the larger grain will become grain 1, etc).
        :param bool overwrite_active_map: if 'True', overwrites the active
            grain map with the renumbered map. If 'False', the active grain map
            is kept and a 'renumbered_grain_map' is added to CellData.
        :param str new_map_name: Used as name for the renumbered grain map
            field if is not None and overwrite_active_map is False.
        :param bool only_grain_map: If `True`, do not modify the grain map
            and GrainDataTable in dataset, but return the renumbered grain_map
            as a numpy array.
        """
        if self._is_empty('grain_map'):
            print('warning: a grain map is needed to renumber the grains')
            return
        self.sync_grain_table_with_grain_map()
        # At this point, the table and the map have the same grain Ids
        grain_map = self.get_grain_map()
        grain_map_renum = grain_map.copy()
        if sort_by_size:
            print('sorting ids by grain size')
            sizes = self.get_grain_volumes()
            new_ids = self.get_grain_ids()[np.argsort(sizes)][::-1]
        else:
            new_ids = range(1, len(np.unique(grain_map)) + 1)
        for i, g in enumerate(tqdm(self.grains, desc='renumbering grains')):
            gid = g['idnumber']
            if not gid > 0:
                # only renumber positive grain ids
                continue
            new_id = new_ids[i]
            grain_map_renum[grain_map == gid] = new_id
            if not only_grain_map:
                g['idnumber'] = new_id
                g.update()
        print('maximum grain id is now %d' % max(new_ids))
        if only_grain_map:
            return grain_map_renum
        # assign the renumbered grain_map to the microstructure
        if new_map_name is None:
            map_name = self.active_grain_map
        else:
            map_name = new_map_name
        self.set_grain_map(grain_map_renum, self.get_voxel_size(),
                           map_name=map_name)
        return

    def compute_grain_volume(self, gid):
        """Compute the volume of the grain given its id.

        The total number of voxels with the given id is computed. The value is
        converted to mm unit using the `voxel_size`. The unit will be squared
        mm for a 2D grain map or cubed mm for a 3D grain map.

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
        if len(voxel_size) == 2:
            voxel_size = np.concatenate((voxel_size, np.array([0])), axis=0)
        offset = bb[:, 0]
        grain_data_bin = (grain_map == gid).astype(np.uint8)
        local_com = ndimage.measurements.center_of_mass(grain_data_bin) + \
                    np.array([0.5, 0.5, 0.5])  # account for first voxel coordinates
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

    def compute_grain_equivalent_diameters(self, id_list=None):
        """Compute the equivalent diameter for a list of grains.

        The equivalent diameter is defined as the diameter of a sphere with
        the same volume as the grain if 3D or the diameter of the circle with
        the same surface as the grain in 2D.

        .. math::

          D_{eq} = \left(\dfrac{6V}{\pi}\right)^{1/3}

        :param list id_list: the list of the grain ids to include (compute
            for all grains by default).
        :return: a 1D numpy array of the grain diameters.
        """
        volumes = self.get_grain_volumes(id_list)
        location = self._get_parent_name(self.active_grain_map)
        if self._get_group_type('CellData') == '2DImage':
            grain_equivalent_diameters = (4 * volumes / np.pi) ** (1 / 2)
        else:
            grain_equivalent_diameters = (6 * volumes / np.pi) ** (1 / 3)
        return grain_equivalent_diameters

    def compute_grain_sphericities(self, id_list=None):
        """Compute the equivalent diameter for a list of grains.

        The sphericity measures how close to a sphere is a given grain.
        It can be computed by the ratio between the surface area of a sphere
        with the same volume and the actual surface area of that grain.

        .. math::

          \psi = \dfrac{\pi^{1/3}(6V)^{2/3}}{A}

        :param list id_list: the list of the grain ids to include (compute
            for all grains by default).
        :return: a 1D numpy array of the grain diameters.
        """
        volumes = self.get_grain_volumes(id_list)
        if not id_list:
            id_list = self.get_grain_ids()
        grain_map = self.get_grain_map()
        if len(grain_map.shape) < 3:
            raise ValueError('Cannot compute grain sphericities on a non'
                             ' tridimensional grain map.')
        surface_areas = np.empty_like(volumes)
        for i, grain_id in enumerate(id_list):
            grain_data = (grain_map == grain_id)
            surface_areas[i] = np.sum(grain_data - ndimage.morphology.binary_erosion(grain_data))
        sphericities = np.pi ** (1 / 3) * (6 * volumes) ** (2 / 3) / surface_areas
        return sphericities

    def compute_grain_aspect_ratios(self, id_list=None):
        """Compute the aspect ratio for a list of grains.

        The aspect ratio is defined by the ratio between the major and minor
        axes of the equivalent ellipsoid of each grain.

        :param list id_list: the list of the grain ids to include (compute
            for all grains by default).
        :return: a 1D numpy array of the grain aspect ratios.
        """
        from skimage.measure import regionprops
        props = regionprops(self.get_grain_map())
        grain_aspect_ratios = np.array([prop.major_axis_length /
                                        prop.minor_axis_length
                                        for prop in props])
        return grain_aspect_ratios

    def recompute_grain_volumes(self):
        """Compute the volume of all grains in the microstructure.

        Each grain volume is computed using the grain map. The value is
        assigned to the volume column of the GrainDataTable node.
        If the voxel size is specified, the grain volumes will be in mm unit,
        if not in voxel unit.

        .. note::

          A grain map need to be associated with this microstructure instance
          for the method to run.

        :return: a 1D array with all grain volumes.
        """
        if self._is_empty('grain_map'):
            print('warning: needs a grain map to recompute the volumes '
                  'of the grains')
            return
        voxel_size = self.get_attribute('spacing', 'CellData')
        labels = self.get_grain_map()
        mask = labels > 0
        volumes = ndimage.sum_labels(mask, 
                                     labels, 
                                     index=self.get_grain_ids()) * np.prod(voxel_size)
        self.set_volumes(volumes)
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

        origin = self.get_attribute('origin', 'CellData')
        if len(origin) == 2:
            origin = np.concatenate((origin, np.array([0])), axis=0)
        voxel_size = self.get_attribute('spacing', 'CellData')
        if len(voxel_size) == 2:
            voxel_size = np.concatenate((voxel_size, np.array([0])), axis=0)

        grain_map = self.get_grain_map()
        centers = ndimage.center_of_mass(grain_map > 0, labels=grain_map, index=self.get_grain_ids())
        # convert to mm
        centers = origin + (centers + np.array([0.5, 0.5, 0.5])) * voxel_size
        #centers = (centers + np.array([0.5, 0.5, 0.5]) - 0.5 * np.array(grain_map.shape)) * voxel_size
        self.set_centers(centers)
        return self.get_grain_centers()

    def recompute_grain_bounding_boxes(self, verbose=False):
        """Compute and assign the center of all grains in the microstructure.

        Each grain bounding box is computed in voxel unit. The value is
        assigned to the grain.bounding_box attribute.

        .. note::

          A grain map need to be associated with this microstructure instance
          for the method to run.

        :param bool verbose: flag for verbose mode.
        """
        if self._is_empty('grain_map'):
            print('warning: need a grain map to recompute the bounding boxes'
                  ' of the grains')
            return
        # find_objects will return a list of N slices, N being the max grain id
        slices = ndimage.find_objects(self.get_grain_map())
        for g in tqdm(self.grains, desc='computing grain bounding boxes'):
            try:
                g_slice = slices[g['idnumber'] - 1]
                x_indices = (g_slice[0].start, g_slice[0].stop)
                y_indices = (g_slice[1].start, g_slice[1].stop)
                z_indices = (g_slice[2].start, g_slice[2].stop)
                bbox = x_indices, y_indices, z_indices
            except (ValueError, TypeError, IndexError):
                '''
                ValueError or TypeError can arise for grains in the data table
                that are not in the grain map (None will be returned from
                find_objects). IndexError can occur if these grain ids are
                larger than the maximum id in the grain map.
                '''
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
        """Compute grain geometry from the grain map.

        This method computes the grain centers, volume and bounding boxes from
        the grain map and update the grain data table. This applies only to
        grains represented in the grain map. If other grains are present, their
        information is unchanged unless the option `overwrite_table` is activated.

        :param bool overwrite_table: if this is True, the grains present in the
        data table and not in the grain map are removed from it.
        """
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

    def compute_grains_map_table_intersection(self, verbose=False):
        """Return grains that are both in grain map and grain table.

        The method also returns the grains that are in the grain map but not
        in the grain table, and the grains that are in the grain table, but
        not in the grain map.

        :return array intersection: array of grain ids that are in both
            grain map and grain data table.
        :return array not_in_map: array of grain ids that are in grain data
            table but not in grain map.
        :return array not_in_table: array of grain ids that are in grain map
            but not in grain data table.
        """
        if self._is_empty('grain_map'):
            print('warning: a grain map is needed to compute grains map'
                  '/table intersection.')
            return
        grain_map = self.get_grain_map()
        map_ids = np.unique(grain_map)
        # only positive integer values are considered as valid grain ids, remove everything else:
        map_ids = np.delete(map_ids, range(0, np.where(map_ids > 0)[0][0]))
        table_ids = self.get_grain_ids()
        intersection = np.intersect1d(map_ids, table_ids)
        not_in_map = table_ids[np.isin(table_ids, map_ids, invert=True,
                             assume_unique=True)]
        not_in_table = map_ids[np.isin(map_ids, table_ids, invert=True,
                             assume_unique=True)]
        if verbose:
            print('Grains ids both in grain map and GrainDataTable:')
            print(str(intersection).strip('[]'))
            print('Grains ids in GrainDataTable but not in grain map:')
            print(str(not_in_map).strip('[]'))
            print('Grains ids in grain map but not in GrainDataTable :')
            print(str(not_in_table).strip('[]'))
        return intersection, not_in_map, not_in_table

    def build_grain_table_from_grain_map(self):
        """Synchronizes and recomputes GrainDataTable from active grain map."""
        # First step: synchronize table with grain map
        self.sync_grain_table_with_grain_map()
        # Second step, recompute grain geometry
        # TODO: use recompute_grain_geometry when method is corrected
        self.recompute_grain_bounding_boxes()
        self.recompute_grain_centers()
        self.recompute_grain_volumes()
        return

    def is_unitary_vector(v):
        # v must be a list or a numpy array with all elements being -1, 0, or 1
        if isinstance(v, (list, np.ndarray)) and all(x in {-1, 0, 1} for x in v):
            # check if there's exactly one non-zero element
            return sum(abs(x) for x in v) == 1
        return False
        
    def change_reference_frame(self, new_x, new_y, cell_data='CellData', in_place=True, suffix='_XYZ'):
        """Change teh local reference frame of this Microstructure instance.

        Args:
            new_x (_type_): _description_
            new_y (_type_): _description_
            cell_data (str, optional): _description_. Defaults to 'CellData'.
            in_place (bool, optional): _description_. Defaults to True.
            suffix (str, optional): _description_. Defaults to '_XYZ'.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # input check
        if not np.all([Microstructure.is_unitary_vector(v) for v in[new_x, new_y]]) or np.dot(new_x, new_y) != 0:
            raise ValueError("input vectors must be unitary, aligned with the cartesian axes and prependicular, please correct your input")
        is_2d = self._get_group_type(cell_data) == '2DImage'
        if in_place is True and is_2d:
            print('warning: cannot change reference frame in place for 2DImage, creating a new Microstructure instance')
            in_place = False
        new_z = np.cross(new_x, new_y)
        T = np.array([new_x, new_y, new_z])
        assert np.linalg.det(T) == 1.0
        swap_indices = np.argwhere(new_x)[0][0], np.argwhere(new_y)[0][0], np.argwhere(new_z)[0][0]
        print('swap indices are', swap_indices)
        # find which direction needs to be reversed
        flip_indices = []
        for i, v in enumerate([new_x, new_y, new_z]):
            if np.dot(v, np.abs(v)) < 0:
                flip_indices.append(i)

        if not in_place:
            # create the new Microstructure instance if needed
            file_xyz = os.path.splitext(self.h5_file)[0] + suffix + '.h5'
            print('new microstructure file name is', file_xyz)
            if is_2d:
                # create a blank new file which will not be 2D anymore
                m2 = Microstructure(filename=file_xyz, 
                                    name=self.get_sample_name() + suffix, 
                                    phase=self.get_phase_list(),
                                    autodelete=False, overwrite_hdf5=True)
            else:
                m2 = Microstructure.copy_sample(self.h5_path, file_xyz, overwrite=True, get_object=True, autodelete=False)
        else:
            m2 = self
        grain_map_xyz = self.get_grain_map().transpose(swap_indices)
        print('flip indices are', flip_indices)
        if len(flip_indices) > 0:
            grain_map_xyz = np.flip(grain_map_xyz, axis=flip_indices)
        m2.set_grain_map(grain_map_xyz, voxel_size=self.get_voxel_size())
        print('new grain_map has shape', m2.get_grain_map().shape)
        m2.sync_grain_table_with_grain_map(sync_geometry=True)
        
        # rotate grain orientations
        rods = self.get_grain_rodrigues()
        rods_xyz = np.empty_like(rods)
        for i in range(len(rods)):
            o = Orientation.from_rodrigues(rods[i])
            g_xyz = np.dot(o.orientation_matrix(), T.T)  # move to new local frame
            rods_xyz[i] = Orientation(g_xyz).rod
        m2.set_orientations(rods_xyz)

        # rotate all the fields in CellData
        image_group = self.get_node(cell_data)
        field_index_path = '%s/Field_index' % image_group._v_pathname
        field_list = self.get_node(field_index_path)
        time.sleep(0.2)
        for name in tqdm(field_list, desc='rotating fields'):
            field_name = name.decode('utf-8')
            field = self.get_field(field_name)
            if not self._is_empty(field_name):
                if self._get_group_type('CellData') == '2DImage':
                    field = np.expand_dims(field, axis=2)
                if field.ndim == 4:
                    field_xyz = field.transpose(swap_indices + (-1,))
                else:
                    field_xyz = field.transpose(swap_indices)
                if len(flip_indices) > 0:
                    field_xyz = np.flip(field_xyz, axis=flip_indices)
                m2.add_field(gridname=cell_data,
                            fieldname=field_name,
                            array=field_xyz,
                            replace=True)
        
        # also rotate the orientation map
        if not m2._is_empty('orientation_map'):
            orientation_map_xyz = m2.get_orientation_map()
            indices = np.where(m2.get_phase_map() > 0)
            time.sleep(0.2)
            for i, j, k in tqdm(zip(*indices), total=len(indices[0]), 
                                desc='changing orientation map reference frame'):
                o_tsl = Orientation.from_rodrigues(orientation_map_xyz[i, j, k, :])
                g_xyz = np.dot(o_tsl.orientation_matrix(), T.T)  # move to XYZ local frame
                orientation_map_xyz[i, j, k, :] = Orientation(g_xyz).rod
            m2.set_orientation_map(orientation_map_xyz)
        
        if not in_place:
           return m2

    def graph(self):
        """Create the graph of this microstructure.

        This method process a `Microstructure` instance using a Region Adgency
        Graph built with the crystal misorientation between neighbors as weights.
        The graph has a node per grain and a connection between neighboring
        grains of the same phase. The misorientation angle is attach to each edge.
        
        :return rag: the region adjency graph of this microstructure.
        """
        try:
            from skimage.future import graph
        except ImportError:
            from skimage import graph

        print('build the region agency graph for this microstructure')
        rag = graph.RAG(self.get_grain_map(), connectivity=1)

        # remove node and connections to the background
        if 0 in rag.nodes:
            rag.remove_node(0)

        # get the grain infos
        grain_ids = self.get_grain_ids()
        rodrigues = self.get_grain_rodrigues()
        centers = self.get_grain_centers()
        volumes = self.get_grain_volumes()
        phases = self.grains[:]['phase']
        for grain_id, d in rag.nodes(data=True):
            d['label'] = [grain_id]
            index = grain_ids.tolist().index(grain_id)
            d['rod'] = rodrigues[index]
            d['center'] = centers[index]
            d['volume'] = volumes[index]
            d['phase'] = phases[index]

        # assign grain misorientation between neighbors to each edge of the graph
        for x, y, d in rag.edges(data=True):
            if rag.nodes[x]['phase'] != rag.nodes[y]['phase']:
                # skip edge between neighboring grains of different phases
                continue
            sym = self.get_phase(phase_id=rag.nodes[x]['phase']).get_symmetry()
            o_x = Orientation.from_rodrigues(rag.nodes[x]['rod'])
            o_y = Orientation.from_rodrigues(rag.nodes[y]['rod'])
            mis = np.degrees(o_x.disorientation(o_y, crystal_structure=sym)[0])
            d['misorientation'] = mis
        
        return rag

    def segment_mtr(self, labels_seg=None, mis_thr=20., min_area=500, store=False):
        """Segment micro-textured regions (MTR).

        This method process a `Microstructure` instance to segment the MTR
        with the specified parameters.

        :param ndarray labels_seg: a pre-segmentation of the grain map, the full
        grain map will be used if not specified.
        :param float mis_thr: threshold in misorientation used to cut the graph.
        :param int min_area: minimum area used to define a MTR.
        :param bool store: flag to store the segmented array in the microstructure.
        :return mtr_labels: array with the labels of the segmented regions.
        """
        rag_seg = self.graph()

        # cut our graph with the misorientation threshold
        rag = rag_seg.copy()
        edges_to_remove = [(x, y) for x, y, d in rag.edges(data=True)
                           if d['misorientation'] >= mis_thr]
        rag.remove_edges_from(edges_to_remove)

        import networkx as nx
        comps = nx.connected_components(rag)
        map_array = np.arange(labels_seg.max() + 1, dtype=labels_seg.dtype)
        for i, nodes in enumerate(comps):
            # compute area of this component
            area = np.sum(np.isin(labels_seg, list(nodes)))
            if area < min_area:
                # ignore small MTR (simply assign them to label zero)
                i = 0
            for node in nodes:
                for label in rag.nodes[node]['label']:
                    map_array[label] = i
        mtr_labels = map_array[labels_seg]
        print('%d micro-textured regions were segmented' % len(np.unique(mtr_labels)))
        if store:
            self.add_field(gridname='CellData', fieldname='mtr_segmentation',
                           array=mtr_labels, replace=True)
        return mtr_labels

    @staticmethod
    def voronoi(shape=(256, 256), n=50):
        """Simple voronoi tesselation to create a grain map.

        The method works both in 2 and 3 dimensions and will create a sample
        with a size of 1 (domain from -0.5 to 0.5). The grains are labeled
        from 1 to `n` (included).

        :param tuple shape: grain map shape in 2 or 3 dimensions.
        :param int n: number of grains to generate.
        :raise: a ValueError if the shape has not size 2 or 3.
        :return: a 2D or 3D numpy array representing the grain map.
        """
        dim = len(shape)
        if dim not in [2, 3]:
            raise ValueError('specified shape must be either 2D or 3D')
        grain_map = np.zeros(shape, dtype=int)
        nx, ny = shape[0], shape[1]
        x = np.linspace(-0.5, 0.5, nx, endpoint=True)
        y = np.linspace(-0.5, 0.5, ny, endpoint=True)
        print('%dD voronoi tesselation' % dim)
        if dim == 2:
            XX, YY = np.meshgrid(x, y)
            seeds = np.random.rand(n, 2) - np.array([0.5, 0.5])
            distance = np.zeros(shape=(n, nx, ny))
            # compute Voronoi distance in 2D
            for i in range(n):
                distance[i] = np.sqrt((XX - seeds[i, 0]) ** 2
                                      + (YY - seeds[i, 1]) ** 2)
            grain_map = 1 + np.argmin(distance, axis=0)
        else:
            nz = shape[2]
            z = np.linspace(-0.5, 0.5, nz, endpoint=True)
            XX, YY, ZZ = np.meshgrid(x, y, z)
            seeds = np.random.rand(n, 3) - np.array([0.5, 0.5, 0.5])
            distance = np.zeros(shape=(n, nx, ny, nz))
            # compute Voronoi distance in 3D
            for i in range(n):
                distance[i] = np.sqrt((XX - seeds[i, 0]) ** 2
                                      + (YY - seeds[i, 1]) ** 2
                                      + (ZZ - seeds[i, 2]) ** 2)
            grain_map = 1 + np.argmin(distance, axis=0)
        return grain_map

    def to_amitex_fftp(self, binary=True, mat_file=True, algo_file=True,
                       char_file=True, elasaniso_path='',
                       add_grips=False, grip_size=10,
                       grip_constants=(104100., 49440.), add_exterior=False,
                       exterior_size=10, use_mask=False):
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
        :param bool mat_file: flag to write the material file for Amitex.
        :param bool algo_file: flag to write the algorithm file for Amitex.
        :param bool char_file: flag to write the loading file for Amitex.
        :param str elasaniso_path: path for the libUmatAmitex.so in
            the Amitex_FFTP installation.
        :param bool add_grips: add a constant region at the beginning and the
            end of the third axis.
        :param int grip_size: thickness of the region.
        :param tuple grip_constants: elasticity values for the grip (lambda, mu).
        :param bool add_exterior: add a constant region around the sample at
            the beginning and the end of the first two axes.
        :param int exterior_size: thickness of the exterior region.
        :param bool use_mask: use mask to define exterior material, and use
            mask to extrude grips with same shapes as microstructure top and
            bottom surfaces.
        """
        n_phases = self.get_number_of_phases()
        ext = 'bin' if binary else 'txt'
        grip_id = n_phases   # material id for the grips
        ext_id = n_phases + 1 if add_grips else n_phases # material id for the exterior
        n1x = open('%s_N1X.%s' % (self.get_sample_name(), ext), 'w')
        n1y = open('%s_N1Y.%s' % (self.get_sample_name(), ext), 'w')
        n1z = open('%s_N1Z.%s' % (self.get_sample_name(), ext), 'w')
        n2x = open('%s_N2X.%s' % (self.get_sample_name(), ext), 'w')
        n2y = open('%s_N2Y.%s' % (self.get_sample_name(), ext), 'w')
        n2z = open('%s_N2Z.%s' % (self.get_sample_name(), ext), 'w')
        files = [n1x, n1y, n1z, n2x, n2y, n2z]
        if binary:
            import struct
            for f in files:
                f.write('%d \ndouble \n' % self.get_number_of_grains())
                f.close()
            n1x = open('%s_N1X.%s' % (self.get_sample_name(), ext), 'ab')
            n1y = open('%s_N1Y.%s' % (self.get_sample_name(), ext), 'ab')
            n1z = open('%s_N1Z.%s' % (self.get_sample_name(), ext), 'ab')
            n2x = open('%s_N2X.%s' % (self.get_sample_name(), ext), 'ab')
            n2y = open('%s_N2Y.%s' % (self.get_sample_name(), ext), 'ab')
            n2z = open('%s_N2Z.%s' % (self.get_sample_name(), ext), 'ab')
            for g in self.grains:
                o = Orientation.from_rodrigues(g['orientation'])
                # handle hexagonal case
                if self.get_phase(g['phase']).get_symmetry() == Symmetry.hexagonal:
                    o = Orientation.from_euler(o.euler - np.array([0, 0, 30]))
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
                if self.get_phase(g['phase']).get_symmetry() == Symmetry.hexagonal:
                    o = Orientation.from_euler(o.euler - np.array([0, 0, 30]))
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

        # if required, write the loading file for Amitex
        if char_file:
            from lxml import etree, builder
            root = etree.Element('Loading_Output')
            output = etree.Element('Output')
            output.append(etree.Element(_tag='vtk_StressStrain', Strain='1', Stress='0'))
            root.append(output)
            loading = etree.Element('Loading', Tag='1')
            loading.append(etree.Element(_tag='Time_Discretization', Discretization='Linear', Nincr='1', Tfinal='1'))
            loading.append(etree.Element(_tag='Output_zone', Number='1'))
            output_vtk_list = etree.Element('Output_vtkList')
            output_vtk_list.text = "1"
            loading.append(output_vtk_list)
            loading.append(etree.Element(_tag='xx', Driving='Stress', Evolution='Constant'))
            loading.append(etree.Element(_tag='yy', Driving='Stress', Evolution='Constant'))
            loading.append(etree.Element(_tag='zz', Driving='Stress', Evolution='Linear', Value='100.0'))
            loading.append(etree.Element(_tag='xy', Driving='Stress', Evolution='Constant'))
            loading.append(etree.Element(_tag='xz', Driving='Stress', Evolution='Constant'))
            loading.append(etree.Element(_tag='yz', Driving='Stress', Evolution='Constant'))
            root.append(loading)
            # write the file
            tree = etree.ElementTree(root)
            tree.write('char.xml', xml_declaration=True, pretty_print=True,
                       encoding='UTF-8')
            print('FFT loading file written in char.xml')

        # if required, write the material file for Amitex
        if algo_file:
            from lxml import etree, builder
            root = etree.Element('Algorithm_Parameters')
            algo = etree.Element('Algorithm', Type='Basic_Scheme')
            algo.append(etree.Element(_tag='Convergence_Criterion', Value='Default'))
            algo.append(etree.Element(_tag='Convergence_Acceleration', Value='true'))
            algo.append(etree.Element(_tag='Nitermin_acv', Value='0'))
            root.append(algo)
            mech = etree.Element('Mechanics')
            mech.append(etree.Element(_tag='Filter', Type='Default'))
            mech.append(etree.Element(_tag='Small_Perturbations', Value='true'))
            root.append(mech)
            # write the file
            tree = etree.ElementTree(root)
            tree.write('algo.xml', xml_declaration=True, pretty_print=True,
                       encoding='UTF-8')
            print('FFT algorithm file written in algo.xml')

        # if required, write the material file for Amitex
        if mat_file:
            from lxml import etree, builder
            root = etree.Element('Materials')
            root.append(etree.Element('Reference_Material',
                                      Lambda0='90000.0',
                                      Mu0='31000.0'))
            # add each phase as a material for Amitex
            phase_ids = self.get_phase_ids_list()
            phase_ids.sort()
            # here phase_ids needs to be equal to [1, ..., n_phases]
            if not phase_ids == list(range(1, n_phases + 1)):
                raise ValueError('inconsistent phase numbering (should be from '
                                 '1 to n_phases): {}'.format(phase_ids))
            for phase_id in phase_ids:
                comment = etree.Comment(' phase %d: %s ' % (phase_id, self.get_phase().name))
                root.append(comment)
                mat = etree.Element('Material', numM=str(phase_id),
                                    Lib='%s/libUmatAmitex.so' % elasaniso_path,
                                    Law='elasaniso')
                # get the C_IJ values
                phase = self.get_phase(phase_id)
                C = phase.get_symmetry().stiffness_matrix(phase.elastic_constants)
                '''
                Note that Amitex uses a different reduced number:
                (1, 2, 3, 4, 5, 6) = (11, 22, 33, 12, 13, 23)
                Because of this indices 4 and 6 are inverted with respect to the Voigt convention.
                '''
                comment = etree.Comment(' orthotropic elasticity coefficients (9 values) ')
                mat.insert(0, comment)
                mat.append(etree.Element(_tag='Coeff', Index='1', Type='Constant', Value=str(C[0, 0])))  # C11
                mat.append(etree.Element(_tag='Coeff', Index='2', Type='Constant', Value=str(C[0, 1])))  # C12
                mat.append(etree.Element(_tag='Coeff', Index='3', Type='Constant', Value=str(C[0, 2])))  # C13
                mat.append(etree.Element(_tag='Coeff', Index='4', Type='Constant', Value=str(C[1, 1])))  # C22
                mat.append(etree.Element(_tag='Coeff', Index='5', Type='Constant', Value=str(C[1, 2])))  # C23
                mat.append(etree.Element(_tag='Coeff', Index='6', Type='Constant', Value=str(C[2, 2])))  # C33
                mat.append(etree.Element(_tag='Coeff', Index='7', Type='Constant', Value=str(C[5, 5])))  # C66
                mat.append(etree.Element(_tag='Coeff', Index='8', Type='Constant', Value=str(C[4, 4])))  # C55
                mat.append(etree.Element(_tag='Coeff', Index='9', Type='Constant', Value=str(C[3, 3])))  # C44
                comment = etree.Comment(' cristal orientation: N1 coeff(10-12), N2 coeff(13-15) ')
                mat.insert(9, comment)
                fmt = "binary" if binary else "ascii"
                mat.append(etree.Element(_tag='Coeff', Index="10", Type="Constant_Zone", File="N1X.bin", Format=fmt))
                mat.append(etree.Element(_tag='Coeff', Index="11", Type="Constant_Zone", File="N1Y.bin", Format=fmt))
                mat.append(etree.Element(_tag='Coeff', Index="12", Type="Constant_Zone", File="N1Z.bin", Format=fmt))
                mat.append(etree.Element(_tag='Coeff', Index="13", Type="Constant_Zone", File="N2X.bin", Format=fmt))
                mat.append(etree.Element(_tag='Coeff', Index="14", Type="Constant_Zone", File="N2Y.bin", Format=fmt))
                mat.append(etree.Element(_tag='Coeff', Index="15", Type="Constant_Zone", File="N2Z.bin", Format=fmt))
                comment = etree.Comment(' internal variable for the elastic strain field ')
                mat.append(comment)
                for i in range(6):
                    mat.append(etree.Element(_tag='IntVar', Index=str(i + 1), Type="Constant", Value="0."))

                root.append(mat)
            # add a material for top and bottom layers
            if add_grips:
                comment = etree.Comment(' Isotropic elastic material for tension test grips ')
                root.append(comment)
                grips = etree.Element('Material', numM=str(grip_id + 1),
                                      Lib='%s/libUmatAmitex.so' % elasaniso_path,
                                      Law='elasiso')
                grips.append(etree.Element(_tag='Coeff', Index='1', Type='Constant', Value=str(grip_constants[0])))
                grips.append(etree.Element(_tag='Coeff', Index='2', Type='Constant', Value=str(grip_constants[1])))
                root.append(grips)
            # add a material for external buffer
            if add_exterior or use_mask:
                comment = etree.Comment(' Isotropic elastic material for the exterior ')
                root.append(comment)
                exterior = etree.Element('Material', numM=str(ext_id + 1),
                                         Lib='%s/libUmatAmitex.so' % elasaniso_path,
                                         Law='elasiso')
                exterior.append(etree.Element(_tag='Coeff', Index='1', Type='Constant', Value='0.'))
                exterior.append(etree.Element(_tag='Coeff', Index='2', Type='Constant', Value='0.'))
                root.append(exterior)

            tree = etree.ElementTree(root)
            tree.write('mat.xml', xml_declaration=True, pretty_print=True,
                       encoding='UTF-8')
            print('material file written in mat.xml')

        # if possible, write the vtk file to run the computation
        if not self._is_empty('grain_map'):
            # convert the grain map to vtk file
            from vtk.util import numpy_support
            # make sure we have a continuous grain map for amitex
            if not self.get_grain_ids().tolist() == list(range(1, self.get_number_of_grains() + 1)):
                print('note: grain ids are not continuous and starting at 1, renumbering them')
                self.renumber_grains(only_grain_map=True)
            grain_ids = self.get_grain_map()
            if not self._is_empty('phase_map'):
                # use the phase map for the material ids
                material_ids = self.get_phase_map().astype(grain_ids.dtype)
            elif use_mask:
                material_ids = self.get_mask().astype(grain_ids.dtype)
            else:
                material_ids = np.ones_like(grain_ids)
            if add_grips:
                # add a layer of new_id (the value must actually be the first
                # grain id) above and below the sample.
                grain_ids = np.pad(grain_ids, ((0, 0),
                                               (0, 0),
                                               (grip_size, grip_size)),
                                   mode='constant', constant_values=1)
                if use_mask:
                    # create top and bottom mask extrusions
                    mask_top = material_ids[:, :, [-1]]
                    mask_bot = material_ids[:, :, [0]]
                    top_grip = np.tile(mask_top, (1, 1, grip_size))
                    bot_grip = np.tile(mask_bot, (1, 1, grip_size))
                    # add grip layers to unit cell matID
                    material_ids = np.concatenate(
                        ((grip_id + 1) * bot_grip, material_ids,
                         (grip_id + 1) * top_grip), axis=2)
                else:
                    material_ids = np.pad(
                        material_ids, ((0, 0), (0, 0), (grip_size, grip_size)),
                        mode='constant',
                        constant_values=grip_id + 1)
            if add_exterior and not use_mask:
                # add a layer of new_id around the first two dimensions
                grain_ids = np.pad(grain_ids, ((exterior_size, exterior_size),
                                               (exterior_size, exterior_size),
                                               (0, 0)),
                                   mode='constant',
                                   constant_values=1)
                material_ids = np.pad(material_ids,
                                      ((exterior_size, exterior_size),
                                       (exterior_size, exterior_size),
                                       (0, 0)),
                                      mode='constant',
                                      constant_values=ext_id + 1)
            if use_mask:
                grain_ids[np.where(grain_ids == 0)] = 1
                material_ids[np.where(material_ids == 0)] = ext_id + 1
            # write both arrays as VTK files for amitex
            voxel_size = self.get_voxel_size()
            for array, array_name in zip([grain_ids, material_ids],
                                         ['grain_ids', 'material_ids']):
                print('array name:', array_name, 'array type:', array.dtype)
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

    def from_amitex_fftp(self, base_name, grip_size=0, ext_size=0,
                         sim_prefix='Amitex', int_var_names={},
                         load_fields=True, compression_options={}):
        """Read the output of a Amitex_fftp simulation and store the field
        in the dataset.

        Read a Amitex_fftp result directory containing a mechanical simulation
        of the microstructure. See method `to_amitex_fftp` to generate input
        files for such simulation of Microstructure instances.

        The results are stored as fields of the CellData group by default.
        If generated by the simulation, the strain and stress tensor fields
        are stored, as well as the internal variables fields.

        Mechanical fields and macroscopic curves are stored. The latter is
        stored in the data group '/Mechanical_simulation' as a structured
        array.

            .. Warning 1::
                For now, this methods can store the results of several
                snapshots but without writing them as a xdmf time serie. This
                feature will be implemented in the future.

            .. Warning 2::
                For now, results are only stored on CellData group. Method will
                be modified in the future to allow to specify a new image data
                group to store de results (created if needed).

        :param base_name: Basename of Amitex .std, .vtk output files to
            load in dataset.
        :type base_name: str
        :param grip_size: Thickness of the grips added to simulation unit cell
            by the method 'to_amitex_fftp' of this class, defaults to 0. This
            value corresponds to a number of voxels on both ends of the cell.
        :type grip_size: int, optional
        :param ext_size: Thickness of the exterior region added to simulation
            unit cell by the method 'to_amitex_fftp' of this class,
            defaults to 0.  This value corresponds to a number of voxels on
            both ends of the cell.
        :type ext_size: int, optional
        :param sim_prefix: Prefix of the name of the fields that will be
            stored on the CellData group from simulation results.
        :type sim_prefix: str, optional
        :param int_var_names: Dictionary whose keys are the names of
            internal variables stored in Amitex output files
            (varInt1, varInt2, ...) and values are corresponding names for
            these variables in the dataset.
        :type int_var_names: dict, optional
        :param bool load_fields: Flag to control if the fields are imported
            from the vtk files.
        :param dict compression_options: Dictionary containing the compression
            options to use for the imported fields.
        """
        # TODO: add grain map to all time steps
        from pymicro.core.utils.SDAmitexUtils import SDAmitexIO
        # get std result file path (.std or .mstd)
        std_suffix = '.std' if grip_size == 0 and ext_size == 0 else '.mstd'
        std_path = Path(str(base_name)).absolute().with_suffix(std_suffix)
        # safety check
        if not std_path.exists():
            raise ValueError('results not found, "base_name" argument'
                             ' not associated with Amitex_fftp simulation'
                             ' results.')
        step = 1
        # for .mstd results skip grid and exterior lines
        if grip_size > 0:
            step += 1
        if ext_size > 0:
            step += 1
        elif not self._is_empty('mask'):
            # if mask used as exterior in computation but exterior size = 0
            # still eed to add 1 to step as .mstd will have three lines per
            # increment
            if np.any(self['mask'] == 0):
                step += 1
        std_res = SDAmitexIO.load_std(std_path, step=step)
        print(std_res[0])
        print(std_res[-1])
        # store macro data in specific group
        self.add_group(groupname=f'{sim_prefix}_Results', location='/',
                       indexname='fft_sim', replace=True)
        # std_res is a numpy structured array whose fields depend on
        # the type of output (finite strain ou infinitesimal strain sim.)
        # ==> we load it into the dataset as a structured table data item.
        self.add_table(location='fft_sim', name='Standard_output',
                       indexname=f'{sim_prefix}_std', replace=True,
                       description=std_res.dtype, data=std_res)
        # idem for zstd --> Add as Mechanical Grain Data Table
        zstd_path = Path(str(base_name) + '_1').with_suffix('.zstd')
        if zstd_path.exists():
            # load .zstd results
            zstd_res = SDAmitexIO.load_std(zstd_path,
                                           int_var_names=int_var_names)
            self.add_table(location='GrainData',
                           name='MechanicalGrainDataTable',
                           indexname='Mech_Grain_Data', replace=True,
                           description=zstd_res.dtype, data=zstd_res)
            grain_ids = self.get_grain_ids()
            n_zone_times = int(zstd_res.shape[0] / len(grain_ids))
            dtype_col = np.dtype([('grain_ID', np.int32)])
            IDs = np.tile(grain_ids, n_zone_times).astype(dtype_col)
            self.add_tablecols(tablename='MechanicalGrainDataTable',
                               description=IDs.dtype, data=IDs)
        # end of macroscopic data loading. Check if field data must be loaded.
        if not load_fields:
            return
        # get results from vtk files
        stress, strain, var_int, incr_list = SDAmitexIO.load_amitex_output_fields(
            base_name, grip_size=grip_size, ext_size=ext_size,
            grip_dim=2)
        # loop over time steps: create group to store results
        self.add_group(groupname=f'{sim_prefix}_fields',
                       location='/CellData', indexname='fft_fields',
                       replace=True)
        # create CellData sub-grids for each time value with a vtk field output
        time_values = std_res['time'][incr_list].squeeze()
        if np.size(time_values) == 0:
            time_values = [0.]
            print(time_values)
        self.add_grid_time('CellData', time_values)

        # add fields to CellData grid collections
        for incr in stress:
            time = std_res['time'][incr].squeeze()
            field_name = f'{sim_prefix}_stress'
            self.add_field(gridname='CellData', fieldname=field_name,
                           array=stress[incr], location='fft_fields',
                           time=time, compression_options=compression_options)
        for incr in strain:
            time = std_res['time'][incr].squeeze()
            field_name = f'{sim_prefix}_strain'
            self.add_field(gridname='CellData', fieldname=field_name,
                           array=strain[incr], location='fft_fields',
                           time=time, compression_options=compression_options)
        for mat in var_int:
            for incr in var_int[mat]:
                time = std_res['time'][incr].squeeze()
                for var in var_int[mat][incr]:
                    var_name = var
                    if int_var_names.__contains__(var):
                        var_name = int_var_names[var]
                    field_name = f'{sim_prefix}_mat{mat}_{var_name}'
                    self.add_field(gridname='CellData', fieldname=field_name,
                                   array=var_int[mat][incr][var],
                                   location='fft_fields', time=time,
                                   compression_options=compression_options)
        return

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
        geom.attrs['GeometryTypeName'] = np.string_('UnknownGeometry')
        # create the data container bundles group
        f.create_group('DataContainerBundles')
        f.close()

    @staticmethod
    def from_dream3d(file_path,
                     main_key='DataContainers',
                     data_container='DataContainer',
                     ensemble_data='EnsembleData',
                     grain_data='FeatureData',
                     grain_orientations='AvgEulerAngles',
                     grain_orientations_type='euler',
                     grain_centroids='Centroids',
                     cell_data='CellData',
                     grain_ids='FeatureIds',
                     mask='Mask',
                     phases='Phases',
                     orientations='EulerAngles',
                     orientations_type='euler',
                     ipf_color='IPFColor'
                     ):
        """Read a microstructure from a Dream3d hdf5 file.

        :param str file_path: the path to the hdf5 file to read.
        :param str main_key: the string describing the main key group.
        :param str data_container: the string describing the data container
            group in the file.
        :param str ensemble_data: the string describing the ensemble data group
            in the file.
        :param str grain_data: the string describing the grain data group in the
            hdf5 file.
        :param str grain_orientations: the string describing the average grain
            orientations in the hdf5 file.
        :param str grain_orientations_type: the string describing the descriptor
            used for average grain orientation data.
        :param str grain_centroids: the string describing the grain centroids in
            the hdf5 file.
        :param str cell_data: the string describing the cell data group in the
            file.
        :param str grain_ids: the string describing the field representing
            grain ids within the cell_data data group.
        :param str mask: the string describing the field representing the sample
            mask within the cell_data data group.
        :param str phases: the string describing the field representing the
            sample phases within the cell_data data group.
        :param str orientations: the string describing the field representing
            voxel wise orientations within the cell_data data group.
        :param str orientations_type: the string describing the descriptor
            used for the orientation field.
        :return: a `Microstructure` instance created from the dream3d file.
        """
        head, tail = os.path.split(file_path)
        tail = tail.strip('.dream3d')
        with h5py.File(file_path, 'r') as f:
            # get information on the material phases
            phases_data_path = '%s/%s/%s' % (main_key, data_container, ensemble_data)
            phase_names = f[phases_data_path]['MaterialName'][()]
            n_phases = len(phase_names) - 1  # skip background
            phase_list = []
            for i_phase in range(n_phases):
                phase_id = i_phase + 1
                n = int(f[phases_data_path]['CrystalStructures'][()][phase_id])
                lattice_constants = f[phases_data_path]['LatticeConstants'][()][phase_id]
                for i in range(3):
                    lattice_constants[i] = lattice_constants[i] / 10  # use nm unit
                # print('found phase with crystal structure %d' % n)
                l = Lattice.from_parameters(*lattice_constants, symmetry=Symmetry.from_dream3d(n))
                phase = CrystallinePhase(phase_id=phase_id, name=phase_names[phase_id], lattice=l)
                phase_list.append(phase)
            # initialize our microstructure
            micro = Microstructure(name=tail, phase=phase_list, overwrite_hdf5=True)
            # if n_phases > 1:
            #     micro.set_phases(phase_list)
            # now get grain data informations
            grain_data_path = '%s/%s/%s' % (main_key, data_container, grain_data)
            grain_orientations = f[grain_data_path][grain_orientations][()]
            # switch to degrees
            grain_orientations = grain_orientations*(180./np.pi)
            # suppose that the grains are number from 0 or 1 to Ngrains by
            # dream3D
            gids = np.array(range(grain_orientations.shape[0]))
            try:
                centroids = f[grain_data_path][grain_centroids][()]
            except:
                centroids = None
            if np.array_equal(grain_orientations[0], [0., 0., 0.]):
                # skip the background
                grain_orientations = grain_orientations[1:]
                gids = gids[1:]
                if centroids is not None:
                    centroids = centroids[1:]
            micro.add_grains(grain_orientations,
                             orientation_type=grain_orientations_type,
                             grain_ids=gids)
            if centroids is not None:
                micro.set_centers(centroids)
            # print('%d grains in the data set' % micro.get_number_of_grains())
            # read voxel size and origin
            geom_path = '%s/%s/_SIMPL_GEOMETRY' % (main_key, data_container)
            voxel_size = f[geom_path]['SPACING'][()]
            origin = f[geom_path]['ORIGIN'][()]
            print(voxel_size)
            # now read cell data
            cell_data_path = '%s/%s/%s' % (main_key, data_container, cell_data)
            grain_ids = f[cell_data_path][grain_ids][:, :, :, 0].transpose()
            cell_data_shape = grain_ids.shape
            print('CellData dimensions:', cell_data_shape)
            mask = f[cell_data_path][mask][:, :, :, 0].transpose()
            # now assign these arrays to the microstructure
            micro.set_grain_map(grain_ids, voxel_size=voxel_size)
            micro.set_origin('CellData', origin)
            micro.set_mask(mask)
            # add phase map array if needed
            if phases in f[cell_data_path].keys():
                phase_map = f[cell_data_path][phases][:, :, :, 0].transpose()
                micro.set_phase_map(phase_map)
            # add voxel wise orientation data
            if orientations in f[cell_data_path].keys():
                print('adding orientation field from key %s' % orientations)
                orientation_map = f[cell_data_path][orientations][:, :, :, :]
                if orientations_type == 'euler':
                    # convert this array to rodrigues vectors
                    orientation_map_euler = orientation_map.reshape((np.prod(cell_data_shape), 3))
                    print('shape of euler angles list:', orientation_map_euler.shape)
                    orientation_map = Orientation.eu2ro(orientation_map_euler).reshape(list(cell_data_shape) + [3])
                elif orientations_type == 'rodrigues':
                    pass
                else:
                    print('warning, only euler and rodrigues type supported at the moment')
                micro.set_orientation_map(orientation_map)
            if ipf_color in f[cell_data_path].keys():
                print('adding IPF color field from key %s' % ipf_color)
                ipf_map = f[cell_data_path][ipf_color][:, :, :, :].transpose(2, 1, 0, 3)
                micro.add_field(gridname='CellData', fieldname=ipf_color, array=ipf_map, replace=True)
            del grain_ids, mask, phases,
        return micro

    @staticmethod
    def copy_sample(src_micro_file, dst_micro_file, overwrite=False,
                    get_object=False, dst_name=None, autodelete=False):
        """ Initiate a new SampleData object and files from existing one"""
        SampleData.copy_sample(src_micro_file, dst_micro_file, overwrite,
                               new_sample_name=dst_name)
        if get_object:
            return Microstructure(filename=dst_micro_file,
                                  autodelete=autodelete)
        else:
            return

# TODO: create ReaderClasses to improve methods factorisation, customization
#       and readability
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
        filename = os.path.join(neper_dir, name)
        assert ext == '.tesr'  # assuming raster tesselation
        micro = Microstructure(name=name, filename=filename, overwrite_hdf5=True)
        with open(neper_file_path, 'r', encoding='latin-1') as f:
            line = f.readline()  # ***tesr
            line = f.readline()  # **format
            format_tokens = f.readline().strip().split()
            print(format_tokens)
            format_version = float(format_tokens[0])
            if format_version <= 2.0:
                data_type = format_tokens[1]
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
            #line = f.readline().strip()
            origin = np.array([0., 0., 0.])
            # look for **cell
            while True:
                line = f.readline().strip()
                if line.startswith('*origin'):
                    origin = np.array(f.readline().split()).astype(float)
                    print('origin will be set to', origin)
                if line.startswith('**cell'):
                    break
            n = int(f.readline().strip())
            print('microstructure contains %d grains' % n)
            f.readline()  # *id
            grain_ids = []
            # read the cell ids
            while True:
                line = f.readline().strip()
                if line.startswith('*'):
                    break
                else:
                    grain_ids.extend(np.array(line.split()).astype(int).tolist())
            print('grain ids are:', grain_ids)
            # current line may be *ori or **data
            has_ori = 'ori' in line
            print('has orientation data:', has_ori)
            if has_ori:
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
            # look for **data and handle *group if present
            phase_ids = None
            while True:
                if line.startswith('*group'):
                    print('multi phase sample')
                    phase_ids = []
                    while True:
                        line = f.readline().strip()
                        if line.startswith('**data'):
                            break
                        else:
                            phase_ids.extend(np.array(line.split()).astype(int).tolist())
                    print('phase ids are:', phase_ids)
                if line.startswith('**data'):
                    if format_version > 2.0:
                        # read the data type
                        data_type = f.readline().strip()
                    print('data type is %s' % data_type)
                    break
                line = f.readline().strip()
            print(f.tell())
            print('reading data from byte %d' % f.tell())
            data = np.fromfile(f, dtype=np.uint16)[:-4]  # leave out the last 4 values
            print(data.shape)
            assert np.prod(dims) == data.shape[0]
            micro.set_grain_map(data.reshape(dims[::-1]).transpose(2, 1, 0),
                                voxel_size=voxel_size)  # swap X/Z axes
            # update the origin grabbed from neper with the shape of the data
            location = micro._get_parent_name(micro.active_grain_map)
            origin = -0.5 * micro.get_voxel_size() * np.array(micro.get_grain_map().shape)
            micro.set_origin(location, origin)
            print('updating grain data table and grain geometry')
            micro.sync_grain_table_with_grain_map(sync_geometry=True)
            # if necessary set the phase_map
            if phase_ids:
                grain_map = micro.get_grain_map()
                phase_map = np.zeros_like(grain_map)
                for grain_id, phase_id in zip(grain_ids, phase_ids):
                    # ignore phase id == 1 as this corresponds to phase_map == 0
                    if phase_id > 1:
                        phase_map[grain_map == grain_id] = phase_id - 1
                micro.set_phase_map(phase_map)
        print('done')
        return micro

    @staticmethod
    def from_labdct(labdct_file, data_dir='.', name=None, include_ipf_map=False,
                    grain_map_key='GrainId', recompute_mean_orientation=False):
        """Create a microstructure from a DCT reconstruction.

        :param str labdct_file: the name of the file containing the labDCT data.
        :param str data_dir: the path to the folder containing the HDF5
            reconstruction file.
        :param str name: the file name to use for this microstructure.
            By default, the suffix `_data` is added to the base name of the
            labDCT scan.
        :param bool include_ipf_map: if True, the IPF maps will be included
            in the microstructure fields.
        :param str grain_map_key: string defining the path to the grain map
            (GrainId by default).
        :param bool recompute_mean_orientation: it True, the orientation of 
            each grain is computed from the rodrigues map (this may take a 
            long time).
        :return: a `Microstructure` instance created from the labDCT
            reconstruction file.
        """
        file_path = os.path.join(data_dir, labdct_file)
        print('creating microstructure for labDCT scan %s' % file_path)
        if not name:
            name, ext = os.path.splitext(labdct_file)
        # get the phase data
        with h5py.File(file_path, 'r') as f:
            #TODO handle multiple phases
            phase01 = f['PhaseInfo']['Phase01']
            #phase_name = phase01['Name'][()].decode('utf-8')
            phase_name = phase01['Name'][0].decode('utf-8')
            parameters = phase01['UnitCell'][()]  # length unit is angstrom
            a, b, c = parameters[:3] / 10  # use nm unit
            alpha, beta, gamma = parameters[3:]
            print(parameters)
            space_group = phase01['SpaceGroup'][()]
            sym = Symmetry.from_space_group(space_group)
            print('found %s symmetry' % sym)
            lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma, symmetry=sym)
            phase = CrystallinePhase(phase_id=1, name=phase_name, lattice=lattice)
        # create the microstructure with the phase infos
        m = Microstructure(name=name, overwrite_hdf5=True, phase=phase)

        # load LabDCT cell data
        with h5py.File(file_path, 'r') as f:
            spacing = f['LabDCT']['Spacing'][0]
            rodrigues_map = f['LabDCT']['Data']['Rodrigues'][()].transpose(2, 1, 0, 3)
            grain_map = f['LabDCT']['Data'][grain_map_key][()].transpose(2, 1, 0)
            print('adding cell data with shape {}'.format(grain_map.shape))
            m.set_grain_map(grain_map, voxel_size=spacing)
            mask = f['LabDCT']['Data']['Mask'][()].transpose(2, 1, 0)
            m.set_mask(mask, voxel_size=spacing)
            phase_map = f['LabDCT']['Data']['PhaseId'][()].transpose(2, 1, 0)
            m.set_phase_map(phase_map, voxel_size=spacing)
            m.set_orientation_map(rodrigues_map)
            if 'Completeness' in f['LabDCT/Data']:
                completeness_map = f['LabDCT']['Data']['Completeness'][()].transpose(2, 1, 0)
                m.add_field(gridname='CellData', fieldname='completeness_map',
                            array=completeness_map)

        # analyze the grain map
        grain_ids_list = range(1, grain_map.max() + 1)  # ids are consecutive
        slices = ndimage.find_objects(grain_map)
        mask = grain_map > 0
        sizes = ndimage.sum(mask, grain_map, grain_ids_list)

        # create grain data table infos
        grain = m.grains.row
        for i in tqdm(range(len(grain_ids_list)), desc='adding grains to the microstructure'):
            gid = grain_ids_list[i]
            # use the bounding box for this grain
            g_slice = slices[i]
            x_indices = (g_slice[0].start, g_slice[0].stop)
            y_indices = (g_slice[1].start, g_slice[1].stop)
            z_indices = (g_slice[2].start, g_slice[2].stop)
            bb = x_indices, y_indices, z_indices
            this_grain_map = grain_map[bb[0][0]:bb[0][1],
                                    bb[1][0]:bb[1][1],
                                    bb[2][0]:bb[2][1]]
            # here orientations are constant per grain, grab the first voxel for each grain
            indices = np.where(this_grain_map == gid)
            rod = rodrigues_map[bb[0][0] + indices[0][0],
                                bb[1][0] + indices[1][0],
                                bb[2][0] + indices[2][0], :]
            # grain center
            grain_data_bin = (this_grain_map == gid).astype(np.uint8)
            local_com = ndimage.measurements.center_of_mass(grain_data_bin) + \
                        np.array([0.5, 0.5, 0.5])  # account for first voxel coordinates
            com = spacing * (np.array(bb)[:, 0] + local_com - 0.5 * np.array(grain_map.shape))

            # create new grain in the data table
            grain['idnumber'] = gid
            grain['orientation'] = rod
            grain['bounding_box'] = bb
            grain['center'] = com
            grain['volume'] = sizes[i] * spacing ** 3
            grain.append()
        m.grains.flush()

        if include_ipf_map:
            print('adding X, Y and Z-IPF maps')
            with h5py.File(file_path, 'r') as f:
                if 'IPF001' in f['LabDCT/Data']:
                    IPF001_map = f['LabDCT/Data/IPF001'][()].transpose(2, 1, 0, 3)
                else:
                    IPF001_map = m.create_IPF_map(axis=np.array([0., 0., 1.]))
                m.add_field(gridname='CellData', fieldname='IPF001_map',
                            array=IPF001_map, 
                            compression_options=m.default_compression_options)
                if 'IPF010' in f['LabDCT/Data']:
                    IPF010_map = f['LabDCT/Data/IPF010'][()].transpose(2, 1, 0, 3)
                else:
                    IPF010_map = m.create_IPF_map(axis=np.array([0., 1., 0.]))
                m.add_field(gridname='CellData', fieldname='IPF010_map',
                            array=IPF010_map, 
                            compression_options=m.default_compression_options)
                if 'IPF100' in f['LabDCT/Data']:
                    IPF100_map = f['LabDCT/Data/IPF100'][()].transpose(2, 1, 0, 3)
                else:
                    IPF100_map = m.create_IPF_map(axis=np.array([1., 0., 0.]))
                m.add_field(gridname='CellData', fieldname='IPF100_map',
                            array=IPF100_map, 
                            compression_options=m.default_compression_options)
                del IPF001_map, IPF010_map, IPF100_map
        return m

    @staticmethod
    def from_dct(data_dir='.', grain_file='index.mat', use_dct_path=True,
                 vol_file='phase_01_vol.mat', vol_key='vol',
                 mask_file='volume_mask.mat', mask_key='vol',
                 phase_file='volume.mat', phase_key='phases',
                 rod_map_file='phase_01_vol.mat', rod_map_key='dmvol',
                 roi=None, verbose=True):
        """Create a microstructure from a DCT reconstruction.

        DCT reconstructions are stored in several files. The indexed grain
        informations are stored in a matlab file in the '4_grains/phase_01'
        folder. Then, the reconstructed volume file (labeled image) is stored
        in the '5_reconstruction' folder as an hdf5 file, possibly stored
        alongside a mask file coming from the absorption reconstruction.

        :param str data_dir: the path to the folder containing the
                              reconstruction data.
        :param str grain_file: the name of the file containing grains info.
        :param bool use_dct_path: if True, the grain_file should be located in
            4_grains/phase_01 folder and the vol_file and mask_file in the
            5_reconstruction folder.
        :param str vol_file: the name of the volume file.
        :param str vol_key: the key to access the volume in the hdf5 file.
        :param str mask_file: the name of the mask file.
        :param str mask_key: the key to access the mask in the hdf5 file.
        :param str phase_file: the name of the file containing the phase map array.
        :param str phase_key: the key to access the phase array in the hdf5 file.
        :param str rod_map_file: the name of the file containing the local
            orientations as a rodrigues vector array.
        :param str rod_map_key: the key to access the phase rodrigues vector
            array in the file.
        :param list roi: specify a region of interest by a list of 6
            integers in the form [x1, x2, y1, y2, z1, z2] to crop the grain map.
        :param bool verbose: activate verbose mode.
        :return: a `Microstructure` instance created from the DCT reconstruction.
        """
        if data_dir == '.':
            data_dir = os.getcwd()
        if isinstance(data_dir, Path):
            data_dir = str(data_dir)
        if data_dir.endswith(os.sep):
            data_dir = data_dir[:-1]
        
        scan = data_dir.split(os.sep)[-1]
        print('creating microstructure for DCT scan %s' % scan)
        filename = os.path.join(data_dir, scan)
        micro = Microstructure(filename=filename, overwrite_hdf5=True)
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
        index = loadmat(index_path, simplify_cells=True)
        voxel_size = index['cryst']['pixelsize']
        # grab the crystal lattice
        lattice_params = index['cryst']['latticepar']
        sym = Symmetry.from_string(index['cryst']['lattice_system'])
        print('creating crystal lattice {} ({}) with parameters {}'
              ''.format(index['cryst']['name'], sym, lattice_params))
        lattice_params[:3] /= 10  # angstrom to nm
        lattice = Lattice.from_parameters(*lattice_params, symmetry=sym)
        # create a crystalline phase
        phase_name = index['cryst']['name']
        phase = CrystallinePhase(name=phase_name, lattice=lattice)
        micro.set_phase(phase)
        # add all grains to the microstructure
        grain = micro.grains.row
        for i in range(len(index['grain'])):
            grain['idnumber'] = index['grain'][i]['id']
            grain['orientation'] = index['grain'][i]['R_vector']
            grain['center'] = index['grain'][i]['center']
            grain.append()
        micro.grains.flush()

        # setup file pathes
        if use_dct_path:
            grain_map_path = os.path.join(data_dir, '5_reconstruction', vol_file)
            mask_path = os.path.join(data_dir, '5_reconstruction', mask_file)
            phase_path = os.path.join(data_dir, '5_reconstruction', phase_file)
            rod_map_path = os.path.join(data_dir, '5_reconstruction', rod_map_file)
        else:
            grain_map_path = os.path.join(data_dir, vol_file)
            mask_path = os.path.join(data_dir, mask_file)
            phase_path = os.path.join(data_dir, phase_file)
            rod_map_path = os.path.join(data_dir, rod_map_file)
        # load the grain map if available
        if os.path.exists(grain_map_path):
            try:
                with h5py.File(grain_map_path, 'r') as f:
                    # because how matlab writes the data, we need to swap X and Z
                    # axes in the DCT volume
                    print(f[vol_key][()].shape)
                    print(voxel_size)
                    grain_map = f[vol_key][()].transpose(2, 1, 0)
            except OSError:
                # fallback on matlab format
                grain_map = loadmat(grain_map_path)[vol_key]
            # work out the ROI
            if roi:
                x1, x2, y1, y2, z1, z2 = roi
                grain_map = grain_map[x1:x2, y1:y2, z1:z2]
            micro.set_grain_map(grain_map, voxel_size)
            location = micro._get_parent_name(micro.active_grain_map)
            origin = -0.5 * micro.get_voxel_size() * np.array(micro.get_grain_map().shape)
            micro.set_origin(location, origin)
            if verbose:
                print('loaded grain ids volume with shape: {}'.format(
                    micro.get_grain_map().shape))
            print('computing grain bounding boxes')
            micro.recompute_grain_bounding_boxes()
        # load the mask if available
        if os.path.exists(mask_path):
            try:
                with h5py.File(mask_path, 'r') as f:
                    mask = f[mask_key][()].transpose(2, 1, 0).astype(np.uint8)
            except OSError:
                # fallback on matlab format
                mask = loadmat(mask_path)[mask_key]
            if roi:
                x1, x2, y1, y2, z1, z2 = roi
                mask = mask[x1:x2, y1:y2, z1:z2]
            # check if mask shape needs to be zero padded
            if not mask.shape == micro.get_grain_map().shape:
                offset = np.array(micro.get_grain_map().shape) - np.array(mask.shape)
                padding = [(o // 2, o // 2) for o in offset]
                print('mask padding is {}'.format(padding))
                mask = np.pad(mask, padding, mode='constant')
            print('now mask shape is {}'.format(mask.shape))
            micro.set_mask(mask, voxel_size)
            if verbose:
                print('loaded mask volume with shape: {}'.format(micro.get_mask().shape))
        # load the phase map if available
        if os.path.exists(phase_path):
            try:
                with h5py.File(phase_path, 'r') as f:
                    phase_map = f[phase_key][()].transpose(2, 1, 0).astype(np.uint8)
            except OSError:
                # fallback on matlab format
                phase_map = loadmat(phase_path)[phase_key]
            if roi:
                x1, x2, y1, y2, z1, z2 = roi
                phase_map = phase_map[x1:x2, y1:y2, z1:z2]
            micro.set_phase_map(phase_map, voxel_size)
            if verbose:
                print('loaded phase_map volume with shape: {}'.format(micro.get_phase_map().shape))
        # load the orientation map if available
        if os.path.exists(rod_map_path):
            try:
                with h5py.File(rod_map_path, 'r') as f:
                    orientation_map = f[rod_map_key][()].transpose(3, 2, 1, 0).astype(float)
                    if roi:
                        x1, x2, y1, y2, z1, z2 = roi
                        orientation_map = orientation_map[x1:x2, y1:y2, z1:z2, :]
                    micro.set_orientation_map(orientation_map)
                    if verbose:
                        print('loaded orientation_map volume with shape: {}'.format(micro.get_orientation_map().shape))
            except KeyError:
                print('warning, local orientation not found in %s' % rod_map_file)
                # give up on local orientations
                pass
        return micro

    @staticmethod
    def from_legacy_h5(file_path, filename=None):
        """read a microstructure object from a HDF5 file created by pymicro
        until version 0.4.5.

        :param str file_path: the path to the file to read.
        :return: the new `Microstructure` instance created from the file.
        """
        with h5py.File(file_path, 'r') as f:
            if filename is None:
                filename = f.attrs['microstructure_name']
            micro = Microstructure(name=filename, overwrite_hdf5=True)
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
                micro.recompute_grain_bounding_boxes()
                micro.recompute_grain_volumes()
            if 'mask' in f['CellData']:
                micro.set_mask(f['CellData/mask'][()],
                               f['CellData/mask'].attrs['voxel_size'])
            return micro

    @staticmethod
    def from_ebsd(file_path, roi=None, ds=1, tol=5., min_ci=0.2, min_size=0.0, 
                  phase_list=None, ref_frame_id=2, grain_ids=None):
        """"Create a microstructure from an EBSD scan.

        :param str file_path: the path to the file to read.
        :param list roi: a list of 4 integers in the form [x1, x2, y1, y2]
            to crop the EBSD scan.
        :param int ds: integer value to downsample the data.
        :param float tol: the misorientation angle tolerance to segment
            the grains (default is 5 degrees).
        :param float min_ci: minimum confidence index for a pixel to be a valid
            EBSD measurement.
        :param list phase_list: a list of CrystallinePhase to overwrite the ones
            in the file, this is particularly useful for osc files as phases
            cannot be read from them at the moment.
        :return: a new instance of `Microstructure`.
        """
        # TODO: segmentation super slow, need to reconsider the process
        #       (Dream3D hundreds to thousand times faster)
        # Get name of file and create microstructure instance
        name = os.path.splitext(os.path.basename(file_path))[0]
        micro = Microstructure(name=name, autodelete=False, overwrite_hdf5=True)
        from pymicro.crystal.ebsd import OimScan
        # Read raw EBSD file, enforce spatial reference frame for orientation data
        scan = OimScan.from_file(file_path, crop=(roi, ds),
                                 use_spatial_ref_frame=True, ref_frame_id=ref_frame_id)
        if phase_list:
            scan.phase_list = phase_list
        # sort the phase list so that it is not renumbered and consistent with the phase map
        phase_ids = [phase.phase_id for phase in scan.phase_list]
        scan.phase_list = np.array(scan.phase_list)[np.argsort(phase_ids)]
        micro.set_phases(scan.phase_list)
        iq = scan.iq
        ci = scan.ci
        euler = scan.euler
        mask = np.ones_like(scan.phase, dtype=np.uint8)
        # check if we use an existing segmentation
        if grain_ids is None:
            # segment the grains
            scan.seg_params['tol'] = tol
            scan.seg_params['min_ci'] = min_ci
            scan.seg_params['min_size'] = min_size
            grain_ids = scan.segment_grains()
        else:
            print('using existing segmentation containing %d grains, size is ' % 
                  len(np.unique(grain_ids)), grain_ids.shape)
        voxel_size = np.array([scan.xStep, scan.yStep])
        micro.set_grain_map(grain_ids, voxel_size)
        micro.set_phase_map(scan.phase)
        micro.set_mask(mask)

        # add each array to the data file to the CellData image Group
        micro.add_field(gridname='CellData', fieldname='iq', array=iq,
                        replace=True)
        micro.add_field(gridname='CellData', fieldname='ci', array=ci,
                        replace=True)
        micro.add_field(gridname='CellData', fieldname='euler',
                        array=euler, replace=True)

        # Fill GrainDataTable
        grains = micro.grains.row
        grain_ids_list = np.unique(grain_ids).tolist()
        time.sleep(0.2)  # prevent tqdm from messing in the output
        for gid in tqdm(grain_ids_list, desc='creating new grains'):
            if gid == 0:
                continue
            # get the phase for this grain
            phase_grain = scan.phase[np.where((grain_ids == gid) & (scan.phase > 0))].astype(int)
            values, counts = np.unique(phase_grain, return_counts=True)
            if len(counts) == 0:
                continue
            grain_phase_id = values[np.argmax(counts)]
            if len(values) > 1:
                # all indexed pixel of this grain must have the same phase id
                print('warning, phase for grain %d is not unique, using value %d' % (gid, grain_phase_id))
            grain_phase_index = [phase.phase_id for phase in scan.phase_list].index(grain_phase_id)
            sym = scan.phase_list[grain_phase_index].get_symmetry()
            # compute the mean orientation for this grain
            euler_grain = scan.euler[np.where((grain_ids == gid) & (scan.phase > 0))]
            rods = Orientation.eu2ro(euler_grain)
            rods = np.atleast_2d(rods)  # for one pixel grains
            o_tsl = Orientation.compute_mean_orientation(rods, symmetry=sym)
            grains['idnumber'] = gid
            grains['phase'] = scan.phase_list[grain_phase_index].phase_id
            grains['orientation'] = o_tsl.rod
            grains.append()
        micro.grains.flush()
        #print('computing grains geometry')
        #micro.recompute_grain_bounding_boxes()
        #micro.recompute_grain_centers(verbose=False)
        #micro.recompute_grain_volumes(verbose=False)
        micro.sync()
        return micro

    @staticmethod
    def merge_microstructures(micros, overlap, translation_offset=[0, 0, 0],
                              key_array_to_merge='/CellData', plot=False):
        """Merge two `Microstructure` instances together.

        The function works for two microstructures with grain maps and an
        overlap between them along the Z direction. Temporary `Microstructures`
        restricted to the overlap regions are created and grains are matched
        between the two based on a disorientation tolerance.

        The method is written such that the end slices of the grain map of the
        first scan in the `micros` list must correspond to the beginning of the
        grain map of the second scan.

        .. note::

          The overlap value can be negative, in that case, the first and last
          layers of the two microstructure are used to match grains and the
          merged microstructure is constructed with a gap that may be filled
          later with the `dilate_grains` method.

        .. note::

          The two microstructure must have the same crystal lattice and the
          same voxel_size for this method to run.

        :param str key_array_to_merge: the path to the arrays to merge in the
            hdf5 files.
        :param list micros: a list containing the two microstructures to merge.
        :param int overlap: the overlap to use.
        :param list translation_offset: a manual translation (in voxels) offset
            to add to the result.
        :param bool plot: a flag to plot some results.
        :return: a new `Microstructure` instance containing the merged
                 microstructure.
        """
        # perform some sanity checks
        for i in range(2):
            if micros[i]._is_empty('grain_map'):
                raise ValueError('microstructure instance %s must have an '
                                 'associated grain_map attribute'
                                 % micros[i].get_sample_name())
        if micros[0].get_phase() != micros[1].get_phase():
            raise ValueError('both microstructure must have the same phase')
        phase = micros[0].get_phase()
        if micros[0].get_voxel_size() != micros[1].get_voxel_size():
            raise ValueError('both microstructure must have the same'
                             ' voxel size')
        voxel_size = micros[0].get_voxel_size()

        if len(micros[0].get_grain_map().shape) == 2:
            raise ValueError('Microstructures to merge must be tridimensional')
        if len(micros[1].get_grain_map().shape) == 2:
            raise ValueError('Microstructures to merge must be tridimensional')

        # create two microstructures for the two overlapping regions
        z1 = micros[0].get_grain_map().shape[2] - max(1, overlap)
        z2 = max(1, overlap)
        micro1_ol = micros[0].crop(z_start=z1, autodelete=True)
        micro2_ol = micros[1].crop(z_end=z2, autodelete=True)
        micros_ol = [micro1_ol, micro2_ol]

        # match grain from micros_ol[1] to micros_ol[0] (the reference)
        matched, _, unmatched = micros_ol[0].match_grains(micros_ol[1],
                                                          verbose=True)
        #np.save('matched.npy', matched)
        #matched = np.load('matched.npy')

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
        #if overlap < 0:
        #    translation_voxel[2] -= overlap
        #    print('translation in voxels updated to {}'.format(translation_voxel))

        # now delete overlapping microstructures
        del micro1_ol, micro2_ol

        # look at ids in the reference volume
        ids_ref = np.unique(micros[0].get_grain_map())
        ids_ref_list = ids_ref.tolist()
        if -1 in ids_ref_list:
            ids_ref_list.remove(-1)  # grain overlap
        if 0 in ids_ref_list:
            ids_ref_list.remove(0)  # background
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

        # prepare a volume with the same size as the second grain map,
        # with grain ids renumbered and (X, Y) translations applied.
        grain_map = micros[1].get_grain_map()
        grain_map_translated = grain_map.copy()
        print('renumbering grains in the overlap region of volume %s'
              % micros[1].get_sample_name())
        for match in matched:
            ref_id, other_id = match
            #TODO get this done faster (could be factorized in the method renumber_grains
            print('replacing %d by %d' % (other_id, ref_id))
            grain_map_translated[grain_map == other_id] = ref_id
            try:
                ids_mrg_list.remove(other_id)
            except ValueError:
                # this can happen if a reference grain was matched to more than 1 grain
                print('%d was not in list anymore' % other_id)
        # also renumber the rest using the offset
        renumbered_grains = []
        for i, other_id in enumerate(ids_mrg_list):
            new_id = id_offset + i + 1
            grain_map_translated[grain_map == other_id] = new_id
            print('replacing %d by %d' % (other_id, new_id))
            renumbered_grains.append([other_id, new_id])

        # apply translation along the (X, Y) axes
        print(grain_map_translated.shape)
        shifts = translation_voxel[:2].tolist() + [0]
        print('shifts:', shifts)
        grain_map_translated = ndimage.shift(grain_map_translated,
                                             shifts, order=0, cval=0)

        check = overlap // 2
        print('check=%d' % check)
        print(overlap)
        print(translation_voxel[2] + check)
        if plot and overlap > 0:
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
            plt.title('voxels that are identical')
            plt.savefig('merging_check1.pdf')

        # merging finished, building the new microstructure instance
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
                                      phase=phase, overwrite_hdf5=True)
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

        # start the merging: the first volume in the list is the reference
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
        print('shape merged: {}'.format(shape_merged))

        if overlap > 0:
            # look at vertices with the same label
            same_voxel = (micros[0].get_grain_map()[:, :, translation_voxel[2]:]
                          == grain_map_translated[:, :, :overlap])
            # look at vertices with a single label
            single_voxels_0 = ((micros[0].get_grain_map()[:, :,
                                translation_voxel[2]:] > 0)
                               & (grain_map_translated[:, :, :overlap] == 0))
            single_voxels_1 = ((grain_map_translated[:, :, :overlap] > 0)
                               & (micros[0].get_grain_map()[:, :,
                                  translation_voxel[2]:] == 0))

        # factorize the merging of all arrays
        array_names = micros[0].get_node('%s/Field_index' % key_array_to_merge)[()]
        for index, array_name in enumerate(array_names):
            array_name = str(array_names[index], 'utf-8)')
            print('merging array %s' % array_name)
            if not micros[0]._is_empty(array_name) and not micros[1]._is_empty(array_name):
                a_bot = micros[0].get_field(array_name)
                array_type = a_bot.dtype
                print(gmap_shape[0][2], a_bot.shape[2])
                if array_name == 'grain_map':
                    # use our renumbered grain map for the second volume
                    a_top = grain_map_translated
                else:
                    # simply translate the second volume
                    #a_top = np.roll(micros[1].get_field(array_name),
                    #                translation_voxel[:2], (0, 1))
                    array_to_shift = micros[1].get_field(array_name)
                    the_shifts = np.zeros_like(array_to_shift.shape)
                    the_shifts[:len(shifts)] = shifts
                    a_top = ndimage.shift(micros[1].get_field(array_name),
                                          the_shifts, order=0, cval=0)

                # merging the two arrays
                shape_array_merged = shape_merged
                merged = np.zeros(shape_array_merged, dtype=array_type)
                if overlap > 0:
                    print('shape same_voxel before test ndim==4', same_voxel.shape)
                    if a_bot.ndim == 4:
                        # handle field arrays (typically local orientation map)
                        shape_array_merged = np.concatenate((shape_merged, [a_bot.shape[3]]))
                        same_voxel = np.expand_dims(same_voxel, axis=same_voxel.ndim)
                        same_voxel = np.concatenate((same_voxel, same_voxel, same_voxel), axis=3)
                        print('shape same_voxel', same_voxel.shape)
                        single_voxels_0 = np.expand_dims(single_voxels_0, axis=single_voxels_0.ndim)
                        single_voxels_0 = np.concatenate((single_voxels_0, single_voxels_0, single_voxels_0), axis=3)
                        print('shape single_voxels_0', single_voxels_0.shape)
                        single_voxels_1 = np.expand_dims(single_voxels_1, axis=single_voxels_1.ndim)
                        single_voxels_1 = np.concatenate((single_voxels_1, single_voxels_1, single_voxels_1), axis=3)
                        print('shape single_voxels_1', single_voxels_1.shape)
                    # add the non-overlapping part of the 2 volumes as is
                    merged[:, :, :gmap_shape[0][2] - overlap] = (a_bot[:, :, :-overlap])
                    merged[:, :, gmap_shape[0][2]:] = (a_top[:, :, overlap:])

                    # copy voxels with the same grain ids
                    #TODO check if we should use the mean value here
                    merged[:, :, translation_voxel[2]:a_bot.shape[2]] = (
                            a_top[:, :, :overlap] * same_voxel)

                    # copy voxels with single values
                    merged[:, :, translation_voxel[2]:a_bot.shape[2]] += (
                        (a_bot[:, :, translation_voxel[2]:]
                         * single_voxels_0).astype(array_type))
                    merged[:, :, translation_voxel[2]:a_bot.shape[2]] += (
                        (a_top[:, :, :overlap]
                         * single_voxels_1).astype(array_type))
                else:
                    # overlap is < 0
                    print('shape merged: {}'.format(merged.shape))
                    print(a_bot.shape, a_top.shape)
                    print('gmap_shape[1][2]: {}'.format(gmap_shape[1][2]))
                    merged[:, :, :gmap_shape[0][2]] = a_bot
                    merged[:, :, -gmap_shape[1][2]:] = a_top

                if plot:
                    fig = plt.figure(figsize=(14, 10))
                    ax1 = fig.add_subplot(1, 2, 1)
                    ax1.imshow(merged[:, shape_merged[1] // 2, :].T)
                    plt.axis('off')
                    plt.title('%s XZ slice' % array_name)
                    ax2 = fig.add_subplot(1, 2, 2)
                    ax2.imshow(merged[shape_merged[0] // 2, :, :].T)
                    plt.axis('off')
                    plt.title('%s YZ slice' % array_name)
                    plt.savefig('merging_check%d.pdf' % (3 + index))

                if array_name == 'grain_map':
                    print('assigning merged grain map')
                    merged_micro.set_grain_map(merged, voxel_size)
                else:
                    try:
                        print('assigning new fused array: %s' % array_name)
                        merged_micro.add_field(gridname=key_array_to_merge,
                                               fieldname=array_name,
                                               array=merged, replace=True,
                                               indexname=array_name,
                                               compression_options=micros[0].default_compression_options)
                    except ValueError:
                        print('skipping array %s' % array_name)

        # recompute the geometry of the grains
        print('updating grain geometry')
        merged_micro.recompute_grain_bounding_boxes()
        merged_micro.recompute_grain_centers()
        merged_micro.recompute_grain_volumes()

        merged_micro.sync()
        return merged_micro

    def get_grain_boundaries_map(self, kernel_size=3):
        """
        method to compute grain boundaries map using microstructure grain_map
        """
        x, y, z = self.get_grain_map().shape
        grain_boundaries_map = np.zeros_like(self.get_grain_map())
        pad_grain_map = np.pad(self.get_grain_map(), pad_width=1)
                
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    kernel = pad_grain_map[i:i+kernel_size//2+1,
                                        j:j+kernel_size//2+1,
                                        k:k+kernel_size//2+1]
                    mean_kernel = np.mean(kernel)
                    inten_level = np.abs(grain_map[i, j, k] - mean_kernel)
                    if inten_level > 0:
                        grain_boundaries_map[i, j, k] = 1

        return grain_boundaries_map
    
    def resample(self, resampling_factor, resample_name=None, autodelete=False,
            recompute_geometry=True, verbose=False):
        """
        Resample the microstructure by a given factor to create a new one.

        This method resamples the CellData image group to a new microstructure,
        and adapts the GrainDataTable to the resampled.

        :param int resample_factor: the factor used for resolution degradation
        :param str resample_name: the name for the resampled microstructure
            (the default is to append '_resampled' to the initial name).
        :param bool autodelete: a flag to delete the microstructure files
            on the disk when it is not needed anymore.
        :param bool recompute_geometry: if `True` (default), recompute the
            grain centers, volumes, and bounding boxes in the resampled
            microstructure. Use `False` when using a resample that do not cut
            grains, for instance when resampling a microstructure within the
            mask, to avoid the heavy computational cost of the grain geometry
            data update.
        :param bool verbose: activate verbose mode.
        :return: a new `Microstructure` instance with the resampled grain map.
        """
        if self._is_empty('grain_map'):
            print('warning: needs a grain map to resample the microstructure')
            return
        # input default values for bounds if not specified
        if not resample_name:
            resample_name = self.get_sample_name() + \
                        (not self.get_sample_name().endswith('_')) * '_' + 'resampled' + \
                            '_' + str(resampling_factor)
        print('RESAMPLING: %s' % resample_name)
        # create new microstructure dataset
        micro_resampled = Microstructure(name=resample_name, overwrite_hdf5=True,
                                    phase=self.get_phase(),
                                    autodelete=autodelete)
        if self.get_number_of_phases() > 1:
            for i in range(2, self.get_number_of_phases()):
                micro_resampled.add_phase(self.get_phase(phase_id=i))
        micro_resampled.default_compression_options = self.default_compression_options
        print('resampling microstructure to %s' % micro_resampled.h5_file)
        # Resize all CellData fields
        image_group = self.get_node('CellData')
        spacing = self.get_attribute('spacing', 'CellData')
        FIndex_path = '%s/Field_index' % image_group._v_pathname
        field_list = self.get_node(FIndex_path)

        dims = self.get_attribute('dimension', 'CellData')
        # Fields dimensions should be multiples of 2 (AMITEX requirement for Zoom Structural purposes, cf L. Gelebart)
        if len(dims) == 3:
            X, Y, Z = dims
            end_X, end_Y, end_Z = resampling_factor * np.array([X//resampling_factor,
                                                                Y//resampling_factor,
                                                                Z//resampling_factor])
        elif len(dims) == 2:
            X, Y = dims
            end_X, end_Y = resampling_factor * np.array([X//resampling_factor,
                                                        Y//resampling_factor])
        else:
            raise ValueError('CellData should be either 2D or 3D')

        resampled_voxel_size = self.get_voxel_size() * resampling_factor  

        for name in field_list:
            field_name = name.decode('utf-8')
            print('resampling field %s' % field_name)
            field = self.get_field(field_name)
            if not self._is_empty(field_name):
                if self._get_group_type('CellData') == '2DImage':
                    field_resampled = field[:end_X:resampling_factor, :end_Y:resampling_factor, ...]

                else:
                    field_resampled = field[:end_X:resampling_factor, :end_Y:resampling_factor,
                                    :end_Z:resampling_factor, ...]
                empty = micro_resampled.get_attribute(attrname='empty',
                                                nodename='CellData')
                if empty:
                    micro_resampled.add_image_from_field(
                        field_array=field_resampled, fieldname=field_name,
                        imagename='CellData', location='/',
                        spacing=spacing, replace=True)
                else:
                    micro_resampled.add_field(gridname='CellData',
                                        fieldname=field_name,
                                        array=field_resampled, replace=True)
                    print(field_resampled.shape)

        # update the origin of the image group according to the resampling
        if verbose:
            print('resampled dataset:')
            print(micro_resampled)
        micro_resampled.set_voxel_size('CellData', resampled_voxel_size)
        print('Updating active grain map')
        print(micro_resampled.get_grain_map().shape)
        micro_resampled.set_active_grain_map('CellData_%s' % self.active_grain_map)
        print(micro_resampled.get_grain_map().shape)
        micro_resampled.add_grains_in_map()
        grain_ids = np.unique(micro_resampled.get_grain_map())
        orientation = []
        for gid in grain_ids:
            if not gid > 0:
                continue
            orientation.append(self.get_grain(gid).orientation.rod)
        orientation = np.array(orientation)
        micro_resampled.set_orientations(orientation)
        micro_resampled.remove_grains_not_in_map()
        max_grain = micro_resampled.get_grain_ids()[-1]
        nb_grain = micro_resampled.get_number_of_grains()
        if max_grain > nb_grain:
            print('renumbering in progress : %i - %i ' % (max_grain, nb_grain))
            micro_resampled.renumber_grains()
            
        print('%d grains in resampled microstructure' % micro_resampled.grains.nrows)
        micro_resampled.grains.flush()
        
        # recompute the grain geometry
        if recompute_geometry:
            print('updating grain geometry')
            micro_resampled.recompute_grain_bounding_boxes(verbose)
            micro_resampled.recompute_grain_centers(verbose)
            micro_resampled.recompute_grain_volumes()
            
        return micro_resampled

