"""The dct module provide helpers functions to work with experimental diffraction contrast tomography data.
"""
import os
import h5py
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt, cm
from pymicro.xray.experiment import ForwardSimulation
from pymicro.crystal.lattice import HklPlane, Symmetry
from pymicro.xray.xray_utils import lambda_keV_to_nm, radiograph, radiographs
from pymicro.crystal.microstructure import Grain, Orientation
from pymicro.file.file_utils import edf_read, edf_write


class Xrd3dForwardSimulation(ForwardSimulation):
    """Class to represent a 3DXRD Forward Simulation."""

    def __init__(self, verbose=False):
        ForwardSimulation.__init__(self, '3dxrd', verbose=verbose)
        self.hkl_planes = []
        self.max_miller = 5
        self.energy_keV = 40.
        self.omega = 0.  # rotation angle in deg
        self.integration_range = 1.  # deg
        self.exp = None

    def set_experiment(self, experiment):
        """Attach an X-ray experiment to this simulation."""
        self.exp = experiment

    def set_energy_keV(self, energy_keV):
        """Set the energy of the 3DXRD experiment."""
        self.energy_keV = energy_keV

    def set_hkl_planes(self, hkl_planes):
        self.hkl_planes = hkl_planes

    def setup(self, include_grains=None):
        """Setup the forward simulation."""
        pass

    def select_hkl_planes(self, hkl_planes, gid=1, margin=10):
        """Filter the list of hkl planes to keep only the ones diffracting
        on the detector.

        :param list hkl_planes: the list of hkl plane to filter.
        :param int gid: the grain id to use.
        :param int margin: a pixel margin to take into account.
        :return: the list of hkl planes diffraction on the detector within
        the margin.
        """
        lambda_nm = lambda_keV_to_nm(self.energy_keV)
        X = np.array([1., 0., 0.]) / lambda_nm
        sample = self.exp.get_sample()
        grain_orientation = sample.get_grain(gid).orientation
        gt = grain_orientation.orientation_matrix().transpose()
        detector = self.exp.get_active_detector()
        position = sample.get_grain_centers(id_list=[gid])[0]
        hkl_planes_dif = []
        omegas = []
        G_vectors = []
        K_vectors = []
        origins = []
        for hkl_plane in hkl_planes:
            try:
                w1, w2 = grain_orientation.dct_omega_angles(hkl_plane,
                                                            self.energy_keV,
                                                            verbose=False)
            except ValueError:
                # skip this plane
                continue
            for omega in [w1, w2]:
                if self.omega < omega < (self.omega + self.integration_range):
                    hkl_planes_dif.append(hkl_plane)
                    omegas.append(omega)
                    omegar = omega * np.pi / 180
                    # grain diffracting at this angle with this hkl plane
                    R = np.array([[np.cos(omegar), -np.sin(omegar), 0],
                                  [np.sin(omegar), np.cos(omegar), 0],
                                  [0, 0, 1]])
                    G = np.dot(R, np.dot(gt, hkl_plane.scattering_vector()))
                    G_vectors.append(G)
                    K = X + G
                    K_vectors.append(K)
                    # add the origin of this diffraction event as the center of the grain at this angle
                    origins.append(np.dot(R, position))
        print('%d diffraction vectors at this omega angle' % len(K_vectors))
        G_vectors = np.array(G_vectors)  # shape (n, 3)
        K_vectors = np.array(K_vectors)  # shape (n, 3)
        origins = np.array(origins)
        OR_vectors = detector.project_along_directions(K_vectors, origins)
        uv = detector.lab_to_pixel(OR_vectors).astype(np.int)
        # look at which hkl plane diffracts on the detector within the given margin
        on_det = np.where((-margin < uv[:, 0]) &
                          (uv[:, 0] < detector.get_size_px()[0] + margin) &
                          (-margin < uv[:, 1]) &
                          (uv[:, 1] < detector.get_size_px()[1] + margin))
        print('%d diffraction planes on the detector among %d' % (len(on_det[0]), len(uv)))
        hkl_planes_dif = [hkl_planes_dif[i] for i in on_det[0]]
        omegas_dif = [omegas[i] for i in on_det[0]]
        '''
        print(hkl_planes_dif)
        print([G_vectors[i] for i in on_det[0]])
        print([K_vectors[i] for i in on_det[0]])
        print([origins[i] for i in on_det[0]])
        print([uv[i] for i in on_det[0]])
        '''
        return hkl_planes_dif, omegas_dif

    def fsim_grain(self, gid=1):
        """Forward simulation for a given grain.

        Optimized version with matrix multiplication.

        :param int gid: the grain id number to simulate.
        """
        #TODO move this to some relevant place
        from pymicro.xray.laue import build_list
        lambda_nm = lambda_keV_to_nm(self.energy_keV)
        X = np.array([1., 0., 0.]) / lambda_nm
        sample = self.exp.get_sample()
        grain_orientation = sample.get_grain(gid).orientation
        gt = grain_orientation.orientation_matrix().transpose()
        lattice = sample.get_lattice()
        detector = self.exp.get_active_detector()
        if self.verbose:
            print('Forward Simulation for grain %d' % gid)
        sample.geo.discretize_geometry(grain_id=gid)

        if len(self.hkl_planes) == 0:
            print('warning: no reflection defined for this simulation, using all planes with max miller=%d' % self.max_miller)
            self.set_hkl_planes(build_list(lattice=lattice, max_miller=self.max_miller))
        # preprocess hkl planes by filtering out the ones not diffracting close to the detector
        hkl_planes_dif, omegas_dif = self.select_hkl_planes(self.hkl_planes, gid=gid)
        n_hkl_dif = len(hkl_planes_dif)
        print('after filtering we have %d hkl planes for this grain' % n_hkl_dif)
        positions = sample.geo.get_positions()  # size n_vox, with 3 elements items
        n_vox = positions.shape[0]

        # look at diffraction events for each position
        # we will create arrays origins and K_vectors with size n_vox * n_hkl_dif
        K_vectors = np.empty((n_hkl_dif * n_vox, 3), dtype=float)
        origins = np.empty((n_hkl_dif * n_vox, 3), dtype=float)

        for i_hkl in range(n_hkl_dif):
            # here we assume a constant orientation per grain
            hkl_plane = hkl_planes_dif[i_hkl]
            omega = omegas_dif[i_hkl]
            omegar = omega * np.pi / 180
            # grain diffracting at this angle with this hkl plane
            R = np.array([[np.cos(omegar), -np.sin(omegar), 0],
                          [np.sin(omegar), np.cos(omegar), 0],
                          [0, 0, 1]])
            G = np.dot(R, np.dot(gt, hkl_plane.scattering_vector()))
            K = X + G
            for i_vox in range(n_vox):
                index = i_vox * n_hkl_dif + i_hkl
                K_vectors[index] = K
                # origin is the voxel position at this angle
                position = positions[i_vox, :]
                origins[index] = np.dot(R, position)

        # call the fsim_3dxrd function for vectorized code
        '''
        omegas, origins, G_vectors = Xrd3dForwardSimulation.fsim_laue(
            grain_orientation, hkl_planes, positions)
        '''

        # with diffraction informations, project them on the detector
        OR_vectors = detector.project_along_directions(K_vectors, origins)
        uv = detector.lab_to_pixel(OR_vectors).astype(np.int)

        # now construct a boolean list to select the diffraction spots
        on_det = np.where((0 < uv[:, 0]) &
                          (uv[:, 0] < detector.get_size_px()[0]) &
                          (0 < uv[:, 1]) &
                          (uv[:, 1] < detector.get_size_px()[1]))
        if self.verbose:
            print('%d diffraction events on the detector among %d' % (len(on_det[0]), len(uv)))

        # now sum the counts on the detector individual pixels
        data = np.zeros_like(detector.data)
        for spot in uv[on_det]:
            data[spot[0], spot[1]] += 1
        return data, uv, on_det

    def fsim(self):
        """run the forward simulation.

        If the sample has a CAD type of geometry, a single grain (the first from the list) is assumed. In the other
        cases all the grains from the microstructure are used. In particular, if the microstructure has a grain map,
        it can be used to carry out an extended sample simulation.
        """
        import time
        t0 = time.time()
        full_data = np.zeros_like(self.exp.get_active_detector().data)
        sample = self.exp.get_sample()
        # for cad geometry we assume only one grain (the first in the list)
        if sample.geo.geo_type == 'cad':
            full_data += self.fsim_grain(gid=sample.grains[0]['idnumber'])
        else:
            # in the other cases, we use all the grains defined in the microstructure
            for grain_id in sample.get_grain_ids():
                full_data += self.fsim_grain(gid=grain_id)
        if self.verbose:
            duration = int(time.time() - t0)
            print('forward simulation duration: {:d} seconds'.format(duration))
        return full_data

class DctForwardSimulation(ForwardSimulation):
    """Class to represent a Forward Simulation."""

    def __init__(self, verbose=False):
        super(DctForwardSimulation, self).__init__('dct', verbose=verbose)
        self.hkl_planes = []
        self.check = 1  # grain id to display infos in verbose mode
        self.omega_start = 0.
        self.omega_end = 360.
        self.omegas = None
        self.reflexions = []

    def set_hkl_planes(self, hkl_planes):
        self.hkl_planes = hkl_planes

    def set_diffracting_famillies(self, hkl_list):
        """Set the list of diffracting hk planes using a set of families."""
        lattice = self.exp.get_sample().get_lattice()
        hkl_planes = HklPlane.from_families(hkl_list, lattice=lattice, friedel_pairs=True)
        self.set_hkl_planes(hkl_planes)

    def setup(self, omega_step, grain_ids=None):
        """Setup the forward simulation.

        :param float omega_step: the angular integration step (in degrees) use to compute the diffraction comditions.
        :param list grain_ids: a list of grain ids to restrict the forward simulation (use all grains by default).
        """
        assert self.exp.source.min_energy == self.exp.source.max_energy  # monochromatic case
        lambda_keV = self.exp.source.max_energy
        sym = self.exp.sample.get_lattice().get_symmetry()
        self.omegas = np.linspace(self.omega_start, self.omega_end,
                                  num=int((self.omega_end - self.omega_start) / omega_step), endpoint=False)
        self.reflexions = []
        for omega in self.omegas:
            self.reflexions.append([])
        if not grain_ids:
            # list of the grains selected for the forward simulation
            grain_ids = self.exp.sample.get_grain_ids()
        for grain_id in grain_ids:
            for plane in self.hkl_planes:
                h, k, l = plane.miller_indices()
                if sym is Symmetry.hexagonal:
                    (h, k, i, l) = HklPlane.three_to_four_indices(h, k, l)
                try:
                    o = Orientation.from_rodrigues(self.exp.sample.get_grain_rodrigues(id_list=[grain_id])[0])
                    (w1, w2) = o.dct_omega_angles(plane, lambda_keV, verbose=False)
                except ValueError:
                    if self.verbose:
                        print('plane {} does not fulfil the Bragg condition for grain {:d}'.format((h, k, l), grain_id))
                    continue
                # add angles for Friedel pairs
                w3 = (w1 + 180.) % 360
                w4 = (w2 + 180.) % 360
                if self.verbose and grain_id == self.check:
                    print('grain %d, angles for plane %d%d%d: w1=%.3f and w2=%.3f | delta=%.1f' % (grain_id, h, k, l, w1, w2, w1-w2))
                    print('(%3d, %3d, %3d, %3d) -- %6.2f & %6.2f' % (h, k, i, l, w1, w2))
                try:
                    self.reflexions[int(w1 / omega_step)].append([grain_id, (h, k, l)])
                    self.reflexions[int(w2 / omega_step)].append([grain_id, (h, k, l)])
                    self.reflexions[int(w3 / omega_step)].append([grain_id, (-h, -k, -l)])
                    self.reflexions[int(w4 / omega_step)].append([grain_id, (-h, -k, -l)])
                except IndexError:
                    # drop reflexions after omega_end
                    pass


    def load_grain(self, gid=1):
        print('loading grain from file 4_grains/phase_01/grain_%04d.mat' % gid)
        with h5py.File(os.path.join(self.exp.get_sample().data_dir, '4_grains/phase_01/grain_%04d.mat' % gid)) as gmat:
            g = Grain(gid, Orientation.from_rodrigues(gmat['R_vector'][()]))
            g.om_exp = gmat['om_exp'][0, :]
            g.uv_exp = gmat['uv_exp'][:, :]
            g.center = gmat['center'][:, 0]
            try:
                ref_included = gmat['proj/included'][0][0]
                g.included = gmat[ref_included][0, :]
                ref_ondet = gmat['proj/ondet'][0][0]
                g.ondet = gmat[ref_ondet][0, :]
                # grab the projection stack
                ref_stack = gmat['proj']['stack'][0][0]
                g.stack_exp = gmat[ref_stack][()].transpose(1, 2, 0)  # now in [ndx, u, v] form
                g.hklsp = gmat['allblobs/hklsp'][:, :]
            except AttributeError:
                # classic file organization
                g.included = gmat['proj/included'][0, :]
                g.ondet = gmat['proj/ondet'][0, :]
                g.stack_exp = gmat['proj/stack'][()].transpose(1, 2, 0)  # now in [ndx, u, v] form
                # for the Ti7AL data set, we have to hack around the DCT + TT work in progress
                #ref_hklsp = gmat['allblobs/hklsp'][()][0][0]
                #g.hklsp = gmat[ref_hklsp][:, :]
                g.hklsp = gmat['allblobs/hklsp'][:, :]
        self.grain = g
        if self.verbose:
            print('experimental proj stack shape: {}'.format(g.stack_exp.shape))

    def grain_projection_image(self, g_uv, g_proj):
        """Produce a 2D image placing all diffraction spots of a given grain at their respective position on the detector.

        Spots outside the detector are are skipped while those only partially on the detector are cropped accordingly.

        :param g_proj: image stack of the diffraction spots. The first axis is so that g_proj[0] is the first spot \
        the second axis is the horizontal coordinate of the detector (u) and the third axis the vertical coordinate \
        of the detector (v).
        :param g_uv: list or array of the diffraction spot position.
        :returns: a 2D composite image of all the diffraction spots.
        """
        print(len(g_proj), g_uv.shape[1])
        assert len(g_proj) == g_uv.shape[1]
        image = np.zeros(self.exp.get_active_detector().get_size_px())
        for i in range(len(g_proj)):
            spot = g_proj[i]
            if self.verbose:
                print('i={0}, size of spot: {1}'.format(i, spot.shape))
                print('placing diffraction spot at location {0}'.format(g_uv[:, i]))
            add_to_image(image, spot, g_uv[:, i], self.verbose)
        return image

    def grain_projection_exp(self, gid=1):
        """Produce a composite image with all the experimental diffraction spots of this grain on the detector.

        :param int gid: the number of the selected grain.
        :returns: a 2D composite image of all the diffraction spots.
        """
        #self.grain = self.exp.get_sample().get_microstructure().get_grain(gid)
        if not hasattr(self, 'grain') or self.grain.id != gid:
            # load the corresponding grain
            self.load_grain(gid=gid)
        return self.grain_projection_image(self.grain.uv_exp, self.grain.stack_exp)

    def grain_projections(self, omegas, gid=1, data=None, hor_flip=False, ver_flip=False):
        """Compute the projections of a grain at different rotation angles.

        The method compute each projection and concatenate them into a single 3D array in the form [n, u, v]
        with n the number of angles.

        :param list omegas: the list of omega angles to use (in degrees).
        :param int gid: the id of the grain to project (1 default).
        :param ndarray data: the data array representing the grain.
        :param bool hor_flip: a flag to apply a horizontal flip.
        :param bool ver_flip: a flag to apply a vertical flip.
        :return: a 3D array containing the n projections.
        """
        from scipy import ndimage
        if data is None:
            grain_ids = self.exp.get_sample().get_grain_ids()
            print('binarizing grain %d' % gid)
            data = np.where(grain_ids[ndimage.find_objects(grain_ids == gid)[0]] == gid, 1, 0)
        print('shape of binary grain is {}'.format(data.shape))
        stack_sim = radiographs(data, omegas)
        stack_sim = stack_sim.transpose(2, 0, 1)[:, ::-1, ::-1]
        # here we need to account for the detector flips (detector is always supposed to be perpendicular to the beam)
        # by default (u, v) correspond to (-Y, -Z)
        if hor_flip:
            print('applying horizontal flip to the simulated image stack')
            stack_sim = stack_sim[:, ::-1, :]
        if ver_flip:
            print('applying vertical flip to the simulated image stack')
            stack_sim = stack_sim[:, :, ::-1]
        return stack_sim

    def grain_projection_simulation(self, gid=1):
        """Function to compute all the grain projection in DCT geometry and create a composite image.

        :param int gid: the id of the grain to project (1 default).
        """
        print('forward simulation of grain %d' % gid)
        detector = self.exp.get_active_detector()
        lambda_keV = self.exp.source.max_energy
        lambda_nm = lambda_keV_to_nm(lambda_keV)
        X = np.array([1., 0., 0.]) / lambda_nm
        lattice = self.exp.get_sample().get_material()

        if not hasattr(self, 'grain'):
            # load the corresponding grain
            self.load_grain(gid=gid)

        # compute all the omega values
        print('simulating diffraction spot positions on the detector')
        omegas = np.zeros(2 * len(self.hkl_planes))
        g_uv = np.zeros((2, 2 * len(self.hkl_planes)))
        for i, plane in enumerate(self.hkl_planes):
            #print(plane.miller_indices())
            try:
                w1, w2 = self.grain.dct_omega_angles(plane, lambda_keV, verbose=False)
            except ValueError:
                # plane does not fulfil the Bragg condition
                continue
            omegas[2 * i] = w1
            omegas[2 * i + 1] = w2
            for j in range(2):
                omega = omegas[2 * i + j]
                omegar = omega * np.pi / 180
                R = np.array([[np.cos(omegar), -np.sin(omegar), 0], [np.sin(omegar), np.cos(omegar), 0], [0, 0, 1]])
                gt = self.grain.orientation_matrix().transpose()
                G = np.dot(R, np.dot(gt, plane.scattering_vector()))
                K = X + G
                # position of the grain at this rotation angle
                g_pos_rot = np.dot(R, self.grain.center)
                pg = detector.project_along_direction(K, g_pos_rot)
                (up, vp) = detector.lab_to_pixel(pg)[0]
                g_uv[:, 2 * i + j] = up, vp
        # check detector flips
        hor_flip = np.dot(detector.u_dir, [0, -1, 0]) < 0
        ver_flip = np.dot(detector.v_dir, [0, 0, -1]) < 0
        if self.verbose:
            print(detector.u_dir)
            print(detector.v_dir)
            print('detector horizontal flip: %s' % hor_flip)
            print('detector vertical flip: %s' % ver_flip)
        # compute the projections
        stack_sim = self.grain_projections(omegas, gid, hor_flip=hor_flip, ver_flip=ver_flip)
        return self.grain_projection_image(g_uv, stack_sim)

    def dct_projection(self, omega, include_direct_beam=True, att=5):
        """Function to compute a full DCT projection at a given omega angle.

        :param float omega: rotation angle in degrees.
        :param bool include_direct_beam: flag to compute the transmission through the sample.
        :param float att: an attenuation factor used to limit the gray levels in the direct beam.
        :return: the dct projection as a 2D numpy array
        """
        if len(self.reflexions) == 0:
            print('empty list of reflexions, you should run the setup function first')
            return None
        grain_ids = self.exp.get_sample().get_grain_map()
        detector = self.exp.get_active_detector()
        lambda_keV = self.exp.source.max_energy
        lattice = self.exp.get_sample().get_lattice()
        index = np.argmax(self.omegas > omega)
        dif_grains = self.reflexions[index - 1]  # grains diffracting between omegas[index - 1] and omegas[index]
        # intialize image result
        full_proj = np.zeros(detector.get_size_px(), dtype=np.float64)
        lambda_nm = lambda_keV_to_nm(lambda_keV)
        omegar = omega * np.pi / 180
        R = np.array([[np.cos(omegar), -np.sin(omegar), 0], [np.sin(omegar), np.cos(omegar), 0], [0, 0, 1]])

        if include_direct_beam:
            # add the direct beam part by computing the radiograph of the sample without the diffracting grains
            data_abs = np.where(grain_ids > 0, 1, 0)
            for (gid, (h, k, l)) in dif_grains:
                mask_dif = (grain_ids == gid)
                data_abs[mask_dif] = 0  # remove this grain from the absorption
            proj = radiograph(data_abs, omega)[:, ::-1]  # (u, v) axes correspond to (Y, -Z) for DCT detector
            add_to_image(full_proj, proj / att, np.array(full_proj.shape) // 2)

        # add diffraction spots
        X = np.array([1., 0., 0.]) / lambda_nm
        for (gid, (h, k, l)) in dif_grains:
            grain_data = np.where(grain_ids == gid, 1, 0)
            if np.sum(grain_data) < 1:
                print('skipping grain %d' % gid)
                continue
            local_com = np.array(ndimage.measurements.center_of_mass(grain_data, grain_ids))
            print('local center of mass (voxel): {0}'.format(local_com))
            g_center_mm = detector.get_pixel_size() * (local_com - 0.5 * np.array(grain_ids.shape))
            print('center of mass (voxel): {0}'.format(local_com - 0.5 * np.array(grain_ids.shape)))
            print('center of mass (mm): {0}'.format(g_center_mm))
            # compute scattering vector
            gt = self.exp.get_sample().get_grain(gid).orientation_matrix().transpose()
            p = HklPlane(h, k, l, lattice)
            G = np.dot(R, np.dot(gt, p.scattering_vector()))
            K = X + G
            # position of the grain at this rotation angle
            g_pos_rot = np.dot(R, g_center_mm)
            pg = detector.project_along_direction(K, g_pos_rot)
            up, vp = detector.lab_to_pixel(pg)[0]
            if self.verbose:
                print('\n* gid=%d, (%d,%d,%d) plane, angle=%.1f' % (gid, h, k, l, omega))
                print('diffraction vector:', K)
                print('postion of the grain at omega=%.1f is ' % omega, g_pos_rot)
                print('up=%d, vp=%d for plane (%d,%d,%d)' % (up, vp, h, k, l))
            data_dif = grain_data[ndimage.find_objects(grain_ids == gid)[0]]
            proj_dif = radiograph(data_dif, omega)  # (Y, Z) coordinate system
            add_to_image(full_proj, proj_dif[:, ::-1], (up, vp), self.verbose)  # (u, v) axes correspond to (Y, -Z)
        return full_proj


def add_to_image(image, inset, uv, verbose=False):
    """Add an image to another image at a specified position.

    The inset image may be of any size and may only overlap partly on the overall image depending on the location
    specified. In such a case, the inset image is cropped accordingly.

    :param np.array image: the master image taht will be modified.
    :param np.array inset: the inset to add to the image.
    :param tuple uv: the location (center) where to add the inset in the form (u, v).
    :param bool verbose: activate verbose mode (False by default).
    """
    # round the center to the closest integer value
    u = int(uv[0])
    v = int(uv[1])
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


def merge_dct_scans(scan_list, samtz_list, use_mask=False, overlap=-1, root_dir='.', write_to_h5=True):
    """Merge two DCT scans.

    This function build a `Microstructure` instance for each DCT scan and calls merge_microstructures.
    The overlap can be deduced from the samtz values or specified directly.

    :param list scan_list: a list with the two DCT scan names.
    :param list samtz_list: a list with the two samtz value (the order should match the scan names).
    :param bool use_mask: a flag to also merge the absorption masks.
    :param int overlap: the value to use for the overlap if not computed automatically.
    :param str root_dir: the root data folder.
    :param bool write_to_h5: flag to write the result of the merging operation to an HDF5 file.
    :return: A new `Microstructure` instance of the 2 merged scans.
    """
    from pymicro.crystal.microstructure import Microstructure
    import numpy as np
    import os
    import h5py

    scan_shapes = []  # warning, shapes will be in (z, y, x) form
    micros = []

    for scan in scan_list:
        scan_path = os.path.join(root_dir, scan, '5_reconstruction', 'phase_01_vol.mat')
        with h5py.File(scan_path) as f:
            scan_shapes.append(f['vol'].shape)
            print(f['vol'].shape)

    # figure out the maximum cross section
    max_shape = np.array(scan_shapes).max(axis=0)[[2, 1, 0]]

    for scan in scan_list:
        # read microstructure for this scan
        dct_analysis_dir = os.path.join(root_dir, scan)
        print('processing scan %s' % scan)
        micro = Microstructure.from_dct(data_dir=dct_analysis_dir)
        print('voxel_size is {}'.format(micro.voxel_size))

        # pad both grain map and mask
        print('max shape is {}'.format(max_shape))
        print('vol shape is {}'.format(micro.grain_map.shape))
        offset = max_shape - micro.grain_map.shape
        offset[2] = 0  # do not pad along Z
        padding = [(o // 2, max_shape[0] - micro.grain_map.shape[0] - o // 2) for o in offset]
        print('padding is {}'.format(padding))
        micro.grain_map = np.pad(micro.grain_map, padding, mode='constant')
        print('has mask ? {}'.format(hasattr(micro, 'mask')))
        if use_mask:
            micro.mask = np.pad(micro.mask, padding, mode='constant')
        elif hasattr(micro, 'mask'):
            print('deleting mask attribute since we do not want to use it')
            delattr(micro, 'mask')
        micros.append(micro)

    # find out the overlap region (based on the difference in samtz)
    overlap_from_samtz = int((samtz_list[1] + scan_shapes[1][0] // 2 * micros[1].voxel_size) / micros[1].voxel_size
                   - (samtz_list[0] - scan_shapes[0][0] // 2 * micros[0].voxel_size) / micros[0].voxel_size)
    print('vertical overlap deduced from samtz positions is %d voxels' % overlap_from_samtz)
    if overlap < 0:
        overlap = overlap_from_samtz
    print('using an actual overlap of %d voxels' % overlap)

    # we have prepared the 2 microstructures, now merge them
    merged_micro = Microstructure.merge_microstructures(micros, overlap, plot=True)

    if write_to_h5:
        # write the result
        merged_micro.to_h5()

    return merged_micro


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


def tt_rock(scan_name, data_dir='.', n_topo=-1, mask=None, dark_factor=1.):
    from pymicro.file.file_utils import edf_read

    # parse the info file
    f = open(os.path.join(data_dir, scan_name, '%s.info' % scan_name))
    infos = dict()
    for line in f.readlines():
        tokens = line.split('=')
        # convert the value into int/float/str depending on the case
        try:
            value = int(tokens[1].strip())
        except ValueError:
            try:
                value = float(tokens[1].strip())
            except ValueError:
                value = tokens[1].strip()
        infos[tokens[0]] = value
    print(infos)

    if n_topo < 0:
        import glob
        # figure out the number of frames per topograph n_topo
        n_frames = len(glob.glob(os.path.join(data_dir, scan_name, '%s*.edf' % scan_name)))
        n_topo = int(n_frames / infos['TOMO_N'])
    print('number of frames to sum for a topograph = %d' % n_topo)

    # handle mask
    if mask is None:
        mask = np.ones((infos['Dim_1'], infos['Dim_2'], infos['TOMO_N']), dtype=np.uint8)
    else:
        # double check mask size
        if not (mask.shape[0] == infos['Dim_1'] and mask.shape[1] == infos['Dim_2']
            and mask.shape[2] == infos['TOMO_N']):
            print('wrong mask size: {}, should be ({}, {}, {})'.format(
                mask.shape, infos['Dim_1'], infos['Dim_2'], infos['TOMO_N']))
            mask = np.ones((infos['Dim_1'], infos['Dim_2'], infos['TOMO_N']), dtype=np.uint8)
    # load dark image
    dark = dark_factor * edf_read(os.path.join(data_dir, scan_name, 'darkend0000.edf'))
    print('dark average: ' % np.mean(dark))

    # build the stack by combining individual images
    tt_rock = np.empty((infos['TOMO_N'], n_topo), dtype=float)
    for n in range(int(infos['TOMO_N'])):
        print('computing rocking curve %d' % (n + 1), end='\r')
        offset = n_topo * n
        for i in range(n_topo):
            index = offset + i + 1
            frame_path = os.path.join(data_dir, scan_name, '%s%04d.edf' % (scan_name, index))
            im = edf_read(frame_path) - dark
            tt_rock[n, i] = np.sum(im * mask[:, :, n])
    print('\ndone')

    return tt_rock


def tt_stack_h5(scan_name, data_dir='.', save_edf=False, n_angles=90,
                data_key='7.1/measurement/frelon16', dark=None):
    """Build a topotomography stack from raw detector images.

    The number of image to sum for a topograph must be specified
    using the variable `n_topo`.

    :param str scan_name: the name of the scan to process.
    :param str data_dir: the path to the data folder.
    :param bool save_edf: flag to save the tt stack as an EDF file.
    :param int n_angles: the number of angles in the acquisition.
    :param str data_key: a string to access the data within the h5 file.
    :param array dark: image to use as dark correction.
    """
    import h5py

    # parse the info file
    h5_path = os.path.join(data_dir, scan_name, '%s.h5' % scan_name)
    f = h5py.File(h5_path, 'r')
    print('loading data from file %s...' % h5_path)
    data = f[data_key][()]
    # check number of images
    if data.shape[0] % n_angles != 0:
        print('warning, number of images not consistent : %d total images for '
              '%d topograph' % (data.shape[0], n_angles))
        return None
    else:
        n_topo = int(data.shape[0] / n_angles)
    print('number of frames to sum for a topograph = %d' % n_topo)
    if dark is None:
        # create a reference image with the median over the first of each topograph
        dark = np.median(data[::2 * n_topo, :, :], axis=0)
    data = data - dark
    # print check dark level
    print('dark level: %.1f' % np.mean(data[::n_angles, :, :]))
    infos = {}
    infos['TOMO_N'] = n_angles
    infos['Dim_1'] = data.shape[1]
    infos['Dim_2'] = data.shape[2]
    print(infos)

    # build the stack by combining individual images
    data = data.reshape((n_topo, infos['TOMO_N'], infos['Dim_1'], infos['Dim_2']))
    tt_stack = np.sum(data, axis=1)
    del data
    print(tt_stack[0].shape)
    tt_stack = tt_stack.transpose((2, 1, 0))
    f.close()
    print('\ndone')

    # save the data as edf if needed
    if save_edf:
        edf_write(tt_stack, os.path.join(data_dir, '%s_stack.edf' % scan_name))
    return tt_stack

def tt_stack(scan_name, data_dir='.', save_edf=False, n_topo=-1, dark_factor=1.):
    """Build a topotomography stack from raw detector images.

    The number of image to sum for a topograph can be determined automatically
    from the total number of images present in `data_dir` or directly specified
    using the variable `TOPO_N`.

    :param str scan_name: the name of the scan to process.
    :param str data_dir: the path to the data folder.
    :param bool save_edf: flag to save the tt stack as an EDF file.
    :param int n_topo: the number of images to sum for a topograph.
    :param float dark_factor: a multiplicative factor for the dark image.
    """
    if n_topo < 0:
        import glob
        # figure out the number of frames per topograph n_topo
        n_frames = len(glob.glob(os.path.join(data_dir, scan_name, '%s*.edf' % scan_name)))
        n_topo = int(n_frames / 90)
    print('number of frames to sum for a topograph = %d' % n_topo)

    # parse the info file
    f = open(os.path.join(data_dir, scan_name, '%s.info' % scan_name))
    infos = dict()
    for line in f.readlines():
        tokens = line.split('=')
        # convert the value into int/float/str depending on the case
        try:
            value = int(tokens[1].strip())
        except ValueError:
            try:
                value = float(tokens[1].strip())
            except ValueError:
                value = tokens[1].strip()
        infos[tokens[0]] = value
    print(infos)

    # load dark image
    dark = dark_factor * edf_read(os.path.join(data_dir, scan_name, 'darkend0000.edf'))

    # build the stack by combining individual images
    tt_stack = np.empty((infos['TOMO_N'], infos['Dim_1'], infos['Dim_2']))
    print(tt_stack[0].shape)
    for n in range(int(infos['TOMO_N'])):
        print('building topograph %d' % (n + 1), end='\r')
        topograph = np.zeros((infos['Dim_1'], infos['Dim_2']))
        offset = n_topo * n
        for i in range(n_topo):
            index = offset + i + 1
            frame_path = os.path.join(data_dir, scan_name, '%s%04d.edf' % (scan_name, index))
            im = edf_read(frame_path) - dark
            topograph += im
        tt_stack[n] = topograph
    tt_stack = tt_stack.transpose((1, 2, 0))
    print('\ndone')

    # save the data as edf if needed
    if save_edf:
        edf_write(tt_stack, os.path.join(data_dir, '%sstack.edf' % scan_name))
    return tt_stack


def tt_sim_rc(micro, gid, hkl_tt, T=None, lambda_keV=40.0, omega_step=4.0,
              base_tilt_range=0.5):
    """Compute the complete topotomography rocking curve for a given grain.

    This method compute the $I_$\omega$(\Theta)$ rocking curve by analyzing
    the lattice rotation field of a grain and computing the diffracting
    conditions of the selected reflection. For each $\omega$ value in [0, 360],
    the scattering vector is rotated by the base tilt around the mean Bragg
    angle. Each voxel will then contribute to the rocking curve depending on
    its orientation.

    :param Microstructure micro: The microstructure to study.
    :param int gid: the id of the selected grain.
    :param HklPlane hkl_tt: the selected reflection.
    :param np.ndarray T: the diffractometer transformation matrix.
    :param float lambda_keV: the X-ray wavelength in keV units.
    :param float omega_step: the angular $\omega$ step.
    :param float base_tilt_range: the base tilt range in degrees.
    :return: a tuple with 3 numpy arrays: the values of omega, the base tilt
    values and the simulated rocking curve.
    """
    from math import cos, sin
    grain = micro.get_grain(gid)
    sym = micro.get_lattice().get_symmetry()  # should get the phase id
    o_xyz = grain.orientation.move_to_FZ(symmetry=sym)
    ut, lt = o_xyz.topotomo_tilts(hkl_tt, T, verbose=True)
    # compute the S and T tilt matrices (constant for this grain and this hkl_tt)
    U = np.array([[1, 0, 0], [0, cos(ut), -sin(ut)], [0, sin(ut), cos(ut)]])
    L = np.array([[cos(lt), 0, sin(lt)], [0, 1, 0], [-sin(lt), 0, cos(lt)]])
    S = np.dot(L, U)

    # grab all the orientation data for this grain
    if micro._is_empty('orientation_map'):
        print('warning: an orientation map is needed to compute the topotomography rocking curve')
        return
    grain_map = np.squeeze(micro.get_grain_map())
    orientation_map = np.squeeze(micro.get_orientation_map())
    rod_gid = orientation_map[grain_map == gid]
    print('%d orientation data points in this grain' % len(rod_gid))
    # now transform all orientation data for this grain to the fundamental zone
    rod_gid_fz = np.empty_like(rod_gid)
    for i in range(len(rod_gid)):
        g = Orientation.from_rodrigues(rod_gid[i]).orientation_matrix()
        g_fz = sym.move_rotation_to_FZ(g, verbose=False)
        o_fz = Orientation(g_fz)
        rod_gid_fz[i] = o_fz.rod

    # find the diffracting condition for each voxel
    omegas = np.arange(0, 360 + omega_step, omega_step)
    base_tilt_step = 0.01
    base_tilt_values = np.arange(-base_tilt_range, base_tilt_range
                                 + base_tilt_step, base_tilt_step)
    rc_sim = np.empty((len(omegas), len(rod_gid_fz)))
    lambda_nm = 1.2398 / lambda_keV
    theta = hkl_tt.bragg_angle(lambda_keV, verbose=False)
    X = np.array([1., 0., 0.]) / lambda_nm
    print('norm of X', np.linalg.norm(X))

    for j in range(len(omegas)):
        # pick a value for omega
        omega = omegas[j]
        omegar = np.radians(omega)
        R = np.array([[cos(omegar), -sin(omegar), 0], [sin(omegar), cos(omegar), 0], [0, 0, 1]])

        for i in range(len(rod_gid_fz)):
            # grab the i-th pixel of this grain
            o_i = Orientation.from_rodrigues(rod_gid_fz[i])

            # apply the different rotation to the sample
            gt = o_i.orientation_matrix().transpose()
            Gs = np.dot(gt, hkl_tt.scattering_vector())  # grain orientation
            Gs = np.dot(T.T, Gs)  # diffractometer offsets
            Gt = np.dot(S, Gs)  # topotomo tilts
            Gl = np.dot(R, Gt)  # omega rotation

            # scan the base tilt angle (no need to offset by theta here)
            hkl_normal_tilted = np.zeros((len(base_tilt_values), 3))
            norm_K = np.zeros(len(base_tilt_values))
            for k, base_tilt in enumerate(base_tilt_values):
                base_tilt = theta + np.radians(base_tilt)
                BT = np.array([[cos(-base_tilt), 0, sin(-base_tilt)], [0, 1, 0],
                               [-sin(-base_tilt), 0, cos(-base_tilt)]])  # base tilt
                hkl_normal_tilted[k] = np.dot(BT, Gl)
                K = X + hkl_normal_tilted[k]
                norm_K[k] = np.linalg.norm(K)

            # to find the diffraction condition, it is equivalent
            # to search when norm(K) is equal to norm(X) or to
            # search for a zero x-component of the tilted scattering vector
            rc_sim[j, i] = base_tilt_values[np.argmin(np.abs(norm_K - np.linalg.norm(X)))]
            # rc_sim[j, i] = base_tilt_values[np.argmin(np.abs(hkl_normal_tilted[:, 0]))]
        progress = 100 * (1 + j) / len(omegas)
        print('simulating topotomography rocking curve: {0:.2f} %'.format(progress), end='\r')
    return omegas, base_tilt_values, rc_sim

