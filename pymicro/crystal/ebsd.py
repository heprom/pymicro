"""ebsd module to manipulate Electron Back Scattered data sets."""
import h5py
import numpy as np
import os
from pymicro.crystal.microstructure import Orientation
from pymicro.crystal.lattice import Symmetry, CrystallinePhase, Lattice


class OimPhase(CrystallinePhase):
    """A class to handle a phase. This is just a child of the class
    `CrystallinePhase` where we add 2 additional attributes: `hklFamilies` and
    `categories`.
    """

    def __init__(self, id):
        CrystallinePhase.__init__(self, phase_id=id, name='unknown', lattice=None)
        self.hklFamilies = []
        self.categories = []


class OimHklFamily:
    def __init__(self):
        self.hkl = [0, 0, 0]
        self.useInIndexing = 0
        self.diffractionIntensity = 0.0
        self.showBands = 0


class OimScan:
    """OimScan class to handle files from EDAX software OIM."""

    def __init__(self, shape, resolution=(1.0, 1.0)):
        """Create an empty EBSD scan."""
        self.x_star = 0
        self.y_star = 0
        self.z_star = 0
        self.working_distance = 0
        self.grid_type = 'SqrGrid'
        self.cols = shape[0]
        self.rows = shape[1]
        self.xStep = resolution[0]
        self.yStep = resolution[1]
        self.operator = ''
        self.sample_id = ''
        self.scan_id = ''
        self.phase_list = []
        self.init_arrays()

    def __repr__(self):
        """Provide a string representation of the class."""
        s = 'EBSD scan of size %d x %d' % (self.cols, self.rows)
        s += '\nspatial resolution: xStep=%.1f, yStep=%.1f' % (self.xStep, self.yStep)
        return s

    def init_arrays(self):
        """Memory allocation for all necessary arrays."""
        self.euler = np.zeros((self.cols, self.rows, 3))
        self.x = np.zeros((self.cols, self.rows))
        self.y = np.zeros((self.cols, self.rows))
        self.iq = np.zeros((self.cols, self.rows))
        self.ci = np.zeros((self.cols, self.rows))
        self.phase = np.zeros((self.cols, self.rows), dtype='int')

    @staticmethod
    def from_file(file_path):
        """Create a new EBSD scan by reading a data file.

        At present, only hdf5 format is supported.

        :param str file_path: the path to the EBSD scan.
        :raise ValueError: if the scan is not in format HDF5.
        :return: a new `OimScan` instance.
        """
        base_name, ext = os.path.splitext(os.path.basename(file_path))
        print(base_name, ext)
        if ext in ['.h5', '.hdf5']:
            scan = OimScan.read_h5(file_path)
        elif ext == '.osc':
            scan = OimScan.read_osc(file_path)
        elif ext == '.ang':
            scan = OimScan.read_ang(file_path)
        else:
            raise ValueError('only HDF5, OSC or ANG formats are supported, '
                             'please convert your scan')
        return scan

    @staticmethod
    def read_osc(file_path):
        """Read a scan in binary OSC format.

        Code inspired from the MTEX project loadEBSD_osc.m function.

        :param str file_path: the path to the osc file to read.
        :param tuple size: the size of the ebsd scan in form (cols, rows).
        :return: a new instance of OimScan populated with the data from the file.
        """
        scan = OimScan((0, 0))
        # the data section is preceded by this pattern
        start_hex = ['B9', '0B', 'EF', 'FF', '02', '00', '00', '00']
        start_bytes = np.array([int(byte, 16) for byte in start_hex])
        with open(file_path, 'r') as f:
            print('reading EBSD scan from file %s' % file_path)
            header = np.fromfile(f, dtype=np.uint32, count=8)
            n = header[6]
            print('%d data points in EBSD scan' % n)
            f.seek(0)
            buffer = np.fromfile(f, dtype=np.uint8, count=2**20)
            # search for the start pattern
            start = np.where(np.correlate(buffer, start_bytes, mode='valid')
                             == np.dot(start_bytes, start_bytes))[0][0]
            print('start sequence located at byte %d' % start)
            f.seek(start + 8)
            # data count
            data_count = np.fromfile(f, dtype=np.uint32, count=1)[0]
            if round(((data_count / 4 - 2) / 10) / n) != 1:
                f.seek(start + 8)
            # the next 8 bytes are float values for xStep and yStep
            scan.xStep = np.fromfile(f, dtype=np.float32, count=1)[0]
            scan.yStep = np.fromfile(f, dtype=np.float32, count=1)[0]
            print('spatial resolution: xStep=%.1f, yStep=%.1f' % (scan.xStep, scan.yStep))
            # now read the payload which contains 10 fields for the n measurements
            data = np.fromfile(f, count=n*10, dtype=np.float32)
            data = np.reshape(data, (n, 10))
            scan.cols = int(max(data[:, 3]) / scan.xStep + 1)
            scan.rows = int(max(data[:, 4]) / scan.yStep + 1)
            print('size of scan is %d x %d' % (scan.cols, scan.rows))
            assert n == scan.cols * scan.rows
            scan.init_arrays()
            scan.euler[:, :, 0] = np.reshape(data[:, 0], (scan.rows, scan.cols)).T
            scan.euler[:, :, 1] = np.reshape(data[:, 1], (scan.rows, scan.cols)).T
            scan.euler[:, :, 2] = np.reshape(data[:, 2], (scan.rows, scan.cols)).T
            scan.x = np.reshape(data[:, 3], (scan.rows, scan.cols)).T
            scan.y = np.reshape(data[:, 4], (scan.rows, scan.cols)).T
            scan.iq = np.reshape(data[:, 5], (scan.rows, scan.cols)).T
            scan.ci = np.reshape(data[:, 6], (scan.rows, scan.cols)).T
            scan.phase = np.reshape(data[:, 7], (scan.rows, scan.cols)).T
        return scan

    @staticmethod
    def read_ang(file_path):
        """Read a scan in ang ascii format.

        :raise ValueError: if the grid type in not square.
        :param str file_path: the path to the ang file to read.
        :return: a new instance of OimScan populated with the data from the file.
        """
        scan = OimScan((0, 0))
        with open(file_path, 'r') as f:
            # start by parsing the header
            line = f.readline().strip()
            while line.startswith('#'):
                tokens = line.split()
                if len(tokens) <= 2:
                    line = f.readline().strip()
                    continue
                if tokens[1] == 'TEM_PIXperUM':
                    pass
                elif tokens[1] == 'x-star':
                    scan.x_star = float(tokens[2])
                elif tokens[1] == 'y-star':
                    scan.y_star = float(tokens[2])
                elif tokens[1] == 'z-star':
                    scan.z_star = float(tokens[2])
                elif tokens[1] == 'WorkingDistance':
                    scan.working_distance = float(tokens[2])
                elif tokens[1] == 'Phase':
                    phase = OimPhase(int(tokens[2]))
                    line = f.readline().strip()
                    phase.name = line.split()[2]
                    line = f.readline().strip()
                    phase.formula = line.split()[2]
                    line = f.readline().strip()
                    line = f.readline().strip()
                    sym = Symmetry.from_tsl(int(line.split()[2]))
                    tokens = f.readline().strip().split()
                    # convert lattice constants to nm
                    lattice = Lattice.from_parameters(float(tokens[2]) / 10,
                                                      float(tokens[3]) / 10,
                                                      float(tokens[4]) / 10,
                                                      float(tokens[5]),
                                                      float(tokens[6]),
                                                      float(tokens[7]),
                                                      symmetry=sym)
                    phase.set_lattice(lattice)
                    scan.phase_list.append(phase)
                elif tokens[1] == 'GRID:':
                    scan.grid_type = tokens[2]
                    print('grid type is %s' % tokens[2])
                    if scan.grid_type != 'SqrGrid':
                        raise ValueError('only square grid is supported, please convert your scan')
                elif tokens[1] == 'XSTEP':
                    scan.xStep = float(tokens[2])
                elif tokens[1] == 'YSTEP':
                    scan.yStep = float(tokens[2])
                elif tokens[1].startswith('NCOLS'):
                    scan.cols = int(tokens[2])
                elif tokens[1].startswith('NROWS'):
                    scan.rows = int(tokens[2])
                elif tokens[1] == 'OPERATOR:':
                    scan.operator = tokens[2]
                elif tokens[1] == 'SAMPLEID:':
                    scan.sample_id = tokens[2] if len(tokens) >= 3 else ''
                elif tokens[1] == 'SCANID:':
                    scan.scan_id = tokens[2] if len(tokens) >= 3 else ''
                line = f.readline().strip()
            print('finished reading header, scan size is %d x %d' % (scan.cols, scan.rows))
            # now read the payload
            data = np.zeros((scan.cols * scan.rows, len(line.split())))
            data[0] = np.fromstring(line, sep=' ')
            i = 1
            for line in f:
                data[i] = np.fromstring(line, sep=' ')
                i += 1
            # we have read all the data, now repack everything into the different arrays
            scan.init_arrays()
            scan.euler[:, :, 0] = np.reshape(data[:, 0], (scan.rows, scan.cols)).T
            scan.euler[:, :, 1] = np.reshape(data[:, 1], (scan.rows, scan.cols)).T
            scan.euler[:, :, 2] = np.reshape(data[:, 2], (scan.rows, scan.cols)).T
            scan.x = np.reshape(data[:, 3], (scan.rows, scan.cols)).T
            scan.y = np.reshape(data[:, 4], (scan.rows, scan.cols)).T
            scan.iq = np.reshape(data[:, 5], (scan.rows, scan.cols)).T
            scan.ci = np.reshape(data[:, 6], (scan.rows, scan.cols)).T
            scan.phase = np.reshape(data[:, 7], (scan.rows, scan.cols)).T
        return scan

    def read_header(self, header):
        # read the header, it contains the following keys: 'Camera Azimuthal Angle', 'Camera Elevation Angle',
        # 'Coordinate System', 'Grid Type', 'Notes', 'Operator', 'Pattern Center Calibration', 'Phase', 'Sample ID',
        # 'Sample Tilt', 'Scan ID', 'Step X', 'Step Y', 'Working Distance', 'nColumns', 'nRows'
        self.x_star = header['Pattern Center Calibration']['x-star'][0]
        self.y_star = header['Pattern Center Calibration']['y-star'][0]
        self.z_star = header['Pattern Center Calibration']['z-star'][0]
        self.working_distance = header['Camera Elevation Angle'][0]
        self.grid_type = header['Grid Type'][0].decode('utf-8')
        if self.grid_type != 'SqrGrid':
            raise ValueError('only square grid is supported, please convert your scan')
        self.cols = header['nColumns'][0]
        self.rows = header['nRows'][0]
        self.xStep = header['Step X'][0]
        self.yStep = header['Step Y'][0]
        self.operator = header['Operator'][0].decode('utf-8')
        self.sample_id = header['Sample ID'][0].decode('utf-8')
        self.scan_id = header['Scan ID'][0].decode('utf-8')
        # get the different phases
        for key in header['Phase'].keys():
            phase = header['Phase'][key]
            # each phase has the following keys: 'Formula', 'Info', 'Lattice Constant a', 'Lattice Constant alpha',
            # 'Lattice Constant b', 'Lattice Constant beta', 'Lattice Constant c', 'Lattice Constant gamma',
            # 'Laue Group', 'MaterialName', 'NumberFamilies', 'Point Group', 'Symmetry', 'hkl Families'
            phase = OimPhase(int(key))
            phase.name = header['Phase'][key]['MaterialName'][0].decode('utf-8')
            phase.formula = header['Phase'][key]['Formula'][0].decode('utf-8')
            phase.description = header['Phase'][key]['Info'][0].decode('utf-8')
            # create a crystal lattice for this phase
            sym = Symmetry.from_tsl(header['Phase'][key]['Symmetry'][0])
            # convert lattice constants to nm
            a = header['Phase'][key]['Lattice Constant a'][0] / 10
            b = header['Phase'][key]['Lattice Constant b'][0] / 10
            c = header['Phase'][key]['Lattice Constant c'][0] / 10
            alpha = header['Phase'][key]['Lattice Constant alpha'][0]
            beta = header['Phase'][key]['Lattice Constant beta'][0]
            gamma = header['Phase'][key]['Lattice Constant gamma'][0]
            lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma, symmetry=sym)
            phase.set_lattice(lattice)
            for row in header['Phase'][key]['hkl Families']:
                family = OimHklFamily()
                family.hkl = [row[0], row[1], row[2]]
                family.useInIndexing = row[4]
                family.diffractionIntensity = row[3]
                family.showBands = row[5]
                phase.hklFamilies.append(family)
            phase.categories = [0, 0, 0, 0, 0]
            self.phase_list.append(phase)

    @staticmethod
    def read_h5(file_path):
        """Read a scan in H5 format.

        :raise ValueError: if the grid type in not square.
        :param str file_path: the path to the h5 file to read.
        :return: a new instance of OimScan populated with the data from the file.
        """
        scan = OimScan((0, 0))
        with h5py.File(file_path, 'r') as f:
            # find out the scan key (the third one)
            key_list = [key for key in f.keys()]
            scan_key = key_list[2]
            print('reading EBSD scan %s from file %s' % (scan_key, file_path))
            header = f[scan_key]['EBSD']['Header']
            scan.read_header(header)
            # now initialize the fields
            scan.init_arrays()
            data = f[scan_key]['EBSD']['Data']
            scan.euler[:, :, 0] = np.reshape(
                data['Phi1'], (scan.rows, scan.cols)).transpose(1, 0)
            scan.euler[:, :, 1] = np.reshape(
                data['Phi'], (scan.rows, scan.cols)).transpose(1, 0)
            scan.euler[:, :, 2] = np.reshape(
                data['Phi2'], (scan.rows, scan.cols)).transpose(1, 0)
            scan.x = np.reshape(data['X Position'],
                                (scan.rows, scan.cols)).transpose(1, 0)
            scan.y = np.reshape(data['Y Position'],
                                (scan.rows, scan.cols)).transpose(1, 0)
            scan.iq = np.reshape(data['IQ'], (scan.rows, scan.cols)).transpose(1, 0)
            scan.ci = np.reshape(data['CI'], (scan.rows, scan.cols)).transpose(1, 0)
            scan.phase = np.reshape(data['Phase'],
                                    (scan.rows, scan.cols)).transpose(1, 0)
        return scan

    def compute_ipf_maps(self):
        """Compute the IPF maps for the 3 cartesian directions.

        .. warning::

          This function is not vectorized and will be slow for large EBSD maps.

        """
        self.ipf001 = np.empty_like(self.euler)
        self.ipf010 = np.empty_like(self.euler)
        self.ipf100 = np.empty_like(self.euler)
        for i in range(self.rows):
            for j in range(self.cols):
                o = Orientation.from_euler(np.degrees(self.euler[j, i]))
                sym = self.phase_list[int(self.phase[j, i])].get_symmetry()
                # compute IPF-Z
                try:
                    self.ipf001[j, i] = o.ipf_color(axis=np.array([0., 0., 1.]),
                                                    symmetry=sym)
                except ValueError:
                    self.ipf001[j, i] = [0., 0., 0.]
                # compute IPF-Y
                try:
                    self.ipf010[j, i] = o.ipf_color(axis=np.array([0., 1., 0.]),
                                                    symmetry=sym)
                except ValueError:
                    self.ipf010[j, i] = [0., 0., 0.]
                # compute IPF-X
                try:
                    self.ipf100[j, i] = o.ipf_color(axis=np.array([1., 0., 0.]),
                                                    symmetry=sym)
                except ValueError:
                    self.ipf100[j, i] = [0., 0., 0.]
            progress = 100 * (i + 1) / self.rows
            print('computing IPF maps: {0:.2f} %'.format(progress), end='\r')

    def segment_grains(self, tol=5., min_ci=0.2):
        """Segment the grains based on the euler angle maps.

        The segmentation is carried out using a region growing algorithm based
        on an orientation similarity criterion.

        The id 0 is reserved to the background which is assigned to pixels with
        a confidence index lower than 0.2. Other pixels are first marqued as
        unlabeled using -1, then pixels are evaluated one by one.
        A new grain is created and non already assigned neighboring pixels are
        evaluated based on the crystal misorientation. If the misorientation is
        lower than `tol`, the pixel is assigned to the current grain and its
        neighbors added to the list of candidates. When no more candidates are
        present, the next pixel is evaluated and a new grain is created.

        :param float tol: misorientation tolerance in degrees.
        :param float min_ci: minimum confidence index for a pixel to be a valid
            EBSD measurement.
        :return: a numpy array of the grain labels.
        """
        # segment the grains
        print('grain segmentation for EBSD scan, misorientation tolerance={:.1f}, '
              'minimum confidence index={:.1f}'.format(tol, min_ci))
        grain_ids = np.zeros_like(self.iq, dtype='int')
        grain_ids += -1  # mark all pixels as non assigned
        # start by assigning bad pixel to grain 0
        grain_ids[self.ci <= min_ci] = 0

        n_grains = 0
        progress = 0
        for j in range(self.rows):
            for i in range(self.cols):
                if grain_ids[i, j] >= 0:
                    continue  # skip pixel
                # create new grain with the pixel as seed
                n_grains += 1
                # print('segmenting grain %d' % n_grains)
                grain_ids[i, j] = n_grains
                candidates = [(i, j)]
                # apply region growing based on the angle misorientation (strong connectivity)
                while len(candidates) > 0:
                    pixel = candidates.pop()
                    # print('* pixel is {}, euler: {}'.format(pixel, np.degrees(euler[pixel])))
                    # get orientation of this pixel
                    o = Orientation.from_euler(np.degrees(self.euler[pixel]))
                    # look around this pixel
                    east = (pixel[0] - 1, pixel[1])
                    north = (pixel[0], pixel[1] - 1)
                    west = (pixel[0] + 1, pixel[1])
                    south = (pixel[0], pixel[1] + 1)
                    neighbors = [east, north, west, south]
                    # look at unlabeled connected pixels
                    neighbor_list = [n for n in neighbors if
                                     0 <= n[0] < self.cols and
                                     0 <= n[1] < self.rows and
                                     grain_ids[n] == -1]
                    # print(' * neighbors list is {}'.format([east, north, west, south]))
                    for neighbor in neighbor_list:
                        # check misorientation
                        o_neighbor = Orientation.from_euler(np.degrees(self.euler[neighbor]))
                        mis, _, _ = o.disorientation(o_neighbor, crystal_structure=Symmetry.hexagonal)
                        if mis * 180 / np.pi < tol:
                            # add to this grain
                            grain_ids[neighbor] = n_grains
                            # add to the list of candidates
                            candidates.append(neighbor)
                    progress = 100 * np.sum(grain_ids >= 0) / (self.cols * self.rows)
            print('segmentation progress: {0:.2f} %'.format(progress), end='\r')
        print('\n%d grains were segmented' % len(np.unique(grain_ids)))
        return grain_ids

    def change_orientation_reference_frame(self):
        """Change the reference frame for orientation data.

        In OIM, the reference frame for orientation data (euler angles) is
        termed A1A2A3 and differs from the sample reference frame XYZ. This can
        be set befor the acquisition but the default case is:

        X = -A2, Y = -A1, Z = -A3.

        This methods change the reference frame used for the euler angles.
        """
        # transformation matrix from A1A2A3 to XYZ
        T = np.array([[0., -1., 0.],  # X is -A2
                      [-1., 0., 0.],  # Y is -A1
                      [0., 0., -1.]])  # Z is -A3
        for j in range(self.rows):
            for i in range(self.cols):
                o_tsl = Orientation.from_euler(np.degrees(self.euler[i, j, :]))
                g_xyz = np.dot(o_tsl.orientation_matrix(), T.T)  # move to XYZ local frame
                o_xyz = Orientation(g_xyz)
                self.euler[i, j, :] = np.radians(o_xyz.euler)
                progress = 100 * (j * self.cols + i) / (self.cols * self.rows)
            print('changing orientation reference frame progress: {0:.2f} %'.format(progress), end='\r')
        print('\n')

    def to_h5(self, file_name):
        """Write the EBSD scan as a hdf5 file compatible OIM software (in
        progress).

        :param str file_name: name of the output file.
        """
        f = h5py.File('%s.h5' % file_name, 'w')
        f.attrs[' Manufacturer'] = np.string_('EDAX')
        f.attrs[' Version'] = np.string_('OIM Analysis 7.3.0 x64  [09-01-15]')
        # create the group containing the data
        data_container = f.create_group('DataContainer')
        ebsd = data_container.create_group('EBSD')
        ebsd_header = ebsd.create_group('Header')
        ebsd_header.create_dataset('Camera Azimuthal Angle', data=np.array([0.0], dtype=np.float32))
        ebsd_header.create_dataset('Camera Elevation Angle', data=np.array([self.working_distance], dtype=np.float32))
        pattern_center = ebsd_header.create_group('Pattern Center Calibration')
        pattern_center.create_dataset('x-star', data=np.array(self.x_star, dtype=np.float32))
        pattern_center.create_dataset('y-star', data=np.array(self.y_star, dtype=np.float32))
        pattern_center.create_dataset('z-star', data=np.array(self.z_star, dtype=np.float32))
        ebsd_data = ebsd.create_group('Data')
        ci = ebsd_data.create_dataset('CI', data=self.ci)
        iq = ebsd_data.create_dataset('IQ', data=self.iq)
        phase = ebsd_data.create_dataset('Phase', data=self.phase)
        phi1 = ebsd_data.create_dataset('Phi1', data=self.euler[:, :, 0])
        phi = ebsd_data.create_dataset('Phi', data=self.euler[:, :, 1])
        phi2 = ebsd_data.create_dataset('Phi2', data=self.euler[:, :, 2])
        x = ebsd_data.create_dataset('X Position', data=self.x)
        y = ebsd_data.create_dataset('Y Position', data=self.y)
        f.close()
