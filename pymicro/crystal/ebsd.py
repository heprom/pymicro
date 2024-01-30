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

    @staticmethod
    def from_OIM_h5(h5_phase_path, phase_id):
        """Create a phase from information stored in a h5 file.

        :param str h5_phase_path: a reference to the Phase node in the h5 file.
        :param str phase_id: the phase id which is also the node in the h5 file
            where the information is stored.
        :return: a new instance OimPhase.
        """
        print(h5_phase_path.keys())
        print(phase_id)
        #assert phase_id == h5_phase_path[phase_id]
        # each phase has the following keys: 'Formula', 'Info', 'Lattice Constant a', 'Lattice Constant alpha',
        # 'Lattice Constant b', 'Lattice Constant beta', 'Lattice Constant c', 'Lattice Constant gamma',
        # 'Laue Group', 'MaterialName', 'NumberFamilies', 'Point Group', 'Symmetry', 'hkl Families'
        phase = OimPhase(int(phase_id))
        # Material Name may or may not have a space
        keys = list(h5_phase_path[phase_id].keys())
        name_key = [item for item in keys if item.endswith('Name')][0]
        phase.name = h5_phase_path[phase_id][name_key][0].decode('utf-8')
        phase.formula = h5_phase_path[phase_id]['Formula'][0].decode('utf-8')
        phase.description = h5_phase_path[phase_id]['Info'][0].decode('utf-8')
        # create a crystal lattice for this phase
        sym = Symmetry.from_tsl(h5_phase_path[phase_id]['Symmetry'][0])
        # convert lattice constants to nm
        key_a = [item for item in keys if item.startswith('Lat') and (item.endswith(' A') or item.endswith(' a'))][0]
        key_b = [item for item in keys if item.startswith('Lat') and (item.endswith(' B') or item.endswith(' b'))][0]
        key_c = [item for item in keys if item.startswith('Lat') and (item.endswith(' C') or item.endswith(' c'))][0]
        key_alpha = [item for item in keys if item.startswith('Lat') and item.endswith('lpha')][0]
        key_beta = [item for item in keys if item.startswith('Lat') and item.endswith('eta')][0]
        key_gamma = [item for item in keys if item.startswith('Lat') and item.endswith('amma')][0]
        a = h5_phase_path[phase_id][key_a][0] / 10
        b = h5_phase_path[phase_id][key_b][0] / 10
        c = h5_phase_path[phase_id][key_c][0] / 10
        alpha = h5_phase_path[phase_id][key_alpha][0]
        beta = h5_phase_path[phase_id][key_beta][0]
        gamma = h5_phase_path[phase_id][key_gamma][0]
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma, symmetry=sym)
        phase.set_lattice(lattice)
        hkl_key = [item for item in keys if item.endswith('Families')][0]
        for row in h5_phase_path[phase_id][hkl_key]:
            family = OimHklFamily()
            family.hkl = [row[0], row[1], row[2]]
            family.useInIndexing = row[4]
            family.diffractionIntensity = row[3]
            family.showBands = row[5]
            phase.hklFamilies.append(family)
        phase.categories = [0, 0, 0, 0, 0]
        return phase


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
        print('init with shape', shape)
        self.x_star = 0
        self.y_star = 0
        self.z_star = 0
        self.working_distance = 0.
        self.sample_tilt_angle = 0.
        self.camera_elevation_angle = 0.
        self.camera_azimuthal_angle = 0.
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
        self.grain_ids = None
        self.god = None

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

    def crop(self, x_start=None, x_end=None, y_start=None, y_end=None,
             in_place=True):
        """Crop an EBSD scan.

        :param int x_start: start value for slicing the first axis (cols).
        :param int x_end: end value for slicing the first axis.
        :param int y_start: start value for slicing the second axis (rows).
        :param int y_end: end value for slicing the second axis.
        :param bool in_place: crop the actual EBSD scan, if False, a new scan
        is returned
        """
        # input default values for bounds if not specified
        if not x_start or x_start < 0:
            x_start = 0
        if not y_start or y_start < 0:
            y_start = 0
        if not x_end or x_end > self.cols:
            x_end = self.cols
        if not y_end or y_end > self.rows:
            y_end = self.rows
        if in_place:
            self.cols = x_end - x_start
            self.rows = y_end - y_start
            self.euler = self.euler[x_start:x_end, y_start:y_end, :]
            self.x = self.x[x_start:x_end, y_start:y_end]
            self.y = self.y[x_start:x_end, y_start:y_end]
            self.iq = self.iq[x_start:x_end, y_start:y_end]
            self.ci = self.ci[x_start:x_end, y_start:y_end]
            self.phase = self.phase[x_start:x_end, y_start:y_end]
            if self.grain_ids is not None:
                self.grain_ids = self.grain_ids[x_start:x_end, y_start:y_end]
        else:
            crop = OimScan((x_end - x_start, y_end - y_start),
                           resolution=(self.xStep, self.yStep))
            crop.operator = self.operator
            crop.sample_id = self.sample_id
            crop.scan_id = self.scan_id + 'cropped'
            crop.phase_list = self.phase_list
            crop.euler = self.euler[x_start:x_end, y_start:y_end, :]
            crop.x = self.x[x_start:x_end, y_start:y_end]
            crop.y = self.y[x_start:x_end, y_start:y_end]
            crop.iq = self.iq[x_start:x_end, y_start:y_end]
            crop.ci = self.ci[x_start:x_end, y_start:y_end]
            crop.phase = self.phase[x_start:x_end, y_start:y_end]
            if self.grain_ids is not None:
                crop.grain_ids = self.grain_ids[x_start:x_end, y_start:y_end]
            return crop

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
        elif ext == '.ctf':
            scan = OimScan.read_ctf(file_path)
        else:
            raise ValueError('only HDF5, OSC, ANG or CTF formats are '
                             'supported, please convert your scan')
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
                print(line)
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
                elif tokens[1] == 'SampleTiltAngle':
                    scan.sample_tilt_angle = float(tokens[2])
                elif tokens[1] == 'CameraElevationAngle':
                    scan.camera_elevation_angle = float(tokens[2])
                elif tokens[1] == 'CameraAzimuthalAngle':
                    scan.camera_azimuthal_angle = float(tokens[2])
                elif tokens[1] == 'Phase' and tokens[2].isdigit():
                    phase = OimPhase(int(tokens[2]))
                    line = f.readline().strip()
                    phase.name = line.split()[2]
                    line = f.readline().strip()
                    try:
                        phase.formula = line.split()[2]
                    except IndexError:
                        phase.formula = ''
                    line = f.readline().strip()
                    line = f.readline().strip()
                    sym = Symmetry.from_tsl(int(line.split()[2]))
                    while not line.startswith('# LatticeConstants'):
                        line = f.readline().strip()
                    tokens = line.split()
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
                elif tokens[1].startswith('XSTEP'):
                    scan.xStep = float(tokens[2])
                elif tokens[1].startswith('YSTEP'):
                    scan.yStep = float(tokens[2])
                elif tokens[1].startswith('NCOLS'):
                    scan.cols = int(tokens[2])
                elif tokens[1].startswith('NROWS'):
                    scan.rows = int(tokens[2])
                elif tokens[1].startswith('OPERATOR'):
                    scan.operator = tokens[2]
                elif tokens[1].startswith('SAMPLEID'):
                    scan.sample_id = tokens[2] if len(tokens) >= 3 else ''
                elif tokens[1].startswith('SCANID'):
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
            '''
            if data.shape[1] > 8:
                print('including SEM signal')
                scan.sem = np.reshape(data[:, 8], (scan.rows, scan.cols)).T
            '''
        return scan

    @staticmethod
    def read_ctf(file_path):
        """Read a scan in Channel Text File format.

        :raise ValueError: if the job mode is not grid.
        :param str file_path: the path to the ctf file to read.
        :return: a new instance of OimScan populated with the data from the file.
        """
        scan = OimScan((0, 0))
        with open(file_path, 'r') as f:
            # start by parsing the header
            line = f.readline().strip()
            while not line.startswith('Phases'):
                tokens = line.split()
                if tokens[0] == 'JobMode':
                    scan.grid_type = tokens[1]
                    if scan.grid_type != 'Grid':
                        raise ValueError('only square grid is supported, please convert your scan')
                elif tokens[0] == 'XCells':
                    scan.cols = int(tokens[1])
                elif tokens[0] == 'YCells':
                    scan.rows = int(tokens[1])
                elif tokens[0] == 'XStep':
                    scan.xStep = float(tokens[1])
                elif tokens[0] == 'YStep':
                    scan.yStep = float(tokens[1])
                line = f.readline().strip()
            # read the phases
            tokens = line.split()
            n_phases = int(tokens[1])
            for i in range(n_phases):
                # read this phase (lengths, angles, name, ?, space group, description)
                line = f.readline().strip()
                tokens = line.split()
                phase = CrystallinePhase(i + 1)
                phase.name = tokens[2]
                phase.name = tokens[5]
                lattice_lengths = tokens[0].split(';')
                lattice_angles = tokens[1].split(';')
                a, b, c = float(lattice_lengths[0]) / 10, \
                          float(lattice_lengths[1]) / 10, \
                          float(lattice_lengths[2]) / 10
                alpha, beta, gamma = float(lattice_angles[0]), \
                                     float(lattice_angles[1]), \
                                     float(lattice_angles[2])
                try:
                    sym = Symmetry.from_space_group(int(tokens[4]))
                except ValueError:
                    # try to guess the symmetry
                    sym = Lattice.guess_symmetry_from_parameters(a, b, c, alpha, beta, gamma)
                    print('guessed symmetry from lattice parameters:', sym)
                # convert lattice constants to nm
                lattice = Lattice.from_parameters(a, b, c, 
                                                  alpha, beta, gamma,
                                                  symmetry=sym)
                phase.set_lattice(lattice)
                print('adding phase %s' % phase)
                scan.phase_list.append(phase)
            # read the line before the data
            line = f.readline().strip()
            # Phase   X       Y       Bands   Error   Euler1  Euler2  Euler3  MAD     BC      BS
            # now read the payload
            data = np.zeros((scan.cols * scan.rows, len(line.split())))
            i = 0
            for line in f:
                data[i] = np.fromstring(line, sep=' ')
                i += 1
            # we have read all the data, now repack everything into the different arrays
            scan.init_arrays()
            x = data[:, 1]
            y = data[:, 2]
            # use x, y values to assign each record to the proper place
            x_indices = (np.round((x - x.min()) / scan.xStep)).astype(int)
            y_indices = (np.round((y - y.min()) / scan.yStep)).astype(int)            
            scan.phase[x_indices, y_indices] = data[:, 0]
            scan.x[x_indices, y_indices] = data[:, 1]
            scan.y[x_indices, y_indices] = data[:, 2]
            scan.euler[x_indices, y_indices, 0] = np.radians(data[:, 5])
            scan.euler[x_indices, y_indices, 1] = np.radians(data[:, 6])
            scan.euler[x_indices, y_indices, 2] = np.radians(data[:, 7])
            scan.iq[x_indices, y_indices] = data[:, 9]
            scan.ci[x_indices, y_indices] = data[:, 10]
            if sym is Symmetry.hexagonal:
                # add a +30 degrees rotation on phi2
                scan.euler[:, :, 2] += np.radians(30)
        return scan

    def read_header(self, header):
        # read the header, it contains the following keys: 'Camera Azimuthal Angle', 'Camera Elevation Angle',
        # 'Coordinate System', 'Grid Type', 'Notes', 'Operator', 'Pattern Center Calibration', 'Phase', 'Sample ID',
        # 'Sample Tilt', 'Scan ID', 'Step X', 'Step Y', 'Working Distance', 'nColumns', 'nRows'
        self.x_star = header['Pattern Center Calibration']['x-star'][0]
        self.y_star = header['Pattern Center Calibration']['y-star'][0]
        self.z_star = header['Pattern Center Calibration']['z-star'][0]
        self.working_distance = header['Working Distance'][0]
        self.camera_elevation_angle = header['Camera Elevation Angle'][0]
        self.camera_azimuthal_angle = header['Camera Azimuthal Angle'][0]
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
            phase = OimPhase.from_OIM_h5(header['Phase'], phase_id=key)
            self.phase_list.append(phase)

    @staticmethod
    def read_h5(file_path, scan_key=None):
        """Read a scan in H5 format.

        :raise ValueError: if the grid type in not square.
        :param str file_path: the path to the h5 file to read.
        :param str scan_key: a string to force the scan key, be default it is
            read as the third key at the file root.
        :return: a new instance of OimScan populated with the data from the file.
        """
        scan = OimScan((0, 0))
        with h5py.File(file_path, 'r') as f:
            if not scan_key:
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
            # the phase value is decreased by 1 wrt the phase_id in the list
            scan.phase = np.reshape(data['Phase'],
                                    (scan.rows, scan.cols)).transpose(1, 0)
        return scan

    def get_phase(self, phase_id=1):
        """Look for a phase with the given id in the list.

        :raise ValueError: if the phase_id cannot be found.
        :param int phase_id: the id of the phase.
        :return: the phase instance with the corresponding id
        """
        try:
            phase_index = [phase.phase_id for phase in self.phase_list].index(phase_id)
        except ValueError:
            raise(ValueError('phase %d not in list' % phase_id))            
        return self.phase_list[phase_index]
        
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
                try:
                    sym = self.get_phase(int(self.phase[j, i])).get_symmetry()
                    # compute IPF-Z
                    self.ipf001[j, i] = o.ipf_color(axis=np.array([0., 0., 1.]),
                                                    symmetry=sym)
                    # compute IPF-Y
                    self.ipf010[j, i] = o.ipf_color(axis=np.array([0., 1., 0.]),
                                                    symmetry=sym)
                    # compute IPF-X
                    self.ipf100[j, i] = o.ipf_color(axis=np.array([1., 0., 0.]),
                                                    symmetry=sym)
                except ValueError:
                    self.ipf001[j, i] = [0., 0., 0.]
                    self.ipf010[j, i] = [0., 0., 0.]
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

        .. warning::

          This function does not account yet for multiple phases. Grains should
          be created separately for each crystallographic phase.

        :param float tol: misorientation tolerance in degrees.
        :param float min_ci: minimum confidence index for a pixel to be a valid
            EBSD measurement.
        :raise ValueError: if no phase is present in the scan.
        :return: a numpy array of the grain labels.
        """
        if not len(self.phase_list) > 0:
            raise ValueError('at least one phase must be present in this EBSD '
                             'scan to segment the grains')
        # segment the grains
        print('grain segmentation for EBSD scan, misorientation tolerance={:.1f}, '
              'minimum confidence index={:.1f}'.format(tol, min_ci))
        grain_ids = np.zeros_like(self.iq, dtype='int')
        grain_ids += -1  # mark all pixels as non assigned
        # start by assigning bad pixel to grain 0
        grain_ids[self.ci <= min_ci] = 0

        n_grains = 0
        progress = 0
        #phase_start = self.phase_list[0].phase_id
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
                    #sym = self.get_phase(phase_start + int(self.phase[pixel])).get_symmetry()
                    sym = self.get_phase(int(self.phase[pixel])).get_symmetry()
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
                        mis, _, _ = o.disorientation(o_neighbor, crystal_structure=sym)
                        if mis * 180 / np.pi < tol:
                            # add to this grain
                            grain_ids[neighbor] = n_grains
                            # add to the list of candidates
                            candidates.append(neighbor)
                    progress = 100 * np.sum(grain_ids >= 0) / (self.cols * self.rows)
            print('segmentation progress: {0:.2f} %'.format(progress), end='\r')
        print('\n%d grains were segmented' % len(np.unique(grain_ids)))
        # assign grain_ids array to the scan
        self.grain_ids = grain_ids
        return grain_ids

    def change_orientation_reference_frame(self):
        """Change the reference frame for orientation data.

        For this we can use a change base matrix

        In OIM, the reference frame for orientation data (euler angles) is
        termed A1A2A3 and differs from the sample reference frame XYZ. This can
        be set before the acquisition but the default case is:

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

    def compute_god_map(self, id_list=None):
        """Create a GOD (grain orientation deviation) map.

        This method computes the grain orientation deviation map. For each
        grain in the list (all grain by default), the mean orientation is
        computed. Then the orientation of each pixel belonging to this grain
        is compared to the mean and the resulting misorientation is assigned
        to the pixel.

        .. note::

          This method needs the grain segmentation to run, a message will be
          displayed if this is not the case.

        :param list id_list: the list of the grain ids to include (compute
            for all grains by default).
        """
        if self.grain_ids is None:
            print('no grain_ids field, please segment your grains first')
            return None
        self.god = np.zeros_like(self.iq)
        if not id_list:
            id_list = np.unique(self.grain_ids)

        for index, gid in enumerate(id_list):
            if gid < 1:
                continue
            progress = 100 * index / len(id_list)
            print('GOD computation progress: {:.2f} % (grain {:d})'.format(progress, gid), end='\r')
            indices = np.where(self.grain_ids == gid)
            # get the symmetry for this grain
            sym = self.get_phase(1 + int(self.phase[indices[0][0], indices[1][0]])).get_symmetry()
            # compute the mean orientation of this grain
            euler_gid = self.euler[np.squeeze(self.grain_ids == gid)]
            rods = Orientation.eu2ro(euler_gid)
            o = Orientation.compute_mean_orientation(rods, symmetry=sym)
            # now compute the orientation deviation wrt the mean for each pixel of the grain
            for i, j in zip(indices[0], indices[1]):
                euler_ij = self.euler[i, j]
                o_ij = Orientation.from_euler(np.degrees(euler_ij))
                self.god[i, j] = np.degrees(o.disorientation(o_ij, crystal_structure=sym)[0])
        print('GOD computation progress: 100.00 %')

    def ang_header(self):
        # compose header
        header = 'HEADER: Start\n'
        header += 'TEM_PIXperUM 1.000000\n'
        header += 'x-star ' + str(self.x_star) + '\n'
        header += 'y-star ' + str(self.y_star) + '\n'
        header += 'z-star ' + str(self.z_star) + '\n'
        header += 'WorkingDistance ' + str(self.working_distance) + '\n'
        header += 'SampleTiltAngle ' + str(self.sample_tilt_angle) + '\n'
        header += '\n'
        for phase in self.phase_list:
            header += 'Phase ' + str(phase.phase_id) + '\n'
            header += 'MaterialName ' + phase.name + '\n'
            header += 'Formula ' + phase.formula + '\n'
            header += 'Info ' + phase.description + '\n'
            header += 'Symmetry ' + Symmetry.to_tsl(phase.get_symmetry()) + '\n'
            lattice_constants = phase.get_lattice().get_lattice_constants(angstrom=True)
            constants = '  '.join(['%.3f' % c for c in lattice_constants])
            header += ('LatticeConstants ' + constants + '\n')
            number_families = 0
            if hasattr(phase, 'hklFamilies'):
                number_families = len(phase.hklFamilies)
            header += ('NumberFamilies ' + str(number_families) + '\n')
            constants = '  '.join(['%.3f' % c for c in phase.elastic_constants])
            if constants:
                header += 'ElasticConstants ' + constants + '\n'
            header += 'Categories'
            if hasattr(phase, 'categories'):
                for category in phase.categories:
                    header += ' ' + str(category)
            header += '\n\n'
        header += 'GRID: ' + self.grid_type + '\n'
        header += 'XSTEP: ' + str(self.xStep) + '\n'
        header += 'YSTEP: ' + str(self.yStep) + '\n'
        header += 'NCOLS_ODD: ' + str(self.cols) + '\n'
        header += 'NCOLS_EVEN: ' + str(self.cols) + '\n'
        header += 'NROWS: ' + str(self.rows) + '\n'
        header += '\n'
        header += 'OPERATOR: ' + self.operator + '\n'
        header += 'SAMPLEID: ' + self.sample_id + '\n'
        header += 'SCANID: ' + self.scan_id + '\n'
        header += 'HEADER: End\n'
        return header

    def to_ang(self, file_name):
        """Write the data as an .ang ascii file.

        This can be read back into the OIM software.

        :param str file_name: name of the file to write.
        """
        # create a single array with all the EBSD data
        data = np.concatenate((self.euler,
                               np.reshape(self.x, (self.x.shape[0], self.x.shape[1], 1)),
                               np.reshape(self.y, (self.y.shape[0], self.y.shape[1], 1)),
                               np.reshape(self.iq, (self.iq.shape[0], self.iq.shape[1], 1)),
                               np.reshape(self.ci, (self.ci.shape[0], self.ci.shape[1], 1)),
                               np.reshape(self.phase, (self.phase.shape[0], self.phase.shape[1], 1))),
                               axis=2).transpose(1, 0, 2)
        assert (self.grid_type == 'SqrGrid')
        data = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))
        print('now data shape is ', data.shape)
        #f.write('\n'.join(map(lambda data_line: ' '.join(map(str, data_line)), data)))
        np.savetxt(file_name, data, header=self.ang_header(), comments='# ', delimiter=' ',
                   fmt=('%9.5f', '%9.5f', '%9.5f', '%12.5f', '%12.5f', '%.1f', '%6.3f', '%2d'))

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
