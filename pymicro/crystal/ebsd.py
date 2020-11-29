"""ebsd module to manipulate Electron Back Scattered data sets."""
import h5py
import numpy as np
import os
from pymicro.crystal.microstructure import Orientation
from pymicro.crystal.lattice import Symmetry

class OimPhase:
    def __init__(self, id):
        self.id = id
        self.materialName = ''
        self.formula = ''
        self.info = ''
        self.symmetry = 0
        self.latticeConstants =[0] * 6 # a, b, c, alpha, beta, gamma (degrees)
        self.hklFamilies = []
        self.elasticConstants = []
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

    def init_arrays(self):
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

        :param str file_path: the path to the ABSD scan.
        :raise ValueError: if the scan is not in format HDF5.
        :return: a new `OimScan` instance.
        """
        base_name, ext = os.path.splitext(os.path.basename(file_path))
        print(base_name, ext)
        if ext not in ['.h5', '.hdf5']:
            raise ValueError('only HDF5 format is supported, please convert your scan')
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
            scan.read_data(data)
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
            phase.materialName = header['Phase'][key]['MaterialName'][0].decode('utf-8')
            phase.formula = header['Phase'][key]['Formula'][0].decode('utf-8')
            phase.info = header['Phase'][key]['Info'][0].decode('utf-8')
            phase.symmetry = header['Phase'][key]['Symmetry'][0]
            phase.latticeConstants[0] = header['Phase'][key]['Lattice Constant a'][0]
            phase.latticeConstants[1] = header['Phase'][key]['Lattice Constant b'][0]
            phase.latticeConstants[2] = header['Phase'][key]['Lattice Constant c'][0]
            phase.latticeConstants[3] = header['Phase'][key]['Lattice Constant alpha'][0]
            phase.latticeConstants[4] = header['Phase'][key]['Lattice Constant beta'][0]
            phase.latticeConstants[5] = header['Phase'][key]['Lattice Constant gamma'][0]
            for row in header['Phase'][key]['hkl Families']:
                family = OimHklFamily()
                family.hkl = [row[0], row[1], row[2]]
                family.useInIndexing = row[4]
                family.diffractionIntensity = row[3]
                family.showBands = row[5]
                phase.hklFamilies.append(family)
            phase.elasticConstants = [[0, 0, 0, 0, 0, 0] * 6 for i in range(6)]
            phase.categories = [0, 0, 0, 0, 0]
            self.phase_list.append(phase)

    def read_data(self, data):
        self.euler[:, :, 0] = np.reshape(data['Phi1'],
                                         (self.rows, self.cols)).transpose(1, 0)
        self.euler[:, :, 1] = np.reshape(data['Phi'],
                                         (self.rows, self.cols)).transpose(1, 0)
        self.euler[:, :, 2] = np.reshape(data['Phi2'],
                                         (self.rows, self.cols)).transpose(1, 0)
        self.x = np.reshape(data['X Position'],
                            (self.rows, self.cols)).transpose(1, 0)
        self.y = np.reshape(data['Y Position'],
                            (self.rows, self.cols)).transpose(1, 0)
        self.iq = np.reshape(data['IQ'], (self.rows, self.cols)).transpose(1, 0)
        self.ci = np.reshape(data['CI'], (self.rows, self.cols)).transpose(1, 0)
        self.phase = np.reshape(data['Phase'],
                                (self.rows, self.cols)).transpose(1, 0)

    def segment_grains(self, tol=5.):
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

        :param tol: misorientation tolerance in degrees
        :return: a numpy array of the grain labels.
        """
        # segment the grains
        grain_ids = np.zeros_like(self.iq, dtype='int')
        grain_ids += -1  # mark all pixels as non assigned
        # start by assigning bad pixel to grain 0
        grain_ids[self.ci <= 0.2] = 0

        n_grains = 0
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

    def to_h5(self, file_name):
        """Write the EBSD scan as a hdf5 file compatible OIM software (in progress)."""
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
