"""ebsd module to manipulate Electron Back Scattered data sets."""
import h5py
import numpy as np
import os


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
    """OimScan class to handle files from EDAX software OIM"""

    def __init__(self, shape, resolution=(1.0, 1.0)):
        """Create an empty EBSD scan."""
        self.xstar = 0
        self.ystar = 0
        self.zstar = 0
        self.workingDistance = 0
        self.gridType = 'SqrGrid'
        self.nRows = shape[0]
        self.nColumns = shape[1]
        self.xStep = resolution[0]
        self.yStep = resolution[1]
        self.operator = ''
        self.sampleId = ''
        self.scanId = ''
        self.phase_list = []
        self.init_arrays()

    def init_arrays(self):
        self.euler = np.zeros((self.nRows, self.nColumns, 3))
        self.x = np.zeros((self.nRows, self.nColumns))
        self.y = np.zeros((self.nRows, self.nColumns))
        self.iq = np.zeros((self.nRows, self.nColumns))
        self.ci = np.zeros((self.nRows, self.nColumns))
        self.phase = np.zeros((self.nRows, self.nColumns), dtype='int')

    @staticmethod
    def from_file(file_path):
        """Create a new EBSD scan by reading a data file. At present, only hdf5 format is supported."""
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
        self.xstar = header['Pattern Center Calibration']['x-star'][0]
        self.ystar = header['Pattern Center Calibration']['y-star'][0]
        self.zstar = header['Pattern Center Calibration']['z-star'][0]
        self.workingDistance = header['Camera Elevation Angle'][0]
        self.gridType = header['Grid Type'][0].decode('utf-8')
        if self.gridType != 'SqrGrid':
            raise ValueError('only square grid is supported, please convert your scan')
        self.nColumns = header['nColumns'][0]
        self.nRows = header['nRows'][0]
        self.xStep = header['Step X'][0]
        self.yStep = header['Step Y'][0]
        self.operator = header['Operator'][0].decode('utf-8')
        self.sampleId = header['Sample ID'][0].decode('utf-8')
        self.scanId = header['Scan ID'][0].decode('utf-8')
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
        self.euler[:, :, 0] = np.reshape(data['Phi1'], (self.nRows, self.nColumns))
        self.euler[:, :, 1] = np.reshape(data['Phi'], (self.nRows, self.nColumns))
        self.euler[:, :, 2] = np.reshape(data['Phi2'], (self.nRows, self.nColumns))
        self.x = np.reshape(data['X Position'], (self.nRows, self.nColumns))
        self.y = np.reshape(data['Y Position'], (self.nRows, self.nColumns))
        self.iq = np.reshape(data['IQ'], (self.nRows, self.nColumns))
        self.ci = np.reshape(data['CI'], (self.nRows, self.nColumns))
        self.phase = np.reshape(data['Phase'], (self.nRows, self.nColumns))

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
        ebsd_header.create_dataset('Camera Elevation Angle', data=np.array([self.workingDistance], dtype=np.float32))
        pattern_center = ebsd_header.create_group('Pattern Center Calibration')
        pattern_center.create_dataset('x-star', data=np.array(self.xstar, dtype=np.float32))
        pattern_center.create_dataset('y-star', data=np.array(self.ystar, dtype=np.float32))
        pattern_center.create_dataset('z-star', data=np.array(self.zstar, dtype=np.float32))
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
