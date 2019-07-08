import json
from json import JSONEncoder
import numpy as np
from pymicro.xray.detectors import Detector2d, RegArrayDetector2d
from pymicro.crystal.lattice import Lattice, Symmetry
from pymicro.crystal.microstructure import Microstructure, Grain, Orientation

class ForwardSimulation:
    """Class to represent a Forward Simulation."""

    def __init__(self, fs_type):
        self.fs_type = fs_type

class XraySource:
    """Class to represent a X-ray source."""

    def __init__(self, position=None):
        self.set_position(position)

    def set_position(self, position):
        if position is None:
            position = (0., 0., 0.)
        self.position = np.array(position)


class ObjectGeometry:
    """Class to represent any object geometry.
    
    The geometry may have multiple form, including just a point, a regular 3D array or it may be described by a CAD 
    file using the STL format."""

    def __init__(self, geo_type='point'):
        self.set_type(geo_type)

    def set_type(self, geo_type):
        assert (geo_type in ['point', 'array', 'cad']) is True
        self.geo_type = geo_type

    def get_bounding_box(self):
        if self.geo_type == 'point':
            return (0., 0., 0.), (0., 0., 0.)
        elif self.geo_type == 'array':
            return (0., 0., 0.), self.array.shape
        elif self.geo_type == 'cad':
            bounds = self.cad.GetBounds()
            return (bounds[0], bounds[2], bounds[4]), (bounds[1], bounds[3], bounds[5])

    def get_positions(self):
        if self.geo_type == 'point':
            return [(0., 0., 0.)]
        elif self.geo_type == 'array':
            return self.array  #FIXME
        elif self.geo_type == 'cad':
            print('discretizing CAD geometry is not yet supported')
            return None

class Sample:
    """Class to describe a material sample.
    
    A sample is made by a given material (that may have multiple phases), has a name and a position in the experimental 
    local frame. A sample also has a geometry (just a point by default), that may be used to discretize the volume 
    in space or display it in 3D.
    
    .. note::

      For the moment, the material is simply a crystal lattice.
    """

    def __init__(self, name=None, position=None, geo=None, material=None, microstructure=None):
        self.name = name
        self.set_position(position)
        self.set_geometry(geo)
        self.set_material(material)
        self.set_microstructure(microstructure)

    def set_name(self, name):
        self.name = name

    def set_position(self, position):
        if position is None:
            position = (0., 0., 0.)
        self.position = np.array(position)

    def set_geometry(self, geo):
        if geo is None:
            geo = ObjectGeometry()
        assert isinstance(geo, ObjectGeometry) is True
        self.geo = geo

    def set_material(self, material):
        if material is None:
            material = Lattice.cubic(1.0)
        self.material = material

    def set_microstructure(self, microstructure):
        if microstructure is None:
            microstructure = Microstructure()
        self.microstructure = microstructure

    def has_grains(self):
        return len(self.microstructure.grains) > 0

class Experiment:
    """Class to represent an actual or a virtual X-ray experiment.
    
    A cartesian coordinate system (X, Y, Z) is associated with the experiment. By default X is the direction of X-rays 
    and the sample is placed at the origin (0, 0, 0).
    """

    def __init__(self):
        self.source = XraySource()
        self.sample = Sample(name='dummy')
        self.detectors = []
        self.active_detector_id = -1

    def set_sample(self, sample):
        assert isinstance(sample, Sample) is True
        self.sample = sample

    def get_sample(self):
        return self.sample

    def set_source(self, source):
        assert isinstance(source, XraySource) is True
        self.source = source

    def get_source(self):
        return self.source

    def add_detector(self, detector, set_as_active=True):
        """Add a detector to this experiment.
        
        If this is the first detector, the active detector id is set accordingly.

        :param Detector2d detector: an instance of the Detector2d class.
        :param bool set_as_active: set this detector as active.
        """
        assert isinstance(detector, Detector2d) is True
        self.detectors.append(detector)
        if set_as_active:
            self.active_detector_id = self.get_number_of_detectors() - 1

    def get_number_of_detectors(self):
        """Return the number of detector for this experiment."""
        return len(self.detectors)

    def get_active_detector(self):
        """Return the active detector for this experiment."""
        return self.detectors[self.active_detector_id]
    
    def forward_simulation(self, fs_type='laue', verbose=False, **kwargs):
        """Perform a forward simulation of the X-ray experiment onto the active detector.
        
        This typically sets the detector.data field with the computed image.

        :param str fs_type: The type of simulation to perform (abs, dct or laue).
        :param bool verbose: activate verbose mode.
        :param **kwargs: additional parameters depending on the `fs_type` chosen.
        """
        self.fs = ForwardSimulation(fs_type)
        detector = self.get_active_detector()
        if self.fs.fs_type == 'laue':
            from pymicro.xray.laue import build_list
            from pymicro.xray.xray_utils import lambda_nm_to_keV
            assert self.sample.has_grains()
            # set the lattice planes to use in the simulation
            self.fs.max_miller = kwargs.get('max_miller', 5)
            if kwargs.has_key('hkl_planes'):
                self.fs.hkl_planes = kwargs['hkl_planes']
            else:
                self.fs.hkl_planes = build_list(lattice=self.sample.material, max_miller=self.fs.max_miller)
            n_hkl = len(self.fs.hkl_planes)
            for grain in self.sample.microstructure.grains:
                if verbose:
                    print('FS for grain %d' % grain.id)
                gt = grain.orientation_matrix().transpose()

                # here we use list comprehension to avoid for loops
                d_spacings = [hkl.interplanar_spacing() for hkl in self.fs.hkl_planes]  # size n_hkl
                G_vectors = [hkl.scattering_vector() for hkl in self.fs.hkl_planes]  # size n_hkl, with 3 elements items
                Gs_vectors = [gt.dot(Gc) for Gc in G_vectors]  # size n_hkl, with 3 elements items
                positions = self.sample.geo.get_positions()  # size n_vox, with 3 elements items
                n_vox = len(positions)  # total number of discrete positions
                Xu_vectors = [(pos - self.source.position) / np.linalg.norm(pos - self.source.position)
                              for pos in positions]  # size n_vox
                thetas = [np.arccos(np.dot(Xu, Gs / np.linalg.norm(Gs))) - np.pi / 2
                          for Xu in Xu_vectors
                          for Gs in Gs_vectors]  # size n_vox * n_hkl
                the_energies = [lambda_nm_to_keV(2 * d_spacings[i_hkl] * np.sin(thetas[i_Xu * n_hkl + i_hkl]))
                                for i_Xu in range(n_vox)
                                for i_hkl in range(n_hkl)]  # size n_vox * n_hkl
                X_vectors = [np.array(Xu_vectors[i_Xu]) / 1.2398 * the_energies[i_Xu * n_hkl + i_hkl]
                             for i_Xu in range(n_vox)
                             for i_hkl in range(n_hkl)]  # size n_vox * n_hkl
                K_vectors = [X_vectors[i_Xu * n_hkl + i_hkl] + Gs_vectors[i_hkl]
                             for i_Xu in range(n_vox)
                             for i_hkl in range(n_hkl)]  # size n_vox * n_hkl
                OR_vectors = [detector.project_along_direction(origin=positions[i_vox],
                                                               direction=K_vectors[i_vox * n_hkl + i_hkl])
                              for i_vox in range(n_vox)
                              for i_hkl in range(n_hkl)]  # size nb_vox * n_hkl
                uv = [detector.lab_to_pixel(OR)[0].astype(np.int) for OR in OR_vectors]
                uv_in = [uv[k][0] > 0 and uv[k][0] < detector.size[0] and uv[k][1] > 0 and uv[k][1] < detector.size[1]
                         for k in range(len(uv))]  # size n, diffraction located on the detector
                if verbose:
                    print('%d diffraction events on the detector among %d' % (sum(uv_in), len(uv)))

                # now sum the counts on the detector individual pixels
                for k in range(len(uv)):
                    if uv_in[k]:
                        detector.data[uv[k][0], uv[k][1]] += 1

    def save(self):
        """Export the parameters to describe the current experiment to a file using json."""
        dict_exp = {}
        dict_exp['Source'] = self.source
        dict_exp['Sample'] = self.sample
        dict_exp['Detectors'] = self.detectors
        dict_exp['Active Detector Id'] = self.active_detector_id
        # save to file using json
        json_txt = json.dumps(dict_exp, indent=4, cls=ExperimentEncoder)
        with open('experiment.txt', 'w') as f:
            f.write(json_txt)

    @staticmethod
    def load(file_path='experiment.txt'):
        with open(file_path, 'r') as f:
            dict_exp = json.load(f)
        sample = Sample()
        sample.set_name(dict_exp['Sample']['Name'])
        sample.set_position(dict_exp['Sample']['Position'])
        if dict_exp['Sample'].has_key('Geometry'):
            sample_geo = ObjectGeometry()
            sample_geo.set_type(dict_exp['Sample']['Geometry']['Type'])
            sample.set_geometry(sample_geo)
        if dict_exp['Sample'].has_key('Material'):
            a, b, c = dict_exp['Sample']['Material']['Lengths']
            alpha, beta, gamma = dict_exp['Sample']['Material']['Angles']
            centering = dict_exp['Sample']['Material']['Centering']
            symmetry = Symmetry.from_string(dict_exp['Sample']['Material']['Symmetry'])
            material = Lattice.from_parameters(a, b, c, alpha, beta, gamma, centering=centering, symmetry=symmetry)
            sample.set_material(material)
        if dict_exp['Sample'].has_key('Microstructure'):
            micro = Microstructure(dict_exp['Sample']['Microstructure']['Name'])
            for i in range(len(dict_exp['Sample']['Microstructure']['Grains'])):
                dict_grain = dict_exp['Sample']['Microstructure']['Grains'][i]
                grain = Grain(dict_grain['Id'], Orientation.from_euler(dict_grain['Orientation']['Euler Angles (degrees)']))
                grain.position = dict_grain['Position']
                grain.volume = dict_grain['Volume']
                micro.grains.append(grain)
            sample.set_microstructure(micro)
        exp = Experiment()
        exp.set_sample(sample)
        source = XraySource()
        source.set_position(dict_exp['Source']['Position'])
        exp.set_source(source)
        for i in range(len(dict_exp['Detectors'])):
            dict_det = dict_exp['Detectors'][i]
            if dict_det['Class'] == 'Detector2d':
                det = Detector2d(size=dict_det['Size (pixels)'])
                det.ref_pos = dict_det['Reference Position (mm)']
            if dict_det['Class'] == 'RegArrayDetector2d':
                det = RegArrayDetector2d(size=dict_det['Size (pixels)'])
                det.pixel_size = dict_det['Pixel Size (mm)']
                det.ref_pos = dict_det['Reference Position (mm)']
                import numpy as np
                det.u_dir = np.array(dict_det['u_dir'])
                det.v_dir = np.array(dict_det['v_dir'])
                det.w_dir = np.array(dict_det['w_dir'])
            exp.add_detector(det)
        return exp


class ExperimentEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, ObjectGeometry):
            dict_geo = {}
            dict_geo['Type'] = o.geo_type
            return dict_geo
        if isinstance(o, Lattice):
            dict_lattice = {}
            dict_lattice['Angles'] = o._angles.tolist()
            dict_lattice['Lengths'] = o._lengths.tolist()
            dict_lattice['Centering'] = o._centering
            dict_lattice['Symmetry'] = o._symmetry.to_string()
            return dict_lattice
        if isinstance(o, Sample):
            dict_sample = {}
            dict_sample['Name'] = o.name
            dict_sample['Position'] = o.position.tolist()
            dict_sample['Geometry'] = o.geo
            dict_sample['Material'] = o.material
            dict_sample['Microstructure'] = o.microstructure
            return dict_sample
        if isinstance(o, RegArrayDetector2d):
            dict_det = {}
            dict_det['Class'] = o.__class__.__name__
            dict_det['Size (pixels)'] = o.size
            dict_det['Pixel Size (mm)'] = o.pixel_size
            dict_det['Data Type'] = str(o.data_type)
            dict_det['Reference Position (mm)'] = o.ref_pos.tolist()
            dict_det['u_dir'] = o.u_dir.tolist()
            dict_det['v_dir'] = o.v_dir.tolist()
            dict_det['w_dir'] = o.w_dir.tolist()
            return dict_det
        if isinstance(o, Detector2d):
            dict_det = {}
            dict_det['Class'] = o.__class__.__name__
            dict_det['Size (pixels)'] = o.size
            dict_det['Data Type'] = o.data_type
            dict_det['Reference Position (mm)'] = o.ref_pos.tolist()
            return dict_det
        if isinstance(o, XraySource):
            dict_source = {}
            dict_source['Position'] = o.position.tolist()
            return dict_source
        if isinstance(o, Microstructure):
            dict_micro = {}
            dict_micro['Name'] = o.name
            dict_micro['Grains'] = o.grains
            return dict_micro
        if isinstance(o, Grain):
            dict_grain = {}
            dict_grain['Id'] = o.id
            dict_grain['Position'] = o.position.tolist()
            dict_grain['Orientation'] = o.orientation
            dict_grain['Volume'] = o.volume
            return dict_grain
        if isinstance(o, Orientation):
            dict_orientation = {}
            dict_orientation['Euler Angles (degrees)'] = o.euler.tolist()
            return dict_orientation
