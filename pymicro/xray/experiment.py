import json
from json import JSONEncoder
import numpy as np
from pymicro.xray.detectors import Detector2d, RegArrayDetector2d

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

class Sample:
    """Class to describe a material sample.
    
    A sample is made by a given material (that may have multiple phases), has a name and a position in the experimental 
    local frame. A sample also have a geometry (just a point by default), that may be used to discretize the volume 
    in space or display it in 3D.
    
    .. note::

      For the moment, the material is simply a crystal lattice.
    """

    def __init__(self, name=None, position=(0., 0., 0.), geo=None, ):
        self.name = name
        self.set_position(position)
        if geo is None:
            geo = ObjectGeometry()
        self.set_geometry(geo)

    def set_name(self, name):
        self.name = name

    def set_position(self, position):
        self.position = np.array(position)

    def set_geometry(self, geo):
        if geo is None:
            geo = ObjectGeometry()
        assert isinstance(geo, ObjectGeometry) is True
        self.geo = geo


class Experiment:
    """Class to represent an actual or a virtual X-ray experiment.
    
    A cartesian coordinate system (X, Y, Z) is associated with the experiment. By default X is the direction of X-rays 
    and the sample is placed at the origin (0, 0, 0).
    """

    def __init__(self):
        self.sample = Sample(name='dummy')
        self.detectors = []
        self.active_detector_id = -1

    def set_sample(self, sample):
        assert isinstance(sample, Sample) is True
        self.sample = sample

    def get_sample(self):
        return self.sample

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

    def save(self):
        """Export the parameters to describe the current experiment to a file using json."""
        dict_exp = {}
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
        exp = Experiment()
        exp.set_sample(sample)
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
        return exp


class ExperimentEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, ObjectGeometry):
            dict_geo = {}
            dict_geo['Type'] = o.geo_type
            return dict_geo
        if isinstance(o, Sample):
            dict_sample = {}
            dict_sample['Name'] = o.name
            dict_sample['Position'] = o.position
            dict_sample['Geometry'] = o.geo
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
