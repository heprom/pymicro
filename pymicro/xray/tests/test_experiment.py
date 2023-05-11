import unittest
import os
from pymicro.xray.experiment import ForwardSimulation, Experiment, Sample, ObjectGeometry
from pymicro.xray.detectors import RegArrayDetector2d
from config import PYMICRO_EXAMPLES_DATA_DIR


class ForwardSimulationTests(unittest.TestCase):

    def setUp(self):
        """testing the ForwardSimulation class:"""
        self.fsim = ForwardSimulation('test')

    def test_set_geo_type(self):
        print(self.fsim.sample_geo)
        print(self.fsim.sample_geo.geo_type)
        self.assertEqual(self.fsim.sample_geo.geo_type, 'point')
        self.fsim.set_sample_geo_type('array')
        self.assertEqual(self.fsim.sample_geo.geo_type, 'array')


class ObjectGeometryTests(unittest.TestCase):

    def setUp(self):
        """testing the ObjectGeometry class:"""
        self.geo = ObjectGeometry()
        self.sample = Sample(os.path.join(PYMICRO_EXAMPLES_DATA_DIR, 'm1_data.h5'))

    def test_discretize_geometry(self):
        self.geo.discretize_geometry()
        self.assertEqual(len(self.geo.get_positions()), 1)
        self.geo.set_type('array')
        self.geo.set_array(self.sample.get_grain_map(),
                           self.sample.get_voxel_size())
        self.geo.discretize_geometry(grain_id=1)
        self.assertEqual(len(self.geo.get_positions()), 1547)

    def tearDown(self):
        del self.sample

class ExperimentTests(unittest.TestCase):

    def setUp(self):
        """testing the experiment module:"""
        self.experiment = Experiment()
        self.experiment.get_sample().set_sample_name('test sample')
        self.experiment.get_sample().autodelete = True
        self.experiment.get_source().set_max_energy(120.)

    def test_add_detector(self):
        """Test the add_detector method."""
        detector = RegArrayDetector2d(size=(512, 512))
        self.assertEqual(self.experiment.get_number_of_detectors(), 0)
        self.experiment.add_detector(detector)
        self.assertEqual(self.experiment.get_number_of_detectors(), 1)
        self.assertTrue(self.experiment.sample.autodelete)
        del detector, self.experiment

    def test_save(self):
        """Test the save method for an experiment."""
        if os.path.exists('experiment.txt'):
            os.remove('experiment.txt')
        detector1 = RegArrayDetector2d(size=(512, 512))
        detector1.pixel_size = 0.1
        detector1.ref_pos[0] = 100.
        self.experiment.add_detector(detector1)
        # set a second detector above the sample in the horizontal plane
        detector2 = RegArrayDetector2d(size=(512, 512))
        detector2.pixel_size = 0.1
        detector2.set_v_dir([0, -90, 0])
        detector2.ref_pos[2] = 100.
        self.experiment.add_detector(detector2)
        self.experiment.active_detector_id = 0
        self.experiment.save()
        self.assertTrue(os.path.exists('experiment.txt'))
        del self.experiment
        exp = Experiment.load('experiment.txt')
        self.assertTrue(exp.get_source().max_energy == 120.)
        del detector1, detector2, exp

    def tearDown(self):
        if os.path.exists('experiment.txt'):
            os.remove('experiment.txt')
        if os.path.exists('dummy_data.h5'):
            os.remove('dummy_data.h5')
            os.remove('dummy_data.xdmf')
