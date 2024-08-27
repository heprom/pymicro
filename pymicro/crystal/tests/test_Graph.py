import unittest
import os
from pymicro.crystal.microstructure import Microstructure
from pymicro.crystal.graph import create_graph
from config import PYMICRO_EXAMPLES_DATA_DIR


class GraphTests(unittest.TestCase):

    def setUp(self):
        print('testing the Microstructure graph utilities')
        
    def test_create_graph(self):
        m = Microstructure(os.path.join(PYMICRO_EXAMPLES_DATA_DIR, 'm1_data.h5'))
        rag = create_graph(m)
        self.assertEqual(len(rag.nodes), m.get_number_of_grains())
        e = rag.edges[15, 26]  # pick two neighboring grains
        self.assertTrue('misorientation' in e)
        self.assertAlmostEqual(e['misorientation'], 43.9954034, 6)

