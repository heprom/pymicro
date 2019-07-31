import unittest
import numpy as np
import vtk
from vtk.util import numpy_support
from pymicro.crystal.lattice import Lattice
from pymicro.crystal.microstructure import Orientation
from pymicro.view.vtk_utils import lattice_grid, lattice_edges, apply_orientation_to_actor


class VtkUtilsTests(unittest.TestCase):
    def setUp(self):
        print('testing vtk_utils')

    def test_apply_orientation_to_actor(self):
        o = Orientation.from_rodrigues([0.0885, 0.3889, 0.3268])
        Bt = o.orientation_matrix().transpose()  # to go from crystal to lab coordinate Vl = Bt.Vc
        l = Lattice.cubic(1.0)
        (a, b, c) = l._lengths
        grid = lattice_grid(l)
        actor = lattice_edges(grid)
        apply_orientation_to_actor(actor, o)
        m = actor.GetUserTransform().GetMatrix()
        for i in range(3):
            for j in range(3):
                self.assertEqual(Bt[i, j], m.GetElement(i, j))

    def tearDown(self):
        pass


class VtkNumpyArrayTests(unittest.TestCase):
    def setUp(self):
        print('testing numpy to vtk array conversion')
        self.data = np.linspace(0, 20 * 10 * 5 - 1, 1000).reshape((20, 10, 5)).astype(np.uint8)
        self.vtk_data_array = numpy_support.numpy_to_vtk(np.ravel(self.data, order='F'), deep=1)

    def test_data_size(self):
        self.assertEqual(np.size(self.data), self.vtk_data_array.GetSize())

    def test_data_type(self):
        self.assertEqual(self.vtk_data_array.GetDataTypeAsString(), 'unsigned char')

    def test_data_access(self):
        size = self.data.shape
        numpy_val = self.data[10, 5, 3]
        vtk_val = self.vtk_data_array.GetValue(3 * size[0] * size[1] + 5 * size[0] + 10)
        self.assertEqual(numpy_val, vtk_val)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
