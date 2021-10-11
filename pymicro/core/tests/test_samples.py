import unittest
import os
import numpy as np
import math
from tables import IsDescription, Int32Col, Float32Col
from pymicro.core.samples import SampleData
from BasicTools.Containers.ConstantRectilinearMesh import ConstantRectilinearMesh
import BasicTools.Containers.UnstructuredMeshCreationTools as UMCT
from config import PYMICRO_EXAMPLES_DATA_DIR


class TestGrainData(IsDescription):
    """
       Description class specifying structured storage for tests
    """
    idnumber = Int32Col()      # Signed 64-bit integer
    volume = Float32Col()    # float
    center = Float32Col(shape=(3,))    # float


class TestDerivedClass(SampleData):
    """ Class to test the datamodel specification mechanism, via definition
        of classes derived from SampleData
    """
    def minimal_data_model(self):
        """
            Specify the minimal contents of the hdf5 (Group names, paths,, and
            group types) in the form of a dictionary {content:Location}
            Extends SampleData Class _minimal_data_model class
        """
        # create a dtype to create a structured array
        Descr = np.dtype([('density', np.float32), ('melting_Pt', np.float32),
                          ('Chemical_comp', 'S', 30)])
        # create data model description dictionaries
        minimal_content_index_dic = {'Image_data': '/CellData',
                                     'grain_map': '/CellData/grain_map',
                                     'Grain_data': '/GrainData',
                                     'GrainDataTable': ('/GrainData/'
                                                        'GrainDataTable'),
                                     'Crystal_data': '/CrystalStructure',
                                     'lattice_params': ('/CrystalStructure'
                                                        '/LatticeParameters'),
                                     'lattice_props': ('/CrystalStructure'
                                                       '/LatticeProps'),
                                     'grain_names': '/GrainData/GrainNames',
                                     'Mesh_data': '/MeshData'}
        minimal_content_type_dic = {'Image_data': '3DImage',
                                    'grain_map': 'field_array',
                                    'Grain_data': 'Group',
                                    'GrainDataTable': Test_GrainData,
                                    'Crystal_data': 'Group',
                                    'lattice_params': 'data_array',
                                    'lattice_props': Descr,
                                    'grain_names': 'string_array',
                                    'Mesh_data': 'Mesh'
                                    }
        return minimal_content_index_dic, minimal_content_type_dic


class SampleDataTests(unittest.TestCase):

    def setUp(self):
        print('testing the SampleData class')
        # Create data to store into SampleData instances
        # dataset sample_name and description
        self.sample_name = 'test_sample'
        self.sample_description = """
        This is a test dataset created by the SampleData class unit tests.
        """
        # Create a mesh of an octahedron with 6 triangles
        self.mesh_nodes = np.array([[-1., -1., 0.],
                                    [-1., 1., 0.],
                                    [1., 1., 0.],
                                    [1., -1., 0.],
                                    [0., 0., 1.],
                                    [0., 0., -1.]])
        self.mesh_elements = np.array([[0, 1, 4],
                                       [0, 1, 5],
                                       [1, 2, 4],
                                       [1, 2, 5],
                                       [2, 3, 4],
                                       [2, 3, 5],
                                       [3, 0, 4],
                                       [3, 0, 5]])
        # Create 2 fields 'shape functions' for the 2 nodes at z=+/-1
        self.mesh_shape_f1 = np.array([0., 0., 0., 0., 1., 0.])
        self.mesh_shape_f2 = np.array([0., 0., 0., 0., 0., 1.])
        # Create 2 element wise fields
        self.mesh_el_Id = np.array([0., 1., 2., 3., 4., 5., 6., 7.])
        self.mesh_alternated = np.array([1., 1., -1., -1., 1., 1., -1., -1.])
        # Create a binary 3D Image
        self.image = np.zeros((10, 10, 10), dtype='int16')
        self.image[:, :, :5] = 1
        self.image_origin = np.array([-1., -1., -1.])
        self.image_voxel_size = np.array([0.2, 0.2, 0.2])
        # Create a data array
        self.data_array = np.array([math.tan(x) for x in
                                    np.linspace(-math.pi/4, math.pi/4, 51)])
        # Create numpy dtype and structure array
        # WARNING: Pytables transforms all strings into bytes
        #          --> use only bytes in dtypes
        self.dtype1 = np.dtype([('density', np.float32),
                                ('melting_Pt', np.float32),
                                ('Chemical_comp', 'S', 30)])
        self.struct_array1 = np.array([(6.0, 1232, 'Cu2O'),
                                       (5.85, 2608, 'ZrO2')],
                                      dtype=self.dtype1)
        # Test file pathes
        self.filename = os.path.join(PYMICRO_EXAMPLES_DATA_DIR,
                                     'test_sampledata')
        self.derived_filename = self.filename+'_derived'
        self.reference_file = os.path.join(PYMICRO_EXAMPLES_DATA_DIR,
                                           'test_sampledata_ref')

    def test_create_sample(self):
        """Test creation of a SampleData instance/file and  data storage."""
        sample = SampleData(filename=self.filename,
                            overwrite_hdf5=True, verbose=False,
                            sample_name=self.sample_name,
                            sample_description=self.sample_description)
        self.assertTrue(os.path.exists(self.filename + '.h5'))
        self.assertTrue(os.path.exists(self.filename + '.xdmf'))
        self.assertEqual(sample.get_sample_name(), self.sample_name)
        self.assertEqual(sample.get_description(), self.sample_description)
        # Add mesh data into SampleData dataset
        mesh = UMCT.CreateMeshOfTriangles(self.mesh_nodes, self.mesh_elements)
        # Add mesh node tags
        mesh.nodesTags.CreateTag('Z0_plane', False).SetIds([0, 1, 2, 3])
        mesh.nodesTags.CreateTag('out_of_plane', False).SetIds([4, 5])
        # Add element tags
        mesh.GetElementsOfType('tri3').GetTag('Top').SetIds([0, 2, 4, 6])
        mesh.GetElementsOfType('tri3').GetTag('Bottom').SetIds([1, 3, 5, 7])
        # Add mesh node fields
        mesh.nodeFields['Test_field1'] = self.mesh_shape_f1
        mesh.nodeFields['Test_field2'] = self.mesh_shape_f2
        # Add mesh element fields
        mesh.elemFields['Test_field3'] = self.mesh_el_Id
        mesh.elemFields['Test_field4'] = self.mesh_alternated
        sample.add_mesh(mesh, meshname='test_mesh', indexname='mesh',
                        location='/', bin_fields_from_sets=True)
        # Add image data into SampleData dataset
        image = ConstantRectilinearMesh(dim=len(self.image.shape))
        image.SetDimensions(self.image.shape)
        image.SetOrigin(self.image_origin)
        image.SetSpacing(self.image_voxel_size)
        image.elemFields['test_image_field'] = self.image
        sample.add_image(image, imagename='test_image', indexname='image',
                         location='/')
        # Add new group and array to SampleData dataset
        sample.add_group(groupname='test_group', location='/', indexname='group')
        sample.add_data_array(location='group', name='test_array',
                              array=self.data_array, indexname='array')
        # close sample data instance
        del sample
        # reopen sample data instance
        sample = SampleData(filename=self.filename)
        # test mesh geometry data recovery
        mesh_nodes = sample.get_mesh_nodes(meshname='mesh', as_numpy=True)
        self.assertTrue(np.all(mesh_nodes == self.mesh_nodes))
        mesh_elements = sample.get_mesh_xdmf_connectivity(meshname='mesh',
                                                          as_numpy=True)
        mesh_elements = mesh_elements.reshape(self.mesh_elements.shape)
        self.assertTrue(np.all(mesh_elements == self.mesh_elements))
        # test mesh field recovery
        shape_f1 = sample.get_field('Test_field1')
        self.assertTrue(np.all(shape_f1 == self.mesh_shape_f1))
        # test image field recovery and dictionary like access
        image_field = sample['test_image_field']
        self.assertTrue(np.all(image_field == self.image))
        # test data array recovery and attribute like access
        array = sample.test_array
        self.assertTrue(np.all(array == self.data_array))
        # test sampledata instance and file autodelete function
        sample.autodelete = True
        del sample
        self.assertTrue(not os.path.exists(self.filename+'.h5'))
        self.assertTrue(not os.path.exists(self.filename+'.xdmf'))

    def test_copy_and_compress(self):
        """ Copy the reference dataset and compress it """
        sample = SampleData.copy_sample(src_sample_file=self.reference_file,
                                        dst_sample_file=self.filename,
                                        overwrite=True, get_object=True,
                                        autodelete=True)
        # get filesizes
        original_filesize, _ = sample.get_file_disk_size(print_flag=False,
                                                         convert=False)
        original_size, _ = sample.get_node_disk_size('test_image_field',
                                                     print_flag=False,
                                                     convert=False)
        # Verify data content
        data_array = sample.get_node('test_array')
        self.assertTrue(np.all(self.data_array == data_array))
        # compress image data
        c_opt = {'complib': 'zlib', 'complevel': 1}
        sample.set_chunkshape_and_compression(nodename='test_image_field',
                                              compression_options=c_opt)
        # assert that node size is smaller after compression
        new_size, _ = sample.get_node_disk_size('test_image_field',
                                                print_flag=False,
                                                convert=False)
        new_filesize, _ = sample.get_file_disk_size(print_flag=False,
                                                    convert=False)
        # repack file and assert file size is lower than original filesize
        sample.repack_h5file()
        new_filesize, _ = sample.get_file_disk_size(print_flag=False,
                                                    convert=False)
        self.assertGreater(original_filesize, new_filesize)
        # delete SampleData instance and assert files deletion
        del sample
        self.assertTrue(not os.path.exists(self.filename + '.h5'))
        self.assertTrue(not os.path.exists(self.filename + '.xdmf'))

    def test_derived_class(self):
        """ Test application specific data model specification through
            derived classes.
            Also test table functionalities.
        """
        derived_sample = TestDerivedClass(filename=self.derived_filename,
                                          autodelete=False,
                                          overwrite_hdf5=True, verbose=False)
        # assert data model Nodes are contained in dataset
        self.assertTrue(derived_sample.__contains__('Image_data'))
        self.assertTrue(derived_sample.__contains__('grain_map'))
        self.assertTrue(derived_sample.__contains__('Grain_data'))
        self.assertTrue(derived_sample.__contains__('GrainDataTable'))
        self.assertTrue(derived_sample.__contains__('Crystal_data'))
        self.assertTrue(derived_sample.__contains__('lattice_params'))
        self.assertTrue(derived_sample.__contains__('lattice_props'))
        self.assertTrue(derived_sample.__contains__('grain_names'))
        self.assertTrue(derived_sample.__contains__('Mesh_data'))
        # # assert data items created are empty, except for Groups
        self.assertTrue(derived_sample._is_empty('Image_data'))
        self.assertTrue(derived_sample._is_empty('grain_map'))
        self.assertTrue(derived_sample._is_empty('GrainDataTable'))
        self.assertTrue(derived_sample._is_empty('lattice_params'))
        self.assertTrue(derived_sample._is_empty('lattice_props'))
        self.assertTrue(derived_sample._is_empty('grain_names'))
        self.assertTrue(derived_sample._is_empty('Mesh_data'))
        # get table node and assert description
        descr = derived_sample.get_table_description('GrainDataTable')
        self.assertEqual(TestGrainData.columns,
                         descr._v_colobjects)
        # add columns to the table
        dtype = np.dtype([('name', np.str_, 16), ('floats', np.float64, (2,))])
        derived_sample.add_tablecols('GrainDataTable', description=dtype)
        tab = derived_sample.get_node('GrainDataTable')
        self.assertTrue('name' in tab.colnames)
        self.assertTrue('floats' in tab.colnames)
        # append other table with numpy array and verify it is no more empty
        derived_sample.append_table('lattice_props', self.struct_array1)
        self.assertFalse(derived_sample._is_empty('lattice_props'))
        # append string array and verify that it is not empty
        derived_sample.append_string_array('grain_names',
                                           ['grain_1', 'grain_2', 'grain_3'])
        self.assertFalse(derived_sample.get_attribute('empty', 'grain_names'))
        del derived_sample
        # reopen file and check that neqw columns have been added
        derived_sample = TestDerivedClass(
            filename=self.derived_filename, autodelete=True,
            overwrite_hdf5=False, verbose=False)
        derived_sample.print_node_info('GrainDataTable')
        tab = derived_sample.get_node('GrainDataTable')
        self.assertTrue('name' in tab.colnames)
        self.assertTrue('floats' in tab.colnames)
        # check other table values
        props = derived_sample['lattice_props']
        self.assertTrue(np.all(props == self.struct_array1))
        # check string array values
        name1 = derived_sample['grain_names'][0].decode('utf-8')
        name2 = derived_sample['grain_names'][1].decode('utf-8')
        self.assertEqual(name1, 'grain_1')
        self.assertEqual(name2, 'grain_2')
        del derived_sample
        self.assertTrue(not os.path.exists(self.derived_filename + '.h5'))
        self.assertTrue(not os.path.exists(self.derived_filename + '.xdmf'))

    def test_BasicTools_binding(self):
        """Test BasicTools to SampleData to BasicTools."""
        # create mesh of triangles
        myMesh = UMCT.CreateSquare(dimensions=[3, 3], ofTris=True)
        # get into a SampleData instance
        sample = SampleData(filename='square', verbose=False, autodelete=True)
        sample.add_mesh(mesh_object=myMesh, meshname='BT_mesh', indexname='BTM',
                        replace=True, bin_fields_from_sets=False)
        # get mesh object from SampleData file/instance
        myMesh2 = sample.get_mesh('BTM')
        # delete SampleData object and test values
        self.assertTrue(np.all(myMesh.nodes == myMesh2.nodes))
        # assert bulk element  connectivity
        connectivity = myMesh.elements['tri3'].connectivity
        connectivity2 = myMesh2.elements['tri3'].connectivity
        self.assertTrue(np.all(connectivity == connectivity2))
        # assert boundary element  connectivity
        connectivity = myMesh.elements['bar2'].connectivity
        connectivity2 = myMesh2.elements['bar2'].connectivity
        self.assertTrue(np.all(connectivity == connectivity2))
        # assert boundary element tags values
        elements_in_tag = myMesh.GetElementsInTag('ExteriorSurf')
        elements_in_tag2 = myMesh2.GetElementsInTag('ExteriorSurf')
        self.assertTrue(np.all(elements_in_tag == elements_in_tag2))
        # assert bulk element tags values
        elements_in_tag = myMesh.GetElementsInTag('2D')
        elements_in_tag2 = myMesh2.GetElementsInTag('2D')
        self.assertTrue(np.all(elements_in_tag == elements_in_tag2))
        del sample

    def test_meshfile_formats(self):
        # TODO: add more mesh formats to load in this test
        from config import PYMICRO_EXAMPLES_DATA_DIR
        sample = SampleData(filename='tmp_meshfiles_dataset',
                            overwrite_hdf5=True, autodelete=True)
        meshfile_name = os.path.join(PYMICRO_EXAMPLES_DATA_DIR,
                                     'cube_ref.geof')
        sample.add_mesh(file=meshfile_name, meshname='geof_mesh',
                        indexname='mesh', bin_fields_from_sets=True)
        # check the number of elements of the mesh
        n_elems = sample.get_attribute('Number_of_elements', 'mesh')
        self.assertTrue(np.all(n_elems == [384, 384]))
        # check the element types in the mesh
        el_type = sample.get_attribute('element_type', 'mesh')
        self.assertEqual(el_type[0], 'tet4')
        self.assertEqual(el_type[1], 'tri3')
        del sample

    def test_mesh_from_image(self):
        """Test BasicTools to SDimage to SDmesh."""
        # 3D image parameters
        dimensions = [11, 11, 11]
        origin = [0., 0., 0.]
        spacing = [1., 1., 1.]
        # create BasicTools image object
        myMesh = ConstantRectilinearMesh(dim=3)
        myMesh.SetDimensions(dimensions)
        myMesh.SetOrigin(origin)
        myMesh.SetSpacing(spacing)
        # create data field
        data = np.zeros(shape=dimensions)
        data[:, 3:8, 3:8] = 1
        myMesh.nodeFields['test_field'] = data
        # create SD instance and image group
        sample = SampleData(filename='cube', verbose=False, autodelete=True)
        sample.add_image(image_object=myMesh, imagename='Image_3D',
                         indexname='Im3D', replace=True)
        # create mesh group of tetra from image group
        sample.add_mesh_from_image('Im3D', with_fields=True, ofTetras=True,
                                   meshname='Tetra_mesh',
                                   indexname='Tmsh', replace=True)
        self.assertTrue(sample.__contains__('Im3D'))
        self.assertTrue(sample.__contains__('Tmsh'))
        field1 = sample.get_node('test_field', as_numpy=True)
        self.assertEqual(field1.shape, (11, 11, 11))
        field2 = sample.get_node('Tmsh_test_field_Tetra_mesh',
                                 as_numpy=True)
        self.assertEqual(field2.shape, (11 * 11 * 11,))
        self.assertEqual(field1.ravel()[37], field2.ravel()[37])
        del sample