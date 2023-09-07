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
                                    'GrainDataTable': TestGrainData,
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
        # Create a vector nodal field array
        self.vector_nodal1 = np.zeros((6,3))
        self.vector_nodal1[4,:] = np.array([np.sqrt(2), np.sqrt(2), 0])
        self.vector_nodal1[5,:] = np.array([np.sqrt(2), -np.sqrt(2), 0])
        self.vector_nodal2 = np.zeros((6,3))
        self.vector_nodal2[4,:] = -self.vector_nodal1[4,:]
        self.vector_nodal2[5,:] = -self.vector_nodal1[5,:]
        # Create 2 element wise fields
        self.mesh_el_Id = np.array([0., 1., 2., 3., 4., 5., 6., 7.])
        self.mesh_alternated = np.array([1., 1., -1., -1., 1., 1., -1., -1.])
        # Create Mesh object
        self.mesh = UMCT.CreateMeshOfTriangles(self.mesh_nodes,
                                               self.mesh_elements)
        # Add mesh node tags
        self.mesh.nodesTags.CreateTag('Z0_plane', False).SetIds([0, 1, 2, 3])
        self.mesh.nodesTags.CreateTag('out_of_plane', False).SetIds([4, 5])
        # Add element tags
        self.mesh.GetElementsOfType('tri3').GetTag('Top').SetIds([0, 2, 4, 6])
        self.mesh.GetElementsOfType('tri3').GetTag('Bottom').SetIds([1, 3, 5, 7])
        # Add mesh node fields
        self.mesh.nodeFields['Test_field1'] = self.mesh_shape_f1
        self.mesh.nodeFields['Test_field2'] = self.mesh_shape_f2
        # Add mesh vector node field
        # Add mesh element fields
        self.mesh.elemFields['Test_field3'] = self.mesh_el_Id
        self.mesh.elemFields['Test_field4'] = self.mesh_alternated
        # Create a binary 3D Image
        self.image = np.zeros((10, 10, 10), dtype='int16')
        self.image[:, :, :5] = 1
        self.image_origin = np.array([-1., -1., -1.])
        self.image_voxel_size = np.array([0.2, 0.2, 0.2])
        # Create a tensorial field defined on the image
        self.tensor_field = np.zeros((*self.image.shape, 9), dtype='float32')
        for i in range(9):
            self.tensor_field[:,:,:,i] = i*self.image
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
        """Test creation of a SampleData instance/file."""
        # Test class instantiation method
        sample = SampleData(filename=self.filename,
                            overwrite_hdf5=True, verbose=False,
                            sample_name=self.sample_name,
                            sample_description=self.sample_description)
        self.assertTrue(os.path.exists(self.filename + '.h5'))
        self.assertEqual(sample.get_sample_name(), self.sample_name)
        self.assertEqual(sample.get_description(), self.sample_description)
        # close sample data instance
        del sample
        # reopen sample data instance
        sample = SampleData(filename=self.filename)
        # test sampledata instance and file autodelete function
        sample.autodelete = True
        del sample
        self.assertTrue(not os.path.exists(self.filename+'.h5'))

    def test_basic_data_handling(self):
        """Test creation of a Group and data array + data recovery."""
        # SampleData object Instantiation
        sample = SampleData(filename=self.filename,
                            overwrite_hdf5=True, verbose=False,
                            sample_name=self.sample_name,
                            sample_description=self.sample_description)
        # Add new group and array to SampleData dataset
        sample.add_group(groupname='test_group', location='/',
                         indexname='group')
        sample.add_data_array(location='group', name='test_array',
                              array=self.data_array, indexname='array')
        # Test add attribute
        sample.add_attributes({'test_attribute':'test'}, 'test_group')
        # close sample data instance
        del sample
        # reopen sample data instance
        sample = SampleData(filename=self.filename, autodelete=True)
        # test data array recovery with get_node and path
        array = sample.get_node('/test_group/test_array', as_numpy=True)
        self.assertTrue(np.all(array == self.data_array))
        # test data array recovery with attribute like access
        array = sample.test_array
        self.assertTrue(np.all(array == self.data_array))
        # test data array recovery with dict like access and indexname
        array = sample['array']
        self.assertTrue(np.all(array == self.data_array))
        # test attribute recovery
        attribute = sample.get_attribute('test_attribute', 'test_group')
        self.assertTrue(attribute == 'test')
        # test remove attribute
        sample.remove_attribute('test_attribute', 'test_group')
        attribute = sample.get_attribute('test_attribute', 'test_group')
        self.assertTrue(attribute is None)
        # test rename node
        sample.rename_node('test_array', 'new_test')
        array = sample['new_test']
        self.assertTrue(np.all(array == self.data_array))
        # test remove node
        sample.remove_node('test_group', recursive=True)
        array = sample['new_test']
        self.assertFalse(sample.__contains__('test_group'))
        self.assertFalse(sample.__contains__('new_test'))
        # Close and Remove dataset
        sample.autodelete = True
        del sample
        self.assertTrue(not os.path.exists(self.filename+'.h5'))

    def test_tables(self):
        """Test creation of structured tables + data recovery."""
        # SampleData object Instantiation
        sample = SampleData(filename=self.filename,
                            overwrite_hdf5=True, verbose=False,
                            sample_name=self.sample_name,
                            sample_description=self.sample_description)
        # Add new group and array to SampleData dataset
        sample.add_group(groupname='test_group', location='/',
                         indexname='group')
        # Add a structured array as table node
        sample.add_table('group', 'test_table', description=self.dtype1,
                         data=self.struct_array1)
        # close sample data instance
        del sample
        # reopen sample data instance
        sample = SampleData(filename=self.filename, autodelete=True)
        # get table data and check content
        density = sample['test_table']['density'][:]
        self.assertTrue(np.all(density == self.struct_array1['density'][:]))
        # Close and Remove dataset
        sample.autodelete = True
        del sample
        self.assertTrue(not os.path.exists(self.filename+'.h5'))

    def test_image_group(self):
        """Test storage and recovery of image data via SampleData."""
        # SampleData object Instantiation
        sample = SampleData(filename=self.filename,
                            overwrite_hdf5=True, verbose=False,
                            sample_name=self.sample_name,
                            sample_description=self.sample_description)
        # Test addition of Image group into dataset
        image = ConstantRectilinearMesh(dim=len(self.image.shape))
        image.SetDimensions(self.image.shape)
        image.SetOrigin(self.image_origin)
        image.SetSpacing(self.image_voxel_size)
        image.elemFields['test_image_field'] = self.image
        sample.add_image(image, imagename='test_image', indexname='image',
                         location='/')
        # Test addition of a tensor fields with time value
        sample.add_field(gridname='image', fieldname='test_tensor',
                         array=self.tensor_field, indexname='tensor9',
                         time=1.)
        sample.add_field(gridname='image', fieldname='test_tensor',
                         array=2*self.tensor_field, indexname='tensor9',
                         time=2.)
        sample.add_field(gridname='image', fieldname='test_tensor',
                         array=3*self.tensor_field, indexname='tensor9',
                         time=3.)
        # close sample data instance
        del sample
        # reopen sample data instance
        sample = SampleData(filename=self.filename, autodelete=True)
        # test recovery of image grid specifications
        origin = sample.get_attribute('origin', 'test_image')
        spacing = sample.get_attribute('spacing', 'test_image')
        self.assertTrue(np.all(origin == self.image_origin))
        self.assertTrue(np.all(spacing == self.image_voxel_size))
        # test image field recovery and dictionary like access
        image_field = sample['test_image_field']
        self.assertTrue(np.all(image_field == self.image))
        # test tensor field recovery
        time_list = sample.get_attribute('time_list', 'image')
        self.assertTrue(np.all(np.isin(time_list, np.array([1.,2.,3.]))))
        image_field = sample['test_tensor_T2_0']
        self.assertTrue(np.all(image_field[:,:,:,4] == 4*2*self.image))
        # Close and Remove dataset
        sample.autodelete = True
        del sample
        self.assertTrue(not os.path.exists(self.filename+'.h5'))

    def test_add_image_from_field(self):
        """Test construction of image data from a numpy array."""
        # SampleData object Instantiation
        sample = SampleData(filename=self.filename, autodelete=True,
                            overwrite_hdf5=True, verbose=False,
                            sample_name=self.sample_name,
                            sample_description=self.sample_description)
        # Add image from numpy array representing a 3D field
        sample.add_image_from_field(field_array=self.image,
                                    fieldname='test_image_field',
                                    imagename='test_image', indexname='image')
        # test image field recovery and dictionary like access
        image_field = sample['test_image_field']
        self.assertTrue(np.all(image_field == self.image))
        # test changing image topology
        sample.set_voxel_size('test_image', np.array([0.1,0.1,0.1]))
        sample.set_origin('test_image', np.array([1.0,1.0,1.0]))
        origin = sample.get_attribute('origin', 'test_image')
        spacing = sample.get_attribute('spacing', 'test_image')
        self.assertTrue(np.all(origin == (-self.image_origin)))
        self.assertTrue(np.all(spacing == (self.image_voxel_size/2.0)))
        # Close and Remove dataset
        del sample
        self.assertTrue(not os.path.exists(self.filename+'.h5'))

    def test_mesh_group(self):
        """Test storage and recovery of mesh data via SampleData."""
        # SampleData object Instantiation
        sample = SampleData(filename=self.filename,
                            overwrite_hdf5=True, verbose=False,
                            sample_name=self.sample_name,
                            sample_description=self.sample_description)
        # Test addition of mesh group into dataset
        sample.add_mesh(self.mesh, meshname='test_mesh', indexname='mesh',
                        location='/', bin_fields_from_sets=True)
        # Test addition of a tensor fields with time value
        sample.add_field(gridname='mesh', fieldname='test_vector',
                         array=self.vector_nodal1, indexname='vector',
                         time=1.)
        sample.add_field(gridname='mesh', fieldname='test_vector',
                         array=self.vector_nodal2, indexname='vector',
                         time=2.)
        # close sample data instance
        del sample
        # reopen sample data instance
        sample = SampleData(filename=self.filename, autodelete=True)
        # test mesh nodes arrays recovery
        mesh_nodes = sample.get_mesh_nodes(meshname='mesh', as_numpy=True)
        nodes_ID = sample.get_mesh_nodesID(meshname='mesh', as_numpy=True)
        self.assertTrue(np.all(mesh_nodes == self.mesh_nodes))
        self.assertTrue(np.all(nodes_ID == np.array(range(6))))
        # test mesh node tags recovery
        nodeTag_names = sample.get_mesh_node_tags_names(meshname='mesh')
        self.assertTrue('out_of_plane' in nodeTag_names)
        nodeTag = sample.get_mesh_node_tag(meshname='mesh',
                                           node_tag='Z0_plane', as_numpy=True)
        nodeTag_coord = sample.get_mesh_node_tag_coordinates(meshname='mesh',
                                                           node_tag='Z0_plane')
        self.assertTrue(np.all(nodeTag == np.array([0,1,2,3])))
        self.assertTrue(np.all(nodeTag_coord == self.mesh_nodes[0:4,:]))
        # test mesh element connectivity array recovery
        mesh_elements = sample.get_mesh_xdmf_connectivity(meshname='mesh',
                                                          as_numpy=True)
        mesh_elements = mesh_elements.reshape(self.mesh_elements.shape)
        self.assertTrue(np.all(mesh_elements == self.mesh_elements))
        # test mesh element type recovery
        elem_types = sample.get_mesh_elem_types_and_number(meshname='mesh')
        self.assertTrue(elem_types['tri3'] == 8)
        # test mesh element tags recovery
        elTag_names = sample.get_mesh_elem_tags_names(meshname='mesh')
        self.assertTrue('Top' in elTag_names)
        elem_Tag = sample.get_mesh_elem_tag(meshname='mesh',
                                            element_tag='Top')
        self.assertTrue(np.all(elem_Tag == np.array([0,2,4,6])))
        elem_Tag_connectivity = sample.get_mesh_elem_tag_connectivity(
                                            meshname='mesh', element_tag='Top')
        self.assertTrue(np.all(elem_Tag_connectivity ==
                                              self.mesh_elements[[0,2,4,6],:]))
        # test mesh fields recovery
        shape_f1 = sample.get_field('Test_field1')
        self.assertTrue(np.all(shape_f1 == self.mesh_shape_f1))
        shape_f1 = sample.get_field('test_vector_T1_0')
        self.assertTrue(np.all(shape_f1 == self.vector_nodal1))
        shape_f1 = sample.get_field('test_vector_T2_0')
        self.assertTrue(np.all(shape_f1 == self.vector_nodal2))
        # Close and Remove dataset
        del sample

    def test_copy(self):
        """Test dataset creation by copying another dataset."""
        # Creation of new dataset from copy of reference file
        sample = SampleData.copy_sample(src_sample_file=self.reference_file,
                                        dst_sample_file=self.filename,
                                        overwrite=True, get_object=True,
                                        autodelete=True)
        # various data content checks
        # get table data and check content
        density = sample['test_table']['density'][:]
        self.assertTrue(np.all(density == self.struct_array1['density'][:]))
        # test recovery of image grid specifications
        origin = sample.get_attribute('origin', 'test_image')
        spacing = sample.get_attribute('spacing', 'test_image')
        # test mesh fields recovery
        self.assertTrue(np.all(origin == self.image_origin))
        self.assertTrue(np.all(spacing == self.image_voxel_size))
        shape_f1 = sample.get_field('test_vector_T1_0')
        self.assertTrue(np.all(shape_f1 == self.vector_nodal1))
        # Close and Remove dataset
        del sample

    def test_print_dataset_content(self):
        """Test method to print information on dataset."""
        # Open reference SampleData file
        sample = SampleData(filename=self.reference_file)
        # test print dataset content
        sample.print_dataset_content(to_file=self.filename+'_content.txt',
                                     max_depth=4, short=True)
        del sample
        self.assertTrue(os.path.exists(self.filename+'_content.txt'))
        os.remove(self.filename+'_content.txt')

    def test_print_index(self):
        """Test methods to print index of dataset."""
        # Open reference SampleData file
        sample = SampleData(filename=self.reference_file)
        # test print index
        sample.print_index(max_depth=4)
        del sample

    def test_specific_prints(self):
        # Open reference SampleData file
        sample = SampleData(filename=self.reference_file)
        # test print index
        sample.print_grids_info()
        sample.print_data_arrays_info()
        sample.print_xdmf()
        del sample

    def test_write_xdmf(self):
        """Test XDMF file writing from reference SampeData dataset."""
        # Open reference SampleData file
        sample = SampleData(filename=self.reference_file)
        # test writing XDMF file
        sample.write_xdmf()
        del sample

    def test_compress(self):
        """Test data array compression."""
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
        # Verify data content
        data_array = sample.get_node('test_image_field', as_numpy=True)
        self.assertTrue(np.all(self.image == data_array))
        # delete SampleData instance and assert files deletion
        del sample
        self.assertTrue(not os.path.exists(self.filename + '.h5'))

    def test_lossy_compression(self):
        """Test data array compression."""
        # TODO: test normalization
        sample = SampleData.copy_sample(src_sample_file=self.reference_file,
                                        dst_sample_file=self.filename,
                                        overwrite=True, get_object=True,
                                        autodelete=True)
        # get filesizes
        original_filesize, _ = sample.get_file_disk_size(print_flag=False,
                                                         convert=False)
        original_size, _ = sample.get_node_disk_size('test_array',
                                                     print_flag=False,
                                                     convert=False)
        # Verify data content
        data_array = sample.get_node('test_array', as_numpy=True)
        self.assertTrue(np.all(self.data_array == data_array))
        # compress image data
        c_opt = {'complib': 'zlib', 'complevel': 1,
                 'least_significant_digit':2}
        sample.set_chunkshape_and_compression(nodename='test_array',
                                              compression_options=c_opt)
        # assert that node size is smaller after compression
        new_size, _ = sample.get_node_disk_size('test_array',
                                                print_flag=False,
                                                convert=False)
        new_filesize, _ = sample.get_file_disk_size(print_flag=False,
                                                    convert=False)
        # repack file and assert file size is lower than original filesize
        sample.repack_h5file()
        new_filesize, _ = sample.get_file_disk_size(print_flag=False,
                                                    convert=False)
        self.assertGreater(original_filesize, new_filesize)
        # Verify data content recovery to prescribed precision
        data_array = sample.get_node('test_array', as_numpy=True)
        self.assertTrue(np.all((self.data_array - data_array) < 10**(-2)))
        # delete SampleData instance and assert files deletion
        del sample
        self.assertTrue(not os.path.exists(self.filename + '.h5'))

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

    def test_import_geof_mesh(self):
        """Test import of Zset mesh file as SampleData dataset."""
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
        field1 = sample.get_field('test_field')
        self.assertEqual(field1.shape, tuple(dimensions))
        field2 = sample.get_field('Tmsh_test_field_Tetra_mesh')
        self.assertEqual(field2.shape, (11 * 11 * 11,))
        self.assertEqual(field1.ravel()[37], field2.ravel()[37])
        del sample