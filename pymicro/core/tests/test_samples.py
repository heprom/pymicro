import unittest
import os
import numpy as np
import math
from tables import IsDescription, Int32Col, Float32Col
from pymicro.core.samples import SampleData
from BasicTools.Containers.ConstantRectilinearMesh import ConstantRectilinearMesh
import BasicTools.Containers.UnstructuredMeshCreationTools as UMCT
from config import PYMICRO_EXAMPLES_DATA_DIR


class Test_GrainData(IsDescription):
    """
       Description class specifying structured storage for tests
    """
    idnumber    = Int32Col()      # Signed 64-bit integer
    volume      = Float32Col()    # float
    center      = Float32Col(shape=(3,))    # float

class Test_DerivedClass(SampleData):
    """ Class to test the datamodel specification mechanism, via definition
        of classes derived from SampleData
    """
    def minimal_data_model(self):
        """
            Specify the minimal contents of the hdf5 (Group names, paths,, and
            group types) in the form of a dictionary {content:Location}
            Extends SampleData Class _minimal_data_model class
        """
        minimal_content_index_dic = {'Image_data':'/CellData',
                                     'grain_map':'/CellData/grain_map',
                                     'Grain_data':'/GrainData',
                                     'GrainDataTable':('/GrainData/'
                                                       'GrainDataTable'),
                                     'Crystal_data':'/CrystalStructure',
                                     'lattice_params':('/CrystalStructure'
                                                       '/LatticeParameters'),}
        minimal_content_type_dic = {'Image_data':'3DImage',
                                    'grain_map':'Array',
                                    'Grain_data':'Group',
                                    'GrainDataTable':Test_GrainData,
                                    'Crystal_data':'Group',
                                    'lattice_params':'Array',}
        return minimal_content_index_dic, minimal_content_type_dic

class SampleDataTests(unittest.TestCase):

    def setUp(self):
        print('testing the SampleData class')
        # Create data to store into SampleData instances
        # Create a mesh of an octahedron with 6 triangles
        self.mesh_nodes = np.array([[-1.,-1., 0.],
                                     [-1., 1., 0.],
                                     [ 1., 1., 0.],
                                     [ 1.,-1., 0.],
                                     [ 0., 0., 1.],
                                     [ 0., 0.,-1.]])
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
        # Create a binary 3D Image
        self.image = np.zeros((10,10,10),dtype='int16')
        self.image[:,:,:5] = 1
        self.image_origin = np.array([-1.,-1.,-1.])
        self.image_voxel_size = np.array([0.2,0.2,0.2])
        # Create a data array
        self.data_array = np.array([ math.tan(x) for x in
                                       np.linspace(-math.pi/4,math.pi/4,51)])
        self.filename = os.path.join(PYMICRO_EXAMPLES_DATA_DIR,
                                     'test_sampledata')
        self.derived_filename = self.filename+'_derived'
        self.reference_file = os.path.join(PYMICRO_EXAMPLES_DATA_DIR,
                                     'test_sampledata_ref')

    def test_create_sample(self):
        """Test creation of a SampleData instance/file and  data storage."""
        sample = SampleData(filename=self.filename,
                            sample_name='validation_test_sample',
                            overwrite_hdf5=True, verbose=False)
        self.assertTrue(os.path.exists(self.filename+'.h5'))
        self.assertTrue(os.path.exists(self.filename+'.xdmf'))
        # Add mesh data into SampleData dataset
        mesh = UMCT.CreateMeshOfTriangles(self.mesh_nodes, self.mesh_elements)
        mesh.nodeFields['Test_field1'] = self.mesh_shape_f1
        mesh.nodeFields['Test_field2'] = self.mesh_shape_f2
        sample.add_mesh(mesh, meshname='test_mesh',indexname='mesh',
                        location='/', extended_data=False)
        # Add image data into SampleData dataset
        image = ConstantRectilinearMesh(dim=len(self.image.shape))
        image.SetDimensions(self.image.shape)
        image.SetOrigin(self.image_origin)
        image.SetSpacing(self.image_voxel_size)
        image.elemFields['test_image_field'] = self.image
        sample.add_image(image,imagename='test_image',indexname='image',
                         location='/')
        # Add new group and array to SampleData dataset
        sample.add_group(groupname='test_group',location='/',indexname='group')
        sample.add_data_array(location='group', name='test_array',
                              array=self.data_array, indexname='array')
        # close sample data instance
        del sample
        # reopen sample data instance
        sample = SampleData(filename=self.filename)
        # test mesh geometry data recovery
        mesh_nodes = sample.get_mesh_nodes(meshname='mesh',as_numpy=True)
        self.assertTrue(np.all(mesh_nodes==self.mesh_nodes))
        mesh_elements = sample.get_mesh_xdmf_connectivity(meshname='mesh',
                                                          as_numpy=True)
        mesh_elements = mesh_elements.reshape(self.mesh_elements.shape)
        self.assertTrue(np.all(mesh_elements==self.mesh_elements))
        # test mesh field recovery
        shape_f1 = sample.get_node('Test_field1', as_numpy=True)
        self.assertTrue(np.all(shape_f1==self.mesh_shape_f1))
        # test image field recovery
        image_field = sample.get_node('test_image_field', as_numpy=True)
        self.assertTrue(np.all(image_field==self.image))
        # test sampledata instance and file autodelete function
        sample.autodelete = True
        del sample
        self.assertTrue(not os.path.exists(self.filename+'.h5'))
        self.assertTrue(not os.path.exists(self.filename+'.xdmf'))

    def test_copy_and_compress(self):
        """ Copy the reference dataset and compress it """
        sample = SampleData.copy_sample(src_sample_file=self.reference_file,
                                        dst_sample_file=self.filename,
                                        overwrite=True,get_object=True,
                                        autodelete=True)
        # get filesizes
        original_filesize, _ = sample.get_file_disk_size(print_flag=False,
                                                      convert=False)
        original_size, _ = sample.get_node_disk_size('test_image_field',
                                                  print_flag=False,
                                                  convert=False)
        # Verify data content
        data_array = sample.get_node('test_array')
        self.assertTrue(np.all(self.data_array==data_array))
        # compress image data
        sample.set_chunkshape_and_compression(node='test_image_field',
                                              complib='zlib',complevel=1)
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
        self.assertGreater(original_filesize,new_filesize)
        # delete SampleData instance and assert files deletion
        del sample
        self.assertTrue(not os.path.exists(self.filename+'.h5'))
        self.assertTrue(not os.path.exists(self.filename+'.xdmf'))

    def test_derived_class(self):
        """ Test application specific data model specification through
            derived classes.
            Also test table functionalities.
        """
        derived_sample = Test_DerivedClass(filename=self.derived_filename,
                                           autodelete=False,
                                           overwrite_hdf5=True, verbose=False)
        # assert data model Nodes are contained in dataset
        self.assertTrue(derived_sample.__contains__('Image_data'))
        self.assertTrue(derived_sample.__contains__('grain_map'))
        self.assertTrue(derived_sample.__contains__('Grain_data'))
        self.assertTrue(derived_sample.__contains__('GrainDataTable'))
        self.assertTrue(derived_sample.__contains__('Crystal_data'))
        self.assertTrue(derived_sample.__contains__('lattice_params'))
        # assert data array nodes created are empty
        self.assertTrue(derived_sample._is_empty('grain_map'))
        self.assertTrue(derived_sample._is_empty('Image_data'))
        self.assertTrue(derived_sample._is_empty('GrainDataTable'))
        self.assertTrue(derived_sample._is_empty('lattice_params'))
        # get table node and assert description
        tab = derived_sample.get_node('GrainDataTable')
        self.assertEqual(Test_GrainData.columns,
                         tab.description._v_colobjects)
        # add columns to the table
        dtype = np.dtype([('name', np.str_, 16),('floats', np.float64, (2,))])
        derived_sample.add_tablecols('GrainDataTable', description=dtype)
        tab = derived_sample.get_node('GrainDataTable')
        self.assertTrue('name' in tab.colnames)
        self.assertTrue('floats' in tab.colnames)
        del derived_sample
        # reopen file and check that neqw columns have been added
        derived_sample = Test_DerivedClass(filename=self.derived_filename,
                                           autodelete=True,
                                           overwrite_hdf5=False, verbose=True)
        derived_sample.get_node_info('GrainDataTable')
        tab = derived_sample.get_node('GrainDataTable')
        self.assertTrue('name' in tab.colnames)
        self.assertTrue('floats' in tab.colnames)
        del derived_sample
        self.assertTrue(not os.path.exists(self.derived_filename+'.h5'))
        self.assertTrue(not os.path.exists(self.derived_filename+'.xdmf'))

    def test_BasicTools_binding(self):
        """Test BasicTools to SampleData to BasicTools."""
        # create mesh of triangles
        myMesh = UMCT.CreateSquare(dimensions=[3,3], ofTris=True)
        # get into a SampleData instance
        sample = SampleData(filename='square', verbose=False, autodelete=True)
        sample.add_mesh(mesh_object=myMesh, meshname='BT_mesh',indexname='BTM',
                          replace=True, extended_data=True)
        # get mesh object from SampleData file/instance
        myMesh2 = sample.get_mesh('BTM')
        # delete SampleData object and test values
        self.assertTrue(np.all(myMesh.nodes == myMesh2.nodes))
        connectivity = myMesh.elements['tri3'].connectivity
        connectivity2 = myMesh2.elements['tri3'].connectivity
        self.assertTrue(np.all(connectivity == connectivity2))
        elements_in_tag = myMesh.GetElementsInTag('ExteriorSurf')
        elements_in_tag2 = myMesh2.GetElementsInTag('ExteriorSurf')
        self.assertTrue(np.all(elements_in_tag == elements_in_tag2))
        del sample

    def test_mesh_from_image(self):
        """Test BasicTools to SDimage to SDmesh."""
        # 3D image parameters
        dimensions = [11,11,11]
        origin = [0.,0.,0.]
        spacing = [1.,1.,1.]
        # create BasicTools image object
        myMesh = ConstantRectilinearMesh(dim=3)
        myMesh.SetDimensions(dimensions)
        myMesh.SetOrigin(origin)
        myMesh.SetSpacing(spacing)
        # create data field
        data = np.zeros(shape=dimensions)
        data[:,3:8,3:8] = 1
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
        self.assertEqual(field1.shape,(11,11,11))
        field2 = sample.get_node('test_field_Tetra_mesh', as_numpy=True)
        self.assertEqual(field2.shape,(11*11*11,1))
        self.assertEqual(field1.ravel()[37], field2.ravel()[37])
        del sample

