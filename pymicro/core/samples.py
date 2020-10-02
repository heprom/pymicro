#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Demonstrator package :
        aimed at exploring the development of a software platform to handle
        multi-modal datasets describing mechanical test samples,
        comprised of CAO, tomographic and simulated datasets.

        The package capabilities include:
            - handling data storage and hierachy through the hdf5 file format

            - enabling a easy visualisation with paraview through the xdmf file
              format

            - prodive an easy interface to import data into from other mesh
              data file format (.vtk, .geof)

            - provide an easy interface to produce input files for the
              simulation softwares AMITEX_FFTP (.vtk files) and Zset
              (.geof files)

@author: amarano
"""

import warnings
import os
from lxml import etree
from lxml.builder import ElementMaker

import numpy as np
import tables as Tb
import h5py as h5

import pymicro.core.geof as geof

# usefull lists to parse keyword arguments
compression_keys = ['complib', 'complevel', 'shuffle', 'bitshuffle', 'checksum',
             'least_significant_digit', 'default_compression']


class SampleData:
    """ Base class to store multi-modal data for a material sample

    Attributes:
    -----------

        - filename (str)
                name of the file used to initiliaze class instance
                (.h5 or .xdmf file)

        - h5_file (str)
                name of .h5 file containing the heavy data associated
                to the sample

        - xdmf_file (str)
                name of .xdmf file containing the light metadata
                associated to the sample, and directly readable with
                the Paraview visualization software

        - h5_dataset (file object from PyTables package)
                file object associated to the heavy data structure contained in
                the h5_file

        - xdmf_tree (ElemenTree object from lxml.etree package)
                ElemenTree associated to the data nodes tree in the .xdmf file
                describing the light metadata assocaited to the h5_dataset

        - name (str)
                Name of the sample

        - description (str)
                description of the sample. Use it to store in the data
                structure significant mechanical information about the sample

    """

    def __init__(self,
                 filename,
                 sample_name='name_to_fill',
                 sample_description="""  """,
                 **keywords):
        """ DataSample initialization

        Create an data structure instance for a sample associated to the data
        file 'filename'.

            - if filename.h5 and filename.xdmf exist, the data structure is
              read from these file and stored into the SampleData instance

            - if the files do not exist, they are created and an empty
              SampleData is instantiated.

        Arguments:
            - filename (str)
                name of the .h5 or .xdmf file used to instantiate the class
                (existent or non existent)

            - sample_name (str)
                name of the mechanical sample described by this class
                instance

            - sample_description (str)
                description string for the mechanical sample. Use it to
                write significant mechanical information about the sample

        """

        # check if filename has a file extension
        if (filename.rfind('.') != -1):
            filename_tmp = filename[:filename.rfind('.')]
        else:
            filename_tmp = filename

        # initialize filenames
        self.h5_file = filename_tmp + '.h5'
        self.xdmf_file = filename_tmp + '.xdmf'

        # initiate verbosity flag
        if 'verbose' in keywords.keys():
            self._verbose = keywords['verbose']
        else:
            self._verbose = False

        # h5 table file object creation
        self.file_exist = False
        self.init_file_object()

        # initialize XML tree for xdmf format handling
        self.init_xml_tree()

        # initialize empty content index dictionnary
        self.init_content_index()

        # set compression options for HDF5 (passed in keywords args)
        if not(self.file_exist):
            Compression_keywords = {k:v for k,v in keywords.items() if k in \
                                    compression_keys}
            self.set_global_compression_opt(**Compression_keywords)

        return

    def __del__(self):
        """ DataSample destructor

        Deletes SampleData instance and:
              - closes h5_file --> writes data structure into the .h5 file
              - writes the .xdmf file

        All methods ensure that h5_dataset & xdmf_tree are consistent, so that
        the resulting .h5 and .xdmf files are consistent as well once class
        instance is deleted

        """
        print('\nDeleting DataSample object ')
        print('\n.... storing content index in "/Index" Group in file \n\t {} '.format(
                self.h5_file))
        self.add_attributes_from_dic(dic = self.content_index,
                                     node_path = '/Index')
        print('\n.... writing hdf5 file :\n\t {}'.format(self.h5_file))
        self.h5_dataset.close()
        print('\n.... writing xdmf file :\n\t {}'.format(self.xdmf_file))
        self.write_xdmf()
        return

    def init_file_object(self,
                         sample_name='',
                         sample_description=''):
        """ Initiate PyTable file object from .h5 file or create it if
            needed
        """

        try:
            print('\n-- Opening file "{}" '.format(self.h5_file))
            self.h5_dataset = Tb.File(self.h5_file, mode='r+')
            self.file_exist = True
        except IOError:
            print('\n-- File "{}" not found : file created'.format(self.h5_file))
            self.h5_dataset = Tb.File(self.h5_file, mode='a')

            # add sample name and description
            self.h5_dataset.root._v_attrs.sample_name = sample_name
            self.h5_dataset.root._v_attrs.description = sample_description

        return

    def init_xml_tree(self):
        """ Read xml tree structured data in .xdmf file or initiate one if
            needed

           The new xdmf trees are created with the minimal nodes for
           XDMF format:
               - root Xdmf node
               - Xinclude namespace in root node
               - Domaine node as children of Xdmf root node
        """

        try:
            self.xdmf_tree = etree.parse(self.xdmf_file)
        except OSError:
            # Non existent xdmf file.
            # A new .xdmf is created with the base node Xdmf and one Domain
            print('\n-- File "{}" not found : file created'.format(self.xdmf_file))

            # create root element of xdmf tree structure
            E = ElementMaker(namespace="http://www.w3.org/2003/XInclude",
                             nsmap={'xi': "http://www.w3.org/2003/XInclude"})
            root = E.root()
            root.tag = 'Xdmf'
            root.set("Version", "2.2")
            self.xdmf_tree = etree.ElementTree(root)

            # create element Domain as a children of root
            self.xdmf_tree.getroot().append(etree.Element("Domain"))

            # write file
            self.write_xdmf()
        return

    def print_xdmf(self):
        """ Print a readable version of xdmf_tree content"""

        print(etree.tostring(self.xdmf_tree, pretty_print=True,
                             encoding='unicode'))
        return

    def write_xdmf(self):
        """ Writes xdmf_tree in .xdmf file with suitable XML declaration """
        self.xdmf_tree.write(self.xdmf_file,
                             xml_declaration=True,
                             pretty_print=True,
                             doctype='<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd"[]>')

        # correct xml declaration to allow Paraview reader compatibility
        with open(self.xdmf_file, 'r') as f:
            lines = f.readlines()

        lines[0] = lines[0].replace("encoding='ASCII'", "")

        with open(self.xdmf_file, 'w') as f:
            f.writelines(lines)

        return

    def init_content_index(self):
        """
            Initialize SampleData dictionary attribute that stores the name and
            the location of dataset/group/attribute stored in the h5 file.
            The dic. is synchronized with the hdf5 Group '/Index' from Root.
            Its items are stored there as HDF5 attributes when the SampleData
            object is destroyed.

            This dic. is used by other methods to easily access/Add/remove/
            modify nodes et contents in the h5 file/Pytables tree structure
        """

        minimal_dic = self.return_minimal_content()
        self.minimal_content = {key:'' for key in minimal_dic}

        if (self.file_exist):
            self.content_index = self.get_dic_from_attributes(
                                             node_path='/Index')
        else:
            self.h5_dataset.create_group('/', name = 'Index')
            self.add_attributes_from_dic(
                    dic = self.minimal_content,
                    node_path = '/Index')
            self.content_index = self.get_dic_from_attributes(
                                             node_path='/Index')
        return


    def print_index(self):
        """ Allow to visualize a list of the datasets contained in the
            file and their status
        """
        print('\nDataset Content Index :')
        print('------------------------:')
        for key,value in self.content_index.items():
            print('\t Name : {} \t H5_Path : {} \t'.format(
                    key,value))
        return


    def set_global_compression_opt(self,**keywords):
        """
            Set compression options applied to all datasets in the h5 file
            as default compression setting

            Options are represented by a PyTables Filter Object, whose
            attributes can be passed as arguments for this method

            example
            --------
                 SampleData.set_global_compression_opt(complib='zlib')
        """

        # initialize general compression Filters object (from PyTables)
        # with no compression (default behavior)
        self.Filters = Tb.Filters(complevel=0)
        default = False
        # ------ check if default compression is required
        if 'default_compression' in keywords:
            # default compression options
            self.Filters.complib = 'zlib'
            self.Filters.complevel = 1
            self.Filters.shuffle = True
            # set flags on
            default = True

        # ------ read specified values of compression options
        for word in keywords:
            if (word == 'complib'):
                self.Filters.complib = keywords[word]
            elif (word == 'complevel'):
                self.Filters.complevel = keywords[word]
            elif (word == 'shuffle'):
                self.Filters.shuffle = keywords[word]
            elif (word == 'bitshuffle'):
                self.Filters.bitshuffle = keywords[word]
            elif (word == 'checksum'):
                self.Filters.fletcher32 = keywords[word]
            elif (word == 'least_significant_digit'):
                self.Filters.least_digit = keywords[word]

        # ----- message and Pytables Filter (comp option container) set up
        print('\n-- General Compression Options for datasets in {} \
              '.format(self.h5_file))

        if (self.Filters.complevel > 0):
            if default:
                print('\t Default Compression Parameters ')
            msg_list = str(self.Filters).strip('Filters(').strip(')').split()
            print(str(msg_list))
            for msg in msg_list:
                print('\t * {}'.format(msg))
        else:
            print('\t * No Compression')
        return

    def set_verbosity(self,verbosity=True):
        self._verbose = verbosity
        return

    def add_mesh_from_file(self,
                           meshfile,
                           meshname,
                           indexname='',
                           location='/',
                           description=' ',
                           **keywords):
        """ add geometry data and fields stored on a mesh from an external file

            Arguments:
                    - meshfile (str)
                         path to the file containing the mesh related heavy
                         data

                    - meshname (str)
                        name of the mesh used to generate hdf5 group and xdmf
                        node Name to store the mesh data

                    - location (str)
                        where to store the mesh inside the .h5 hierarchy in
                        h5_dataset structure.
                        defaut is root

                    - description (str)
                        mesh description to store mechanically significant
                        information about the data
                        Stored as a group attribute in the h5_dataset structure
                        and

                Mesh formats specific **keywords:
                    - .mat
                    -----------
                        --> h5 file with unknown hierarchy, must be passed as
                            __init__method keywords for fields location in
                            hdf5 structure

                                **keywords (optional):
                                    * matlab_variables :  list of (str)
                                              indicating the nodes storing the
                                              various fields to read
                                    * matlab_mesh_transpose : True if data need
                                        to be transposed

            Currently supported file formats :
                .geof
                .mat

        """
        print('Lauching mesh reader on  {}...'.format(meshfile))
        Reader = MeshReader(meshfile, **keywords)
        Mesh_object = Reader.mesh
        self.add_mesh(
                      mesh_object=Mesh_object,
                      meshname=meshname,
                      indexname=indexname,
                      location=location,
                      description=description)
        return

    def add_image_from_file(self,
                            imagefile,
                            imagename,
                            location='/',
                            description=' ',
                            indexname='',
                            **keywords):
        """ add fields stored on a 3D image from an external file

            Arguments:
                    - imagefile (str)
                         path to the file containing the image related heavy
                         data

                    - imagename (str)
                        name of the image used to generate hdf5 group and xdmf
                        node Name to store the image data

                    - location (str)
                        where to store the image inside the .h5 hierarchy in
                        h5_dataset structure.
                        defaut is root

                    - description (str)
                        image description to store mechanically significant
                        information about the data
                        Stored as a group attribute in the h5_dataset structure
                        and

                Handled image formats:
                    - .mat
                    -----------
                        --> h5 file with unknown hierarchy, must be passed as
                            __init__method keywords for fields location in
                            hdf5 structure

                                **keywords required :
                                    * matlab_variables :  list of (str)
                                                    indicating the
                                                    nodes storing the various
                                                    fields to read

                                    * matlab_mesh_transpose : True if data need
                                        to be transposed

        """

        print('Lauching image reader on  {}...'.format(imagefile))
        Reader = ImageReader(imagefile, **keywords)
        Image_object = Reader.image
        self.add_image(
                       image_object=Image_object,
                       imagename=imagename,
                       indexname=indexname,
                       location=location,
                       description=description)
        return

    def add_mesh(self,
                 mesh_object,
                 meshname,
                 indexname='',
                 location='/',
                 description=' '):
        """ add geometry data and fields stored on a mesh from a MeshObject

            Arguments:
                    - mesh_object (samples.MeshObject)
                         mesh object containing all mesh data (geometry,
                         topology fields)

                    - meshname (str)
                        name of the mesh used to generate hdf5 group and xdmf
                        node Name to store the mesh data

                    - location (str)
                        where to store the mesh inside the .h5 hierarchy in
                        h5_dataset structure.
                        defaut is root

                    - description (str)
                        mesh description to store mechanically significant
                        information about the data
                        Stored as a group attribute in the h5_dataset structure
                        and

        """

        if (indexname == ''):
            warn_msg = (' (add_mesh) indexname not provided, '
                        ' the meshname {} is used instead to fill'
                        'content index'.format(meshname))
            print(warn_msg)
            indexname = meshname
        mesh_path = os.path.join(location,meshname)
        # store mesh location in dataset index
        self.content_index[indexname] = mesh_path

        print('Creating hdf5 group {} in file {}'.format(mesh_path,
                  self.h5_file))
        mesh_group = self.add_group(
                                    path=location,
                                    groupname=meshname,
                                    indexname=indexname)

        # store mesh metadata as HDF5 attributes
        mesh_group._v_attrs.element_topology = mesh_object.element_topology[0]
        mesh_group._v_attrs.mesh_description = description
        mesh_group._v_attrs.field_dim = {}
        mesh_group._v_attrs.group_type = 'mesh'

        print('Creating Nodes data set in group {} in file {}'.format(
                mesh_path, self.h5_file))
        self.h5_dataset.create_carray(
                                where=mesh_path,
                                name='Nodes',
                                filters=self.Filters,
                                obj=mesh_object.nodes,
                                title=indexname+'_Nodes')

        # safety check
        if (len(mesh_object.element_topology) > 1):
            warnings.warn('''  number of element type found : {} \n
                          add_mesh_from_file current implementation works only
                          with meshes with only one element type
                          '''.format(len(mesh_object.element_topology)))

        print('Creating Elements data set in group {} in file {}'.format(
                location+'/'+meshname, self.h5_file))
        self.h5_dataset.create_carray(
                                where=mesh_path,
                                name='Elements',
                                filters=self.Filters,
                                obj=mesh_object.element_connectivity[0],
                                title=indexname+'_Elements')

        print('Updating xdmf tree...')
        mesh_xdmf = etree.Element(
                                  _tag='Grid',
                                  Name=meshname,
                                  GridType='Uniform')

        NElements = str(mesh_object.element_connectivity[0].shape[0])
        Dim = str(mesh_object.element_connectivity[0].shape).strip(
                  '(').strip(')')
        Dim = Dim.replace(',', ' ')

        topology_xdmf = etree.Element(_tag='Topology',
                                      TopologyType=mesh_object.
                                      element_topology[0][0],
                                      NumberOfElements=NElements)

        topology_data = etree.Element(_tag='DataItem',
                                      Format='HDF',
                                      Dimensions=Dim,
                                      NumberType='Int',
                                      Precision='64')

        topology_data.text = self.h5_file + ':' + mesh_path + '/Elements'
        # Add node DataItem as children of node Topology
        topology_xdmf.append(topology_data)

        # create Geometry element
        geometry_xdmf = etree.Element(_tag='Geometry',
                                      Type='XYZ')

        Dim = str(mesh_object.nodes.shape).strip('(').strip(')').replace(
                  ',', ' ')
        geometry_data = etree.Element(_tag='DataItem',
                                      Format='HDF',
                                      Dimensions=Dim,
                                      NumberType='Float',
                                      Precision='64')

        geometry_data.text = self.h5_file + ':' + mesh_path + '/Nodes'

        # Add node DataItem as children of node Geometry
        geometry_xdmf.append(geometry_data)

        # Add Geometry and Topology as childrens of Grid
        mesh_xdmf.append(topology_xdmf)
        mesh_xdmf.append(geometry_xdmf)

        # Add Grid to node Domain
        self.xdmf_tree.getroot()[0].append(mesh_xdmf)

        # Get xdmf elements paths as attributes
        #   mesh group
        el_path = self.xdmf_tree.getelementpath(mesh_xdmf)
        mesh_group._v_attrs.xdmf_path = el_path
        #   topology element
        el_path = self.xdmf_tree.getelementpath(topology_xdmf)
        mesh_group._v_attrs.xdmf_topology_path = el_path
        mesh_group.Elements._v_attrs.xdmf_path = el_path
        #   geometry  element
        el_path = self.xdmf_tree.getelementpath(geometry_xdmf)
        mesh_group._v_attrs.xdmf_geometry_path = el_path
        mesh_group.Nodes._v_attrs.xdmf_path = el_path

        # Add mesh fields if some are stored
        for Mesh_fields in mesh_object.fields:
            self.add_field_to_mesh(mesh_location=location,
                                   meshname=meshname,
                                   field=list(Mesh_fields.values())[0],
                                   fieldname=list(Mesh_fields.keys())[0],
                                   field_indexname='')

        return

    def add_image(self,
                  image_object,
                  imagename,
                  indexname='',
                  location='/',
                  description=' '):
        """ add geometry data and fields stored on a mesh from a MeshObject

            Arguments:
                    - image_object (samples.MeshObject)
                         ImageObject containing the data stored in the image

                    - imagename (str)
                        name of the image used to generate hdf5 group and xdmf
                        node Name to store the associated data

                    - location (str)
                        where to store the image inside the .h5 hierarchy in
                        h5_dataset structure (hdf5 group).
                        defaut is root

                    - description (str)
                        image description to store mechanically significant
                        information about the data
                        Stored as a group attribute in the h5_dataset structure
                        and

        """

        if (indexname == ''):
            warn_msg = (' (add_mesh_from_field) indexname not provided, '
                        ' the meshname {} is used instead to fill'
                        'content index'.format(imagename))
            print(warn_msg)
            indexname = imagename
        image_path = os.path.join(location,imagename)
        # store image location in dataset index
        self.content_index[indexname] = image_path

        # create group in the h5 structure for the mesh
        print('Creating hdf5 group {} in file {}'.format(
              image_path, self.h5_file))
        image_group = self.add_group(
                                    path=location,
                                    groupname=imagename,
                                    indexname=indexname)

        # store image metadata as HDF5 attributes
        image_group._v_attrs.element_topology = '3d_image'
        image_group._v_attrs.dimension = image_object.dimension
        image_group._v_attrs.image_description = description
        image_group._v_attrs.field_dim = {}
        image_group._v_attrs.group_type = '3Dimage'

        print('Updating xdmf tree...')
        image_xdmf = etree.Element(_tag='Grid',
                                   Name=imagename,
                                   GridType='Uniform')

        # create Topology element
        # Add 1 to each dimension of the image
        #    --> image cell data is stored (i.e. value at voxel centers)
        #        but xdmf intends 3DCoRectMesh dimension as number of points
        #        which is equal to voxel number + 1 in each direction
        Im_dim = (image_object.dimension +
                  np.ones(len(image_object.dimension), dtype='int64'))
        Dim = str(Im_dim).strip('(').strip(')')
        Dim = Dim.strip('[').strip(']')
        Dim = Dim.replace(',', ' ')

        topology_xdmf = etree.Element(_tag='Topology',
                                      TopologyType='3DCoRectMesh',
                                      Dimensions=Dim)

        geometry_xdmf = etree.Element(_tag='Geometry',
                                      Type='ORIGIN_DXDYDZ')

        origin_data = etree.Element(_tag='DataItem',
                                    Format='XML',
                                    Dimensions='3')
        Origin = str(image_object.origin).strip('[').strip(']').replace(
                     ',', ' ')
        origin_data.text = Origin

        spacing_data = etree.Element(_tag='DataItem',
                                     Format='XML',
                                     Dimensions='3')
        Spacing = str(image_object.spacing).strip('[').strip(']').replace(
                      ',', ' ')
        spacing_data.text = Spacing

        # Add nodes DataItem as childrens of node Geometry
        geometry_xdmf.append(origin_data)
        geometry_xdmf.append(spacing_data)

        # Add Geometry and Topology as childrens of Grid
        image_xdmf.append(topology_xdmf)
        image_xdmf.append(geometry_xdmf)

        # Add Grid to node Domain
        self.xdmf_tree.getroot()[0].append(image_xdmf)

        # Get xdmf elements paths as HDF5 attributes
        #     image group
        el_path = self.xdmf_tree.getelementpath(image_xdmf)
        image_group._v_attrs.xdmf_path = el_path
        #     topology group
        el_path = self.xdmf_tree.getelementpath(topology_xdmf)
        image_group._v_attrs.xdmf_topology_path = el_path
        #     geometry group
        el_path = self.xdmf_tree.getelementpath(geometry_xdmf)
        image_group._v_attrs.xdmf_geometry_path = el_path

        # Add mesh fields if some are stored
        for Image_fields in image_object.fields:
            self.add_field_to_image(image_location=location,
                                    imagename=imagename,
                                    field=list(Image_fields.values())[0],
                                    fieldname=list(Image_fields.keys())[0],
                                    field_indexname='')
        return

    def add_field_to_mesh(self,
                          mesh_location,
                          meshname,
                          field,
                          fieldname,
                          field_indexname=''):
        """ Add field nodal values associated to a mesh in the data structure

            New fields must be added to already existing mesh

            Arguments:
                - mesh_location (str)
                    location of the hdf5 group containing the mesh on which the
                    field is defined

                - meshname (str)
                    name of the hdf5 group containing the mesh

                - field  (np.array shape : Nnodes x Ncomponents)
                    np.array containing the nodal values of the field. First
                    dimension length must match first dimension of nodes array
                    in the mesh

                - fieldname (str)
                    name of the field
        """

        if (field_indexname == ''):
            warn_msg = (' (add_field_to_mesh) indexname not provided, '
                        ' the fieldname {} is used instead to fill'
                        'content index'.format(fieldname))
            print(warn_msg)
            field_indexname = fieldname


        mesh_path = os.path.join(mesh_location,meshname)
        field_path = os.path.join(mesh_path,fieldname)
        # store mesh location in dataset index
        self.content_index[field_indexname] = field_path

        Field_type = { 1:'Scalar',
                       2:'Vector',
                       3:'Vector',
                       6:'Tensor6',
                       9:'Tensor'
                     }

        # if field has only 1 dimension, reshape it to a 2dim array
        if (field.ndim < 2):
            field = field.reshape([field.shape[0],1])

        # get mesh node in hdf5 data structure
        mesh_group = self.h5_dataset.get_node(mesh_path)

        # get corresponding grid node in xdmf tree
        for grid in self.xdmf_tree.getroot()[0]:
            if (grid.attrib['Name'] == meshname):
                mesh_xdmf_grid = grid

        # verify that number of nodes and number of nodal values match
        if (mesh_group.Nodes.shape[0] != field.shape[0]):
            raise ValueError(''' The nodal value array given for the field
                              contains {} nodal values, which do not match the
                              number of nodes in the mesh object ({} nodes)
                             '''.format(field.shape[0],
                                 mesh_group.Nodes.shape[0] ))

        # Add field data to the hdf5 data structure
        print('Creating field "{}" data set in group {} in file {}'.format(
                fieldname,mesh_location + meshname,self.h5_file))
        field_node = self.h5_dataset.create_carray(
                                        where= mesh_path,
                                        name = fieldname,
                                        filters= self.Filters,
                                        obj= field)

        # determine AttributeType
        for Ftype in Field_type:
            if (Ftype==field.shape[1]):
                Attribute_type= Field_type[Ftype]
                break
            else:
                Attribute_type = 'Matrix'

        # Store field attributes : nothing for now
        mesh_group._v_attrs.field_dim.update( {fieldname:Attribute_type} )

        # Update xdmf tree
        # 1- create Attribute element
        print('Updating xdmf tree...')
        Attribute_xdmf = etree.Element(_tag='Attribute',
                                       Name=fieldname,
                                       AttributeType=Attribute_type,
                                       Center='Node')

        Dim = str(field.shape).strip('(').strip(')')
        Dim = Dim.replace(',',' ')
        Attribute_data = etree.Element(_tag = 'DataItem',
                                       Format = 'HDF',
                                       Dimensions = Dim,
                                       NumberType = 'Float',
                                       Precision = '64')
        Attribute_data.text = self.h5_file + ':' + field_path

        # add data item to attribute
        Attribute_xdmf.append(Attribute_data)

        # add attribute to Grid
        mesh_xdmf_grid.append(Attribute_xdmf)

        # add element path to h5 structure
        el_path = self.xdmf_tree.getelementpath(Attribute_xdmf)
        field_node._v_attrs.xdmf_path = el_path
        return

    def add_field_to_image(self,
                          image_location,
                          imagename,
                          field,
                          fieldname,
                          field_indexname=''):
        """ Add field nodal values associated to a 3D image in the data structure

            New fields must be added to already existing image

            Arguments:
                - image_location (str)
                    location of the hdf5 group containing the image on which the
                    field is defined

                - imagename (str)
                    name of the hdf5 group containing the image

                - field  (np.array shape : Nnodes x Ncomponents)
                    np.array containing the values of the field at voxels centers.
                    Dimension must match image dimension

                - fieldname (str)
                    name of the field
        """

        if (field_indexname == ''):
            warn_msg = (' (add_field_to_image) indexname not provided, '
                        ' the fieldname {} is used instead to fill'
                        'content index'.format(fieldname))
            print(warn_msg)
            field_indexname = fieldname

        image_path = os.path.join(image_location,imagename)
        field_path = os.path.join(image_path,fieldname)
        # store mesh location in dataset index
        self.content_index[field_indexname] = field_path

        Field_type = { 1:'Scalar',
                       2:'Vector',
                       3:'Vector',
                       6:'Tensor6',
                       9:'Tensor'
                     }

        # get image node in hdf5 data structure
        image_group = self.h5_dataset.get_node(image_path)

        # get corresponding grid node in xdmf tree
        for grid in self.xdmf_tree.getroot()[0]:
            if (grid.attrib['Name'] == imagename):
                image_xdmf_grid = grid

        # verify that number of nodes and number of nodal values match
        if (image_group._v_attrs.dimension != field.shape):
            raise ValueError(''' The array given for the field
                              contains {} cell values, which do not match the
                              dimensions of the image object ({} voxels)
                             '''.format(field.shape[0],
                                 image_group._v_attrs.dimension ))

        # Add field data to the hdf5 data structure
        print('Creating field "{}" data set in group {} in file {}'.format(
                fieldname,image_location + imagename,self.h5_file))
        field_node = self.h5_dataset.create_carray(
                                         where= image_path,
                                         name = fieldname,
                                         filters= self.Filters,
                                         obj= field)

        for Ftype in Field_type:
            if (Ftype==field.shape[1]):
                Attribute_type= Field_type[Ftype]
                break
            else:
                Attribute_type = 'Matrix'

        # Store field attributes : nothing for now
        image_group._v_attrs.field_dim.update( {fieldname:Attribute_type} )

        # Update xdmf tree
        # 1- create Attribute element
        print('Updating xdmf tree...')
        Attribute_xdmf = etree.Element(_tag='Attribute',
                                       Name=fieldname,
                                       AttributeType=Attribute_type,
                                       Center='Cell')

        Dim = str(field.shape).strip('(').strip(')')
        Dim = Dim.replace(',',' ')

        if (np.issubdtype(field.dtype, np.floating) ):
            NumberType = 'Float'
            if (str(field.dtype) == 'float'):
                Precision = '32'
            else:
                Precision = '64'
        elif (np.issubdtype(field.dtype, np.integer) ):
            NumberType = 'Int'
            Precision = str(field.dtype).strip('int')


        Attribute_data = etree.Element(_tag = 'DataItem',
                                       Format = 'HDF',
                                       Dimensions = Dim,
                                       NumberType = NumberType,
                                       Precision = Precision)
        Attribute_data.text = self.h5_file + ':' + field_path

        # add data item to attribute
        Attribute_xdmf.append(Attribute_data)

        # add attribute to Grid
        image_xdmf_grid.append(Attribute_xdmf)

        # add element path to h5 structure
        el_path = self.xdmf_tree.getelementpath(Attribute_xdmf)
        field_node._v_attrs.xdmf_path = el_path
        return

    def add_group(self,
                  path,
                  groupname,
                  indexname=''):
        """ Create a (hdf5) group at desired location in the data format"""
        if (indexname == ''):
            warn_msg = (' (add_group) indexname not provided, '
                        ' the groupname {} is used instead to fill'
                        'content index'.format(groupname))
            print(warn_msg)
            indexname = groupname
        self.content_index[indexname] = path+groupname
        Group = self.h5_dataset.create_group(
                                   where=path,
                                   name=groupname,
                                   title=indexname)
        return Group


    def get_indexname_from_path(self,
                                node_path):
        """Return the key of the given node_path in content index """

        key = ''
        for k in self.content_index.keys():
            if (self.content_index[k] == node_path):
                key = k
                break
        if not (key == ''):
            return key
        else:
            msg = 'No node with path {} referenced in content index'.format(
                       node_path)
            raise ValueError(msg)

    def get_path_with_indexname(self,indexname):
        if indexname in self.content_index.keys():
            return self.content_index[indexname]
        else:
            raise ValueError('Index contains no item named {}'.format(
                    indexname))
            return

    def get_node(self,
                 indexname=None,
                 node_path=None):
        """ get a Node object from h5 data tree from indexname or path """

        if ((indexname is not None) and (node_path is None)):
            node_path = self.content_index[indexname]
            print(node_path)
            return self.h5_dataset.get_node(node_path)
        if node_path is None:
            warnings.warn('\n (get_node) neither indexname nor node_path'
                  ' passed, node return aborted')
            return

    def get_node_info(self,
                 indexname=None,
                 node_path=None):
        """ get information on a node in hdf5 tree from indexname or path """

        Node = self.get_node(indexname=indexname,node_path=node_path)
        print('\n Information on Node {}'.format(Node._v_pathname))
        print('----------------')
        print(' -- node path : {}'.format(Node._v_pathname))
        print(' -- node name : {}'.format(Node._v_name))
        print(' -- node attributes : ')
        for attr in Node._v_attrs._f_list():
            print('\t {}'.format(attr))
        print(' -- content : {}'.format(str(Node)))
        print('----------------')
        return

    def _remove_from_index(self,
                          node_path):
        """Remove a hdf5 node from content index dictionary"""

        try:
            key = self.get_indexname_from_path(node_path)
            removed_path = self.content_index.pop(key)
            print('\nitem {} : {} removed from context index dictionary'.format(
                    key,removed_path))
        except ValueError:
            print('\nnode {} not found in content index values for removal'
                  ''.format(node_path))
        return

    def remove_node(self,
                    indexname=None,
                    node_path=None,
                    recursive=False):
        """ Remove a node from the Sample h5 data structure and xdmf tree

            Node to remove can be passed by their indexname, or their path in
            the hdf5 tree structure

            Arguments:
                - indexname : key containing node path in content_index dic
                - node_path : location of the node to remove in the h5
                                   data structure
                - recursive : boolean control allowing to force recursive
                              suppression if the node is a Group
        """

        if ((indexname is not None) and (node_path is None)):
            pass
            node_path = self.content_index[indexname]

        if node_path is None:
            warnings.warn('\n (remove_node) neither indexname nor node_path'
                  ' passed, node removal aborted')
            return

        Node = self.h5_dataset.get_node(node_path)
        isGroup = False
        remove_flag = False

        # determine if node is an HDF5 Group
        if (Node._v_attrs.CLASS == 'GROUP'):
            isGroup = True

        if (isGroup):
            print('\nWARGNING : node {} is a hdf5 group with  {} children(s)'
                  ' :'.format(node_path,Node._v_nchildren) )
            count = 1
            for child in Node._v_children:
                print('\t child {}:'.format(count),Node[child])
                count = count+1
            if not recursive:
                print('\nDeleting the group will delete children data.',
                         'Are you sure you want to remove it ? ', end='')
                text_input = input('(y/n) ')

                if (text_input == 'y'):
                    remove_flag = True
                elif (text_input == 'n'):
                    remove_flag = False
                else:
                    print('''\t Unknown response, y or n expected. Group has not been
                                removed.
                          ''')
                    return
            else:
                remove_flag = True

        # remove node
        if (remove_flag or not(isGroup)):

            # remove node in xdmf tree
            xdmf_path = Node._v_attrs.xdmf_path
            group_xdmf = self.xdmf_tree.find(xdmf_path)
            print('\nRemoving  node {} in xdmf tree....'.format(xdmf_path))
            parent = group_xdmf.getparent()
            parent.remove(group_xdmf)

            print('\nRemoving  node {} in h5 structure ....'.format(node_path))
            if (isGroup):
                for child in Node._v_children:
                    remove_path = Node._v_children[child]._v_pathname
                    self._remove_from_index(node_path=remove_path)
                print('\nRemoving  node {} in content index....'.format(
                        Node._v_pathname))
                self._remove_from_index(node_path=Node._v_pathname)
                Node._f_remove(recursive=True)

            else:
                self._remove_from_index(node_path=Node._v_pathname)
                Node.remove()
            print('\nNode sucessfully removed, '
                  'new data structure is:\n')
            print(str(self.h5_dataset))

        return

    def return_minimal_content(self):
        """
            Specify the minimal contents of the hdf5 (Node names, organization)
            in the form of a dictionary {content:Location}

            This method is central to define derived classes from SampleData
            that are associated with a minimal content and organization
            for the hdf5 file storing the data.

            This dictionary is used to assess the state of completion of the
            datafile, and serves as indication to inform the minimal content
            that is required to have a usefull dataset for the physical purpose
            for which the subclass is designed

        """
        minimal_content_index_dic = ['test_key1','test_key2']
        return minimal_content_index_dic


    def add_attributes_from_dic(self, dic, node_path):
        """
            private method used to store a Python dictionary as a set of
            HDF5 Attributes compatible with Pytables AttributeSet class

            Arguments
            ---------
                * dic : Python dictionary to store in HDF5 file
                * node_path : path of the node where to store the attributes
                              in the hdf5 file

            Warnings
            --------
                * Attributes are not meant for heavy data storage. Keep the
                  content of dic to strings and small numerical arrays
                *
        """
        Node = self.h5_dataset.get_node(node_path)

        for key,value in dic.items():
            Node._v_attrs[key] = value
        return

    def get_dic_from_attributes(self, node_path):
        """
            private method used to store a Python dictionary as a set of
            HDF5 Attributes compatible with Pytables AttributeSet class

            Arguments
            ---------
                * dic : Python dictionary to store in HDF5 file
                * node_path : path of the node where to store the attributes
                              in the hdf5 file

            Warnings
            --------
                * Attributes are not meant for heavy data storage. Keep the
                  content of dic to strings and small numerical arrays
                *
        """
        Node = self.h5_dataset.get_node(node_path)
        dic = {}
        for key in Node._v_attrs._f_list():
            dic[key] = Node._v_attrs[key]
        return dic

class MeshObject():
    """ Base class to store mesh data

        Attributes:
            - nodes (np.Array float64, shape (Nnodes, mesh_dim= 1 or 2 or 3) )
                coordinates of mesh nodes

            - element_topology (list of [str])
                List of element types in the mesh.
                Each element is another list containing
                  [(str) 'element_topology', element_degree (int)]
                Element type is consistent with XDMF format

            - element_ID (list of [ np.array int64 shape (Nelem, 1) ])
                List of elements ID_numbers arrays in the mesh.
                Used for element sets definition
                One array for each element type in the mesh

            - element_connectivity (list of [np.array int64 shape (Nelem, NVertex)])
                List of numpy Integer arrays of element node connectivity list.
                      1 row = 1 element of element If_numbers forming the element
                One array for each element type in the mesh

            - fields (list)
                contains a list of fields defined on the mesh. Each field is
                represented by a dictionnary : {field_name:field_values}
                field_values is a np.array containing the field nodel values
                corresponding to the nodes in MeshObject.nodes
    """

    def __init__(self,
                 **keywords):
        """ init Mesh Object with provided data or empty attributes

            Attributes:
                - grid_type (str)
                    Type of grid. Supported : 'Uniform', 'Collection'

                - topology_type (str)
                    type of grid elements (only for Uniform grids)

                - **keywords
                    Not implemented yet. Intended to pass on Nodes and
                    Element arrays, Node and Element Sets Dictionnaries etc...

        """

        # mesh heavy data defaut emtpy instantiation
        self.nodes = np.array([], dtype='float64')
        self.element_topology = []
        self.element_Id = []
        self.element_connectivity = []

        # mesh sets defaut emtpy instantiation
        self.node_sets = {}

        # mesh fields defaut empty instantiation
        self.fields = []

        return

    def add_nodes(self,
                     Nodes):
        """ Adds Nodes coordinates to the mesh

            Arguments:
                - Nodes (np.array - double - shape : NNodes, Dim)
                    coordinates arrays of the Nodes to add to the MeshObject
                    one line is one vertex
        """

        if (len(self.nodes)==0):
            self.nodes = Nodes
            print('\t \t  Added {} nodes to the mesh'.format(len(Nodes)))
            print('\t \t \t First node  : {}'.format(
                    Nodes[0,:]))
            print('\t \t \t Last node  : {}'.format(
                    Nodes[-1,:]))
        else:
            if (self.nodes.shape[1] == Nodes.shape[1]):
                self.nodes =  np.append(self.nodes,Nodes)
                print('\t \t Added {} nodes to the mesh'.format(len(Nodes)))
                print('\t \t \t First node  : {}'.format(
                        Nodes[0,:]))
                print('\t \t \t Last node   : {}'.format(
                        Nodes[-1,:]))
            else:
                raise ValueError(''' shape of added Nodes array {} do not
                                 match MeshObject.nodes shape {}
                                 '''.format(Nodes.shape,self.nodes.shape))

        return

    def add_elements(self,
                     Elements_topology,
                     Elements_Id,
                     Elements_connectivity):
        """ Add a set of element to the MeshObject

            Arguments:
                    - Elements_type list : [(str) , int]
                        List describing the type of elements in the element set
                        added to the MeshObject, according to XDMF format
                        elements naming conventions
                        string : element topology, int : element degree

                    - Elements_Id (np.array - int64 - shape : Nelements)
                        Integer array containing the Id numbe of the elements in
                        the element set to add to the MeshObject

                    - Elements_connectivity (np.array - int64 -
                                              shape : Nelements, NNodes)
                        Integer array containing the ocnnectivity matrix for the
                        element set to add to the MeshObject
                        Each line contains the Nodes Id forming the element

            NOTE : the correspondance between Id and connectivit arrays is
                   line by line. Elements_Id[K] is the ID of the element
                   composed of the Nodes Elements_connectivity[K,:]

        """
        self.element_topology.append(Elements_topology)
        self.element_Id.append(Elements_Id)
        self.element_connectivity.append(Elements_connectivity)
        print('\t \t Added {} {} elements of degree {} to the mesh'.format(
                Elements_Id.size,Elements_topology[0],Elements_topology[1]))
        print('\t \t \t First element nodes  : {}'.format(
                Elements_connectivity[0,:]))
        print('\t \t \t Last  element nodes  : {}'.format(
                Elements_connectivity[-1,:]))
        return

    def add_sets(self,
                 Node_sets = {}):
        """ Adds nodes/elements sets to the MeshObject

            Arguments:
                - Node_sets (dict :  'nset name':np.array.int64)
                  dictionnary linking node set names (keys) to integer arrays
                  of the node set verteces ID numbers (values)
                  default : empty dict

        """
        self.node_sets.update(Node_sets)
        print('\t Added the foolowing node sets to the mesh: ')
        for nset in Node_sets:
            print('\t\t - {}'.format(nset))
        return

    def add_field(self,
                  Field,
                  Field_name):
        """ add field nodal values to the MeshObject

            Arguments:
                - Field (np.array shape : Nnodes x Ncomponents)
                  ex: 3D vector field shape is Nnodes x 3
                  Array containing the nodal values of the field, defined on the
                  nodes coordinates contained in self.nodes

                - Field_name (str)
                    Name of the field

            Fields are added to the mesh object as a dict :
                field_name:field_values
        """

        # verify that number of field nodes and self.nodes are coherent
        if (Field.shape[0] != self.nodes.shape[0]):
            raise ValueError(''' The nodal value array given for the field
                              contains {} nodal values, which do not match the
                              number of nodes in the mesh object ({} nodes)
                             '''.format(Field.shape[0],self.nodes.shape[0]))


        # in case of a two dimensional array : reshape to 3dim array with
        # singleton last dimension
        if (Field.ndim < 2):
            Field = Field.reshape([Field.shape[0],1])

        self.fields.append({Field_name:Field})
        print('\t field {} of dimension {} added to the mesh'.format(Field_name,
              Field.shape))
        return

class ImageObject():
    """ Base class to store 3D image data

        Attributes:
            - dimension [Nx,Ny,Nz] int64
                number of voxels for each dimension of the 3D image

            - origin [Ox, Oy, Oz] np.double
                coordinates of the origin (center of the [0,0,0] voxel)

            - spacing [Dx, Dy, Dz] np.double
                voxel size in each dimension

                 DEFAUT DIMENSIONS : voxels size [1,1,1] and origin at
                 [0.5, 0.5, 0.5] --> the first node of the first voxel is
                 the point with coordinates [0 0 0]

            - fields (list)
                contains a list of fields defined on the mesh. Each field is
                represented by a dictionnary : {field_name:field_values}
                field_values is a np.array containing the field nodel values
                corresponding to the nodes in MeshObject.nodes

    """

    def __init__(self,
                 dimension = [1,1,1],
                 origin = np.array([0., 0., 0.], dtype='double'),
                 spacing = np.array([1, 1, 1], dtype='double'),
                 **keywords):
        """ init Image Object with provided data or empty attributes

            Attributes:
                - grid_type (str)
                    Type of grid. Supported : 'Uniform', 'Collection'

                - topology_type (str)
                    type of grid elements (only for Uniform grids)

                - **keywords
                    Not implemented yet. Intended to pass on Nodes and
                    Element arrays, Node and Element Sets Dictionnaries etc...

        """

        # mesh heavy data defaut emtpy instantiation
        self.dimension = dimension
        self.origin = origin
        self.spacing = spacing

        # mesh fields defaut empty instantiation
        self.fields = []
        return

    def add_field(self,
                  Field,
                  Field_name):
        """ add field nodal values to the MeshObject

            Arguments:
                - Field (np.array shape : Nnodes x Ncomponents)
                  ex: 3D vector field shape is Nnodes x 3
                  Array containing the nodal values of the field, defined on the
                  nodes coordinates contained in self.nodes

                - Field_name (str)
                    Name of the field

            Fields are added to the mesh object as a dict :
                field_name:field_values
        """

        # verify that number of field nodes and self.nodes are coherent
        if (Field.shape != self.dimension):
            raise ValueError(''' The array given for the field shape is {}
                               which does not match ImageObject dimension : {}
                             '''.format(Field.shape,self.dimension))

        self.fields.append({Field_name:Field})
        print('\t field {} of dimension {} added to the 3D image'.format(Field_name,
              Field.shape))
        return



class MeshReader():
    """ Class to read mesh data from a file into a mesh object

        Contains a mesh object following hdf5/xdfm format conventions

        Currently supported file formats
            .geof

        Attributes:
            - filename (str)
                name/path to file containing the mesh data

            - Geof_to_Xdmf_topology (dictionnary)
                translation table between element names in .geof and xdmf format

            - Mesh (MeshObject)
                mesh object class instance containing mesh data read from file

    """

    # dict for Geof to xdmf element topology naming convention
    Geof_to_Xdmf_topology = {
                              'c2d3r':['Triangle',1],
                              'c2d3':['Triangle',1],
                              'c2d4r':['Quadrilateral',1],
                              'c2d4':['Quadrilateral',1],
                              'c2d6r':['Tri_6',2],
                              'c2d6':['Tri_6',2],
                              'c2d8r':['Quad_8',2],
                              'c2d8':['Quad_8',2],
                              's3d3r':['Triangle',1],
                              's3d3':['Triangle',1],
                              's2d4r':['Quadrilateral',1],
                              's2d4':['Quadrilateral',1],
                              's2d6r':['Tri_6',2],
                              's2d6':['Tri_6',2],
                              's2d8r':['Quad_8',2],
                              's2d8':['Quad_8',2],
                              'c3d4r':['Tetrahedron',1],
                              'c3d4':['Tetrahedron',1],
                              'c3d10r':['Tetrahedron',2],
                              'c3d10':['Tetrahedron',2],
                              'c3d6r':['Wedge',1],
                              'c3d6':['Wedge',1],
                              'c3d15r':['Wedge_15',1],
                              'c3d15':['Wedge_15',1],
                              'c3d8r':['Hexahedron',1],
                              'c3d8':['Hexahedron',1],
                              'c3d20r':['Hex_20',2],
                              'c3d20':['Hex_20',2]
                        }

    def __init__(self,
                 mesh_filename,
                 ** keywords
                 ):
        """ init MeshReader with filename and reading options

            Attributes:
                - mesh_filename (str)
                    path to the mesh file to be read by the MeshReader. Extension
                    must be a known format

                - keywords
                    allow to pass options required to read data for some mesh
                    formats. See below

                Handled mesh formats:
                    - .geof
                    -----------
                        --> only geometrical mesh description data
                        --> all information is in file, no keywords required

                    - .mat
                    -----------
                        --> geometrical and/or field data
                        --> h5 file with unknown hierarchy, must be passed as
                            __init__method keywords for nodes, elements and
                            fields location in hdf5 structure

                                Keywords required :
                                    * matlab_mesh_nodes : Hdf5 node storing the mesh
                                                  nodes coordinates (str)

                                    * matlab_mesh_elements : Hdf5 node storing the mesh
                                                    element connectivity array
                                                    (str)

                                    * matlab_mesh_element_type (list)
                                            two elements [0] string indicating
                                            elements topology
                                            [1] int indicating elements degree
                                            See Geof_to_Xdmf_topology Dict in
                                            the class for a comprehensive list

                                    * matlab_variables :  list of (str) indicating the
                                                   nodes storing the various
                                                   fields to read

                                    * matlab_mesh_transpose : True if data need
                                        to be transposed
                    - .vtk
                    ----------

        """
        self.filename = mesh_filename
        self.get_mesh_type()

        # empty Matlab read keywords init
        self.matlab_mesh_nodes = ''
        self.matlab_mesh_elements = ''
        self.matlab_mesh_element_type = ''
        self.matlab_variables = []
        self.matlab_mesh_transpose= False

        # instantiate mesh object
        self.mesh = MeshObject()

        # get mesh reading keywords
        for words in keywords:
            if (words == 'matlab_mesh_nodes'):
                self.matlab_mesh_nodes = keywords[words]
            elif (words == 'matlab_mesh_elements'):
                self.matlab_mesh_elements = keywords[words]
            elif (words == 'matlab_variables'):
                self.matlab_variables = keywords[words]
            elif (words == 'matlab_mesh_element_type'):
                self.matlab_mesh_element_type = keywords[words]
            elif (words == 'matlab_mesh_transpose'):
                self.matlab_mesh_transpose = True

        # get mesh data
        self.read_mesh()

        return
    #############################

    def get_mesh_type(self):
        """ Find out mesh file format type from mesh file extension"""

        ext = self.filename[self.filename.rfind('.'):]

        print(' MeshReader : reading extension of file {} to find file format'.format(self.filename))
        if (ext == '.mat'):
            self.file_type = 'matlab'
        elif (ext == '.geof'):
            self.file_type = 'geof'
        elif (ext == '.vtk'):
            self.file_type = 'vtk'
        else:
            print(' MeshReader : File format not handled for file {}'.format(self.filename))
        print(' MeshReader : file format --> {}'.format(self.file_type))
        return
    #############################

    def read_mesh(self):
        """ Call the appropriate read method for the considered mesh format """

        if (self.file_type == 'matlab'):
            self.read_mesh_matlab()
        elif (self.file_type == 'geof'):
            self.read_geof_mesh()
        elif (self.file_type == 'vtk'):
            warnings.warn(" vtk reading not implemented yet for MeshReader, no data has been read ")
        return

    def read_mesh_matlab(self):
        """ Read .mat formatted mesh data """

        # verification of Matlab read keywords
        if (self.matlab_mesh_nodes == ''):
            raise ValueError(''' Matlab mesh nodes hdf5 location not specified,
                                 got empty location ''')

        if (self.matlab_mesh_elements == ''):
            raise ValueError(''' Matlab mesh elements hdf5 location not
                                 specified, got empty location ''')

        if (self.matlab_mesh_element_type == []):
            raise ValueError(''' Matlab mesh element type not specified,
                                 got empty list''')

        print(' MeshReader : reading mesh in {} ....'.format(self.filename))
        # open .mat file as hdf5 file
        matfile_data = h5.File(self.filename, 'r')

        # read mesh nodes
        print('\n \t reading mesh nodes in h5 node {} ....'.format(
                self.matlab_mesh_nodes))
        Nodes = np.array(matfile_data[self.matlab_mesh_nodes][()]).astype('double')
        if (self.matlab_mesh_transpose):
            Nodes = Nodes.transpose()
        self.mesh.add_nodes(Nodes)

        # read mesh elements
        print('\n \t reading mesh elements in h5 node {} ....'.format(
                self.matlab_mesh_elements))
        Connectivity = np.array(matfile_data[self.matlab_mesh_elements][()]).astype('int64')
        # correction accounting for different indexing convention in Matlab
        Connectivity = Connectivity - np.ones(Connectivity.shape, dtype='int64')
        if (self.matlab_mesh_transpose):
            Connectivity = Connectivity.transpose()
        Nelements = Connectivity.shape[0]
        self.mesh.add_elements(self.matlab_mesh_element_type,np.arange(Nelements),
                               Connectivity)

        # read fields
        for fieldloc in self.matlab_variables:
            fieldname = fieldloc[fieldloc.rindex('/')+1:]
            field = matfile_data[fieldloc][()]
            print(' \t reading field in h5 node {} ....'.format(fieldloc))
            self.mesh.add_field(field,fieldname)

        # close .mat file
        matfile_data.close()

        return

    def read_geof_mesh(self):
        """ Read .geof formatted meshes with .geof package support """

        print(' MeshReader : reading geof mesh  in {} ....'.format(self.filename))
        # read mesh nodes
        self.mesh.add_nodes(geof.read_geof_nodes(self.filename))

        # read mesh elements and translate type into xdmf format
        TopologyType, Elements_Id, Element_connectivity = \
        geof.read_geof_elements(self.filename)
        for j in range(len(TopologyType)):
            self.mesh.add_elements( Elements_topology= \
                                    self.Geof_to_Xdmf_topology[TopologyType[j]],
                                    Elements_Id = Elements_Id[j],
                                    Elements_connectivity = Element_connectivity[j])

        # read nsets
        self.mesh.add_sets(Node_sets= geof.read_geof_nset(self.filename))
        return


class ImageReader():
    """ Class to read 3D image data from a file into an ImageObject

        Contains an ImageObject following hdf5/xdfm format conventions

        Currently supported file formats
            .mat (Matlab)

        Attributes:
            - filename (str)
                name/path to file containing the mesh data

            - Mesh (ImageObject)
                ImageObject class instance containing image data from file

    """

    def __init__(self,
                 image_filename,
                 **keywords):
        """ init ImageReader with filename and reading options

            Attributes:
                - mesh_filename (str)
                    path to the file to be read by the ImageReader.
                    Extension must be a known format

                - keywords
                    allow to pass options required to read data for some mesh
                    formats. See below

                Handled image formats:
                    - .mat
                    -----------
                        --> h5 file with unknown hierarchy, must be passed as
                            __init__method keywords for fields location in
                            hdf5 structure

                                Keywords required :
                                    * matlab_variables :  list of (str) indicating the
                                                   nodes storing the various
                                                   fields to read

                Other keywords
                ------------------

                        * dimension : [Nx,Ny,Nz]
                            number of voxels in each dimension

                        * spacing : [Dx,Dy,Dz]
                           size of voxels in each direction

                        * origin : [Ox, Oy, Oz]
                            coordinates of the center of the first
                            voxel in the image
        """

        self.filename = image_filename
        self.get_image_type()

        # empty Matlab read keywords init
        self.matlab_variables = []

        # instantiate Image Object
        self.image = ImageObject()

        # get image reading keywords
        for words in keywords:
            if (words == 'dimension'):
                self.image.dimension = keywords[words]
            elif (words == 'origin'):
                self.image.origin = keywords[words]
            elif (words == 'spacing'):
                self.image.spacing = keywords[words]
            elif (words == 'matlab_variables'):
                self.matlab_variables = keywords[words]
            elif (words == 'matlab_field_names'):
                self.matlab_field_names = keywords[words]

        # read image
        self.read_image()

        return

    def get_image_type(self):
        """ Find out mesh file format type from mesh file extension"""

        ext = self.filename[self.filename.rfind('.'):]

        print(' ImageReader : reading extension of file {} to find file format'.format(self.filename))
        if (ext == '.mat'):
            self.file_type = 'matlab'
        elif (ext == '.vtk'):
            self.file_type = 'vtk'
        else:
            print(' ImageReader : File format not handled for file {}'.format(self.filename))
        print(' ImageReader : file format --> {}'.format(self.file_type))
        return

    def read_image(self):
        """ Call the appropriate read method for the considered 3D image format """

        if (self.file_type == 'matlab'):
            self.read_image_matlab()
        elif (self.file_type == 'vtk'):
            warnings.warn(" vtk reading not implemented yet for ImageReader, no data has been read ")
        return

    def read_image_matlab(self):
        """ Read .mat formatted 3D image data """

        print(' ImageReader : reading 3D image data in {} ....'.
              format(self.filename))
        # open .mat file as hdf5 file
        matfile_data = h5.File(self.filename, 'r')


        # get image dimension
        if (self.image.dimension == [1,1,1]):
            Dim = matfile_data[self.matlab_variables[0]][()].shape
            self.image.dimension = Dim
        else:
            Dim = self.image.dimension

        # read fields
        for fieldloc,fieldname in zip(self.matlab_variables,self.matlab_field_names):
            #fieldname = fieldloc[fieldloc.rindex('/')+1:]
            field = matfile_data[fieldloc][()]
            print(' \t reading field in h5 node {} ....'.format(fieldloc))

            # check dimension
            loc_dim = field.shape
            if (loc_dim != Dim):
                raise ValueError(''' Dimension of the added field {} : {}, do not
                                 match image dimension {}
                                 '''.format(fieldname, loc_dim, Dim))
            self.image.add_field(field,fieldname)

        # close .mat file
        matfile_data.close()

        return




