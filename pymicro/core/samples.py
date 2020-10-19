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
# TODO : Update documentation !!!

import warnings
import os
from lxml import etree
from lxml.builder import ElementMaker

import numpy as np
import tables as Tb
import h5py as h5

import pymicro.core.geof as geof

# Module global variables for xdmf compatibility
FIELD_TYPE = {1: 'Scalar', 2: 'Vector', 3: 'Vector', 6: 'Tensor6', 9: 'Tensor'}
CENTERS_XDMF = {'3DImage': 'Cell', 'mesh': 'Node'}

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

    # =============================================================================
    # SampleData magic methods
    # =============================================================================
    def __init__(self,
                 filename,
                 sample_name='name_to_fill',
                 sample_description="""  """,
                 verbose=False,
                 overwrite_hdf5=False,
                 **keywords):
        """ DataSample initialization

        Create an data structure instance for a sample associated to the data
        file 'filename'.

            - if filename.h5 and filename.xdmf exist, the data structure is
              read from these file and stored into the SampleData instance
              unless the overwrite_hdf5 flag is True, in which case, the file is deleted.

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
        self._verbose = verbose

        # if file exists and overwrite flag is True, delete it
        if os.path.exists(self.h5_file) and overwrite_hdf5:
            self._verbose_print('-- File "{}" exists  and will be overwritten'.format(self.h5_file))
            os.remove(self.h5_file)

        # h5 table file object creation
        self.file_exist = False
        self.init_file_object(sample_name, sample_description)

        # initialize XML tree for xdmf format handling
        self.init_xml_tree()

        # set compression options for HDF5 (passed in keywords args)
        if not self.file_exist:
            Compression_keywords = {k: v for k, v in keywords.items() if k in \
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
        self._verbose_print('Deleting DataSample object ')
        self.sync()
        self.h5_dataset.close()
        self._verbose_print('Dataset and Datafiles closed')
        return

    def __repr__(self):
        """ Return a string representation of the dataset content"""
        s = self.print_index(as_string=True)
        s += self.print_dataset_content(as_string=True)
        return s

    def init_file_object(self,
                         sample_name='',
                         sample_description=''):
        """ Initiate PyTable file object from .h5 file or create it if
            needed
        """

        try:
            self.h5_dataset = Tb.File(self.h5_file, mode='r+')
            self._verbose_print('-- Opening file "{}" '.format(self.h5_file),
                                line_break=False)
            self.file_exist = True
            self.init_content_index()
            self.Filters = self.h5_dataset.filters
            self._verbose_print(' ****** File content *****')
            self._verbose_print(repr(self))
        except IOError:
            self._verbose_print('-- File "{}" not found : file'
                                ' created'.format(self.h5_file))
            self.h5_dataset = Tb.File(self.h5_file, mode='a')

            # add sample name and description
            self.h5_dataset.root._v_attrs.sample_name = sample_name
            self.h5_dataset.root._v_attrs.description = sample_description
            self.init_content_index()
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
            self._verbose_print('-- File "{}" not found : file'
                                ' created'.format(self.xdmf_file),
                                line_break=False)

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

        self._verbose_print('.... writing xdmf file : {}'
                            ''.format(self.xdmf_file),
                            line_break=False)
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

    # =========================================================================
    # TODO : implement add to content_index with check of existing name and
    #        aliases mechanism
    #
    # TODO : implement alias mechanism (get path with name & )
    # =========================================================================
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
        self.minimal_content = {key: '' for key in minimal_dic}
        self.content_index = {}

        if self.file_exist:
            self.content_index = self.get_dic_from_attributes(
                node_path='/Index')
        else:
            self.h5_dataset.create_group('/', name='Index')
            self.add_attributes(
                 dic=self.minimal_content,
                 nodename='/Index')
            self.content_index = self.get_dic_from_attributes(
                node_path='/Index')
        return

    def print_dataset_content(self, as_string=False):
        """ Print information on all nodes in hdf5 file"""
        s = '\n****** DATA SET CONTENT ******\n   {} \n'.format(self.h5_file)
        if not(as_string):
            print(s)
        for node in self.h5_dataset:
            if not(node._v_name == 'Index'):
                s += self.get_node_info(node._v_name, as_string)
        return s

    def print_data_arrays(self, as_string=False):
        """ Print information on all data array nodes in hdf5 file"""
        s = ''
        for node in self.h5_dataset:
            if self._is_array(node._v_name):
                s += self.get_node_info(node._v_name,as_string)
        return s

    def print_index(self, as_string=False):
        """ Allow to visualize a list of the datasets contained in the
            file and their status
        """
        s = ''
        s += str('Dataset Content Index :\n')
        s += str('------------------------:\n')
        for key, value in self.content_index.items():
            s += str('\t Name : {:20}  H5_Path : {} \t\n'.format(
                        key, value))
        if not(as_string):
            print(s)
            s = ''
        return s

    def sync(self):
        """ Synchronises .h5 and .xdmf files with dataset content """

        message = ('.... Storing content index in {}:/Index attributes'
                   ''.format(self.h5_file))
        self._verbose_print(message,
                            line_break=False)
        self.add_attributes(dic=self.content_index,
                                     nodename='/Index')
        self.write_xdmf()
        self._verbose_print('.... flushing data in file {}'.format(
            self.h5_file),
            line_break=False)
        self.h5_dataset.flush()
        self._verbose_print('File {} synchronized with in memory data tree'
                            ''.format(self.h5_file),
                            line_break=False)
        return

    def switch_verbosity(self):
        self._verbose = not (self._verbose)
        return

    def add_mesh_from_file(self,
                           meshfile,
                           meshname,
                           indexname='',
                           location='/',
                           description=' ',
                           replace=False,
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
        self._verbose_print('Lauching mesh reader on  {}...'.format(meshfile))
        Reader = MeshReader(meshfile, **keywords)
        Mesh_object = Reader.mesh
        self.add_mesh(
            mesh_object=Mesh_object,
            meshname=meshname,
            indexname=indexname,
            location=location,
            description=description,
            replace=replace)
        return

    def add_image_from_file(self,
                            imagefile,
                            imagename,
                            location='/',
                            description=' ',
                            indexname='',
                            replace=False,
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

        self._verbose_print('Lauching image reader on  {}...'.format(imagefile))
        Reader = ImageReader(imagefile, **keywords)
        Image_object = Reader.image
        self.add_image(
            image_object=Image_object,
            imagename=imagename,
            indexname=indexname,
            location=location,
            description=description,
            replace=replace)
        return

    def add_mesh(self,
                 mesh_object,
                 meshname,
                 indexname='',
                 location='/',
                 description=' ',
                 replace=False):
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
                        ' the meshname {} is used in '
                        'content index'.format(meshname))
            self._verbose_print(warn_msg)
            indexname = meshname
        mesh_path = os.path.join(location, meshname)

        # Check mesh group existence and replacement
        if self.h5_dataset.__contains__(mesh_path):
            msg = ('(add_mesh) Mesh group {} already exists.'
                   ''.format(mesh_path))
            if replace:
                msg += ('--- It will be replaced by the new MeshObject content')
                self.remove_node(node_path=mesh_path, recursive=True)
                self._verbose_print(msg)
            else:
                msg += ('\n--- If you want to replace it by new MeshObject'
                        ' content, please add `replace=True` to keyword'
                        ' arguments of `add_mesh`')
                self._verbose_print(msg)
                return

        self._verbose_print('Creating hdf5 group {} in file {}'.format(
            mesh_path, self.h5_file))
        mesh_group = self.add_group(path=location,
                                    groupname=meshname,
                                    indexname=indexname)
        self.content_index[indexname] = mesh_path

        # store mesh metadata as HDF5 attributes
        mesh_group._v_attrs.element_topology = mesh_object.element_topology[0]
        mesh_group._v_attrs.mesh_description = description
        mesh_group._v_attrs.field_dim = {}
        mesh_group._v_attrs.group_type = 'mesh'

        self._verbose_print('Creating Nodes data set in group {} in file {}'
                            ''.format(mesh_path, self.h5_file))
        self.h5_dataset.create_carray(
            where=mesh_path,
            name='Nodes',
            filters=self.Filters,
            obj=mesh_object.nodes,
            title=indexname + '_Nodes')

        # safety check
        if (len(mesh_object.element_topology) > 1):
            warnings.warn('''  number of element type found : {} \n
                          add_mesh_from_file current implementation works only
                          with meshes with only one element type
                          '''.format(len(mesh_object.element_topology)))

        self._verbose_print('Creating Elements data set in group {} in file {}'
                            ''.format(location + '/' + meshname, self.h5_file))
        self.h5_dataset.create_carray(
            where=mesh_path,
            name='Elements',
            filters=self.Filters,
            obj=mesh_object.element_connectivity[0],
            title=indexname + '_Elements')

        self._verbose_print('...Updating xdmf tree...', line_break=False)
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
                  description=' ',
                  replace=False):
        """ add geometry data and fields stored on a mesh from a MeshObject

            Arguments:
                    - image_object (samples.ImageObject)
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

        if indexname == '':
            warn_msg = ('(add_image) indexname not provided, '
                        ' the image name {} is used instead in '
                        'content index'.format(imagename))
            self._verbose_print(warn_msg)
            indexname = imagename
        image_path = os.path.join(location, imagename)

        # Check mesh group existence and replacement
        if self.h5_dataset.__contains__(image_path):
            msg = ('\n(add_image) Image group {} already exists.'
                   ''.format(image_path))
            if replace:
                msg += ('--- It will be replaced by the new ImageObject'
                        ' content')
                # self.remove_node(node_path=image_path, recursive=True)
                self.remove_node(name=imagename, recursive=True)
                self._verbose_print(msg)
            else:
                msg += ('\n--- If you want to replace it by new ImageObject '
                        'content, please add `replace=True` to keyword '
                        'arguments of `add_image`.'
                        'WARNING This will erase the current Group and'
                        'its content')
                print(msg)
                return

        # create group in the h5 structure for the mesh
        self._verbose_print('Creating hdf5 group {} in file {}'.format(
            image_path, self.h5_file))
        image_group = self.add_group(
            path=location,
            groupname=imagename,
            indexname=indexname)

        # store image metadata as HDF5 attributes
        image_group._v_attrs.dimension = image_object.dimension
        image_group._v_attrs.spacing = image_object.spacing
        image_group._v_attrs.origin = image_object.origin
        image_group._v_attrs.image_description = description
        image_group._v_attrs.field_dim = {}
        image_group._v_attrs.group_type = '3DImage'

        self._verbose_print('Updating xdmf tree...', line_break=False)
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
            self._verbose_print(warn_msg)
            field_indexname = fieldname

        mesh_path = os.path.join(mesh_location, meshname)
        field_path = os.path.join(mesh_path, fieldname)
        # store mesh location in dataset index
        self.content_index[field_indexname] = field_path

        Field_type = {1: 'Scalar',
                      2: 'Vector',
                      3: 'Vector',
                      6: 'Tensor6',
                      9: 'Tensor'
                      }

        # if field has only 1 dimension, reshape it to a 2dim array
        if (field.ndim < 2):
            field = field.reshape([field.shape[0], 1])

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
                                        mesh_group.Nodes.shape[0]))

        # Add field data to the hdf5 data structure
        self._verbose_print('Creating field "{}" data set in group {} in'
                            ' file {}'.format(fieldname,
                                              mesh_location + meshname, self.h5_file))
        field_node = self.h5_dataset.create_carray(
            where=mesh_path,
            name=fieldname,
            filters=self.Filters,
            obj=field)

        # determine AttributeType
        for Ftype in Field_type:
            if (Ftype == field.shape[1]):
                Attribute_type = Field_type[Ftype]
                break
            else:
                Attribute_type = 'Matrix'

        # Store field attributes : nothing for now
        mesh_group._v_attrs.field_dim.update({fieldname: Attribute_type})

        # Update xdmf tree
        # 1- create Attribute element
        self._verbose_print('Updating xdmf tree...', line_break=False)
        Attribute_xdmf = etree.Element(_tag='Attribute',
                                       Name=fieldname,
                                       AttributeType=Attribute_type,
                                       Center='Node')

        Dim = str(field.shape).strip('(').strip(')')
        Dim = Dim.replace(',', ' ')
        Attribute_data = etree.Element(_tag='DataItem',
                                       Format='HDF',
                                       Dimensions=Dim,
                                       NumberType='Float',
                                       Precision='64')
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
            warn_msg = ('(add_field_to_image) indexname not provided, '
                        ' the fieldname {} is used instead to fill'
                        'content index'.format(fieldname))
            self._verbose_print(warn_msg)
            field_indexname = fieldname

        image_path = os.path.join(image_location, imagename)
        field_path = os.path.join(image_path, fieldname)
        # store mesh location in dataset index
        self.content_index[field_indexname] = field_path

        Field_type = {1: 'Scalar',
                      2: 'Vector',
                      3: 'Vector',
                      6: 'Tensor6',
                      9: 'Tensor'
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
                                        image_group._v_attrs.dimension))

        # Add field data to the hdf5 data structure
        self._verbose_print('Creating field "{}" data set in group {} in'
                            ' file {}'.format(
            fieldname, image_location + imagename, self.h5_file))
        self._verbose_print('Creating field with following compression'
                            ' Filters : ')
        self._verbose_print(str(self.Filters))
        field_node = self.h5_dataset.create_carray(
            where=image_path,
            name=fieldname,
            filters=self.Filters,
            obj=field)
        self._verbose_print('CREATED field compression Filters : ')
        self._verbose_print(str(field_node.filters))

        for Ftype in Field_type:
            if (Ftype == field.shape[1]):
                Attribute_type = Field_type[Ftype]
                break
            else:
                Attribute_type = 'Matrix'

        # Store field attributes : nothing for now
        image_group._v_attrs.field_dim.update({fieldname: Attribute_type})

        # Update xdmf tree
        # 1- create Attribute element
        self._verbose_print('Updating xdmf tree...', line_break=False)
        Attribute_xdmf = etree.Element(_tag='Attribute',
                                       Name=fieldname,
                                       AttributeType=Attribute_type,
                                       Center='Cell')

        Dim = str(field.shape).strip('(').strip(')')
        Dim = Dim.replace(',', ' ')

        if (np.issubdtype(field.dtype, np.floating)):
            NumberType = 'Float'
            if (str(field.dtype) == 'float'):
                Precision = '32'
            else:
                Precision = '64'
        elif (np.issubdtype(field.dtype, np.integer)):
            NumberType = 'Int'
            Precision = str(field.dtype).strip('int')

        Attribute_data = etree.Element(_tag='DataItem',
                                       Format='HDF',
                                       Dimensions=Dim,
                                       NumberType=NumberType,
                                       Precision=Precision)
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
                  indexname='',
                  replace=False):
        """ Create a (hdf5) group at desired location in the data format"""

        if (indexname == ''):
            warn_msg = ('\n(add_group) indexname not provided, '
                        ' the groupname {} is used in content '
                        'index'.format(groupname))

            self._verbose_print(warn_msg)
            indexname = groupname
        self.content_index[indexname] = path + groupname

        try:
            Group = self.h5_dataset.create_group(
                where=path,
                name=groupname,
                title=indexname)
            return Group
        except Tb.NodeError:
            node_path = os.path.join(path, groupname)
            if (self.h5_dataset.__contains__(node_path)):
                if replace:
                    Group = self.h5_dataset.create_group(
                        where=path,
                        name=groupname,
                        title=indexname)
                    return Group
                else:
                    warn_msg = ('\n(add_group) group {} already exists,'
                                ' group creation aborted'
                                ''.format(path + groupname))
                    warn_msg += ('\n--- If you want to replace it by an empty {}'
                                 ' Group please add `replace=True` to keyword '
                                 'arguments of `add_group`.'
                                 'WARNING This will erase the current Group and'
                                 'its content'.format(groupname))
                    print(warn_msg)
                return None
            else:
                raise

    # =========================================================================
    # TODO : implement this
    #       -- Test
    #       -- Replace add_field_to_mesh add_image_to_mesh by add_data_array
    #==========================================================================
    def add_data_array(self,
                       location,
                       name,
                       array,
                       indexname=None,
                       createparents=False,
                       replace=False,
                       **keywords):
        """Add the data array at the given location in hte HDF5 data tree.

        """

        # get location path
        location_path = self._name_or_node_to_path(location)
        if location_path is None:
            if createparents:
                location_exists = False
            else:
                msg = ('(add_data_array): location {} does not exist, array'
                       ' cannot be added. Use optional argument'
                       ' "createparents=True" to force location Group creation'
                       ''.format(location))
                self._verbose_print(msg)
                return
        else:
            location_exists = True
            # check location nature
            if not(self._get_node_class(location) == 'GROUP') :
                    msg = ('(add_data_array): location {} is not a Group nor '
                           'empty. Please choose an empty location or a HDF5 '
                           'Group to store data array'.format(location))
                    self._verbose_print(msg)
                    return
            # check if array location exists and remove node if asked
            array_path = location_path + name
            if self.h5_dataset.__contains__(array_path):
                if replace:
                    msg = ('(add_data_array): existing node {} will be '
                           'overwritten and all of its childrens removed'
                           ''.format(array_path))
                    self._verbose_print(msg)
                    self.remove_node(array_path, recursive=True)
                else:
                    msg = ('(add_data_array): node {} already exists. To '
                           'overwrite, use optional argument "replace=True"'
                           ''.format(array_path))
                    self._verbose_print(msg)

        # add to index
        if indexname is None:
            warn_msg = (' (add_data_array) indexname not provided, '
                        ' the array name `{}` is used as index name '
                        ''.format(name))
            self._verbose_print(warn_msg)
            indexname = name
        self.content_index[indexname] = os.path.join(location_path,name)

        # get compression options
        Filters = self._get_local_compression_opt(**keywords)
        self._verbose_print('-- Compression Options for dataset {}'
                            ''.format(name))
        if (self.Filters.complevel > 0):
            msg_list = str(self.Filters).strip('Filters(').strip(')').split()
            for msg in msg_list:
                self._verbose_print('\t * {}'.format(msg), line_break=False)
        else:
            self._verbose_print('\t * No Compression')

        # get location type
        if location_exists:
            self._check_field_compatibility(location,array)
        self._verbose_print('Adding array `{}` into `{}`'
                            ''.format(name,location))

        # Create dataset node to store array
        print('location_path is {}'.format(location_path))
        print('name is {}'.format(name))
        Node = self.h5_dataset.create_carray(where=location_path, name=name,
                                      filters=Filters, obj=array)

        if self._is_grid(location):
            ftype = self._get_xdmf_field_type(field=array,
                                              grid_location=location_path)
            dic = {'field_type':ftype}
            self.add_attributes(dic=dic, nodename=name)
            self._add_field_to_xdmf(name,array)
        return

    def add_attributes(self, dic, nodename):
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
        """
        Node = self.get_node(nodename)
        for key, value in dic.items():
            Node._v_attrs[key] = value
        return

    def get_indexname_from_path(self,
                                node_path):
        """Return the key of the given node_path in the content index."""

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

    def get_data_array(self,
                       name,
                       as_numpy=False):
        """ Searchs and returns data array with given name in hdf5 data tree

                Parameters
                ----------
                    name (str):
                        string interpreted as the data path in the hdf5 tree
                        structure or data name in self.content_index directory

                    as_numpy (bool):
                        if True, returns the data as a numpy array.
                        if False (default) returns Data as a Pytables node

        """

        data = None
        data_path = self._name_or_node_to_path(name)
        if data_path is None:
            msg = ('(get_data_array) `name` not matched with a path or an'
                   ' indexname. No data returned')
            self._verbose_print(msg)
        else:
            if self._is_array(name=data_path):
                msg = '(get_data_array) Getting data from : {}:{}'.format(self.h5_file, data_path)
                self._verbose_print(msg)
                data = self.get_node(name=data_path, as_numpy=as_numpy)
            else:
                msg = ('(get_data_array) Data is not an array node.'
                       ' Use `get_node` method instead.')
                self._verbose_print(msg)
        return data

    def get_node(self,
                 name,
                 as_numpy=False):
        """ get a Node object from the HDF5 data tree from indexname or path.

            If the node is not found, returns None.
            Else, returns the Node object as corresponding Pytable class, or as
            a Numpy array if requested, and if the Node stores array data.
        """

        node = None
        node_path = self._name_or_node_to_path(name)
        if node_path is None:
            msg = '(get_node) ERROR : Node name does not fit any hdf5 path nor index name.'
            self._verbose_print(msg)
        else:
            node = self.h5_dataset.get_node(node_path)
            if as_numpy and self._is_array(name):
                node = node.read()
        return node

    def get_node_info(self, name, as_string=False):
        """ get information on a node in HDF5 tree from indexname or path."""

        s = ''
        node_path = self._name_or_node_to_path(name)
        if self._is_array(node_path):
            s += self._get_array_node_info(name,as_string)
        else:
            s += self._get_group_info(name,as_string)
        return s

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

    def get_attribute(self,
                      attrname,
                      node_name):
        """ Get a specific attribute of a specific hdf5 data tree element"""

        attribute = None
        data_path = self._name_or_node_to_path(node_name)
        if (data_path is None):
            self._verbose_print(' (get_attribute) neither indexname nor'
                                ' node_path passed, node return aborted')
        else:
            attribute = self.h5_dataset.get_node_attr(where=data_path,
                                                      attrname=attrname)
        return attribute

    def set_global_compression_opt(self, **keywords):
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
        default = False
        # ------ check if default compression is required
        if 'default_compression' in keywords:
            default = True
            self.Filters = self._set_defaut_compression()
        else:
            self.Filters = self._get_compression_opt(**keywords)

        # ----- message and Pytables Filter (comp option container) set up
        self._verbose_print('-- General Compression Options for datasets'
                            ' in {}'.format(self.h5_file))

        self.h5_dataset.filters = self.Filters

        if (self.Filters.complevel > 0):
            if default:
                self._verbose_print('\t Default Compression Parameters ')
            msg_list = str(self.Filters).strip('Filters(').strip(')').split()
            self._verbose_print(str(msg_list))
            for msg in msg_list:
                self._verbose_print('\t * {}'.format(msg), line_break=False)
        else:
            self._verbose_print('\t * No Compression')
        return


    def set_verbosity(self, verbosity=True):
        self._verbose = verbosity
        return

    def remove_node(self,
                    name,
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

        node_path = self._name_or_node_to_path(name)
        if node_path is None:
            msg = ('(remove_node) Node name does not fit any hdf5 path'
                   ' nor index name. Node removal aborted.')
            self._verbose_print(msg)
            return

        Node = self.h5_dataset.get_node(node_path)
        isGroup = False
        remove_flag = False

        # determine if node is an HDF5 Group
        if (Node._v_attrs.CLASS == 'GROUP'):
            isGroup = True

        if (isGroup):
            print('WARGNING : node {} is a hdf5 group with  {} children(s)'
                  ' :'.format(node_path, Node._v_nchildren))
            count = 1
            for child in Node._v_children:
                print('\t child {}:'.format(count), Node[child])
                count = count + 1
            if not recursive:
                print('Deleting the group will delete children data.',
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
        if (remove_flag or not (isGroup)):

            # remove node in xdmf tree
            xdmf_path = Node._v_attrs.xdmf_path
            group_xdmf = self.xdmf_tree.find(xdmf_path)
            self._verbose_print('Removing  node {} in xdmf'
                                ' tree....'.format(xdmf_path))
            parent = group_xdmf.getparent()
            parent.remove(group_xdmf)

            self._verbose_print('Removing  node {} in h5 structure'
                                ' ....'.format(node_path))
            if (isGroup):
                for child in Node._v_children:
                    remove_path = Node._v_children[child]._v_pathname
                    self._remove_from_index(node_path=remove_path)
                self._verbose_print('Removing  node {} in content'
                                    ' index....'.format(
                    Node._v_pathname))
                self._remove_from_index(node_path=Node._v_pathname)
                Node._f_remove(recursive=True)

            else:
                self._remove_from_index(node_path=Node._v_pathname)
                Node.remove()
            self._verbose_print('Node sucessfully removed, '
                                'new data structure is:\n')
            self._verbose_print(str(self.h5_dataset))
            self._verbose_print('New data structure is:\n')
            self._verbose_print(self.print_index())

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
        minimal_content_index_dic = []
        return minimal_content_index_dic

    # =============================================================================
    #  SampleData private utilities
    # =============================================================================
    def _remove_from_index(self,
                           node_path):
        """Remove a hdf5 node from content index dictionary"""

        try:
            key = self.get_indexname_from_path(node_path)
            removed_path = self.content_index.pop(key)
            self._verbose_print('item {} : {} removed from context index'
                                ' dictionary'.format(key, removed_path))
        except ValueError:
            print('node {} not found in content index values for removal'
                  ''.format(node_path))
        return

    def _name_or_node_to_path(self, name_or_node):
        """ Match `name` with content_index dic or hdf5 pathes and return path
        """
        path = None
        # name_or_node is a Node
        if isinstance(name_or_node,Tb.Node):
            return name_or_node._v_pathname
        # name_or_node is a string or else
        name_tmp = os.path.join('/', name_or_node)
        if self.h5_dataset.__contains__(name_tmp):
            # name is a path in hdf5 tree data
            path = name_tmp
        if name_or_node in self.content_index:
            # name is a key in content index
            path = self._get_path_with_indexname(name_or_node)
        # if not found with indexname or is not a path
        # find nodes with this name. If several nodes have this name
        # return warn message and return None
        if path is None:
            count = 0
            path_list = []
            for node in self.h5_dataset:
                if (name_or_node == node._v_name):
                    count = count +1
                    path_list.append(node._v_pathname)
            if count == 1:
                path = path_list[0]
            elif count > 1:
                msg = ('(_name_or_node_to_path) : more than 1 node ({}) with '
                       'name {} have been found :\n {} \n Use indexname to '
                       'distinguish nodes.'.format(count,name_or_node,
                                                   path_list))
                warnings.warn(msg)
                path = None
        return path

    def _is_array(self, name):
        """ find out if name or path references an array dataset"""
        Class = self._get_node_class(name)
        List = ['CARRAY', 'EARRAY', 'VLARRAY', 'ARRAY', ]
        if (Class in List):
            return True
        else:
            return False

    def _is_group(self, name):
        """ find out if name or path references a HDF5 Group"""
        Class = self._get_node_class(name)
        if Class == 'GROUP':
            return True
        else:
            return False

    def _is_grid(self, name):
        """ find out if name or path references a image or mesh HDF5 Group """
        gtype = self._get_group_type(name)
        List = ['mesh','3DImage']
        if (gtype in List):
            return True
        else:
            return False

    def _check_field_compatibility(self, name, field):
        """ check if field dimension is compatible with the storing location"""
        group_type = self._get_group_type(name)
        if group_type == '3DImage':
            compatibility = self._compatibility_with_image(name, field)
        elif group_type == 'mesh':
            compatibility = self._compatibility_with_mesh(name,field)

        if not(compatibility):
            msg = ('Array dimensions not compatible with location {}'
                   ''.format(name))
            raise ValueError(msg)
        return


    def _compatibility_with_mesh(self, name, field):
        """ check if field has a number of values compatible with the mesh"""
        mesh = self.get_node(name)
        compatibility = (mesh.Nodes.shape[0] == field.shape[0])
        if not(compatibility):
            msg = ('Field number of values ({}) is not conformant with mesh '
                   '`{}` node numbers ({})'.format(field.shape[0],name,
                                                 mesh.Nodes.shape[0]))
            self._verbose_print(msg)
        return compatibility

    def _compatibility_with_image(self, name, field):
        """ check if field has a number of values compatible with the image"""
        image = self.get_node(name)
        compatibility = (image._v_attrs.dimension == field.shape[0:3])
        if not(compatibility):
            msg = ('Field number of values ({}) is not conformant with image '
                   '`{}` dimensions ({})'.format(field.shape,name,
                                                 image._v_attrs.dimension))
            self._verbose_print(msg)
        return compatibility

    def _add_field_to_xdmf(self, name, field):
        """ Write field data as Grid Attribute in xdmf tree/file """

        Node = self.get_node(name)
        Grid_type = self._get_parent_type(name)
        Grid_name = self._get_parent_name(name)
        attribute_center = CENTERS_XDMF[Grid_type]
        xdmf_path = self.get_attribute(attrname='xdmf_path',
                                       node_name=Grid_name)
        field_type = self.get_attribute(attrname='field_type',
                                         node_name=name)
        Xdmf_grid_node = self.xdmf_tree.find(xdmf_path)

        # create Attribute element
        Attribute_xdmf = etree.Element(_tag='Attribute',
                                       Name=name,
                                       AttributeType=field_type,
                                       Center=attribute_center)

        Dim = str(Node.shape).strip('(').strip(')').replace(',', ' ')
        # Create data item element

        if (np.issubdtype(field.dtype, np.floating)):
            NumberType = 'Float'
            if (str(field.dtype) == 'float'):
                Precision = '32'
            else:
                Precision = '64'
        elif (np.issubdtype(field.dtype, np.integer)):
            NumberType = 'Int'
            Precision = str(field.dtype).strip('int')

        Attribute_data = etree.Element(_tag='DataItem',
                                       Format='HDF',
                                       Dimensions=Dim,
                                       NumberType=NumberType,
                                       Precision=Precision)
        Attribute_data.text = (self.h5_file + ':'
                               + self._name_or_node_to_path(name) )

        # add data item to attribute
        Attribute_xdmf.append(Attribute_data)
        # add attribute to Grid
        Xdmf_grid_node.append(Attribute_xdmf)
        el_path = self.xdmf_tree.getelementpath(Attribute_xdmf)
        dic = {'xdmf_path':el_path}
        self.add_attributes(dic,name)
        return

    def _get_xdmf_field_type(self, field, grid_location):
        """ Determine field's XDMF nature w.r.t. grid dimension"""

        field_type = None
        # get grid dimension
        if not(self._is_grid(grid_location)):
            msg = ('(_get_xdmf_field_type) name or path `{}` is not a grid.'
                   'Field type detection impossible'.format(grid_location))
            self._verbose_print(msg)
            return None
        Type = self._get_group_type(grid_location)
        # analyze field shape
        if Type == 'mesh':
            if len(field.shape) != 2:
                msg = ('(_get_xdmf_field_type) wrong field shape. For a mesh'
                       'field, shape should be (Nnodes,Ndim) or (NIntPts,Ndim)'
                       '. None type returned')
                warnings.warn(msg)
                return field_type
            field_dim = field.shape[1]
        elif Type == '3DImage':
            if (len(field.shape) < 3) or (len(field.shape) > 4):
                msg = ('(_get_xdmf_field_type) wrong field shape. For a 3DImage'
                       'field, shape should be (Nx1,Nx2,Nx3) or'
                       ' (Nx1,Nx2,Nx3,Ndim). None type returned')
                warnings.warn(msg)
                return field_type
            if (len(field.shape) == 3): field_dim= 1
            if (len(field.shape) == 4): field_dim = field.shape[3]
        # determine field dimension and get name in dictionary
        field_type = FIELD_TYPE[field_dim]
        return field_type

    def _get_node_class(self,
                        name):
        """ returns the Pytables Class type associated to the node name or path
        """
        return self.get_attribute(attrname='CLASS', node_name=name)

    def _get_path_with_indexname(self, indexname):
        if indexname in self.content_index.keys():
            return self.content_index[indexname]
        else:
            raise ValueError('Index contains no item named {}'.format(
                indexname))
            return

    def _get_group_type(self, groupname):
        """  """
        if self._is_group(groupname):
            grouptype = self.get_attribute(attrname='group_type',
                                           node_name=groupname)
            if grouptype is None:
                return 'GROUP'
            else:
                return grouptype
        else:
            return None

    def _get_parent_type(self, name):
        """ get the type of the node parent group"""
        groupname = self._get_parent_name(name)
        return self._get_group_type(groupname)

    def _get_parent_name(self, name):
        """ get the name of the node parent group"""
        Node = self.get_node(name)
        Group = Node._g_getparent()
        return Group._v_name

    def _get_group_info(self, groupname, as_string=False):
        """ Print a human readable information on the Pytables Group object"""

        s = ''
        Group = self.get_node(groupname)
        gname = Group._v_name
        s += str('\n Group {}\n'.format(gname))
        s += str('=====================\n')
        gparent_name = Group._v_parent._v_name
        s += str(' -- Parent Group : {}\n'.format(gparent_name))
        s += str(' -- Group attributes : \n')
        for attr in Group._v_attrs._v_attrnamesuser:
            value = Group._v_attrs[attr]
            s += str('\t {} : {}\n'.format(attr, value))
        s += str(' -- Childrens : ')
        for child in Group._v_children:
            s += str('{}, '.format(child))
        s += str('\n----------------\n')
        if not(as_string):
            print(s)
            s = ''
        return s

    def _get_array_node_info(self, nodename, as_string=False):
        """ Print a human readable information on the Pytables Group object"""

        Node = self.get_node(nodename)
        s = ''
        s += str('\n Node {}\n'.format(Node._v_pathname))
        s += str('====================\n')
        nparent_name = Node._v_parent._v_name
        s += str(' -- Parent Group : {}\n'.format(nparent_name))
        s += str(' -- Node name : {}\n'.format(Node._v_name))
        s += str(' -- Node attributes : \n')
        for attr in Node._v_attrs._v_attrnamesuser:
            value = Node._v_attrs[attr]
            s += str('\t {} : {}\n'.format(attr, value))
        s += str(' -- content : {}\n'.format(str(Node)))
        s += str('----------------\n')
        if not(as_string):
            print(s)
            s = ''
        return s

    def _get_compression_opt(self, **keywords):
        """ Get compression options in keywords and return a Filters instance

            See PyTables documentation of Filters class for keywords and use
            details
        """
        Filters = self._set_defaut_compression()
        # ------ read specified values of compression options
        for word in keywords:
            if (word == 'complib'):
                Filters.complib = keywords[word]
            elif (word == 'complevel'):
                Filters.complevel = keywords[word]
            elif (word == 'shuffle'):
                Filters.shuffle = keywords[word]
            elif (word == 'bitshuffle'):
                Filters.bitshuffle = keywords[word]
            elif (word == 'checksum'):
                Filters.fletcher32 = keywords[word]
            elif (word == 'least_significant_digit'):
                Filters.least_significant_digit = keywords[word]
        return Filters

    def _get_local_compression_opt(self, **keywords):
        """ Get compression options in keywords and return a Filters instance

            See PyTables documentation of Filters class for keywords and use
            details
        """
        Filters = self.Filters
        # ------ read specified values of compression options
        for word in keywords:
            if (word == 'complib'):
                Filters.complib = keywords[word]
            elif (word == 'complevel'):
                Filters.complevel = keywords[word]
            elif (word == 'shuffle'):
                Filters.shuffle = keywords[word]
            elif (word == 'bitshuffle'):
                Filters.bitshuffle = keywords[word]
            elif (word == 'checksum'):
                Filters.fletcher32 = keywords[word]
            elif (word == 'least_significant_digit'):
                Filters.least_significant_digit = keywords[word]
        return Filters

    def _set_defaut_compression(self):
        """ Returns a Filter object with defaut compression parameters """
        Filters = Tb.Filters(complib='zlib', complevel=1, shuffle=True)
        return Filters

    def _verbose_print(self, message, line_break=True):
        """ Print message if verbose flag True"""
        Msg = message
        if line_break: Msg = ('\n' + Msg)
        if self._verbose: print(Msg)
        return

# =============================================================================
#  Utility classes
#   TODO : Integrate with Pymicro and Basic Tools codes for mesh handling
#   TODO : handle Mesh/Image Objects and Readers verbosity
# =============================================================================
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

        if (len(self.nodes) == 0):
            self.nodes = Nodes
        #            print('\t \t  Added {} nodes to the mesh'.format(len(Nodes)))
        #            print('\t \t \t First node  : {}'.format(
        #                    Nodes[0,:]))
        #            print('\t \t \t Last node  : {}'.format(
        #                    Nodes[-1,:]))
        else:
            if (self.nodes.shape[1] == Nodes.shape[1]):
                self.nodes = np.append(self.nodes, Nodes)
            #                print('\t \t Added {} nodes to the mesh'.format(len(Nodes)))
            #                print('\t \t \t First node  : {}'.format(
            #                        Nodes[0,:]))
            #                print('\t \t \t Last node   : {}'.format(
            #                        Nodes[-1,:]))
            else:
                raise ValueError(''' shape of added Nodes array {} do not
                                 match MeshObject.nodes shape {}
                                 '''.format(Nodes.shape, self.nodes.shape))

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
        #        print('\t \t Added {} {} elements of degree {} to the mesh'.format(
        #                Elements_Id.size,Elements_topology[0],Elements_topology[1]))
        #        print('\t \t \t First element nodes  : {}'.format(
        #                Elements_connectivity[0,:]))
        #        print('\t \t \t Last  element nodes  : {}'.format(
        #                Elements_connectivity[-1,:]))
        return

    def add_sets(self,
                 Node_sets={}):
        """ Adds nodes/elements sets to the MeshObject

            Arguments:
                - Node_sets (dict :  'nset name':np.array.int64)
                  dictionnary linking node set names (keys) to integer arrays
                  of the node set verteces ID numbers (values)
                  default : empty dict

        """
        self.node_sets.update(Node_sets)
        # print('\t Added the foolowing node sets to the mesh: ')

    #        for nset in Node_sets:
    #            print('\t\t - {}'.format(nset))
    #        return

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
                             '''.format(Field.shape[0], self.nodes.shape[0]))

        # in case of a two dimensional array : reshape to 3dim array with
        # singleton last dimension
        if (Field.ndim < 2):
            Field = Field.reshape([Field.shape[0], 1])

        self.fields.append({Field_name: Field})
        #        print('\t field {} of dimension {} added to the mesh'.format(Field_name,
        #              Field.shape))
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
                 dimension=[1, 1, 1],
                 origin=np.array([0., 0., 0.], dtype='double'),
                 spacing=np.array([1, 1, 1], dtype='double'),
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
                             '''.format(Field.shape, self.dimension))

        self.fields.append({Field_name: Field})
        #        print('\t field {} of dimension {} added to the 3D image'.format(Field_name,
        #              Field.shape))
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
        'c2d3r': ['Triangle', 1],
        'c2d3': ['Triangle', 1],
        'c2d4r': ['Quadrilateral', 1],
        'c2d4': ['Quadrilateral', 1],
        'c2d6r': ['Tri_6', 2],
        'c2d6': ['Tri_6', 2],
        'c2d8r': ['Quad_8', 2],
        'c2d8': ['Quad_8', 2],
        's3d3r': ['Triangle', 1],
        's3d3': ['Triangle', 1],
        's2d4r': ['Quadrilateral', 1],
        's2d4': ['Quadrilateral', 1],
        's2d6r': ['Tri_6', 2],
        's2d6': ['Tri_6', 2],
        's2d8r': ['Quad_8', 2],
        's2d8': ['Quad_8', 2],
        'c3d4r': ['Tetrahedron', 1],
        'c3d4': ['Tetrahedron', 1],
        'c3d10r': ['Tetrahedron', 2],
        'c3d10': ['Tetrahedron', 2],
        'c3d6r': ['Wedge', 1],
        'c3d6': ['Wedge', 1],
        'c3d15r': ['Wedge_15', 1],
        'c3d15': ['Wedge_15', 1],
        'c3d8r': ['Hexahedron', 1],
        'c3d8': ['Hexahedron', 1],
        'c3d20r': ['Hex_20', 2],
        'c3d20': ['Hex_20', 2]
    }

    def __init__(self,
                 mesh_filename,
                 **keywords
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
        self.matlab_mesh_transpose = False

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

        #        print(' MeshReader : reading extension of file {} to find file format'.format(self.filename))
        if (ext == '.mat'):
            self.file_type = 'matlab'
        elif (ext == '.geof'):
            self.file_type = 'geof'
        elif (ext == '.vtk'):
            self.file_type = 'vtk'
        else:
            print(' MeshReader : File format not handled for file {}'.format(self.filename))
        # print(' MeshReader : file format --> {}'.format(self.file_type))
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

        #        print(' MeshReader : reading mesh in {} ....'.format(self.filename))
        # open .mat file as hdf5 file
        matfile_data = h5.File(self.filename, 'r')

        # read mesh nodes
        #        print(' \t reading mesh nodes in h5 node {} ....'.format(
        #                self.matlab_mesh_nodes))
        Nodes = np.array(matfile_data[self.matlab_mesh_nodes][()]).astype('double')
        if (self.matlab_mesh_transpose):
            Nodes = Nodes.transpose()
        self.mesh.add_nodes(Nodes)

        # read mesh elements
        #        print(' \t reading mesh elements in h5 node {} ....'.format(
        #                self.matlab_mesh_elements))
        Connectivity = np.array(matfile_data[self.matlab_mesh_elements][()]).astype('int64')
        # correction accounting for different indexing convention in Matlab
        Connectivity = Connectivity - np.ones(Connectivity.shape, dtype='int64')
        if (self.matlab_mesh_transpose):
            Connectivity = Connectivity.transpose()
        Nelements = Connectivity.shape[0]
        self.mesh.add_elements(self.matlab_mesh_element_type, np.arange(Nelements),
                               Connectivity)

        # read fields
        for fieldloc in self.matlab_variables:
            fieldname = fieldloc[fieldloc.rindex('/') + 1:]
            field = matfile_data[fieldloc][()]
            #            print(' \t reading field in h5 node {} ....'.format(fieldloc))
            self.mesh.add_field(field, fieldname)

        # close .mat file
        matfile_data.close()

        return

    def read_geof_mesh(self):
        """ Read .geof formatted meshes with .geof package support """

        #        print(' MeshReader : reading geof mesh  in {} ....'.format(self.filename))
        # read mesh nodes
        self.mesh.add_nodes(geof.read_geof_nodes(self.filename))

        # read mesh elements and translate type into xdmf format
        TopologyType, Elements_Id, Element_connectivity = \
            geof.read_geof_elements(self.filename)
        for j in range(len(TopologyType)):
            self.mesh.add_elements(Elements_topology= \
                                       self.Geof_to_Xdmf_topology[TopologyType[j]],
                                   Elements_Id=Elements_Id[j],
                                   Elements_connectivity=Element_connectivity[j])

        # read nsets
        self.mesh.add_sets(Node_sets=geof.read_geof_nset(self.filename))
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

        #        print(' ImageReader : reading extension of file {} to find file format'.format(self.filename))
        if (ext == '.mat'):
            self.file_type = 'matlab'
        elif (ext == '.vtk'):
            self.file_type = 'vtk'
        else:
            print(' ImageReader : File format not handled for file {}'.format(self.filename))
        # print(' ImageReader : file format --> {}'.format(self.file_type))
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

        #        print(' ImageReader : reading 3D image data in {} ....'.
        #              format(self.filename))
        # open .mat file as hdf5 file
        matfile_data = h5.File(self.filename, 'r')

        # get image dimension
        if (self.image.dimension == [1, 1, 1]):
            Dim = matfile_data[self.matlab_variables[0]][()].shape
            self.image.dimension = Dim
        else:
            Dim = self.image.dimension

        # read fields
        for fieldloc, fieldname in zip(self.matlab_variables, self.matlab_field_names):
            # fieldname = fieldloc[fieldloc.rindex('/')+1:]
            field = matfile_data[fieldloc][()]
            #            print(' \t reading field in h5 node {} ....'.format(fieldloc))

            # check dimension
            loc_dim = field.shape
            if (loc_dim != Dim):
                raise ValueError(''' Dimension of the added field {} : {}, do not
                                 match image dimension {}
                                 '''.format(fieldname, loc_dim, Dim))
            self.image.add_field(field, fieldname)

        # close .mat file
        matfile_data.close()

        return
