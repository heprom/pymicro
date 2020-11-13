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

import shutil
import numpy as np
import tables as Tb
from tables import dtype_from_descr

from pymicro.core.images import ImageReader
from pymicro.core.meshes import MeshReader

# Module global variables for xdmf compatibility
FIELD_TYPE = {1: 'Scalar', 2: 'Vector', 3: 'Vector', 6: 'Tensor6', 9: 'Tensor'}
CENTERS_XDMF = {'3DImage': 'Cell', 'Mesh': 'Node'}

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

    # =========================================================================
    # SampleData magic methods
    # =========================================================================
    def __init__(self,
                 filename,
                 sample_name='name_to_fill',
                 sample_description="""  """,
                 verbose=False,
                 overwrite_hdf5=False,
                 autodelete=False,
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

        self.h5_file = filename_tmp + '.h5'
        self.xdmf_file = filename_tmp + '.xdmf'
        self._verbose = verbose
        self.autodelete = autodelete
        if os.path.exists(self.h5_file) and overwrite_hdf5:
            self._verbose_print('-- File "{}" exists  and will be '
                                'overwritten'.format(self.h5_file))
            os.remove(self.h5_file)
            os.remove(self.xdmf_file)
        self.init_file_object(sample_name, sample_description, **keywords)
        self.sync()
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
        self.repack_h5file()
        self.h5_dataset.close()
        self._verbose_print('Dataset and Datafiles closed')
        if self.autodelete:
            print('{} Autodelete: \n Removing hdf5 file {} and xdmf file {}'
                  ''.format(self.__class__.__name__, self.h5_file,
                            self.xdmf_file))
            os.remove(self.h5_file)
            os.remove(self.xdmf_file)
            if (os.path.exists(self.h5_file) or os.path.exists(self.xdmf_file)):
                raise RuntimeError('HDF5 and XDMF not removed')
        return

    def __repr__(self):
        """ Return a string representation of the dataset content"""
        s = self.print_index(as_string=True)
        s += self.print_dataset_content(as_string=True)
        return s

    def __contains__(self, name):
        """ Check if inputed name/indexname/path is a HDF5 node in dataset"""
        path = self._name_or_node_to_path(name)
        if path is None:
            return False
        else:
            return (self.h5_dataset.__contains__(path)
                    and not self._is_empty(path))

    def init_file_object(self,
                         sample_name='',
                         sample_description='',
                         **keywords):
        """ Initiate PyTable file object from .h5 file or create it if
            needed
        """

        try:
            self.h5_dataset = Tb.File(self.h5_file, mode='r+')
            self._verbose_print('-- Opening file "{}" '.format(self.h5_file),
                                line_break=False)
            self.file_exist = True
            self._init_xml_tree()
            self.Filters = self.h5_dataset.filters
            self._init_data_model()
            self._verbose_print('**** FILE CONTENT ****')
            self._verbose_print(SampleData.__repr__(self))
        except IOError:
            self.file_exist = False
            self._verbose_print('-- File "{}" not found : file'
                                ' created'.format(self.h5_file),
                                line_break=True)
            self.h5_dataset = Tb.File(self.h5_file, mode='a')
            self._init_xml_tree()
            # add sample name and description
            self.h5_dataset.root._v_attrs.sample_name = sample_name
            self.h5_dataset.root._v_attrs.description = sample_description
            # get compression options
            Compression_keywords = {k: v for k, v in keywords.items() if k in
                                    compression_keys}
            self.set_global_compression_opt(**Compression_keywords)
            # Generic Data Model initialization
            self._init_data_model()
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

    def print_dataset_content(self, as_string=False):
        """ Print information on all nodes in hdf5 file"""
        s = '\n****** DATA SET CONTENT ******\n File: \n  {}'.format(
                self.h5_file)
        if not(as_string):
            print(s)
        s += self.get_node_info('/', as_string)
        s += '\n************************************************'
        if not(as_string):
            print('\n************************************************')
        for node in self.h5_dataset.root:
            if not(node._v_name == 'Index'):
                s += self.get_node_info(node._v_name, as_string)
                s += self.print_group_content(node._v_name,
                                              recursive = True,
                                              as_string = as_string)
                s += '\n************************************************'
                if not(as_string):
                    print('\n************************************************')
        return s

    def print_group_content(self, groupname,
                            recursive=False,
                            as_string=False):
        """  Print information on all nodes in hdf5 group """
        s = '\n\n****** Group {} CONTENT ******'.format(groupname)
        group = self.get_node(groupname)
        if group._v_nchildren == 0:
            return ''
        else:
            if not(as_string):
                print(s)

        for node in group._f_iter_nodes():
            s += self.get_node_info(node._v_name, as_string)
            if (self._is_group(node._v_name) and recursive):
                s += self.print_group_content(node._v_pathname,
                                         recursive=True,
                                         as_string=as_string)
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
            col = None
            if isinstance(value,list):
                path = value[0]
                col = value[1]
            else:
                path = value
            if col is None:
                s += str('\t Name : {:20}  H5_Path : {} \t\n'.format(
                         key, path))
            else:
                s += str('\t Name : {:20}  H5_Path : {}|col:{} \t\n'.format(
                         key, path,col))
            if key in self.aliases:
                s += str('\t        {} aliases -->'.format(key))
                for aliasname in self.aliases[key]:
                    s += str(' `'+aliasname+'`')
                s += '\n'
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
        self.add_attributes(dic=self.aliases,
                                     nodename='/Index/Aliases')
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
                 mesh_object=None, meshname='', indexname='', location='/',
                 description=' ', replace=False, **keywords):
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
        location = self._name_or_node_to_path(location)
        mesh_path = os.path.join(location, meshname)
        # Check mesh group existence and replacement
        if self.h5_dataset.__contains__(mesh_path):
            msg = ('(add_mesh) Mesh group {} already exists.'
                   ''.format(mesh_path))
            empty = self.get_attribute('empty',mesh_path)
            if empty:
                mesh_group = self.get_node(mesh_path)
                empty = (mesh_object is None)
            elif (not(empty) and replace):
                msg += ('--- It will be replaced by the new MeshObject content')
                self.remove_node(node_path=mesh_path, recursive=True)
                self._verbose_print(msg)
                self._verbose_print('Creating hdf5 group {} in file {}'.format(
                                     mesh_path, self.h5_file))
                mesh_group = self.add_group(path=location, groupname=meshname,
                                            indexname=indexname)
            else:
                msg += ('\n--- If you want to replace it by new MeshObject'
                        ' content, please add `replace=True` to keyword'
                        ' arguments of `add_mesh`')
                self._verbose_print(msg)
                return
        else:
            mesh_group = self.add_group(path=location, groupname=meshname,
                                        indexname=indexname)
            empty = (mesh_object is None)
        if empty:
            Attribute_dic = {'empty':True,
                             'group_type':'Mesh'}
            self.add_attributes(Attribute_dic,indexname)
            return
        # store mesh metadata as HDF5 attributes
        Attribute_dic = {'element_topology':mesh_object.element_topology[0],
                         'mesh_description':description,
                         'group_type':'Mesh',
                         'empty':False}

        self._verbose_print('Creating Nodes data set in group {} in file {}'
                            ''.format(mesh_path, self.h5_file))
        Nodes = self.h5_dataset.create_carray(where=mesh_path,
                                      name='Nodes',
                                      filters=self.Filters,
                                      obj=mesh_object.nodes,
                                      title=indexname + '_Nodes')
        Nodes_path = Nodes._v_pathname

        # safety check
        if (len(mesh_object.element_topology) > 1):
            warnings.warn('''  number of element type found : {} \n
                          add_mesh_from_file current implementation works only
                          with meshes with only one element type
                          '''.format(len(mesh_object.element_topology)))

        self._verbose_print('Creating Elements data set in group {} in file {}'
                            ''.format(location + '/' + meshname, self.h5_file))
        Elements =self.h5_dataset.create_carray(where=mesh_path,
                                      name='Elements',
                                      filters=self.Filters,
                                      obj=mesh_object.element_connectivity[0],
                                      title=indexname + '_Elements')
        Elements_path = Elements._v_pathname

        self._verbose_print('...Updating xdmf tree...', line_break=False)
        mesh_xdmf = etree.Element(_tag='Grid',
                                  Name=meshname,
                                  GridType='Uniform')

        NElements = str(mesh_object.element_connectivity[0].shape[0])
        Dim = str(mesh_object.element_connectivity[0].shape).strip(
            '(').strip(')')
        Dim = Dim.replace(',', ' ')

        topology_xdmf = etree.Element(_tag='Topology',
                                TopologyType=mesh_object.element_topology[0][0],
                                NumberOfElements=NElements)
        topology_data = etree.Element(_tag='DataItem', Format='HDF',
                                      Dimensions=Dim, NumberType='Int',
                                      Precision='64')
        topology_data.text = self.h5_file + ':' + mesh_path + '/Elements'
        # Add node DataItem as children of node Topology
        topology_xdmf.append(topology_data)

        # create Geometry element
        geometry_xdmf = etree.Element(_tag='Geometry', Type='XYZ')
        Dim = str(mesh_object.nodes.shape).strip('(').strip(')').replace(
                  ',', ' ')
        geometry_data = etree.Element(_tag='DataItem', Format='HDF',
                                      Dimensions=Dim, NumberType='Float',
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
        Attribute_dic['xdmf_path'] = el_path
        #   topology element
        el_path = self.xdmf_tree.getelementpath(topology_xdmf)
        Attribute_dic['xdmf_topology_path'] = el_path
        Topology_attribute_dic = {'xdmf_path':el_path}
        #   geometry  element
        el_path = self.xdmf_tree.getelementpath(geometry_xdmf)
        Attribute_dic['xdmf_geometry_path'] = el_path
        Geometry_attribute_dic = {'xdmf_path':el_path}

        self.add_attributes(Attribute_dic,indexname)
        self.add_attributes(Topology_attribute_dic,Elements_path)
        self.add_attributes(Geometry_attribute_dic,Nodes_path)

        # Add mesh fields if some are stored
        for field_name, field in mesh_object.fields.items():
            self.add_data_array(location=mesh_path,
                                name=field_name,
                                array=field,
                                **keywords)
        return

    def add_image(self,
                  image_object=None, imagename='', indexname='', location='/',
                  description=' ', replace=False, **keywords):
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
        location = self._name_or_node_to_path(location)
        image_path = os.path.join(location, imagename)
        # Check image group existence and replacement
        if self.h5_dataset.__contains__(image_path):
            msg = ('\n(add_image) Image group {} already exists.'
                   ''.format(image_path))
            empty = self.get_attribute('empty',image_path)
            if empty:
                image_group = self.get_node(image_path)
                empty = (image_object is None)
            elif (not(empty) and replace):
                msg += ('--- It will be replaced by the new ImageObject'
                        ' content')
                # self.remove_node(node_path=image_path, recursive=True)
                self.remove_node(name=imagename, recursive=True)
                self._verbose_print(msg)
                self._verbose_print('Creating hdf5 group {} in file {}'.format(
                                     image_path, self.h5_file))
                image_group = self.add_group(path=location,
                                             groupname=imagename,
                                             indexname=indexname)
            else:
                msg += ('\n--- If you want to replace it by new ImageObject '
                        'content, please add `replace=True` to keyword '
                        'arguments of `add_image`.')
                print(msg)
                return
        else:
            image_group = self.add_group(path=location, groupname=imagename,
                                         indexname=indexname)
            empty = (image_object is None)
        if empty:
            Attribute_dic = {'empty':True,
                             'group_type':'3DImage'}
            self.add_attributes(Attribute_dic,indexname)
            return

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

        # store image metadata as HDF5 attributes
        Attribute_dic = {'dimension':image_object.dimension,
                         'spacing':image_object.spacing,
                         'origin':image_object.origin,
                         'description':description,
                         'group_type':'3DImage',
                         'empty':False}
        # Get xdmf elements paths as HDF5 attributes
        #     image group
        el_path = self.xdmf_tree.getelementpath(image_xdmf)
        Attribute_dic['xdmf_path'] = el_path
        #     topology group
        el_path = self.xdmf_tree.getelementpath(topology_xdmf)
        Attribute_dic['xdmf_topology_path'] = el_path
        #     geometry group
        el_path = self.xdmf_tree.getelementpath(geometry_xdmf)
        Attribute_dic['xdmf_geometry_path'] = el_path
        self.add_attributes(Attribute_dic,indexname)

        # Add fields if some are stored in the image object
        for field_name,field in image_object.fields.items():
            self.add_data_array(location=image_path,
                                name=field_name,
                                array=field,
                                replace=True,
                                **keywords)
        return

    def add_group(self,
                  path, groupname, indexname='',
                  replace=False, createparents=False):
        """ Create a (hdf5) group at desired location in the data format"""

        if (indexname == ''):
            warn_msg = ('\n(add_group) indexname not provided, '
                        ' the groupname {} is used in content '
                        'index'.format(groupname))

            self._verbose_print(warn_msg)
            indexname = groupname
        self._verbose_print('Creating hdf5 group `{}` {} in file {}'.format(
            groupname, path, self.h5_file))
        group_path = os.path.join(path,groupname)
        self.add_to_index(indexname, group_path)

        try:
            Group = self.h5_dataset.create_group(where=path, name=groupname,
                                                 title=indexname,
                                                  createparents=createparents)
            self.add_attributes(dic={'group_type':'Data'},
                                nodename=Group)
            return Group
        except Tb.NodeError:
            node_path = os.path.join(path, groupname)
            if (self.h5_dataset.__contains__(node_path)):
                if replace:
                    self.remove_node(node_path,recursive=True)
                    Group = self.h5_dataset.create_group(where=path,
                                                name=groupname, title=indexname,
                                                  createparents=createparents)
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
                    self._verbose_print(warn_msg)
                return None
            else:
                raise

    def add_data_array(self, location, name, array,
                       indexname=None, chunkshape=None, createparents=False,
                       replace=False, filters=None, empty=False,
                       **keywords):
        """Add the data array at the given location in hte HDF5 data tree.

        """
        self._verbose_print('Adding array `{}` into Group `{}`'
                            ''.format(name,location))
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
            if not(self._get_node_class(location) == 'GROUP'):
                    msg = ('(add_data_array): location {} is not a Group nor '
                           'empty. Please choose an empty location or a HDF5 '
                           'Group to store data array'
                           ' flag'.format(location))
                    self._verbose_print(msg)
                    return
            # check if array location exists and remove node if asked
            array_path = os.path.join(location_path, name)
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

        # get compression options
        # keywords compression options prioritized over passed filters instances
        if (filters is None) or bool(keywords):
            Filters = self._get_local_compression_opt(**keywords)
        else:
            Filters = filters
        self._verbose_print('-- Compression Options for dataset {}'
                            ''.format(name))
        if (self.Filters.complevel > 0):
            msg_list = str(self.Filters).strip('Filters(').strip(')').split()
            for msg in msg_list:
                self._verbose_print('\t * {}'.format(msg), line_break=False)
        else:
            self._verbose_print('\t * No Compression')

        # get location type
        if (location_exists and self._is_grid(location) and not empty
           and not self._is_empty(location)):
            self._check_field_compatibility(location,array)
        # add to index
        if indexname is None:
            warn_msg = (' (add_data_array) indexname not provided, '
                        ' the array name `{}` is used as index name '
                        ''.format(name))
            self._verbose_print(warn_msg)
            indexname = name
        self.add_to_index(indexname,os.path.join(location_path,name))

        # Create dataset node to store array
        if empty:
            Node = self.h5_dataset.create_carray(where=location_path, name=name,
                                                 obj=np.array([0]))
            self.add_attributes(dic={'empty':True}, nodename=name)
        else:
            Node = self.h5_dataset.create_carray(where=location_path, name=name,
                                             filters=Filters, obj=array,
                                             chunkshape=chunkshape)
            self.add_attributes(dic={'empty':False}, nodename=name)

        if (self._is_grid(location) and not empty
                                    and not self._is_empty(location)):
            ftype = self._get_xdmf_field_type(field=array,
                                              grid_location=location_path)
            dic = {'field_type':ftype}
            self.add_attributes(dic=dic, nodename=name)
            self._add_field_to_xdmf(name,array)
        return Node

    def add_table(self, location, name, description,
                  indexname=None, chunkshape=None, replace=False,
                  createparents=False, data=None,
                  filters=None, **keywords):
        """ Add a structured array dataset (tables.Table) in HDF5 tree"""

        self._verbose_print('Adding table `{}` into Group `{}`'
                            ''.format(name,location))
        # get location path
        location_path = self._name_or_node_to_path(location)
        if (location_path is None) and not(createparents):
            msg = ('(add_table): location {} does not exist, table'
                   ' cannot be added. Use optional argument'
                   ' "createparents=True" to force location Group creation'
                   ''.format(location))
            self._verbose_print(msg)
            return
        else:
            # check location nature
            if not(self._get_node_class(location) == 'GROUP') :
                    msg = ('(add_table): location {} is not a Group nor '
                           'empty. Please choose an empty location or a HDF5 '
                           'Group to store table'.format(location))
                    self._verbose_print(msg)
                    return
            # check if array location exists and remove node if asked
            table_path = location_path + name
            if self.h5_dataset.__contains__(table_path):
                if replace:
                    msg = ('(add_table): existing node {} will be '
                           'overwritten and all of its childrens removed'
                           ''.format(table_path))
                    self._verbose_print(msg)
                    self.remove_node(table_path, recursive=True)
                else:
                    msg = ('(add_table): node {} already exists. To '
                           'overwrite, use optional argument "replace=True"'
                           ''.format(table_path))
                    self._verbose_print(msg)

        # get compression options
        # keywords compression options prioritized over passed filters instances
        if (filters is None) or bool(keywords):
            Filters = self._get_local_compression_opt(**keywords)
            self._verbose_print('-- Compression Options for dataset {}'
                                ''.format(name))
        else:
            Filters = filters
        self._verbose_print('-- Compression Options for dataset {}'
                            ''.format(name))
        if (self.Filters.complevel > 0):
            msg_list = str(self.Filters).strip('Filters(').strip(')').split()
            for msg in msg_list:
                self._verbose_print('\t * {}'.format(msg), line_break=False)
        else:
            self._verbose_print('\t * No Compression')

        table = self.h5_dataset.create_table(where=location_path,name=name,
                                     description=description,
                                     filters=Filters,chunkshape=chunkshape,
                                     createparents=createparents)

        if data is not None:
            table.append(data)
            table.flush()

        # add to index
        if indexname is None:
            warn_msg = (' (add_table) indexname not provided, '
                        ' the table name `{}` is used as index name '
                        ''.format(name))
            self._verbose_print(warn_msg)
            indexname = name
        self.add_to_index(indexname,table._v_pathname)
        return table

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

    def add_alias(self, aliasname, path=None, indexname = None):
        """Add alias name to the input path"""
        if (path is None) and (indexname is None):
            msg = ('(add_alias) None path nor indexname inputed. Alias'
                   'addition aborted')
            self._verbose_print(msg)
            return
        Is_present = (self._is_in_index(aliasname)
                      and self._is_alias(aliasname))
        if Is_present:
            msg =('Alias`{}` already exists : duplicates not allowed'
                  ''.format(aliasname))
            self._verbose_print(msg)
        else:
            if (indexname is None):
                indexname = self.get_indexname_from_path(path)
            if indexname in self.aliases:
                self.aliases[indexname].append(aliasname)
            else:
                self.aliases[indexname] = [aliasname]
        return

    def add_to_index(self, indexname, path, colname = None):
        """ Add path to index if indexname is not already in content_index"""
        Is_present = (self._is_in_index(indexname)
                      and self._is_alias(indexname))
        if Is_present:
            raise ValueError('Name `{}` already in '
                             'content_index : duplicates not allowed in Index'
                             ''.format(indexname))
        else:
            for item in self.content_index:
                if isinstance(self.content_index[item],list):
                    index_path = self.content_index[item][0]
                    index_colname = self.content_index[item][1]
                else:
                    index_path = self.content_index[item]
                    index_colname = None
                if (path == index_path) and (colname is index_colname):
                    msg = (' (add_to_index) indexname provided for a path ({})'
                           'that already in index --> stored as alias name.'
                           ''.format(path))
                    self._verbose_print(msg)
                    self.add_alias(aliasname=indexname, indexname=item)
                    return
            if colname is None:
                self.content_index[indexname] = path
            else:
                self.content_index[indexname] = [path, colname]
        return

    def get_indexname_from_path(self, node_path):
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

    def get_tablecol(self, tablename, colname):
        """ Returns a column of a Pytables.Table as a numpy array"""

        data = None
        data_path = self._name_or_node_to_path(tablename)
        if data_path is None:
            msg = ('(get_tablecol) `tablename` not matched with a path or an'
                   ' indexname. No data returned')
            self._verbose_print(msg)
        else:
            if self._is_table(name=data_path):
                msg = '(get_tablecol) Getting column {} from : {}:{}'.format(
                        colname,self.h5_file, data_path)
                self._verbose_print(msg)
                table = self.get_node(name=data_path)
                data = table.col(colname)
            else:
                msg = ('(get_tablecol) Data is not an table node.')
                self._verbose_print(msg)
        return data

    def get_node(self, name, as_numpy=False):
        """ get a Node object from the HDF5 data tree from indexname or path.
        """
        node = None
        colname = None
        node_path = self._name_or_node_to_path(name)
        if node_path is None:
            msg = ('(get_node) ERROR : Node name does not fit any hdf5 path'
                   ' nor index name.')
            self._verbose_print(msg)
        else:
            if self._is_table(node_path):
                node = self.h5_dataset.get_node(node_path)
                if name in self.content_index:
                    if isinstance(self.content_index[name],list):
                        colname = self.content_index[name][1]
                elif name in node.colnames:
                   colname = name
                if (colname is not None): node = node.col(colname)
            else:
                node = self.h5_dataset.get_node(node_path)
            if as_numpy and self._is_array(name) and (colname is None):
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
            private method used to get HDF5 Attributes of a node as a dic

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
            try:
                attribute = self.h5_dataset.get_node_attr(where=data_path,
                                                          attrname=attrname)
            except AttributeError:
                self._verbose_print(' (get_attribute) node {} has no attribute'
                                    ' `empty`'.format(node_name))
                return None
            if isinstance(attribute,bytes):
                attribute = attribute.decode()
        return attribute

    def get_file_disk_size(self):
        units = ['bytes','Kb','Mb','Gb','Tb','Pb']
        fsize = os.path.getsize(self.h5_file)
        k=0
        unit = units[k]
        while fsize/1024 > 1:
            k = k+1
            fsize = fsize/1024
            unit = units[k]
        print('File size is {:4.3f} {} for file \n {}'.format(fsize,unit,
                                                       self.h5_file))
        return fsize, unit

    def set_sample_name(self, sample_name):
        self.add_attributes({'sample_name':sample_name},'/')

    def set_description(self, node, description):
        """ """
        self.add_attributes({'description':description},node)
        return

    def set_voxel_size(self, image_data_group, voxel_size):
        """ Set voxel size for an HDF5 image data group"""
        self.add_attributes({'spacing':voxel_size},image_data_group)
        xdmf_geometry_path = self.get_attribute('xdmf_geometry_path',
                                                image_data_group)
        xdmf_geometry = self.xdmf_tree.find(xdmf_geometry_path)
        spacing_node = xdmf_geometry.getchildren()[1]
        spacing_text = str(voxel_size).strip('[').strip(']').replace(',', ' ')
        spacing_node.text = spacing_text
        self.sync()
        return

    def set_origin(self, image_data_group, origin):
        """ Set voxel size for an HDF5 image data group"""
        self.add_attributes({'origin':origin},image_data_group)
        xdmf_geometry_path = self.get_attribute('xdmf_geometry_path',
                                                image_data_group)
        xdmf_geometry = self.xdmf_tree.find(xdmf_geometry_path)
        origin_node = xdmf_geometry.getchildren()[0]
        origin_text = str(origin).strip('[').strip(']').replace(',', ' ')
        origin_node.text = origin_text
        self.sync()
        return

    def set_chunkshape_and_compression(self, node, chunkshape=None,
                                       filters=None, **keywords):
        """ Changes the chunkshape for a HDF5 array node

            Removes the array and recreate it with a different chunkshape

            Note:
                Disk space of HDF5 will not be reduced until `repack_h5`
                method is used, or `ptrepack` executable utility is used on
                h5_file.
                This is default behavior of hdf5 files
        """
        if not self._is_array(node):
            msg = ('(set_chunkshape) Cannot set chunkshape or compression'
                   ' settings for a non array node')
            raise ValueError(msg)
        node_tmp = self.get_node(node)
        node_name = node_tmp._v_name
        node_indexname = self.get_indexname_from_path(node_tmp._v_pathname)
        node_path = os.path.dirname(node_tmp._v_pathname)
        node_chunkshape = node_tmp.chunkshape
        if chunkshape is not None:
            node_chunkshape = chunkshape
        if filters is None:
            node_filters = node_tmp.filters
        else:
            node_filters = filters
        if self.aliases.__contains__(node_indexname):
            node_aliases = self.aliases[node_indexname]
        else:
            node_aliases = []
        array = node_tmp.read()
        if self._is_table(node):
            description = node.description
            new_array = self.add_table(location=node_path, name=node_name,
                                     description=description,
                                     indexname=node_indexname,
                                     chunkshape=node_chunkshape, replace=True,
                                     data=array, filters=node_filters,
                                     **keywords)
        else:
            new_array = self.add_data_array(location=node_path, name=node_name,
                                            indexname=node_indexname,
                                            array=array, filters=node_filters,
                                            chunkshape=node_chunkshape,
                                            replace=True, **keywords)
        for alias in node_aliases:
            self.add_alias(aliasname=alias, indexname=node_indexname)
        return

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
            self.Filters = self._set_default_compression()
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

            Note:
                Disk space of HDF5 will not be reduced until `repack_h5`
                method is used, or `ptrepack` executable utility is used on
                h5_file.
                This is default behavior of hdf5 files
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
            xdmf_path = self.get_attribute('xdmf_path',Node)
            if xdmf_path is not None:
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
            self.sync()
            self._verbose_print('Node {} sucessfully removed'.format(name))
        return

    def repack_h5file(self):
        """ Overwrite hdf5 file with a copy of itself to recover disk space

            Manipulation to recover space leaved empty when removing data from
            the HDF5 tree.
        """
        head, tail = os.path.split(self.h5_file)
        tmp_file = os.path.join(head,'tmp_'+tail)
        self.h5_dataset.copy_file(tmp_file)
        shutil.move(tmp_file,self.h5_file)
        return

    @staticmethod
    def copy_sample(src_sample_file, dst_sample_file, overwrite=False,
                    get_object=False, new_sample_name=None, autodelete=False):
        """ Initiate a new SampleData object and files from existing one"""
        sample = SampleData(filename=src_sample_file)
        if new_sample_name is None:
            new_sample_name = sample.get_attribute('sample_name','/')
        # copy HDF5 file
        dst_sample_file_h5 = os.path.splitext(dst_sample_file)[0] + '.h5'
        dst_sample_file_xdmf = os.path.splitext(dst_sample_file)[0] + '.xdmf'
        sample.h5_dataset.copy_file(dst_sample_file_h5, overwrite=overwrite)
        # copy XDMF file
        dst_xdmf_lines = []
        with open(sample.xdmf_file,'r') as f:
            src_xdmf_lines = f.readlines()
        for line in src_xdmf_lines:
            dst_xdmf_lines.append(line.replace(sample.h5_file,
                                               dst_sample_file_h5))
        with open(dst_sample_file_xdmf,'w') as f:
            f.writelines(dst_xdmf_lines)
        del sample
        new_sample = SampleData(filename=dst_sample_file_h5,
                                autodelete=autodelete)
        new_sample.set_sample_name(new_sample_name)
        if get_object:
            return new_sample
        else:
            del new_sample
            return

    # =========================================================================
    #  SampleData private utilities
    # =========================================================================

    def _minimal_data_model(self):
        """
            Specify the minimal contents of the hdf5 (Node names, organization)
            in the form of a dictionary {content:Location}

            This method is central to defien derived classes from SampleData
            that are associated with a minimal content and organization
            for the hdf5 file storing the data.

            This dictionary is used to assess the state of completion of the
            datafile, and serves as indication to inform the minimal content
            that is required to have a usefull dataset for the physical purpose
            for which the subclass is designed

        """
        index_dic = {}
        type_dic = {}
        return index_dic, type_dic

    def _init_data_model(self):
        """ Initialization of the minimal data model specified for the class"""
        content_paths, content_type = self._minimal_data_model()
        self._init_content_index(content_paths)
        self._verbose_print('Minimal data model initialization....')
        for key,value in content_paths.items():
            head, tail = os.path.split(value)
            if self.h5_dataset.__contains__(content_paths[key]):
                if self._is_table(content_paths[key]):
                   self._update_table_columns(tablename=content_paths[key],
                                              Description=content_type[key])
                if self._is_empty(content_paths[key]):
                    self._verbose_print('Warning: node {} specified in the'
                                        'minimal data model for this class'
                                        'is empty'.format(content_paths[key]))
                continue
            elif content_type[key] == 'Group':
                self.add_group(path=head, groupname=tail, indexname=key,
                               replace=False,createparents=True)
            elif content_type[key] == '3DImage':
                self.add_image(imagename=tail, indexname=key, location=head)
            elif content_type[key] == 'Mesh':
                self.add_mesh(meshname=tail, indexname=key, location=head)
            elif content_type[key] == 'Array':
                empty_array = np.array([0])
                self.add_data_array(location=head, name=tail,
                                    array=empty_array, empty=True,
                                    indexname=key)
            elif (isinstance(content_type[key],Tb.IsDescription)
                  or issubclass(content_type[key],Tb.IsDescription) ):
                self.add_table(location=head,name=tail,indexname=key,
                               description=content_type[key])
        self._verbose_print('Minimal data model initialization done\n')
        return

    def _init_xml_tree(self):
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

    def _init_content_index(self, path_dic):
        """
            Initialize SampleData dictionary attribute that stores the name and
            the location of dataset/group/attribute stored in the h5 file.
            The dic. is synchronized with the hdf5 Group '/Index' from Root.
            Its items are stored there as HDF5 attributes when the SampleData
            object is destroyed.

            This dic. is used by other methods to easily access/Add/remove/
            modify nodes et contents in the h5 file/Pytables tree structure
        """

        self.minimal_content = {key: '' for key in path_dic}
        self.content_index = {}
        self.aliases = {}

        if self.file_exist:
            self.content_index = self.get_dic_from_attributes(
                                                    node_path='/Index')
            self.aliases = self.get_dic_from_attributes(
                                                    node_path='/Index/Aliases')
        else:
            self.h5_dataset.create_group('/', name='Index')
            self.add_attributes(dic=self.minimal_content, nodename='/Index')
            self.content_index = self.get_dic_from_attributes(
                    node_path='/Index')
            self.h5_dataset.create_group('/Index', name='Aliases')
        return

    def _compatible_descriptions(self, desc_items1, desc_items2 ):
        """ """
        if not(desc_items1.keys() <= desc_items2.keys()):
            return False
        else:
            for key in desc_items1:
                kind_comp = (desc_items1[key].kind == desc_items2[key].kind )
                shape_comp = (desc_items1[key].shape == desc_items2[key].shape)
                if not(kind_comp and shape_comp):
                    return False
            return True



    def _update_table_columns(self, tablename, Description):
        """ Update table if associated table description Class has evolved"""
        table = self.get_node(tablename)
        current_desc = table.description._v_colobjects
        desc_dtype = dtype_from_descr(Description)
        # Check if current table description is contained or equal to
        # Class defined table description
        compatibility = self._compatible_descriptions(current_desc,
                                                      Description.columns)
        if not(compatibility):
            msg = ('Table `{}` has a Description (column specification) '
                   'that does not match the Class `{}` Description `{}` for '
                   'this table\n'.format(tablename, self.__class__.__name__,
                                   Description.__name__))
            msg += ('Current table description: \n {} \nClass table description'
                    ': \n {}'.format(current_desc.items,
                                     Description.columns))
            raise ValueError(msg)
        elif compatibility and (current_desc.keys()
                                 < Description.columns.keys()):
            msg = ('Updating `{}` with current class Table description'.format(
                    tablename))
            self._verbose_print(msg)
            Nrows = table.nrows
            tab_name = table._v_name
            tab_indexname = self.get_indexname_from_path(table._v_pathname)
            tab_path = os.path.dirname(table._v_pathname)
            tab_chunkshape = table.chunkshape
            tab_filters = table.filters
            data = np.array(np.zeros((Nrows,)), dtype=desc_dtype)
            for key in current_desc:
                data[key] = self.get_tablecol(tablename=tablename,
                                                        colname=key)
            if self.aliases.__contains__(tab_indexname):
                tab_aliases = self.aliases[tab_indexname]
            else:
                tab_aliases = []
            self.remove_node(tab_name)
            new_tab = self.add_table(location=tab_path, name=tab_name,
                                     description=Description,
                                     indexname=tab_indexname,
                                     chunkshape=tab_chunkshape, replace=True,
                                     data=data, filters=tab_filters)
            for alias in tab_aliases:
                self.add_alias(aliasname=alias, indexname=tab_indexname)
        return

    def _remove_from_index(self,
                           node_path):
        """Remove a hdf5 node from content index dictionary"""
        try:
            key = self.get_indexname_from_path(node_path)
            removed_path = self.content_index.pop(key)
            if key in self.aliases:
                self.aliases.pop(key)
            self._verbose_print('item {} : {} removed from context index'
                                ' dictionary'.format(key, removed_path))
        except ValueError:
            self._verbose_print('node {} not found in content index values for'
                                'removal'.format(node_path))
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
        if self._is_alias(name_or_node):
            # name is a an alias for a node path in content_index
            temp_name = self._is_alias_for(name_or_node)
            path = self._get_path_with_indexname(temp_name)
        # if not found with indexname or is not a path
        # find nodes or table column with this name.
        # If several nodes have this name return warn message and return None
        if path is None:
            count = 0
            path_list = []
            for node in self.h5_dataset:
                if (name_or_node == node._v_name):
                    count = count +1
                    path_list.append(node._v_pathname)
                if node._v_attrs.CLASS == 'TABLE':
                    if name_or_node in node.colnames:
                        count = count + 1
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

    def _is_empty(self, name):
        """ find out if name or path references an empty node"""
        name2 = self._name_or_node_to_path(name)
        if self._is_table(name):
            tab = self.get_node(name2)
            return tab.nrows > 0
        else:
            return self.get_attribute('empty',name2)

    def _is_array(self, name):
        """ find out if name or path references an array dataset"""
        name2 = self._name_or_node_to_path(name)
        Class = self._get_node_class(name2)
        List = ['CARRAY', 'EARRAY', 'VLARRAY', 'ARRAY', 'TABLE']
        if (Class in List):
            return True
        else:
            return False

    def _is_table(self, name):
        """ find out if name or path references an array dataset"""
        if (self._get_node_class(name) == 'TABLE'):
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
        List = ['Mesh','3DImage']
        if (gtype in List):
            return True
        else:
            return False

    def _is_in_index(self,name):
        return (name in self.content_index)

    def _is_alias(self,name):
        """ Chekc if name is an HDF5 node alias"""
        Is_alias = False
        for item in self.aliases:
            if (name in self.aliases[item]):
                Is_alias = True
                break
        return Is_alias

    def _is_alias_for(self,name):
        """ Returns the indexname for which input name is an alias"""
        Indexname = None
        for item in self.aliases:
            if (name in self.aliases[item]):
                Indexname = item
                break
        return Indexname

    def _check_field_compatibility(self, name, field):
        """ check if field dimension is compatible with the storing location"""
        group_type = self._get_group_type(name)
        msg = ''
        if group_type == '3DImage':
            compatibility = self._compatibility_with_image(name, field)
            if self._is_empty(name):
                msg += ('{} is an empty image. Use add_image to initialize'
                        ''.format(name))
        elif group_type == 'Mesh':
            compatibility = self._compatibility_with_mesh(name,field)
            if self._is_empty(name):
                msg += ('{} is an empty mesh. Use add_mesh to initialize'
                        ''.format(name))
        else:
            msg = ('location {} is not a grid.'.format(name))
            raise ValueError(msg)

        if not(compatibility):
            msg = ('Array dimensions not compatible with {} `{}`'
                   ''.format(group_type,name))
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
        if Type == 'Mesh':
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
            if isinstance(self.content_index[indexname],list):
                return self.content_index[indexname][0]
            else:
                return self.content_index[indexname]
        else:
            raise ValueError('Index contains no item named {}'.format(
                indexname))
            return

    def _get_group_type(self, groupname):
        """  """
        if groupname == '/':
            return 'GROUP'
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
        s += str('\n----------------')
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
        if self._is_table(nodename):
            s += ' -- table description : \n'
            s += repr(Node.description)+'\n'
        s += str('----------------')
        if not(as_string):
            print(s)
            s = ''
        return s

    def _get_compression_opt(self, **keywords):
        """ Get compression options in keywords and return a Filters instance

            See PyTables documentation of Filters class for keywords and use
            details
        """
        Filters = self._set_default_compression()
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
        if 'default' in keywords:
            if keywords['default']:
                Filters = self._set_default_compression()
            else:
                Filters = self.Filters
        else:
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

    def _set_default_compression(self):
        """ Returns a Filter object with defaut compression parameters """
        Filters = Tb.Filters(complib='zlib', complevel=0, shuffle=True)
        return Filters

    def _verbose_print(self, message, line_break=True):
        """ Print message if verbose flag True"""
        Msg = message
        if line_break: Msg = ('\n' + Msg)
        if self._verbose: print(Msg)
        return
