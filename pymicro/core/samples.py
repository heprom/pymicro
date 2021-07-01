#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DataPlatform module for the management of multimodal mechanical datasets.

The `samples` module provides a base class `SampleData` implementing a generic
data model and data paltform API, allowing to define by inheritance specific
data platform classes for specific mechanical applications. It is designed to
gather and manipulate easily multimodal datasets gathering data from 3D/4D
imaging experiments and simulations of mechanical material samples. Such data
consists in volumic data defined on 3D meshes/images, classical data arrays,
and associated metadata.

"""
import os
import subprocess
import shutil
import numpy as np
import tables
import lxml.builder
from lxml import etree
from pathlib import Path
# BasicTool imports
from BasicTools.Containers.ConstantRectilinearMesh import (
    ConstantRectilinearMesh)
from BasicTools.Containers.UnstructuredMesh import (UnstructuredMesh,
                                                    AllElements)
import BasicTools.Containers.UnstructuredMeshCreationTools as UMCT
from BasicTools.Containers.MeshBase import MeshBase
from BasicTools.IO.XdmfTools import XdmfName,XdmfNumber
# Import variables for XDMF binding
from pymicro.core.global_variables import (XDMF_FIELD_TYPE,
                                           XDMF_IMAGE_GEOMETRY,
                                           XDMF_IMAGE_TOPOLOGY)
# Import variables for SampleData data model
from pymicro.core.global_variables import (SD_GROUP_TYPES, SD_GRID_GROUPS,
                                           SD_IMAGE_GROUPS, SD_MESH_GROUPS)
# Import variables for SampleData utilities
from pymicro.core.global_variables import (COMPRESSION_KEYS)


class SampleData:
    """Base class to store organized multi-modal datasets for material science.

    SampleData is a high level API to add/modifiy/remove data into a HDF5 data
    file accordingly to a user defined data model, including volumic data
    (data organized on a geometrical grid) and classical data arrays. The class
    ensures creation and synchronization of a XDMF file with the HDF5 dataset
    to allow visualization of volumic data with the Paraview software.
    The various types of data items that can be handled by SampleData, as well
    as the SampleData constructor parameters and attributes are documented
    below.

    The HDF5 and XDMF data tree structure and content in both files are
    accessible through the `h5_dataset` and `xdmf_tree` class attributes,
    that are respectively instances of classes imported from the
    `Pytables <https://www.pytables.org/index.html>`_ and
    `lxml <https://lxml.de/>`_ packages.

    .. note:: The SampleData class relies on the
        `Pytables <https://www.pytables.org/index.html>`_ package, on the
        **HDF5** and **XDMF** file formats. A priori knowledge of these
        elements is not mandatory to use SampleData but is strongly
        recommanded.

    - **SampleData datasets can be composed of three types of HDF5 groups:**

        :Group: Classical HDF5 Group node, used to organize data arrays.
        :3DImage: A Group to store data arrays representing fields defined on
            a 3D image (voxelized fields).
        :2DImage: A Group to store data arrays representing fields defined on
            a 2D image (pixelized fields).
        :Mesh: A Group to store data arrays representing fields defined on
            a mesh, and store mesh geometry.

    .. seealso::
        See :func:`add_group`, :func:`add_image` and :func:`add_mesh`.

    `3DImage`, `2DImage` and `Mesh` group types are synchronized with the XDMF
    file to allow visualization of their content with Paraview

    - **SampleData datasets can be composed of three types of data items:**

    :arrays: classical HDF5 node containing a data array.
    :tables: structured array class, analogous to numpy.void data arrays,
        imported from :py:class:`tables.Filters`.
    :attributes: classical HDF5 attributes, to store metadata.

    .. seealso::
        See :func:`add_data_array`, :func:`add_table` and
        :func:`add_attributes`.

    .. rubric:: INDEX NAMES AND ALIASES

    An index of the dataset content is stored into a dictionary `content_index`
    as an attribute of the class. Additional names can be defined by users
    for data items, and are stored in an alias dictionary. The `content_index`
    dic is synchronized with the hdf5 Group '/Index'.
    The `aliases` dic is synchronized with the '/Index/Aliases' Group.
    When an existing dataset is opened to create a SampleData instance, these
    attributes are initialized from these Groups in the dataset HDF5 file.
    Each data item can be accessed in the API methods via:

        :path: it's path in the HDF5 data tree
        :indexname: it's name in the `content_index` dic (the key associated
            to it's path)
        :alias: an alias of it's indexname
        :name: its name as a HDF5 Node if it is unique

    .. seealso::
        | See :func:`get_node` for more details on this mechanism.
        | See :func:`print_index` to visualize Index names and Aliases.

    .. rubric:: DATA COMPRESSION

    HDF5 compression algorithm are available through the
    `Pytables <https://www.pytables.org/index.html>`_ package. SampleData
    offers an interface to it with the :func:`set_chunkshape_and_compression`
    method.

    .. rubric:: CONSTRUCTOR PARAMETERS

    :filename: `str`
        name of HDF5/XDMF files to create/read
    :sample_name: `str`, optional ('')
        name of the sample associated to data
    :sample_description: `str`, optional ('')
        short description of the mechanical sample (material, type of
        tests....)
    :verbose: `bool`, optional (False)
        set verbosity flag
    :overwrite_hdf5: `bool`, optional (False)
        set to `True` to overwrite existing HDF5/XDMF couple of files
    :autodelete: `bool`, optional (False)
        set to `True` to remove HDF5/XDMF files when deleting SampleData
        instance

    .. rubric:: Examples

    ::

        # create 'my_dataset.h5' and 'my_dataset.xdmf' files to store data
        sample = SampleData(filename='my_dataset',sample_name='my_sample_name')
        # create a temporary dataset 'tmp_dataset.h5/xdmf', to be removed when
        # sample instance is destroyed
        sample = SampleData(filename='tmp_dataset.h5/xdmf', autodelete=True)
        del sample # files are removed
        # create a dataset and overwrites 'my_dataset.h5' and 'my_dataset.xdmf'
        # files
        sample = SampleData(filename='my_dataset', overwrite_hdf5=True)

    .. note:: additional keywords arguments can be passed to specify global
            compression options, see :func:`set_chunkshape_and_compression`
            documentation for their definition

    .. rubric:: CLASS ATTRIBUTES

    :h5_file: name of HDF5 file containing dataset (`str`)
    :h5_dataset: :py:class:`tables.File` instance associated to the
        `h5_file`
    :xdmf_file: name of XDMF file associated with `h5_file` (`str`)
    :xdmf_tree: :py:class:`lxml.etree` XML tree associated with `xdmf_file`
    :autodelete: autodelete flag (`bool`)
    :Filters: instance of :py:class:`tables.Filters` specifying
        general compression options
    :content_index: Dictionnary of data items (nodes/groups)
        names and pathes in HDF5 dataset (`dic`)
    :aliases: Dictionnary of list of aliases for each item in
        content_index (`dic`)

    .. important:: **Inheritance -- Data Platforms construction for specific
        Applications**

        SampleData provides a mechanism to derive classes with a specific data
        model, by redefining the method :func:`minimal_data_model` in the
        derived class. This method defines two dictionaris specifying a list of
        data item/group names, pathes, and data item types constituting the
        data model. Each instance of this derived class will automatically
        create the data model elements in any new dataset, or ad them into a
        pre-existing dataset. The data items created in this way are empty
        until actual data are added at these locations.

        This mechanisms ensures retro-compatibility. When the data model of a
        class is enriched, any old dataset opened with the class will be
        udpated with ne new elements of the data model.

        This usage is the original purpose of the SampleData class. It allows
        to implement data platforms dedicated to a specific usage by
        subclassing SampleData. This process can be breakdown into the
        following steps:

            #. Create a `class` inherited from SampleData
            #. Define the class data model and implement it in
               `minimal_data_model` method
            #. If needed, develop dedicated methods to offer a specific API to
               support the derived class practical application

        When creating a derived Class from SampleData, users can defined their
        own default compression settings by overwritting the method
        :func:`set_default_compression`.

        | See documentation of :func:`minimal_data_model` for further details
        | To see examples of such derived classes see:
        | - the :py:class:`pymicro.crystal.microstructure`
        | - the :py:class:`Test_DerivedClass` used in the `samples` module
    """

    def __init__(self, filename='sample_data', sample_name='',
                 sample_description=' ', verbose=False, overwrite_hdf5=False,
                 autodelete=False, **keywords):
        """Sample Data constructor."""
        # get file directory and file name
        path_file = Path(filename).absolute()
        filename_tmp = path_file.stem
        file_dir = str(path_file.parent)

        self.h5_file = filename_tmp + '.h5'
        self.xdmf_file = filename_tmp + '.xdmf'
        self.file_dir = file_dir
        self.h5_path = os.path.join(self.file_dir,self.h5_file)
        self.xdmf_path = os.path.join(self.file_dir,self.xdmf_file)
        self._verbose = verbose
        self.autodelete = autodelete
        if os.path.exists(self.h5_path) and overwrite_hdf5:
            self._verbose_print('-- File "{}" exists  and will be '
                                'overwritten'.format(self.h5_path))
            os.remove(self.h5_path)
            os.remove(self.xdmf_path)
        self._init_file_object(sample_name, sample_description, **keywords)
        self.sync()
        return

    def __del__(self):
        """Sample Data destructor.

        Deletes SampleData instance and:
              - closes h5_file --> writes data structure into the .h5 file
              - writes the .xdmf file
        """
        self._verbose_print('Deleting DataSample object ')
        self.sync()
        # self.repack_h5file()
        self.h5_dataset.close()
        self._verbose_print('Dataset and Datafiles closed')
        if self.autodelete:
            print('{} Autodelete: \n Removing hdf5 file {} and xdmf file {}'
                  ''.format(self.__class__.__name__, self.h5_file,
                            self.xdmf_file))
            os.remove(self.h5_path)
            os.remove(self.xdmf_path)
            if os.path.exists(self.h5_path) or os.path.exists(self.xdmf_path):
                raise RuntimeError('HDF5 and XDMF not removed')
        return

    def __repr__(self):
        """Return a string representation of the dataset content."""
        s = ''
        s += self.print_index(as_string=True, max_depth=3)
        s += self.print_dataset_content(as_string=True, max_depth=3)
        return s

    def __contains__(self, name):
        """Check if name refers to an existing HDF5 node in the dataset.

        :param str name: a string for the name / indexname / path
        :return bool: True if the dataset has a node associated with this name,
            False if not.
        """
        path = self._name_or_node_to_path(name)
        if path is None:
            return False
        else:
            return self.h5_dataset.__contains__(path)

    def minimal_data_model(self):
        """Specify minimal data model to store in class instance.

        This method is designed to construct derived classes from SampleData
        to serve as data platforms for a specific data model, that is specified
        by the two dictionaries returned.

        The class constructor searches through these dictionaries to determine
        the name, pathes and types of data items constituting the data model of
        the class, and creates them at each instance creation. If the
        constructor is used to create an instance from an existing file, the
        compatibility with the data model is verified, and the missing data
        items are created if needed. The constructor ensures compatibility of
        previously created datasets with the class if the current data model
        of the class has been enriched. Note that the associated datasets can
        contain additional data items, those defined in this method are the
        minimal required content for this class of datasets.

        The return dictionaries keys are the Index names of the data items of
        the data model. Their values are defined hereafter:

        :return dic index_dic:
            | Dictionary specifying the indexnames and pathes of data items in
            | the data model. Each entry must be of the form
            | {'indexname':'/data_item/path'}. The data item path is its path
            | in the HDF5 tree structure.
        :return dic type_dic:
            | Dictionary specifying the indexnames and types of data items in
            | the data model. Each entry must be of the form
            | {'indexname':'grouptype'}. 'grouptype' can be:

        :'Group': creates a classical HDF5 group (str)
        :'3DImage': creates a HDF5 group containing datasets (fields) defined
            on the same 3D image (str)
        :'Mesh': creates a HDF5 group containing datasets (fields) defined on
            the same mesh (str)
        :'Array': creates a HDF5 node containing a data array (str)
        :Table_description: creates a structured storage array
            (:py:class:`tables.Filters` class) with the given Description.
            `Table_description` must be a subclass of
            :py:class:`tables.IsDescription` class.

        .. rubric:: Example

        ::

            class MyDesc(IsDescription):
                int_data    = Int32Col()
                float_data  = Float32Col()
                float_array = Float32Col(shape=(3,))

            class MyDerivedClass(SampleData):
                def _minimal_data_model(self):
                # MyDerivedClass will handle datasets containing at least
                # a set of data associated to a 3D image, one associated to
                # a mesh, a classical group of data containing one Array, and a
                # node containing a structured storage table whose lines are
                # comprised of one scalar integer, one scalar float and one
                # float vector (shape 3,)
                    minimal_content_index_dic = {'3DImage':'/ImagePath',
                                                 'DataGroup':'/GroupPath',
                                                 'MeshGroup':'/MeshPath',
                                                 'DataArray':'/GroupPath/Data',
                                                 'DataTable':'/GroupPath/Tab'}
                    minimal_content_type_dic = {'3DImage':'3DImage',
                                                'DataGroup':'Group',
                                                'MeshGroup':'Mesh',
                                                'DataArray':'Array',
                                                'DataTable':MyDesc}
        """
        index_dic = {}
        type_dic = {}
        return index_dic, type_dic

    def print_xdmf(self):
        """Print a readable version of xdmf_tree content."""
        print(etree.tostring(self.xdmf_tree, pretty_print=True,
                             encoding='unicode'))
        return

    def write_xdmf(self):
        """Write xdmf_tree in .xdmf file with suitable XML declaration."""
        self._verbose_print('.... writing xdmf file : {}'
                            ''.format(self.xdmf_file),
                            line_break=False)
        self.xdmf_tree.write(self.xdmf_path,
                             xml_declaration=True,
                             pretty_print=True,
                             doctype='<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd"[]>')

        # correct xml declaration to allow Paraview reader compatibility
        with open(self.xdmf_path, 'r') as f:
            lines = f.readlines()

        lines[0] = lines[0].replace("encoding='ASCII'", "")

        with open(self.xdmf_path, 'w') as f:
            f.writelines(lines)

        return

    def print_dataset_content(self, as_string=False, max_depth=3):
        """Print information on all nodes in the HDF5 file.

        :param bool as_string: If `True` solely returns string representation.
            If `False`, prints the string representation.
        :param int max_depth: Control the maximum depth of the node/groups
            informations that are printed. Depth is the number of parents that
            the node has to root group, including root group. For instance,
            max_depth=2 will print info on the root group children and their
            childrens.
        :return str s: string representation of HDF5 nodes information
        """
        size, unit = self.get_file_disk_size(print_flag=False)
        s = ('\n****** DATA SET CONTENT ******\n -- File: {}\n '
             '-- Size: {:9.3f} {}\n -- Data Model Class: {}\n'
             ''.format(self.h5_file, size, unit, self.__class__.__name__))
        if not(as_string):
            print(s)
        s += self.get_node_info('/', as_string)
        s += '\n************************************************'
        if not(as_string):
            print('\n************************************************')
        for node in self.h5_dataset.root:
            if node._v_depth > max_depth:
                continue
            if not(node._v_name == 'Index'):
                s += self.get_node_info(node._v_pathname, as_string)
                if self._is_group(node._v_pathname):
                    s += self.print_group_content(node._v_pathname,
                                                  recursive=True,
                                                  as_string=as_string,
                                                  max_depth=max_depth)
                s += '\n************************************************'
                if not(as_string):
                    print('\n************************************************')
        return s

    def print_group_content(self, groupname, recursive=False, as_string=False,
                            max_depth=1000):
        """Print information on all nodes in a HDF5 group.

        :param str groupname: Name, Path, Index name or Alias of the HDF5 group
        :param bool recursive: If `True`, print content of children groups
        :param bool as_string: If `True` solely returns string representation.
            If `False`, prints the string representation.
        :return str s: string representation of HDF5 nodes information
        """
        s = '\n\n****** Group {} CONTENT ******'.format(groupname)
        group = self.get_node(groupname)
        if group._v_depth > max_depth:
            return ''
        if group._v_nchildren == 0:
            return ''
        else:
            if not(as_string):
                print(s)
        for node in group._f_iter_nodes():
            s += self.get_node_info(node._v_pathname, as_string)
            if (self._is_group(node._v_pathname) and recursive):
                s += self.print_group_content(node._v_pathname, recursive=True,
                                              as_string=as_string,
                                              max_depth=max_depth-1)
        return s

    def print_data_arrays_info(self, as_string=False):
        """Print information on all data array nodes in hdf5 file.

        :param bool as_string: If `True` solely returns string representation.
            If `False`, prints the string representation.
        :return str s: string representation of HDF5 nodes information
        """
        s = ''
        for node in self.h5_dataset:
            if self._is_array(node._v_name):
                s += self.get_node_info(node._v_name, as_string)
        return s

    def print_index(self, as_string=False, max_depth=3, node_type=[],
                    local_root='/'):
        """Print a list of the datasets in HDF5 and their Index names.

        :param bool as_string: If `True` solely returns string representation.
            If `False`, prints the string representation.
        :return str s: string representation of HDF5 nodes information
        """
        s = ''
        s += str('Dataset Content Index :\n')
        s += str('------------------------:\n')
        s += str('index printed with max depth `{}` and under local root'
                 ' `{}`\n\n'.format(max_depth, local_root))
        for key, value in self.content_index.items():
            col = None
            if isinstance(value, list):
                path = value[0]
                col = value[1]
            else:
                path = value
            node = self.get_node(path)
            if node is None:
                # precaution to avoid errors when content index references an
                # empty node
                continue
            if not(self._is_children_of(node, local_root)):
                continue
            if node._v_depth > max_depth:
                continue
            if col is None:
                s += str('\t Name : {:40}  H5_Path : {} \t\n'.format(
                         key, path))
            else:
                s += str('\t Name : {:40}  H5_Path : {}|col:{} \t\n'.format(
                         key, path, col))
            if key in self.aliases:
                s += str('\t        {} aliases -->'.format(key))
                for aliasname in self.aliases[key]:
                    s += str(' `'+aliasname+'`')
                s += '\n'
        if not(as_string):
            print(s)
            return
        return s

    def sync(self):
        """Synchronize and flush .h5 and .xdmf files with dataset content.

        After using the `sync` method, the XDMF file can be opened in Paraview
        and 3DImage and/or Mesh data visualized, even if the files are still
        open in the class instance.

        .. important::
            Paraview >=5 cannot read data from synchronized files, you must
            close them first. In this case, use method
            :func:`pause_for_visualization`.
        """
        message = ('.... Storing content index in {}:/Index attributes'
                   ''.format(self.h5_file))
        self._verbose_print(message,
                            line_break=False)
        self.add_attributes(dic=self.content_index, nodename='/Index')
        self.add_attributes(dic=self.aliases, nodename='/Index/Aliases')
        self.write_xdmf()
        self._verbose_print('.... flushing data in file {}'.format(
                                self.h5_file), line_break=False)
        self.h5_dataset.flush()
        self._verbose_print('File {} synchronized with in memory data tree'
                            ''.format(self.h5_file),
                            line_break=False)
        return

    def pause_for_visualization(self, Vitables=False, Paraview=False,
                                **keywords):
        """Flushes data, close files and pause interpreter for visualization.

        This method pauses the interpreter until you press the <Enter> key.
        During the pause, the HDF5 file object is closed, so that it can be
        read by visualization softwares like Paraview or ViTables. Two
        optional arguments allow to directly open the dataset with Paraview
        and/or Vitables, as a subprocess of Python. In these cases, the Python
        interpreter is paused until you close the visualization software.

        Paraview allows to visualize the volumic data that is stored in the
        SampleData dataset, *i.e.* Mesh and Images groups (geometry and
        stored fields). Vitables allows to visualize the content of the HDF5
        dataset in term of data tree, arrays content and nodes attributes. If
        both are requested, Vitables is executed before Paraview.


        :param bool Vitables: set to `True` to launch Vitables on the HDF5 file
            of the instance HDF5 dataset.
        :param bool Paraview: set to `True` to launch Paraview on the XDMF file
            of the instance.
        """
        Pause = True
        self.sync()
        self.h5_dataset.close()
        print('File objects are now closed, you can visualize dataset'
              ' content.')
        if Vitables:
            software_cmd = 'vitables'
            if 'Vitables_path' in keywords:
                software_cmd = keywords['Vitables_path']
            print('--- Lauching Vitables on file {} ---'.format(
                   self.h5_file))
            print('Once you will close Vitables, you may resume data'
                  ' management with your SampleData instance.')
            subprocess.run(args=[software_cmd,self.h5_path])
            Pause = False
        if Paraview:
            software_cmd = 'paraview'
            if 'Paraview_path' in keywords:
                software_cmd = keywords['Paraview_path']
            print('--- Lauching Paraview on file {} ---'.format(
                   self.xdmf_file))
            print('Once you will close Paraview, you may resume data'
                  ' management with your SampleData instance.')
            subprocess.run(args=[software_cmd,self.xdmf_path])
            Pause = False
        if Pause:
            input('Paused interpreter, you may open {} and {} files with'
                  ' other softwares during this pause.'
                  ' Press <Enter> when you want to resume data management'
                  ''.format(self.h5_file, self.xdmf_file))
        self.h5_dataset = tables.File(self.h5_path, mode='r+')
        print('File objects {} and {} are opened again.\n You may use this'
              ' SampleData instance normally.'.format(self.h5_file,
                                                      self.xdmf_file))
        return

    def switch_verbosity(self):
        """Change the verbosity flag to its opposite."""
        self._verbose = not (self._verbose)
        return

    def add_mesh(self, mesh_object=None, meshname='', indexname='',
                 location='/', description=' ', replace=False,
                 bin_fields_from_sets=True, file=None, **keywords):
        """Create a Mesh group in the dataset from a MeshObject.

        A Mesh group is a HDF5 Group that contains arrays describing mesh
        geometry, and fields defined on this mesh. The mesh geometry HDF5 nodes
        are: This methods adds a Mesh group to the dataset from a BasicTools
        :py:class:`UnstructuredMesh` class instance.

            :Nodes: array of shape `(Nnodes,Ndim)` with path
               ``'/Mesh_Path/Geometry/Nodes'`` and Index name
               ``'Meshname_Nodes'``
            :Elements: array of shape `(Nelements,Nelement_nodes)` with path
               ``'/Mesh_Path/Geometry/Elements'`` and Index name
               ``'Meshname_Elements'``

        Mesh group may also contain data array to describe fields, whose
        pathes, index names and content can be set using the class method
        :func:`add_data_array`. Fields defined on nodes must have a shape equal
        to (Nnodes,Field_dimension). Fields defined on integration points must
        have a shape equal to (Nintegration_points,Field_dimension).

        :param mesh_object: mesh to add to dataset. It is an instance from the
            :py:class:`pymicro.core.meshes.MeshObject` class
        :param str meshname: name used to create the Mesh group in dataset
        :param indexname: Index name used to reference the Mesh group
        :location str: Path, Name, Index Name or Alias of the parent group
            where the Mesh group is to be created
        :param str description: Description metadata for this mesh
        :param bool replace: remove Mesh group in the dataset with the same
            name/location if `True` and such group exists
        :param bool bin_fields_from_sets: If `True`, stores all Node and
            Element Sets in mesh_object as binary fields (1 on Set, 0 else)
        """
        # Check if the input array is in an external file
        if file is not None:
            mesh_object = self._read_mesh_from_file(file)
        ### Create or fetch mesh group
        mesh_group = self.add_group(meshname, location, indexname, replace)
        ### empty meshes creation
        if (mesh_object is None):
            self.add_attributes({'empty': True, 'group_type': 'emptyMesh'},
                                mesh_group._v_pathname)
            return
        else:
            self._check_mesh_object_support(mesh_object)
        ### Add Mesh Geometry to HDF5 dataset
        self._add_mesh_geometry(mesh_object,mesh_group, replace,
                                bin_fields_from_sets)
        ### Add mesh Grid to xdmf file
        self._add_mesh_to_xdmf(mesh_group)
        # store mesh metadata as HDF5 attributes
        Attribute_dic = {'description': description,
                         'empty': False,
                         'xdmf_gridname': mesh_group._v_name}
        self.add_attributes(Attribute_dic, mesh_group._v_pathname)
        ### Add node and element tags, eventually as fields if extended=True
        self._add_nodes_elements_tags(mesh_object, mesh_group, replace,
                                      bin_fields_from_sets)
        ### Add fields if some are stored in the mesh object
        for field_name, field in mesh_object.nodeFields.items():
            self.add_field(gridname=mesh_group._v_pathname,
                           fieldname=field_name, array=field,
                           replace=replace, **keywords)
        for field_name, field in mesh_object.elemFields.items():
            self.add_field(gridname=mesh_group._v_pathname,
                           fieldname=field_name, array=field,
                           replace=replace, **keywords)
        return mesh_object

    def add_mesh_from_image(self, imagename, with_fields=True, ofTetras=False,
                            meshname='', indexname='', location='/',
                            description=' ', replace=False,
                            bin_fields_from_sets=True, **keywords):
        """Create a Mesh group in the dataset from an Image dataset.

        The mesh group created can represent a mesh of tetrahedra or a mesh of
        hexaedra, of the image domain (square/triangles in 2D). The fields in
        the mesh groups are restored, with an adequate shape, and a suffix
        '_msh' in their indexname.

        :param str imagename: Name, Path or Indexname of the mesh group to get
        :param bool with_fields: If `True`, load the nodes and elements fields
            from the image group into the mesh object.
        :param bool ofTetras: if `True`, returns a mesh with tetrahedron
            elements. If `False`, return a rectilinera mesh of hexaedron
            elements.
        :param str meshname: name used to create the Mesh group in dataset
        :param indexname: Index name used to reference the Mesh group
        :location str: Path, Name, Index Name or Alias of the parent group
            where the Mesh group is to be created
        :param str description: Description metadata for this mesh
        :param bool replace: remove Mesh group in the dataset with the same
            name/location if `True` and such group exists
        :param bool bin_fields_from_sets: If `True`, stores all Node and
            Element Sets in mesh_object as binary fields
        """

        Mesh_o = self.get_mesh_from_image(imagename, with_fields, ofTetras)
        # Rename mesh fields to avoid duplicate in content_index
        field_names = list(Mesh_o.nodeFields.keys())
        for key in field_names:
            Mesh_o.nodeFields[key+'_'+meshname] = Mesh_o.nodeFields.pop(key)
        field_names = list(Mesh_o.elemFields.keys())
        for key in field_names:
            Mesh_o.elemFields[key+'_'+meshname] = Mesh_o.elemFields.pop(key)
        self.add_mesh(Mesh_o, meshname, indexname, location, description,
                      replace, bin_fields_from_sets)
        return

    def add_image(self, image_object=None, imagename='', indexname='',
                  location='/', description=' ', replace=False,
                  **keywords):
        """Create a 2D/3D Image group in the dataset from an ImageObject.

        An Image group is a HDF5 Group that contains arrays describing fields
        defined on an image (uniform grid of voxels/pixels). This methods adds
        an Image group to the dataset from a BasicTools
        :py:class:`ConstantRectilinearMesh` class instance.This class
        represents regular meshes of square/cubes, *i.e.* pixels/voxels.


        The image geometry and topology is defined by HDF5 attributes of the
        Image Group, that are:

            :nodes_dimension: np.array, number of grid points along each
                dimension of the Image. This number is array is equal to
                the `dimension` attribute array +1 for each value.
            :dimension: np.array, number of voxels along each dimension of the
                Image (Nx,Ny,Nz) or (Nx,Ny)
            :spacing: np.array, voxel size along each dimension (dx,dy,dz) or
                (dx, dy)
            :origin: np.array, coordinates of the image grid origin,
                corresponding to the first vertex of the voxel [0,0,0]
                or pixel [0,0]

        The Image group may also contain arrays of field values on the image.
        These fields can be elementFields (defined at pixel/voxel centers) or
        nodefields (defined at pixel/voxel vertexes). Their
        pathes, index names and content can be set using the class method
        :func:`add_field`. Fields defined on nodes must have a shape equal
        to the `nodes_dimension` attribute. Fields defined on elements must
        have a shape equal to the `dimension` attribute. Both can have an
        additional last dimension if they have a higher dimensionality than
        scalar fields (for instance [Nx,Ny,Nz,3] for a vector field).

        :param image_object: image to add to dataset. It is an instance from
            the :py:class:`ConstantRectilinearMesh` class of the BasicTools
            Python package.
        :param str imagename: name used to create the Image group in dataset
        :param str indexname: Index name used to reference the Image. If none
            is provided, `imagename` is used.
        :location str: Path, Name, Index Name or Alias of the parent group
            where the Image group is to be created
        :param str description: Description metadata for this 3D image
        :param bool replace: remove Image group in the dataset with the same
            name/location if `True` and such group exists
        """
        ### Create or fetch image group
        image_group = self.add_group(imagename, location, indexname, replace)
        ### empty images creation
        if (image_object is None):
            self.add_attributes({'empty': True, 'group_type': 'emptyImage'},
                                image_group._v_pathname)
            return
        else:
            self._check_image_object_support(image_object)
        ### Add image Grid to xdmf file
        self._add_image_to_xdmf(imagename, image_object)
        ### store image metadata as HDF5 attributes
        image_type = self._get_image_type(image_object)
        image_nodes_dim = np.array(image_object.GetDimensions())
        image_cell_dim = image_nodes_dim - np.ones(image_nodes_dim.shape,
                                                   dtype=image_nodes_dim.dtype)
        if len(image_nodes_dim) == 2:
            image_xdmf_dim = image_nodes_dim[[1,0]]
        elif len(image_nodes_dim) == 3:
            image_xdmf_dim = image_nodes_dim[[1,0,2]]
        Attribute_dic = {'nodes_dimension': image_nodes_dim,
                         'nodes_dimension_xdmf': image_xdmf_dim,
                         'dimension': image_cell_dim,
                         'spacing': np.array(image_object.GetSpacing()),
                         'origin': np.array(image_object.GetOrigin()),
                         'description': description,
                         'group_type': image_type,
                         'empty': False,
                         'xdmf_gridname': imagename}
        self.add_attributes(Attribute_dic, image_group._v_pathname)
        ### Add fields if some are stored in the image object
        for field_name, field in image_object.nodeFields.items():
            self.add_field(gridname=image_group._v_pathname,
                           fieldname=field_name, array=field,
                           replace=replace, **keywords)
        for field_name, field in image_object.elemFields.items():
            self.add_field(gridname=image_group._v_pathname,
                           fieldname=field_name, array=field,
                           replace=replace, **keywords)
        return image_object

    def add_image_from_field(self, field_array, fieldname, imagename='',
                             indexname='', location='/', description=' ',
                             replace=False, origin=None, spacing=None,
                             is_scalar=True, is_elemField=True,
                             **keywords):
        """Create a 2D/3M Image group in the dataset from a field data array.

        Construct an image object from the inputed field array. This array is
        interpreted by default as an element field of a pixelized/voxelized
        grid. Hence, if the field is of shape (Nx,Ny), the image group will
        store a (Nx,Ny) image (*i.e.* a regular grid of Nx+1,Ny+1 nodes). If
        specified, the field can be interpreted as a nodal field (values at
        pixels/voxels vertexes). In this case the method will create a
        (Nx-1,Ny-1) image of (Nx,Ny) nodes. The same applies in 3D.

        If the field is not a scalar field, the last dimension of the field
        array is interpreted as the dimension containing the field components

        :param numpy.array field_array: data array of the field values on the
            image regular grid.
        :param str fieldname: add the field to HDF5 dataset and image Group
            with this name.
        :param str imagename: name used to create the Image group in dataset
        :param str indexname: Index name used to reference the Image. If none
            is provided, `imagename` is used.
        :location str: Path, Name, Index Name or Alias of the parent group
            where the Image group is to be created
        :param str description: Description metadata for this 3D image
        :param bool replace: remove Image group in the dataset with the same
            name/location if `True` and such group exists
        :param np.array(3,) origin: Coordinates of the first node of the
            regular grid of squares/cubes constituting the image geometry
        :param np.array(3,) spacing: Size along each dimension of the
            pixels/voxels composing the image.
        :param bool is_scalar: If `True` (default value), the field is
            considered as a scalar field to compute the image dimensions from
            the field array shape.
        :param bool is_elemField: If `True` (default value), the array is
            considered as a pixel/voxel wise field value array. If `False`, the
            field is considered as a nodal value array.

        """
        if is_scalar:
            field_dim = len(field_array.shape)
            field_dimensions = field_array.shape
        else:
            field_dim = len(field_array.shape)-1
            field_dimensions = field_array.shape[:-1]
        if spacing is None:
            spacing = np.ones((len(field_dimensions),))
        if origin is None:
            origin = np.zeros((len(field_dimensions),))
        if is_elemField:
            field_dimensions = field_dimensions + np.ones((field_dim,))
        image_object = ConstantRectilinearMesh(dim=field_dim)
        image_object.SetDimensions(field_dimensions)
        image_object.SetOrigin(origin)
        image_object.SetSpacing(spacing)
        image_object.elemFields[fieldname] = field_array
        self.add_image(image_object, imagename, indexname, location,
                       description, replace, **keywords)
        return

    def add_group(self, groupname, location, indexname='', replace=False):
        """Create a standard HDF5 group at location with no grid properties.

        If the group parents in `location` do not exist, they are created.

        :param str groupname: Name of the group to create
        :param str location: Path where the group will be added in the HDF5
            dataset
        :param str indexname: Index name used to reference the Group. If none
            is provided, `groupname` is used.
        :param bool replace: remove 3DImage group in the dataset with the same
            name/location if `True` and such group exists
        :param bool createparents: if `True`, create parent nodes in `path` if
            they are not present in the dataset
        """
        if (indexname == ''):
            indexname = groupname
        Group = self._init_SD_group(groupname, location,
                                    group_type='Group', replace=replace)
        if Group is None:
            raise tables.NodeError('Group {} could not be created or fetched.'
                                   ' Unknown error.'.format(groupname))
        self.add_to_index(indexname, Group._v_pathname)
        return Group

    def add_field(self, gridname, fieldname, array, location=None,
                  indexname=None, chunkshape=None, replace=False,
                  filters=None, empty=False, visualisation_type='Elt_mean',
                  **keywords):
        """Add a field to a grid (Mesh or 2D/3DImage) group from a numpy array.

        This methods checks the compatibility of the input field array with the
        grid dimensionality and geometry, adds it to the HDF5 dataset, and
        the XDMF file. Metadata describing the field type, dimensionality are
        stored as field HDF node attributes. The path of the field is added to
        the grid Group as a HDF5 attribute.

        :param str gridname: Path, name or indexname of the grid Group on which
            the field will be added
        :param str fieldname: Name of the HDF5 node to create that will contain
            the field value array
        :param np.array array: Array containing the field values to add in the
            dataset
        ;param str location: Path, name or indexname of the Group in which the
            field array will be stored. This Group must be a children of the
            `gridname` Group. If not provided, the field is stored in the
            `gridname` Group.
        :param str indexname: Index name used to reference the field node
        :param tuple  chunkshape: The shape of the data chunk to be read or
            written in a single HDF5 I/O operation
        :param bool replace: remove 3DImage group in the dataset with the same
            name/location if `True` and such group exists
        :param Filters filters: instance of :py:class:`tables.Filters` class
            specifying compression settings.
        :param bool empty: if `True` create the path, Index Name in dataset and
            store an empty array. Set the node attribute `empty` to True.
        :param str visualisation_type: Type of visualisation used to represent
            integration point fields with an element wise constant field.
            Possibilities are 'Elt_max' (maximum value per element), 'Elt_mean'
            (mean value per element), 'None' (no visualisation field).
            Default value is 'Elt_mean'
        .. note:: additional keywords arguments can be passed to specify global
                compression options, see :func:`set_chunkshape_and_compression`
                documentation for their definition. If some are passed, they
                are prioritised over the settings in the inputed Filter object.

        """
        self._verbose_print('Adding field `{}` into Grid `{}`'
                            ''.format(fieldname, gridname))
        # Fields can only be added to grid Groups --> sanity check
        if not(self._is_grid(gridname)):
            raise tables.NodeError('{} is not a grid, cannot add a field data'
                                   ' array in this group.'.format(gridname))
        # If needed, pad the field with 0s to comply with number of bulk and
        # boundary elements
        array, padding, vis_array = self._mesh_field_padding(array, gridname)
        # Check if the array shape is consistent with the grid geometry
        # and returns field dimension, xdmf Center attribute
        field_type, dimensionality = self._check_field_compatibility(
                                                        gridname,array.shape)
        if location is None:
            # FIELD STORAGE DEFAULT CONVENTION :
            # fields are stored directly into the HDF5 grid group
            array_location = gridname
        else:
            # check if the given location is a subgroup of the grid group
            if self._is_children_of(location, gridname):
                array_location = location
            elif self.get_node(location) == self.get_node(gridname):
                array_location = location
            else:
                raise tables.NodeError('Cannot add field at location `{}`.'
                                       ' Field location must be a grid group'
                                       ' (Mesh or Image), or a grid group'
                                       ' children'.format(location))
        # Add data array into HDF5 dataset
        if self._is_image(gridname):
            array, transpose_indices = self._transpose_image_array(
                dimensionality, array)
        if dimensionality in ['Tensor6','Tensor']:
            array, transpose_components = self._transpose_field_comp(
                dimensionality, array)
        if indexname is None:
            grid_path = self._name_or_node_to_path(gridname)
            grid_indexname = self.get_indexname_from_path(grid_path)
            indexname = grid_indexname+'_'+fieldname
        node = self.add_data_array(array_location, fieldname, array, indexname,
                                   chunkshape, replace, filters, empty,
                                   **keywords)

        Attribute_dic = {'field_type': field_type,
                         'field_dimensionality': dimensionality,
                         'parent_grid_path': self._name_or_node_to_path(
                             gridname),
                         'xdmf_gridname': self.get_attribute('xdmf_gridname',
                                                             gridname),
                         'padding': padding
                         }
        if self._is_image(gridname):
            Attribute_dic['transpose_indices'] = transpose_indices
        if dimensionality in ['Tensor6','Tensor']:
            Attribute_dic['transpose_components'] = transpose_components
        if (field_type == 'IP_field') and not (visualisation_type=='None'):
            vis_array = self._IP_field_for_visualisation(vis_array,
                                                         visualisation_type)
            visname = fieldname+f'_{visualisation_type}'
            visindexname = indexname+f'_{visualisation_type}'
            if dimensionality in ['Tensor6','Tensor']:
                vis_array, _ = self._transpose_field_comp(
                    dimensionality, vis_array)
            node_vis = self.add_data_array(array_location, visname, vis_array,
                                           visindexname, chunkshape, replace,
                                           filters, empty, **keywords)
            Attribute_dic['visualisation_type'] = visualisation_type
            self.add_attributes(Attribute_dic, nodename=visindexname)
            Attribute_dic['visualisation_field_path'] = node_vis._v_pathname
        self.add_attributes(Attribute_dic, nodename=indexname)
        # Add field description to XDMF file
        if (field_type == 'IP_field') and not (visualisation_type=='None'):
            self._add_field_to_xdmf(visindexname, vis_array)
        else:
            self._add_field_to_xdmf(indexname, array)
        # Add field path to grid node Field_list attribute
        self._append_field_index(gridname, fieldname)
        return node

    def add_data_array(self, location, name, array=None, indexname=None,
                       chunkshape=None, replace=False, filters=None,
                       empty=False, **keywords):
        """Add a data array node at the given location in the HDF5 dataset.

        The method uses the :py:class:`CArray` and
        :py:class:`tables.Filters` classes of the
        `Pytables <https://www.pytables.org/index.html>`_ package to add
        data arrays in the dataset and control their chunkshape and compression
        settings.

        :param str location: Path where the array will be added in the dataset
        :param str name: Name of the array to create
        :param np.array array: Array to store in the HDF5 node
        :param str indexname: Index name used to reference the node
        :param tuple  chunkshape: The shape of the data chunk to be read or
            written in a single HDF5 I/O operation
        :param bool replace: remove 3DImage group in the dataset with the same
            name/location if `True` and such group exists
        :param Filters filters: instance of :py:class:`tables.Filters` class
            specifying compression settings.
        :param bool empty: if `True` create the path, Index Name in dataset and
            store an empty array. Set the node attribute `empty` to True.
        :param file:
        :type str, optional:

        .. note:: additional keywords arguments can be passed to specify global
            compression options, see :func:`set_chunkshape_and_compression`
            documentation for their definition. If some are passed, they are
            prioritised over the settings in the inputed Filter object.

        """
        self._verbose_print('Adding array `{}` into Group `{}`'
                            ''.format(name, location))
        # Safety checks
        self._check_SD_array_init(name, location, replace)
        # Check if the input array is in an external file
        if 'file' in keywords:
            array = self._read_array_from_file(**keywords)
        if array is None:
            raise ValueError('Received a `None` array. Cannot add data array.')
        # get location path
        location_path = self._name_or_node_to_path(location)
        # get compression options
        Filters = self._get_compression_opt(filters, **keywords)
        # add to index
        if indexname is None:
            indexname = name
        self.add_to_index(indexname, os.path.join(location_path, name))
        # Create dataset node to store array
        if empty:
            Node = self.h5_dataset.create_carray(
                    where=location_path, name=name, obj=np.array([0]),
                    title=indexname)
            self.add_attributes({'empty': True}, Node._v_pathname)
        else:
            Node = self.h5_dataset.create_carray(
                    where=location_path, name=name, filters=Filters,
                    obj=array, chunkshape=chunkshape,
                    title=indexname)
            self.add_attributes({'empty': False}, Node._v_pathname)
        return Node

    def add_table(self, location, name, description, indexname=None,
                  chunkshape=None, replace=False, data=None, filters=None,
                  **keywords):
        """Add a structured storage table in HDF5 dataset.

        :param str location: Path where the array will be added in the dataset
        :param str name: Name of the array to create
        :param IsDescription description: Definition of the table rows
        :param str indexname: Index name used to reference the node
            composition as a sequence of named fields (analogous to Numpy
            structured arrays). It must be an instance of the
            :py:class:`tables.IsDescription` class from the
            `Pytables <https://www.pytables.org/index.html>`_ package
        :param tuple chunkshape: The shape of the data chunk to be read or
            written in a single HDF5 I/O operation
        :param bool replace: remove 3DImage group in the dataset with the same
            name/location if `True` and such group exists
        :param np.array(np.void) data: Array to store in the HDF5 node. `dtype`
            must be consistent with the table `description`.
        :param Filters filters: instance of :py:class:`tables.Filters` class
            specifying compression settings.
        :param bool empty: if `True` create the path, Index Name in dataset and
            store an empty table. Set the table attribute `empty` to True.

        .. note:: additional keywords arguments can be passed to specify global
                compression options, see :func:`set_chunkshape_and_compression`
                documentation for their definition. If some are passed, they
                are prioritised over the settings in the inputed Filter object.
        """
        self._verbose_print('Adding table `{}` into Group `{}`'
                            ''.format(name, location))
        # get location path
        location_path = self._name_or_node_to_path(location)
        if (location_path is None):
            msg = ('(add_table): location {} does not exist, table'
                   ' cannot be added. Use optional argument'
                   ' "createparents=True" to force location Group creation'
                   ''.format(location))
            self._verbose_print(msg)
            return
        else:
            # check location nature
            if not(self._get_node_class(location) == 'GROUP'):
                msg = ('(add_table): location {} is not a Group nor '
                       'empty. Please choose an empty location or a HDF5 '
                       'Group to store table'.format(location))
                self._verbose_print(msg)
                return
            # check if array location exists and remove node if asked
            table_path = os.path.join(location_path, name)
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
        # keywords compression options prioritized over input filters instances
        if (filters is None) or bool(keywords):
            Filters = self._get_compression_opt(**keywords)
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

        table = self.h5_dataset.create_table(where=location_path, name=name,
                                             description=description,
                                             filters=Filters,
                                             chunkshape=chunkshape)
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
        self.add_to_index(indexname, table._v_pathname)
        return table

    def add_tablecols(self, tablename, description, data=None):
        """Add new columns to a table node.

        :param tablename: Name, Path or Indexname of the table where the
            columns must be added.
        :type tablename: str
        :param description: Description of the fields constituting the
            new columns to add the table.
        :type description: np.dtype or tables.IsDescription
        :param data: values to add into the new columns, defaults to None. The
            dtype of this array must be constitent with the `description`
            argument.
        :type data: np.array, optional
        :raises ValueError: If `data.dtype` and `description` do no match
        """
        table = self.get_node(tablename)
        current_dtype = tables.dtype_from_descr(table.description)
        if isinstance(description, tables.IsDescription):
            descr_dtype = tables.dtype_from_descr(description)
        elif isinstance(description, np.dtype):
            descr_dtype = description
        else:
            raise ValueError('description must be a tables.IsDescription'
                             ' instance or a numpy.dtype instance.')
        new_dtype = SampleData._merge_dtypes(current_dtype,descr_dtype)
        new_desc = tables.descr_from_dtype(new_dtype)[0]
        self._update_table_columns(tablename, new_desc)
        # if data is provided, safety check. Must be adequate array dtype
        if data is not None:
            if not(data.dtype == descr_dtype):
                raise ValueError('Data provided to add to the new columns'
                                 ' dtype is not consistent with the new'
                                 ' columns description inputed.\n'
                                 'Provided data dtype : {}\n'
                                 'Provided description : {}\n'
                                 ''.format(data.dtype, descr_dtype))
            for colname in descr_dtype.names:
                column = data[colname]
                self.set_tablecol(tablename, colname, column)
        return

    def add_string_array(self, name, location, indexname=None,
                         replace=False, data=[]):
        """Add an enlargeable array to store strings of max 255 characters.

        String arrays are typically used to store large list of strings that
        are too large to be stored as HDF5 attributes into the dataset.

            .. Warning::
                The string are stored as byte strings. You will need to
                use the str.decode() method to get the elements of the
                string_array as UTF-8 or  ASCII formatted strings.

        To manipulate a string array use the 'get_node' method to get the
        array, and then manipulate as a list of binary strings.

        :param str name: Name of the array to create
        :param str location: Path where the array will be added in the dataset
        :param str indexname: Index name used to reference the node
            composition as a sequence of named fields (analogous to Numpy
            structured arrays). It must be an instance of the
            :py:class:`tables.IsDescription` class from the
            `Pytables <https://www.pytables.org/index.html>`_ package
        :param bool replace: remove array in the dataset with the same
            name/location if `True`
        :param list[str] data: List of strings to add to the string array upon
            creation.
        """
        self._verbose_print('Adding String Array `{}` into Group `{}`'
                            ''.format(name, location))
        # get location path
        location_path = self._name_or_node_to_path(location)
        if (location_path is None):
            msg = ('(add_string_array): location {} does not exist, string'
                   ' array cannot be added.'
                   ''.format(location))
            self._verbose_print(msg)
            return
        else:
            # check location nature
            if not(self._get_node_class(location) == 'GROUP'):
                msg = ('(add_string_array): location {} is not a Group. '
                       'Please choose an empty location or a HDF5 '
                       'Group to store table'.format(location))
                self._verbose_print(msg)
                return
            # check if array location exists and remove node if asked
            array_path = os.path.join(location_path, name)
            if self.h5_dataset.__contains__(array_path):
                if replace:
                    msg = ('(add_string_array): existing node {} will be '
                           'overwritten and all of its childrens removed'
                           ''.format(array_path))
                    self._verbose_print(msg)
                    self.remove_node(array_path, recursive=True)
                else:
                    msg = ('(add_string_array): node {} already exists. To '
                           'overwrite, use optional argument "replace=True"'
                           ''.format(array_path))
                    self._verbose_print(msg)

        # Create String array
        string_atom = tables.StringAtom(itemsize=255)
        str_array = self.h5_dataset.create_earray(where=location_path,
                                                  name=name,
                                                  atom=string_atom,
                                                  shape=(0,))
        # Append input string list
        if data is not None:
            str_array.append(data)
            # add to index
        if indexname is None:
            warn_msg = (' (add_string_array) indexname not provided, '
                        ' the string array name `{}` is used as index name '
                        ''.format(name))
            self._verbose_print(warn_msg)
            indexname = name
        self.add_to_index(indexname, str_array._v_pathname)
        return str_array

    def add_attributes(self, dic, nodename):
        """Add a dictionary entries as HDF5 Attributes to a Node or Group.

        :param dic dic: Python dictionary of items to store in HDF5 file as
            HDF5 Attributes
        :param str nodename: Path, Index name or Alias of the HDF5 node or
            group receiving the Attributes
        """
        Node = self.get_node(nodename)
        for key, value in dic.items():
            Node._v_attrs[key] = value
        return

    def add_alias(self, aliasname, path=None, indexname=None):
        """Add alias name to reference Node with inputed path or index name.

        :param str aliasname: name to add as alias to reference the node
        :param str path: Path of the node to reference with `aliasname`
        :param str indexname: indexname of the node to reference with
            `aliasname`
        """
        if (path is None) and (indexname is None):
            msg = ('(add_alias) None path nor indexname inputed. Alias'
                   'addition aborted')
            self._verbose_print(msg)
            return
        Is_present = (self._is_in_index(aliasname)
                      and self._is_alias(aliasname))
        if Is_present:
            msg = ('Alias`{}` already exists : duplicates not allowed'
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

    def add_to_index(self, indexname, path, colname=None):
        """Add path to index if indexname is not already in content_index.

        :param str indexname: name to add as indexname to reference the node
        :param str path: Path of the node to reference with `aliasname`
        :param str colname: if the node is a `table` node, set colname to
            reference a column (named field) of the table with this indexname
        """
        Is_present = (self._is_in_index(indexname)
                      or self._is_alias(indexname))
        if Is_present:
            raise ValueError('Name `{}` already in '
                             'content_index : duplicates not allowed in Index'
                             ''.format(indexname))
        else:
            for item in self.content_index:
                if isinstance(self.content_index[item], list):
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

    def compute_mesh_elements_normals(
            self, meshname, element_tag, Normal_fieldname=None,
            align_vector=np.random.rand(3), as_nodal_field=False):
        """Compute the normals of a set of boundary elements of a mesh group.

        The normals are stored as en element wise constant field in the mesh
        group.

        :param meshname: Name, Path, Index name or Alias of the Mesh group
            in dataset
        :type meshname: str
        :param element_tag: name of the element tag or element type whose
            normals must be computed
        :type element_tag: str
        :param Normal_fieldname: Name of the normals field to store on the
            mesh group. Defaults to 'Normals_element_tag_name'
        :type Normal_fieldname: TYPE, optional
        :param align_vector: All normals are oriented to have positive dot
            product with `align_vector`, defaults to [0,0,1]
        :type align_vector: np.array(3), optional
        :raises ValueError: Can only process elements of bidimensional
            topology (surface elements, like triangles, quadrilaterals...)
        """
        import BasicTools.Containers.ElementNames as EN
        if Normal_fieldname is None:
            Normal_fieldname = 'Normals_' + element_tag

        # Step 0: identify to which element type or element tag
        mesh_elements = self.get_mesh_elem_types_and_number(meshname)
        mesh_el_tags = self.get_mesh_elem_tags_names(meshname)
        if element_tag in mesh_elements.keys():
            el_type = mesh_elements[element_tag]
        if element_tag in mesh_el_tags.keys():
            el_type = mesh_el_tags[element_tag]

        # Step 1: Find out type of element and assert that it is a 2D element
        # type
        if EN.dimension[el_type] != 2:
            raise ValueError('Can compute normals only for sets of elements'
                             ' with a 2D topology.')

        # Step 2: Extract element set connectivity and global IDs
        if element_tag in mesh_elements.keys():
            connectivity = self.get_mesh_elements(
                meshname, with_tags=False, get_eltype_connectivity=element_tag)
        if element_tag in mesh_el_tags.keys():
            connectivity = self.get_mesh_elem_tags_connectivity(
                                    meshname, element_tag)
        element_IDs = self.get_mesh_elem_tag(meshname, element_tag)

        # Step 3: Compute normals
        mesh_nodes = self.get_mesh_nodes(meshname, as_numpy=True)
        vect_1 = (  mesh_nodes[connectivity[:,1],:]
                  - mesh_nodes[connectivity[:,0],:])
        vect_2 = (  mesh_nodes[connectivity[:,2],:]
                  - mesh_nodes[connectivity[:,1],:])
        normals = np.cross(vect_1, vect_2)
        #     * normalize normals
        norms = np.linalg.norm(normals, axis=-1)
        normals[:,0] = normals[:,0] / norms
        normals[:,1] = normals[:,1] / norms
        normals[:,2] = normals[:,2] / norms
        #     * align orientation of surface vectors
        idx = np.where(np.dot(normals,align_vector) < 0)
        normals[idx,:] = - normals[idx,:]

        # Step 4: Create element field to store normals on mesh group
        Nelem = np.sum(self.get_attribute('Number_of_elements', meshname))
        Elem_normals_field = np.zeros(shape=(Nelem,3))
        Elem_normals_field[element_IDs,:] = normals
        if as_nodal_field:
            Nodal_normals_field = np.zeros(shape=mesh_nodes.shape)
            for nodeId in np.unique(connectivity):
                E_idx, _ = np.where(connectivity == nodeId)
                idxx = element_IDs[E_idx]
                Nodal_normals_field[nodeId,:] = (
                                    np.sum(Elem_normals_field[idxx,:], axis=0)
                                                 ) / len(idxx)
                # print('E_idx :', E_idx, 'idxx :', idxx)
                # print('local Normals :', Elem_normals_field[idxx,:])
                # print(Nodal_normals_field[nodeId,:])
                # print(np.linalg.norm(Nodal_normals_field[nodeId,:]))
            norms = np.linalg.norm(Nodal_normals_field, axis=-1)
            idx = np.where(norms > 0)
            Nodal_normals_field[idx,0] = Nodal_normals_field[idx,0]/norms[idx]
            Nodal_normals_field[idx,1] = Nodal_normals_field[idx,1]/norms[idx]
            Nodal_normals_field[idx,2] = Nodal_normals_field[idx,2]/norms[idx]
            Normals_field = Nodal_normals_field
        else:
            Normals_field = Elem_normals_field
        # Step 5: store Normal field in mesh group
        self.add_field(meshname, Normal_fieldname, Normals_field)
        return

    def get_indexname_from_path(self, node_path):
        """Return the Index name of the node at given path.

        :param str node_path: Path of the node in the HDF5 data tree
        :return: Returns the Index name of the node
        """
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
            self._verbose_print(msg)

    def get_mesh(self, meshname, with_tags=True, with_fields=True,
                 as_numpy=True):
        """Return data of a mesh group as BasicTools UnstructuredMesh object.

        This methods gathers the data of a 2DMesh or 3DMesh group, including
        nodes coordinates, elements types and connectivity and fields, into a
        BasicTools :class:`ConstantRectilinearMesh` object.

        :param str meshname: Name, Path or Indexname of the mesh group to get
        :param bool with_tags: If `True`, store the nodes and element tags
            (sets) into the mesh object
        :param bool with_fields: If `True`, store mesh group fields into the
            mesh_object, defaults to True
        :type with_fields: bool, optional
        :return: Mesh_object containing all data (nodes, elements, nodes and
                elements sets, fields), contained in the mesh data group.
        :rtype:  BasicTools :class:`UnstructuredMesh` object.

        """
        # Create mesh object
        mesh_object = UnstructuredMesh()
        # Get mesh nodes
        mesh_object.nodes = self.get_mesh_nodes(meshname, as_numpy)
        # No mesh ID for now --> create mesh Ids
        mesh_object.originalIDNodes = self.get_mesh_nodesID(meshname, as_numpy)
        # Get node tags
        if with_tags:
            self._load_nodes_tags(meshname, mesh_object, as_numpy=as_numpy)
        # Get mesh elements and element tags
        mesh_object.elements = self.get_mesh_elements(meshname,
                                                      with_tags=with_tags,
                                                      as_numpy=as_numpy)
        # Set mesh originalIds from 0 to Nelems
        Nelems = np.sum(self.get_attribute('Number_of_elements', meshname))
        originalIds = np.arange(Nelems) + 1
        mesh_object.SetElementsOriginalIDs(originalIds)
        # Get mesh fields
        mesh_group = self.get_node(meshname)
        FIndex_path = os.path.join(mesh_group._v_pathname,'Field_index')
        Field_index = self.get_node(FIndex_path)
        if with_fields and (Field_index is not None):
            for fieldname in Field_index:
                name = fieldname.decode('utf-8')
                field_type = self.get_attribute('field_type', name)
                if field_type == 'Nodal_field':
                    data = self.get_field(name, unpad_field=True)
                    mesh_object.nodeFields[name] = data
                elif field_type == 'Element_field':
                    data = self.get_field(name, unpad_field=True)
                    mesh_object.elemFields[name] = data
        mesh_object.PrepareForOutput()
        return mesh_object

    def get_mesh_from_image(self, imagename, with_fields=True, ofTetras=False):
        """Return an UnstructuredMesh instance from an Image data group.

        :param str imagename: Name, Path or Indexname of the mesh group to get
        :param bool with_fields: If `True`, load the nodes and elements fields
            from the image group into the mesh object.
        :param bool ofTetras: if `True`, returns a mesh with tetrahedron
            elements. If `False`, return a rectilinera mesh of hexaedron
            elements.
        :return: Mesh_object containing all data (nodes, elements, nodes and
                elements sets, fields), corresponding to the Image data group
                content.
        :rtype:  BasicTools :class:`UnstructuredMesh` object.
        """
        mesh_CR = self.get_image(imagename, with_fields)
        mesh_obj = UMCT.CreateMeshFromConstantRectilinearMesh(mesh_CR,
                                                              ofTetras)
        n_elems = mesh_obj.GetNumberOfNodes()
        if with_fields:
            for key,val in mesh_CR.nodeFields.items():
                mesh_obj.nodeFields[key] = val.reshape((n_elems,1))
        return mesh_obj

    def get_mesh_nodes(self, meshname, as_numpy=False):
        """Return the mesh node coordinates as a HDF5 node or Numpy array.

        :param str meshname: Name, Path, Index name or Alias of the Mesh group
            in dataset
        :param bool as_numpy: if `True`, returns the Node as a `numpy.array`.
            If `False`, returns the node as a Node or Group object.
        :return: Return the mesh Nodes coordinates array as a
            :py:class:`tables.Node` object or a `numpy.array`
        """
        nodes_path = self.get_attribute('nodes_path', meshname)
        return self.get_node(nodes_path, as_numpy)

    def get_mesh_nodesID(self, meshname, as_numpy=False):
        """Return the mesh node ID as a HDF5 node or Numpy array.

        :param str meshname: Name, Path, Index name or Alias of the Mesh group
            in dataset
        :param bool as_numpy: if `True`, returns the Node as a `numpy.array`.
            If `False`, returns the node as a Node or Group object.
        :return: Return the mesh Nodes ID array as a
            :py:class:`tables.Node` object or a `numpy.array`
        """
        nodes_path = self.get_attribute('nodesID_path', meshname)
        nodesID = self.get_node(nodes_path, as_numpy)
        if nodesID is None:
            Nnodes = self.get_mesh_nodes(meshname).shape[0]
            nodesID = np.arange(Nnodes)
        return nodesID

    def get_mesh_xdmf_connectivity(self, meshname, as_numpy=False):
        """Return the mesh elements connectivity as HDF5 node or Numpy array.

        :param str meshname: Name, Path, Index name or Alias of the Mesh group
            in dataset
        :param bool as_numpy: if `True`, returns the Node as a `numpy.array`.
            If `False`, returns the node as a Node or Group object.
        :return: Return the mesh elements connectivity referenced in the XDMF
            file as a :py:class:`tables.Node` object or a `numpy.array`
        """
        elems_path = self.get_attribute('elements_path', meshname)
        return self.get_node(elems_path, as_numpy)

    def get_mesh_elements(self, meshname, with_tags=True, as_numpy=True,
                          get_eltype_connectivity=None):
        """Return the mesh elements connectivity as HDF5 node or Numpy array.

        :param str meshname: Name, Path, Index name or Alias of the Mesh group
            in dataset
        :param bool with_tags: if `True`, loads the element tags in the
            returned elements container
        :param bool as_numpy: if `True`, returns arrays in elements container
            as numpy array
        :param str get_eltype_connectivity: if this argument is set to the name
            of an element type contained in the mesh, the method retuns the
            connectivity array of these elements, and not the BasicTools
            elements container.
        :return: Return the mesh elements container as a
            BasicTools :py:class:`AllElements` object
        """
        # Create AllElementsContainer
        AElements = AllElements()
        connectivity = self.get_mesh_xdmf_connectivity(meshname, as_numpy)
        # Get Elements Metadata
        Mesh_attrs = self.get_dic_from_attributes(meshname)
        Topology = Mesh_attrs['Topology']
        element_type = Mesh_attrs['element_type']
        Nelems = Mesh_attrs['Number_of_elements']
        Xdmf_code = Mesh_attrs['Xdmf_elements_code']
        offset = 0
        # For each element type, create an Element container and fill
        # connectivity
        for i in range(len(element_type)):
            # Create elements container
            Elements = AElements.GetElementsOfType(element_type[i])
            # get parameters for element type elements in mesh
            Nnode_per_el = Elements.GetNumberOfNodesPerElement()
            Nvalues = (1+Nnode_per_el)*Nelems[i]
            id_offset = 1
            local_code = Xdmf_code[i]
            # For bar2 and point1 elements, 2 integers are stored as XDMF code
            # before each element connectivity
            if (element_type[i] == 'bar2') or (element_type[i] == 'point1'):
                Nvalues += Nelems[i]
                id_offset += 1
                Nnode_per_el += 1
            if Topology == 'Mixed':
                # Get connectivity chunk for this element type and reshape it
                local_connect = connectivity[offset:offset+Nvalues]
                local_connect = local_connect.reshape((Nelems[i],
                                                       Nnode_per_el+1))
                # Safety check
                if not(np.all(local_connect[:,0] == local_code)):
                    raise ValueError('Local connectivity for element type {}'
                                     ' is ill-shaped : Xdmf code value wrong'
                                     ' for at least one element.'
                                     ''.format(element_type[i]))
                if get_eltype_connectivity == element_type[i]:
                    return local_connect[:,id_offset:]
                Elements.connectivity = local_connect[:,id_offset:]
                Elements.cpt = Nelems[i]
                offset = Nvalues
            elif Topology == 'Uniform':
                Elements.connectivity = connectivity.reshape((Nelems[i],
                                                              Nnode_per_el))
                Elements.cpt = Nelems[i]
        if with_tags:
            self._load_elements_tags(meshname, AElements, as_numpy)
        return AElements

    def get_mesh_elem_types_and_number(self, meshname):
        """Returns the list and types of elements tags defined on a mesh.

        :param str meshname: Name, Path, Index name or Alias of the Mesh group
            in dataset
        :return dict: keys are element types in the mesh and values are the
            number of elements for each element type.
        """
        elem_types = self.get_attribute('element_type', meshname)
        elem_number = self.get_attribute('Number_of_elements', meshname)
        return {elem_types[i]:elem_number[i] for i in range(len(elem_types))}

    def get_mesh_elem_tag(self, meshname, element_tag, as_numpy=True,
                          local_IDs=False):
        """Returns the elements IDs of an element tag for the mesh group.

        :param str meshname: Name, Path, Index name or Alias of the Mesh group
            in dataset
        :param str element_tag: name of the element tag whose connectivity
            must be returned. Can also be one of the element types contained
            in the mesh. In this case, the complete global element IDs for
            these elements is return
        :param bool as_numpy: if `True`, returns arrays in elements container
            as numpy array
        :param bool local_IDs: if `True`, returns the local elements IDs for
            the element type, i.e. the indexes of elements in the local
            connectivity array that can be obtain with get_mesh_elements.
        """
        # find out if the required element_tag is an element type of the mesh
        mesh_elements = self.get_mesh_elem_types_and_number(meshname)
        if element_tag in mesh_elements.keys():
            # return element type IDs
            offsets = self._get_mesh_elements_offsets(meshname)
            el_numbers = self.get_mesh_elem_types_and_number(meshname)
            tag = np.arange(el_numbers[element_tag])
            if not local_IDs:
                tag = tag + offsets[element_tag]
        else:
            # Return element tag IDs
            # Add prefix to element_tag name if needed
            if not element_tag.startswith('ET_'):
                el_tag_nodename = 'ET_' + element_tag
                el_tag_true_name = element_tag
            else:
                el_tag_true_name = element_tag[3:]
            # get mesh group
            mesh_group = self.get_node(meshname)
            # get path of Element tag
            ET_path = os.path.join(mesh_group._v_pathname,'Geometry',
                                         'ElementsTags',el_tag_nodename)
            # get element tag type
            tag = self.get_node(ET_path, as_numpy=as_numpy)
            if local_IDs:
                tag_names = self.get_mesh_elem_tags_names(meshname)
                offsets = self._get_mesh_elements_offsets(meshname)
                tag = tag - offsets[tag_names[el_tag_true_name]]
        return tag

    def get_mesh_elem_tags_names(self, meshname):
        """Returns the list and types of elements tags defined on a mesh.

        :param str meshname: Name, Path, Index name or Alias of the Mesh group
            in dataset
        :return dict: keys are element tag names in the mesh and values are the
            element type for each element tag.
        """
        mesh_group = self.get_node(meshname)
        Etag_list_indexname = mesh_group._v_name+'_ElTagsList'
        Etag_list = self.get_node(Etag_list_indexname)
        Etag_Etype_list_indexname = mesh_group._v_name+'_ElTagsTypeList'
        Etag_Etype_list = self.get_node(Etag_Etype_list_indexname)
        Tags_dict = {
            Etag_list[i].decode('utf-8'):Etag_Etype_list[i].decode('utf-8')
                     for i in range(len(Etag_list))}
        return Tags_dict

    def get_mesh_elem_tags_connectivity(self, meshname, element_tag):
        """Returns the list and types of elements tags defined on a mesh.

        :param str meshname: Name, Path, Index name or Alias of the Mesh group
            in dataset
        :param str element_tag: name of the element tag whose connectivity
            must be returned.
        """
        # get element tag type
        tags = self.get_mesh_elem_tags_names(meshname)
        el_type = tags[element_tag]
        # get connectivity of element type
        type_connectivity = self.get_mesh_elements(
                                    meshname, get_eltype_connectivity=el_type)
        # get local element IDs in element tag
        local_IDs = self.get_mesh_elem_tag(meshname, element_tag,
                                           local_IDs=True)
        return type_connectivity[local_IDs,:]

    def get_mesh_node_tags_names(self, meshname):
        """Returns the list of node tags defined on a mesh.

        :param str meshname: Name, Path, Index name or Alias of the Mesh group
            in dataset
        :return list node_tags: list of node tag names defined on this mesh
        """
        mesh_group = self.get_node(meshname)
        Ntag_list_indexname = mesh_group._v_name+'_NodeTagsList'
        NTags_list = self.get_node(Ntag_list_indexname).read()
        node_tags = []
        for tag in NTags_list:
            node_tags.append(tag.decode('utf-8'))
        return node_tags


    def get_image(self, imagename, with_fields=True):
        """Return data of an image group as a BasicTools mesh object.

        This methods gathers the data of a 2DImage or 3DImage group, including
        grid geometry and fields, into a BasicTools
        :class:`ConstantRectilinearMesh` object.

        :param imagename: Name, Path or Indexname of the image group to get
        :type imagename: str
        :param bool with_fields: If `True`, load the nodes and elements fields
            from the image group into the mesh object.
        :return: Returns a BasicTools rectilinear mesh object with image group
            data.
        :rtype: :class:`ConstantRectilinearMesh`
        """
        # Get image informations
        dimensions = self.get_attribute('nodes_dimension', imagename)
        spacing =  self.get_attribute('spacing', imagename)
        origin =  self.get_attribute('origin', imagename)
        # Create ConstantRectilinearMesh to serve as image_object
        image_object = ConstantRectilinearMesh(dim=len(dimensions))
        image_object.SetDimensions(dimensions)
        image_object.SetSpacing(spacing)
        image_object.SetOrigin(origin)
        # Get image fields
        image_group = self.get_node(imagename)
        FIndex_path = os.path.join(image_group._v_pathname,'Field_index')
        Field_index = self.get_node(FIndex_path)
        if with_fields and (Field_index is not None):
            for fieldname in Field_index:
                name = fieldname.decode('utf-8')
                field_type = self.get_attribute('field_type', name)
                if field_type == 'Nodal_field':
                    data = self.get_field(fieldname=name)
                    image_object.nodeFields[name] = data
                elif field_type == 'Element_field':
                    data = self.get_field(fieldname=name)
                    image_object.elemFields[name] = data
        return image_object

    def get_tablecol(self, tablename, colname):
        """Return a column of a table as a numpy array.

        :param str tablename: Name, Path, Index name or Alias of the table in
            dataset
        :param str colname: Name of the column to get in the table (analogous
            to name of the field to get in a Numpy structured array)
        :return numpy.array data: returns the queried column data as a
            `numpy.array`
        """
        data = None
        data_path = self._name_or_node_to_path(tablename)
        if data_path is None:
            msg = ('(get_tablecol) `tablename` not matched with a path or an'
                   ' indexname. No data returned')
            self._verbose_print(msg)
        else:
            if self._is_table(name=data_path):
                msg = '(get_tablecol) Getting column {} from : {}:{}'.format(
                        colname, self.h5_file, data_path)
                self._verbose_print(msg)
                table = self.get_node(name=data_path)
                data = table.col(colname)
            else:
                msg = ('(get_tablecol) Data is not an table node.')
                self._verbose_print(msg)
        return data

    def get_field(self, fieldname, unpad_field=True,
                  get_visualisation_field=False):
        """Return a padded or unpadded field from a grid data group as array.

        Use this method to get a mesh element wise field in its original form,
        i.e. bulk element fields (defined on elements of the same dimensonality
        than the mesh) or a boundary field (defined on elements of a lower
        dimensionality than the mesh).

        :param str fieldname: Name, Path, Index, Alias or Node of the field in
            dataset
        :param bool unpad_field: if `True` (default), remove the zeros added to
            to the field to comply with the mesh topology and return it with
            its original size (bulk or boundary field).
        """
        field_type = self.get_attribute('field_type', fieldname)
        if (field_type == 'IP_field') and get_visualisation_field:
            field_path = self.get_attribute('visualisation_field_path',
                                            fieldname)
            field = self.get_node(field_path, as_numpy=True)
        else:
            field = self.get_node(fieldname, as_numpy=True)
        padding = self.get_attribute('padding', fieldname)
        parent_mesh =  self.get_attribute('parent_grid_path', fieldname)
        if padding is not None:
            return self._mesh_field_unpadding(field, parent_mesh, padding)
        else:
            return field


    def get_node(self, name, as_numpy=False):
        """Return a HDF5 node in the dataset.

        :param str name: Name, Path, Index name or Alias of the Node in dataset
        :param bool as_numpy: if `True`, returns the Node as a `numpy.array`.
            If `False`, returns the node as a Node or Group object.
        :return: Return the node as a a :py:class:`tables.Node` or
            :py:class:`tables.Group` object depending on the nature of the
            node, or, returns it as a `numpy.array` if required and if the node
            is an array node.
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
                    if isinstance(self.content_index[name], list):
                        colname = self.content_index[name][1]
                elif name in node.colnames:
                    colname = name
                if (colname is not None):
                    node = node.col(colname)
            else:
                node = self.h5_dataset.get_node(node_path)
            if as_numpy and self._is_array(name) and (colname is None):
                # note : np.atleast_1d is used here to avoid returning a 0d
                # array when squeezing a scalar node
                transpose_indices = self.get_attribute('transpose_indices',
                                                       name)
                node = np.atleast_1d(node.read())
                if transpose_indices is not None:
                    node = node.transpose(transpose_indices)
                transpose_components = self.get_attribute('transpose_components',
                                                       name)
                if transpose_components is not None:
                    node = node[...,transpose_components]
        return node

    def get_node_info(self, name, as_string=False):
        """Print information on a node in the HDF5 tree.

        Prints node name, content, attributes, compression settings, path,
        childrens list if it is a group.

        :param str name: Name, Path, Index name or Alias of the HDF5 Node
        :param bool as_string: If `True` solely returns string representation.
            If `False`, prints the string representation.
        :return str s: string representation of HDF5 Node information
        """
        s = ''
        node_path = self._name_or_node_to_path(name)
        if self._is_array(node_path):
            s += self._get_array_node_info(name, as_string)
        else:
            s += self._get_group_info(name, as_string)
        return s

    def get_node_compression_info(self, name, as_string=False):
        """Print the compression settings of an array node.

        :param str name:Name, Path, Index name or Alias of the HDF5 Node
        :param bool as_string: If `True` solely returns string representation.
        If `False`, prints the string representation.
        :return str s: string representation of HDF5 Node compression settings
        """
        s = ''
        if not self.__contains__(name):
            raise tables.NodeError('node `{}` not in {} instance'.format(
                                   name, self.__class__.__name__))
        node_path = self._name_or_node_to_path(name)
        if self._is_array(node_path):
            N = self.get_node(name)
            s += 'Compression options for node `{}`:\n\t'.format(name)
            s += repr(N.filters).strip('Filters(').strip(')')
        else:
            s += '{} is not a data array node'.format(name)
        if not as_string:
            print(s)
        return s+'\n'

    def get_dic_from_attributes(self, nodename):
        """Get all attributes from a HDF5 Node in the dataset as a dictionary.

        :param str nodename: Name, Path, Index name or Alias of the HDF5 group
        :return: Dictionary of the form ``{'Attribute_name': Attribute_value}``
        """
        Node = self.get_node(nodename)
        dic = {}
        for key in Node._v_attrs._f_list():
            dic[key] = Node._v_attrs[key]
        return dic

    def get_attribute(self, attrname, nodename):
        """Get a specific attribute value from a HDF5 Node in the dataset.

        :param str attrname: name of the attribute to get
        :param str nodename: Name, Path, Index name or Alias of the HDF5 group
        :return: Value of the attribute
        """
        attribute = None
        data_path = self._name_or_node_to_path(nodename)
        if (data_path is None):
            self._verbose_print(' (get_attribute) neither indexname nor'
                                ' node_path passed, node return aborted')
        else:
            try:
                attribute = self.h5_dataset.get_node_attr(where=data_path,
                                                          attrname=attrname)
            except AttributeError:
                self._verbose_print(' (get_attribute) node {} has no attribute'
                                    ' `{}`'.format(nodename,attrname))
                return None
            if isinstance(attribute, bytes):
                attribute = attribute.decode()
        return attribute

    def get_file_disk_size(self, print_flag=True, convert=True):
        """Get the disk size of the dataset.

        :param bool print_flag: print the disk size if `True`
        :param bool convert: convert disk size to a suitable memory unit if
            `True`. If `False`, return result in bytes.
        :return float fsize: Disk size of the dataset in `unit`
        :return str unit: Unit of `fsize`. `bytes` is the default
        """
        units = ['bytes', 'Kb',  'Mb', 'Gb', 'Tb', 'Pb']
        fsize = os.path.getsize(self.h5_path)
        k = 0
        unit = units[k]
        if convert:
            while fsize/1024 > 1:
                k = k+1
                fsize = fsize/1024
                unit = units[k]
        if print_flag:
            print('File size is {:9.3f} {} for file \n {}'
                  ''.format(fsize, unit, self.h5_file))
        return fsize, unit

    def get_node_disk_size(self, nodename, print_flag=True, convert=True):
        """Get the disk size of a HDF5 node.

        :param str nodename: Name, Path, Index name or Alias of the HDF5 node
        :param bool print_flag: print the disk size if `True`
        :param bool convert: convert disk size to a suitable memory unit if
            `True`. If `False`, return result in bytes.
        :return float fsize: Disk size of the dataset in `unit`
        :return str unit: Unit of `fsize`. `bytes` is the default
        """
        units = ['bytes', 'Kb', 'Mb', 'Gb', 'Tb', 'Pb']
        if not self.__contains__(nodename):
            raise tables.NodeError('node `{}` not in {} instance'
                                   ''.format(nodename,
                                             self.__class__.__name__))
        if not self._is_array(nodename):
            print('Node {} is not a data array node'.format(nodename))
            return None
        node = self.get_node(nodename)
        nsize = node.size_on_disk
        k = 0
        unit = units[k]
        if convert:
            while nsize/1024 > 1:
                k = k+1
                nsize = nsize/1024
                unit = units[k]
        if print_flag:
            print('Node {} size on disk is {:9.3f} {}'.format(nodename,
                  nsize, unit))
        return nsize, unit

    def get_sample_name(self):
        """Return the sample name."""
        return self.get_attribute(attrname='sample_name', nodename='/')

    def set_sample_name(self, sample_name):
        """Set the sample name.

        :param str sample_name: a string for the sample name.
        """
        self.add_attributes({'sample_name': sample_name}, '/')

    def get_description(self, node):
        """Get the string describing this node.

        By defaut the sample description is returned, from the root HDF5 Group.

        :param str nodename: the path or name of the node of interest.
        """
        return self.get_attribute(attrname='description', nodename=node)

    def set_description(self, description, node='/'):
        """Set the description of a node.

        By defaut this method sets the description of the complete sample, in
        the root HDF5 Group.

        :param str description: a string for the description of the node or
            sample.
        :param str node: the path or name of the node of interest ('/' by
                                                                      default).
        """
        self.add_attributes({'description': description}, node)
        return

    def set_new_indexname(self, nodename, new_indexname):
        """Change the indexname of a node in the dataset.

        Usefull to solve indexname duplicates issues that can arises when
        automatically adding elements to the dataset, that have the same name.

        :param nodename: Name, Path, Indexname or Alias of the node whose
            indexname is to be changed
        :type nodename: str
        :param new_indexname: New indexname for the node
        :type new_indexname: str
        """
        path = self._name_or_node_to_path(nodename)
        old_indexname = self.get_indexname_from_path(path)
        index_content = self.content_index.pop(old_indexname)
        self.content_index[new_indexname] = index_content
        return


    def set_voxel_size(self, image_group, voxel_size):
        """Set voxel size for an HDF5/XDMF image data group.

        The values are registered in the `spacing` Attribute of the 3DImage
        group.

        :param str image_data_group: Name, Path, Index name or Alias of the
            3DImage group
        :param np.array voxel_size: (dx, dy, dz) array of the voxel size in
            each dimension of the 3Dimage
        """
        old_spacing = self.get_attribute('spacing', image_group)
        if isinstance(voxel_size, float):
            voxel_size = np.ones(shape=(len(old_spacing),))*voxel_size
        if len(old_spacing) != len(voxel_size):
            raise ValueError('Dimension mismatch between image group old'
                             f' grid spacing {old_spacing} and inputed'
                             f' new grid spacing {voxel_size}')
        self.add_attributes({'spacing': np.array(voxel_size)},
                            image_group)
        xdmf_geometry = self._find_xdmf_geometry(image_group)
        spacing_node = xdmf_geometry.getchildren()[1]
        if len(voxel_size) == 2:
            VoxSize = voxel_size[[1,0]]
        elif len(voxel_size) == 3:
            VoxSize = voxel_size[[2,1,0]]
        spacing_text = str(VoxSize).strip('[').strip(']').replace(',', ' ')
        spacing_node.text = spacing_text
        self.sync()
        return

    def set_origin(self, image_group, origin):
        """Set origin coordinates for an HDF5/XDMF image data group.

        The origin corresponds to the first vertex of the first voxel, that is
        referenced by the [0,0,0] elements of arrays in the 3DImage group. The
        values are registered in the `origin` Attribute of the 3DImage group.

        :param str image_data_group: Name, Path, Index name or Alias of the
            3DImage group
        :param np.array voxel_size: (Ox, Oy, Oz) array of the coordinates in
            each dimension of the origin of the 3Dimage
        """
        old_origin = self.get_attribute('origin', image_group)
        if len(old_origin) != len(origin):
            raise ValueError('Dimension mismatch between image group origin'
                             f' {old_origin} and inputed new origin'
                             f' {origin}')
        self.add_attributes({'origin': origin}, image_group)
        xdmf_geometry = self._find_xdmf_geometry(image_group)
        if len(origin) == 2:
            Or = origin[[1,0]]
        elif len(origin) == 3:
            Or = origin[[2,1,0]]
        origin_node = xdmf_geometry.getchildren()[0]
        origin_text = str(Or).strip('[').strip(']').replace(',', ' ')
        origin_node.text = origin_text
        self.sync()
        return

    def set_tablecol(self, tablename, colname, column):
        """Store an array into a structured table column.

        If the column is not in the table description, a new field
        corresponding to the inputed column is added to the table description.

        :param str tablename: Name, Path, Index name or Alias of the table
        :param str colname: Name of the column to set (analogous to name of a
            field in a `Numpy` structured array)
        :param np.array column: array of values to set as column of the table.
            It's shape must match the column shape in table description.
        """
        if not self._is_table(tablename):
            raise tables.NodeError('{} is not a structured table node'
                                   ''.format(tablename))
        table = self.get_node(tablename)
        col_shape = self.get_tablecol(tablename, colname).shape
        if (column.shape != col_shape):
            raise ValueError('inputed column shape {} does not match the shape'
                             '{} of column {} in table {}'
                             ''.format(column.shape, col_shape, colname,
                                       tablename))
        table.modify_column(column=column, colname=colname)
        table.flush()
        return

    def set_nodes_compression_chunkshape(self, node_list=None, chunkshape=None,
                                         filters=None, **keywords):
        """Set compression options for a list of nodes in the dataset.

        :param list node_list: list of Name, Path, Index name or Alias of the
            HDF5 array nodes where to set the compression settings.
        :param tuple  chunkshape: The shape of the data chunk to be read or
            written in a single HDF5 I/O operation
        :param Filters filters: instance of :py:class:`tables.Filters` class
            specifying compression settings.

        .. rubric:: Additional keyword arguments

        Compression settings can be passed as keyword arguments to this method.
        They are the `Filters
        <https://www.pytables.org/_modules/tables/filters.html#Filters>`_
        class constructor parameters (see `PyTables` documentation for details)

        :param str complevel: Compression level for data. Allowed range is 0-9.
            A value of 0 (the default) disables compression.
        :param str complib: Compression library to use. Possibilities are:
            zlib' (the default), 'lzo', 'bzip2' and 'blosc'.
        :param bool shuffle:  Whether or not to use the *Shuffle* filter in the
            HDF5 library (may improve compression ratio).
        :param bool bitshuffle: Whether or not to use the *BitShuffle* filter
            in the Blosc library (may improve compression ratio).
        :param bool fletcher32: Whether or not to use the *Fletcher32* filter
            in the HDF5 library. This is used to add a checksum on each data
            chunk.
        :param int least_significant_digit:
            If specified, data will be truncated using
            ``around(scale*data)/scale``, where
            ``scale = 2**least_significant_digit``.
            In conjunction with enabling compression, this produces 'lossy',
            but significantly more efficient compression.

        .. important:: If the new compression settings reduce the size of the
            node in the dataset, the file size will not be changed. This is a
            standard behavior for HDF5 files, that preserves freed space in
            disk to add additional data in the future. If needed, use the
            :func:`repack_file` method to reduce file disk size after changing
            compression settings. This method is also called by the class
            instance destructor.

        .. note:: If compression settings are passed as additional keyword
            arguments, they are prioritised over the settings in the inputed
            Filter object.
        """
        if node_list is None:
            node_list = []
            for node in self.h5_dataset.root:
                if self._is_array(node):
                    node_list.append(node)
        for nodename in node_list:
            self.set_chunkshape_and_compression(nodename, chunkshape, filters,
                                                **keywords)
        return

    def set_chunkshape_and_compression(self, node, chunkshape=None,
                                       filters=None, **keywords):
        """Set the chunkshape and compression settings for a HDF5 array node.

        :param str node: Name, Path, Index name or Alias of the node
        :param tuple  chunkshape: The shape of the data chunk to be read or
            written in a single HDF5 I/O operation
        :param Filters filters: instance of :py:class:`tables.Filters` class
            specifying compression settings.

        .. rubric:: Additional keyword arguments

        Compression settings can be passed as keyword arguments to this method.
        See :func:`set_nodes_compression_chunkshape`.

        .. important:: If the new compression settings reduce the size of the
            node in the dataset, the file size will not be changed. This is a
            standard behavior for HDF5 files, that preserves freed space in
            disk to add additional data in the future. If needed, use the
            :func:`repack_file` method to reduce file disk size after changing
            compression settings. This method is also called by the class
            instance destructor.

        .. note:: If compression settings are passed as additional keyword
            arguments, they are prioritised over the settings in the inputed
            Filter object.
        """
        if not self._is_array(node):
            msg = ('(set_chunkshape) Cannot set chunkshape or compression'
                   ' settings for a non array node')
            raise tables.NodeError(msg)
        node_tmp = self.get_node(node)
        nodename = node_tmp._v_name
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
            description = node_tmp.description
            new_array = self.add_table(location=node_path, name=nodename,
                                       description=description,
                                       indexname=node_indexname,
                                       chunkshape=node_chunkshape,
                                       replace=True, data=array,
                                       filters=node_filters, **keywords)
        else:
            new_array = self.add_data_array(location=node_path, name=nodename,
                                            indexname=node_indexname,
                                            array=array, filters=node_filters,
                                            chunkshape=node_chunkshape,
                                            replace=True, **keywords)
        for alias in node_aliases:
            self.add_alias(aliasname=alias, indexname=node_indexname)
        if self._verbose:
            self._verbose_print(self.get_node_compression_info(
                new_array._v_pathname))
        return

    def set_default_compression(self):
        """Return a Filter object with defaut compression parameters."""
        Filters = tables.Filters(complib='zlib', complevel=0, shuffle=True)
        return Filters

    def set_global_compression_opt(self, **keywords):
        """Set default compression settings for the dataset.

        .. rubric:: Additional keyword arguments

        Compression settings can be passed as keyword arguments to this method.
        See :func:`set_nodes_compression_chunkshape`.
        """
        # initialize general compression Filters object (from PyTables)
        # with no compression (default behavior)
        default = False
        # ------ check if default compression is required
        if 'default_compression' in keywords:
            default = True
        self.Filters = self._get_compression_opt(default=default, **keywords)

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
        """Set the verbosity of the instance methods to inputed boolean."""
        self._verbose = verbosity
        return

    def rename_node(self, nodename, newname, replace=False,
                    new_indexname=None):
        """Rename a node in the HDF5 tree, XDMF file and content index.

        This method do not change the indexname of the node, if one exists.

        :param nodename: Name, Path or Index name of the node to modify
        :type nodename: str
        :param newname: New name to give to the HDF5 node
        :type newname: str
        :param replace: If `True`, overwrite a possibily existing node with
            name `newname` defaults to False
        :type replace: bool, optional
        """
        self.sync()
        node = self.get_node(nodename)
        indexname = self.get_indexname_from_path(node._v_pathname)
        if new_indexname is not None:
            self.content_index.pop(indexname)
            indexname = new_indexname
        elif indexname == node._v_name:
            self.content_index.pop(indexname)
            indexname = newname
        # change nodename in XDMF file
        xdmf_lines = []
        with open(self.xdmf_path, 'r') as f:
            old_xdmf_lines = f.readlines()
        for line in old_xdmf_lines:
            xdmf_lines.append(line.replace(node._v_name, newname))
        with open(self.xdmf_path, 'w') as f:
            f.writelines(xdmf_lines)
        # change HDF5 node name
        self.h5_dataset.rename_node(node, newname, overwrite=replace)
        # change index
        self.content_index[indexname] = node._v_pathname
        self.xdmf_tree = etree.parse(self.xdmf_path)
        self.sync()
        return


    def remove_node(self, name, recursive=False):
        """Remove a node from the dataset.

        :param str name: Name, Path, Index name or Alias of the node to remove
        :param bool recursive: if `True` and the node is a Group, removes all
            childrens of the node as well.

        .. important:: After node removal, the file size will not be changed.
            This is a standard behavior for HDF5 files, that preserves freed
            space in disk to add additional data in the future. If needed, use
            the :func:`repack_file` method to reduce file disk size after
            changing compression settings. This method is also called by the
            class instance destructor.
        """
        node_path = self._name_or_node_to_path(name)
        if node_path is None:
            msg = ('(remove_node) Node name does not fit any hdf5 path'
                   ' nor index name. Node removal aborted.')
            self._verbose_print(msg)
            return
        Node = self.get_node(node_path)
        isGroup = (Node._v_attrs.CLASS == 'GROUP')
        if (isGroup) and not recursive:
            msg = ('Node {} is a hdf5 group. Use `recursive=True` keyword'
                  ' argument to remove it and its childrens.'
                  ''.format(node_path))
            self._verbose_print(msg)
            return

        # Remove visualisation field if node is a IP field node with an
        # additionnal visualisation values array
        IP_vis = self.get_attribute('visualisation_field_path', Node)
        if IP_vis is not None:
            IP = self.get_node(IP_vis)
            self.remove_node(IP)

        # Remove HDF5 node and its childrens
        self._verbose_print('Removing  node {} in content index....'
                            ''.format(Node._v_pathname))
        if isGroup and recursive:
            for child, child_node in Node._v_children.items():
                self.remove_node(child_node, recursive=True)
            self._remove_from_index(node_path=Node._v_pathname)
            # remove node in xdmf tree
            self._remove_from_xdmf(Node)
            Node._f_remove(recursive=True)
        else:
            self._remove_from_index(node_path=Node._v_pathname)
            # remove node in xdmf tree
            self._remove_from_xdmf(Node)
            Node.remove()
        # synchronize HDF5 and XDMF file with node removal
        self.sync()
        self._verbose_print('Node {} sucessfully removed'.format(name))
        return

    def repack_h5file(self):
        """Overwrite hdf5 file with a copy of itself to recover disk space.

        Manipulation to recover space leaved empty when removing data from
        the HDF5 tree or reducing a node space by changing its compression
        settings. This method is called also by the class destructor.
        """
        # BUG: copy_file from tables returns exception with large mesh and lot
        # of elsets. Use external utility ptrepack ? For now, taken out of
        # class destructor
        # TODO: improve this method --> add sync(), use Pathlib
        head, tail = os.path.split(self.h5_path)
        tmp_file = os.path.join(head, 'tmp_'+tail)
        self.h5_dataset.copy_file(tmp_file)
        self.h5_dataset.close()
        shutil.move(tmp_file, self.h5_path)
        self.h5_dataset = tables.File(self.h5_path, mode='r+')
        return

    @staticmethod
    def copy_sample(src_sample_file, dst_sample_file, overwrite=False,
                    get_object=False, new_sample_name=None, autodelete=False):
        """Initiate a new SampleData object and files from existing dataset.

        :param src src_sample_file: name of the dataset file to copy.
        :param src dst_sample_file: name of the new dataset files.
        :param bool overwrite: set to `True` to overwrite an existing dataset
            file with name `dst_sample_file` when copying.
        :param bool get_object: if `True` returns the SampleData instance
        :param str new_sample_name: name of the sample in the new dataset
        :param bool autodelete: remove copied dataset files when copied
            instance is destroyed.
        """
        # TODO: improve method --> use Pathlib
        sample = SampleData(filename=src_sample_file)
        if new_sample_name is None:
            new_sample_name = sample.get_attribute('sample_name', '/')
        # copy HDF5 file
        dst_sample_file_h5 = os.path.splitext(dst_sample_file)[0] + '.h5'
        dst_sample_file_xdmf = os.path.splitext(dst_sample_file)[0] + '.xdmf'
        sample.h5_dataset.copy_file(dst_sample_file_h5, overwrite=overwrite)
        # copy XDMF file
        dst_xdmf_lines = []
        with open(sample.xdmf_path, 'r') as f:
            src_xdmf_lines = f.readlines()
        _, new_file = os.path.split(dst_sample_file_h5)
        for line in src_xdmf_lines:
            dst_xdmf_lines.append(line.replace(sample.h5_file, new_file))
        with open(dst_sample_file_xdmf, 'w') as f:
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

    def create_elset_ids_field(self, meshname=None, store=True, fieldname=None,
                               get_sets_IDs=True, tags_prefix='elset',
                               remove_elset_fields=False):
        """Create an element tag Id field on the inputed mesh.

        Creates a element wise field from the provided mesh,
        adding to each element the value of the Elset it belongs to.

        .. warning::

            - CAUTION : the methods is designed to work with non intersecting
              element tags/sets. In this case, the produce field will indicate
              the value of the last elset containing it for each element.

        :param str mesh: Name, Path or index name of the mesh on which an
            orientation map element field must be constructed
        :param bool store: If `True`, store the field on the mesh
        :param bool get_sets_IDs: If `True`, get the sets ID numbers from their
            names by substracting the input prefix. If `False`, use the set
            position in the mesh elset list as ID number.
        :param str tags_prefix: Remove from element sets/tags names
            prefix to determine the set/tag ID. This supposes that sets
            names have the form prefix+ID
        :param bool remove_elset_fields: If `True`, removes the elset
            indicator fields after construction of the elset id field.
            (default is `False`)
        """
        if meshname is None:
                raise ValueError('meshname do not refer to an existing mesh')
        if not(self._is_mesh(meshname)) or  self._is_empty(meshname):
                raise ValueError('meshname do not refer to a non empty mesh'
                                 'group')
        # create empty element vector field
        Nelements = int(self.get_attribute('Number_of_elements',meshname))
        mesh = self.get_node(meshname)
        El_tag_path = os.path.join(mesh._v_pathname,'Geometry','ElementsTags')
        ID_field = np.zeros((Nelements,1),dtype=float)
        elem_tags = self.get_mesh_elem_tags_names(meshname)
        # if mesh is provided
        i = 0
        for set_name, set_type in elem_tags.items():
            elset_path = os.path.join(El_tag_path, 'ET_'+set_name)
            element_ids = self.get_node(elset_path, as_numpy=True)
            if get_sets_IDs:
                set_ID = int(set_name.strip(tags_prefix))
            else:
                set_ID = i
                i += 1
            ID_field[element_ids] = set_ID
            if remove_elset_fields:
                field_path = os.path.join(El_tag_path, 'field_'+set_name)
                self.remove_node(field_path)
        if store:
            if fieldname is None:
                fieldname = meshname+'_elset_ids'
            self.add_field(gridname=meshname, fieldname=fieldname,
                           array=ID_field, replace=True,
                           complib='zlib', complevel=1, shuffle=True)
        return ID_field

    # =========================================================================
    #  SampleData private methods
    # =========================================================================
    def _init_file_object(self, sample_name='', sample_description='',
                          **keywords):
        """Initiate or create PyTable HDF5 file object."""
        try:
            self.h5_dataset = tables.File(self.h5_path, mode='r+')
            self._verbose_print('-- Opening file "{}" '.format(self.h5_file),
                                line_break=False)
            self._file_exist = True
            self._init_xml_tree()
            self.Filters = self.h5_dataset.filters
            self._init_data_model()
            self._verbose_print('**** FILE CONTENT ****')
            self._verbose_print(SampleData.__repr__(self))
        except IOError:
            self._file_exist = False
            self._verbose_print('-- File "{}" not found : file'
                                ' created'.format(self.h5_file),
                                line_break=True)
            self.h5_dataset = tables.File(self.h5_path, mode='a')
            self._init_xml_tree()
            # add sample name and description
            self.h5_dataset.root._v_attrs.sample_name = sample_name
            self.h5_dataset.root._v_attrs.description = sample_description
            # get compression options
            Compression_keywords = {k: v for k, v in keywords.items() if k in
                                    COMPRESSION_KEYS}
            self.set_global_compression_opt(**Compression_keywords)
            # Generic Data Model initialization
            self._init_data_model()
        return

    def _init_data_model(self):
        """Initialize the minimal data model specified for the class."""
        content_paths, content_type = self.minimal_data_model()
        self.minimal_content = content_paths
        self._init_content_index()
        self._verbose_print('Minimal data model initialization....')
        # Determine maximum path level in data model elements
        max_path_level = 0
        for key, value in content_paths.items():
            max_path_level = max(value.count('/'), max_path_level)
        for level in range(max_path_level+1):
            for key, value in content_paths.items():
                if value.count('/') != level:
                    continue
                head, tail = os.path.split(value)
                if self.h5_dataset.__contains__(content_paths[key]):
                    if self._is_table(content_paths[key]):
                        msg = ('Updating table {}'.format(content_paths[key]))
                        self._verbose_print(msg)
                        self._update_table_columns(
                            tablename=content_paths[key],
                            Description=content_type[key])
                    if self._is_empty(content_paths[key]):
                        self._verbose_print('Warning: node {} specified in the'
                                            ' minimal data model for this class'
                                            ' is empty'
                                            ''.format(content_paths[key]))
                    continue
                elif content_type[key] == 'Group':
                    msg = ('Adding empty Group {}'.format(content_paths[key]))
                    self.add_group(groupname=tail, location=head,
                                   indexname=key, replace=False)
                elif content_type[key] == '3DImage':
                    msg = ('Adding empty 3DImage {}'
                           ''.format(content_paths[key]))
                    self.add_image(imagename=tail, indexname=key, location=head)
                elif content_type[key] == 'Mesh':
                    msg = ('Adding empty Mesh  {}'.format(content_paths[key]))
                    self.add_mesh(meshname=tail, indexname=key, location=head)
                elif content_type[key] == 'Array':
                    msg = ('Adding empty Array  {}'.format(content_paths[key]))
                    empty_array = np.array([0])
                    self.add_data_array(location=head, name=tail,
                                        array=empty_array, empty=True,
                                        indexname=key)
                elif (isinstance(content_type[key], tables.IsDescription)
                      or issubclass(content_type[key], tables.IsDescription)):
                    msg = ('Adding empty Table  {}'.format(content_paths[key]))
                    self.add_table(location=head, name=tail, indexname=key,
                                   description=content_type[key])
        self._verbose_print('Minimal data model initialization done\n')
        return

    def _init_xml_tree(self):
        """Read xml tree structured in .xdmf file or initiate one."""
        try:
            file_parser = etree.XMLParser(remove_blank_text=True)
            with open(self.xdmf_path, 'rb') as source:
                self.xdmf_tree = etree.parse(source, parser=file_parser)
        except OSError:
            # Non existent xdmf file.
            # A new .xdmf is created with the base node Xdmf and one Domain
            self._verbose_print('-- File "{}" not found : file'
                                ' created'.format(self.xdmf_file),
                                line_break=False)
            # create root element of xdmf tree structure
            E = lxml.builder.ElementMaker(
                    namespace="http://www.w3.org/2003/XInclude",
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

    def _init_content_index(self):
        """Initialize content_index dictionary."""
        self.content_index = {}
        self.aliases = {}
        if self._file_exist:
            self.content_index = self.get_dic_from_attributes(
                                                    nodename='/Index')
            self.aliases = self.get_dic_from_attributes(
                                                    nodename='/Index/Aliases')
        else:
            self.h5_dataset.create_group('/', name='Index')
            self.h5_dataset.create_group('/Index', name='Aliases')
        return

    def _check_SD_array_init(self, arrayname='', location='/', replace=False):
        """Safety check to create data array into location in dataset."""
        location_path = self._name_or_node_to_path(location)
        if location_path is None:
            msg = ('No location `{}`, cannot create array/table {}.'
                   ' Create parent groups before adding the array/table.'
                   ''.format(arrayname, location))
            raise tables.NodeError(msg)
        else:
            # check location nature
            if not(self._get_node_class(location) == 'GROUP'):
                msg = ('Location {} is not a HDF5 Group. Cannot create array/'
                       'table there.'.format(location))
                raise tables.NodeError(msg)
            # check if array location exists and remove node if asked
            array_path = os.path.join(location_path, arrayname)
            if self.__contains__(array_path):
                empty = self.get_attribute('empty', array_path)
                if replace:
                    msg = ('Existing node {} will be overwritten to recreate '
                           'array/table'.format(array_path))
                    if not(empty):
                        self._verbose_print(msg)
                    self.remove_node(array_path, recursive=True)
                else:
                    msg = ('Array/table {} already exists. To overwrite, use '
                           'optional argument "replace=True"'
                           ''.format(array_path))
                    raise tables.NodeError(msg)

    def _init_SD_group(self, groupname='', location='/',
                       group_type='Group', replace=False):
        """Create or fetch a SampleData Group and returns it."""
        Group = None
        # init flags
        fetch_group = False
        # sanity checks
        if groupname == '':
            raise ValueError('Cannot create Group. Groupname must be'
                             ' specified')
        if group_type not in SD_GROUP_TYPES:
            raise ValueError('{} is not a valid group type. Use on of {}'
                             ''.format(group_type,SD_GROUP_TYPES))
        location = self._name_or_node_to_path(location)
        group_path = os.path.join(location, groupname)
        # check existence and emptiness
        if self.__contains__(group_path):
            if self._is_grid(group_path):
                if self._is_empty(group_path):
                    fetch_group = True
            if not(fetch_group) and replace:
                self._verbose_print('Removing group {} to replace it by new'
                                    ' one.'.format(groupname))
                self.remove_node(name=group_path,recursive=True)
            if not(fetch_group or replace):
                msg = ('Group {} already exists. Set arg. `replace=True` to'
                       'to replace it by a new Group.'.format(groupname))
                raise tables.NodeError(msg)
        # create or fetch group
        if fetch_group:
            Group = self.get_node(group_path)
        else:
            self._verbose_print('Creating {} group `{}` in file {} at {}'
                                ''.format(group_type, groupname,
                                          self.h5_file,location))
            Group = self.h5_dataset.create_group(where=location,
                                                 name=groupname,
                                                 title=groupname,
                                                 createparents=True)
            self.add_attributes(dic={'group_type': group_type}, nodename=Group)
        return Group

    @staticmethod
    def _merge_dtypes(dtype1, dtype2):
        """Merge 2 numpy.void dtypes to creates a new one."""
        descr = dtype1.descr
        for item in dtype2.descr:
            if not(item in descr) and not(item[0]==''):
                descr.append(item)
        return np.dtype(descr)

    def _update_table_columns(self, tablename, Description):
        """Extends table with new fields in input Description."""
        table = self.get_node(tablename)
        current_desc = table.description
        current_dtype = tables.dtype_from_descr(table.description)
        desc_dtype = tables.dtype_from_descr(Description)
        new_dtype = SampleData._merge_dtypes(current_dtype, desc_dtype)
        new_descr = tables.descr_from_dtype(new_dtype)[0]
        if current_dtype == new_dtype:
            self._verbose_print('Nothing to update for table `{}`'
                                ''.format(tablename))
            return
        self._verbose_print('Updating `{}` with fields {}'
                            ''.format(tablename, desc_dtype.fields))
        self._verbose_print('New table description is `{}`'
                            ''.format(new_dtype.fields))
        Nrows = table.nrows
        tab_name = table._v_name
        tab_indexname = self.get_indexname_from_path(table._v_pathname)
        tab_path = os.path.dirname(table._v_pathname)
        tab_chunkshape = table.chunkshape
        tab_filters = table.filters
        # Create a new array with modified dtype
        data = np.array(np.zeros((Nrows,)), dtype=new_dtype)
        self._verbose_print('data is:',data)
        # Get data from old table
        for key in current_desc._v_names:
            data[key] = self.get_tablecol(tablename=tablename,
                                          colname=key)
        # get table aliases
        if self.aliases.__contains__(tab_indexname):
            tab_aliases = self.aliases[tab_indexname]
        else:
            tab_aliases = []
        # remove old table
        self.remove_node(tab_name)
        # create new table
        new_tab = self.add_table(location=tab_path, name=tab_name,
                                 description=new_descr,
                                 indexname=tab_indexname,
                                 chunkshape=tab_chunkshape, replace=True,
                                 data=data, filters=tab_filters)
        for alias in tab_aliases:
            self.add_alias(aliasname=alias, indexname=new_tab._v_pathname)
        return

    def _remove_from_index(self, node_path):
        """Remove a hdf5 node from content index dictionary."""
        try:
            key = self.get_indexname_from_path(node_path)
            removed_path = self.content_index.pop(key)
            if key in self.aliases:
                self.aliases.pop(key)
            self._verbose_print('item {} : {} removed from context index'
                                ' dictionary'.format(key, removed_path))
        except:
            self._verbose_print('node {} not found in content index values for'
                                'removal'.format(node_path))
        return

    def _remove_from_xdmf(self, nodename):
        """Remove a Grid or Attribute Node from the xdmf tree."""
        if self._is_grid(nodename):
            xdmf_grid = self._find_xdmf_grid(nodename)
            p = xdmf_grid.getparent()
            p.remove(xdmf_grid)
        elif self._is_field(nodename):
            xdmf_field, xdmf_grid = self._find_xdmf_field(nodename)
            xdmf_grid.remove(xdmf_field)
        return

    def _find_xdmf_grid(self, gridname):
        name = self.get_attribute('xdmf_gridname', gridname)
        for el in self.xdmf_tree.iterfind('.//Grid'):
            if el.get('Name') == name:
                return el
        return None

    def _find_xdmf_geometry(self, gridname):
        xdmf_grid = self._find_xdmf_grid(gridname)
        for el in xdmf_grid:
            if el.tag == 'Geometry':
                return el
        return None

    def _find_xdmf_topologyy(self, gridname):
        xdmf_grid = self._find_xdmf_grid(gridname)
        for el in xdmf_grid:
            if el.tag == 'Topology':
                return el
        return None

    def _find_xdmf_field(self, fieldnodename):
        gridname = self.get_attribute('xdmf_gridname', fieldnodename)
        fieldname = self.get_attribute('xdmf_fieldname', fieldnodename)
        xdmf_grid = self._find_xdmf_grid(gridname)
        for el in xdmf_grid:
            if el.get('Name') == fieldname:
                return el, xdmf_grid
        return None

    def _name_or_node_to_path(self, name_or_node):
        """Return path of `name` in content_index dic or HDF5 tree."""
        path = None
        # name_or_node is a Node
        if isinstance(name_or_node, tables.Node):
            return name_or_node._v_pathname
        # name or node is none
        if name_or_node is None:
            return None
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
                    count = count + 1
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
                       'distinguish nodes.'.format(count, name_or_node,
                                                   path_list))
                self._verbose_print(msg)
                path = None
        return path

    def _is_empty(self, name):
        """Find out if name or path references an empty node."""
        if not self.__contains__(name):
            return True
        if self._is_table(name):
            tab = self.get_node(name)
            return tab.nrows <= 0
        else:
            return self.get_attribute('empty', name)

    def _is_image(self, name):
        """Find out if name or path references an image groupe."""
        return self._get_group_type(name) in SD_IMAGE_GROUPS.values()

    def _is_mesh(self, name):
        """Find out if name or path references an image groupe."""
        return self._get_group_type(name) in SD_MESH_GROUPS.values()

    def _is_array(self, name):
        """Find out if name or path references an array dataset."""
        name2 = self._name_or_node_to_path(name)
        Class = self._get_node_class(name2)
        List = ['CARRAY', 'EARRAY', 'VLARRAY', 'ARRAY', 'TABLE']
        return Class in List

    def _is_table(self, name):
        """Find out if name or path references an array dataset."""
        return self._get_node_class(name) == 'TABLE'

    def _is_group(self, name):
        """Find out if name or path references a HDF5 Group."""
        return self._get_node_class(name) == 'GROUP'

    def _is_grid(self, name):
        """Find out if name or path references a image or mesh HDF5 Group."""
        return self._get_group_type(name) in SD_GRID_GROUPS

    def _is_children_of(self, node, parent_group):
        """Find out if location is a subgroup of the grid."""
        if not self.__contains__(node):
            return False
        if not self.__contains__(parent_group):
            return False
        node_path = self._name_or_node_to_path(node)
        bool_return = False
        group = self.get_node(parent_group)
        for n in group._f_iter_nodes():
            if n._v_pathname == node_path:
                bool_return = True
                break
            else:
                if self._is_group(n._v_pathname):
                    bool_return = self._is_children_of(node, n._v_pathname)
                if bool_return:
                    break
        return bool_return


    def _is_field(self, fieldname):
        """Checks conditions to consider node `name` as a field data node."""
        test = self.get_attribute('field_type', fieldname)
        return (test is not None)

    def _is_in_index(self, name):
        return (name in self.content_index)

    def _is_alias(self, name):
        """Check if name is an HDF5 node alias."""
        Is_alias = False
        for item in self.aliases:
            if (name in self.aliases[item]):
                Is_alias = True
                break
        return Is_alias

    def _is_alias_for(self, name):
        """Return the indexname for which input name is an alias."""
        Indexname = None
        for item in self.aliases:
            if (name in self.aliases[item]):
                Indexname = item
                break
        return Indexname

    def _check_image_object_support(self, image_object):
        """Return `True` if image_object type is supported by the class."""
        if not(isinstance(image_object, ConstantRectilinearMesh)):
            raise ValueError('WARNING : unknown image object. Supported objects'
                             ' are BasicTools ConstantRectilinearMesh'
                             ' instances.')

    def _check_mesh_object_support(self, mesh_object):
        """Return `True` if image_object type is supported by the class."""
        if not(isinstance(mesh_object, MeshBase)):
            raise ValueError('WARNING : unknown mesh object. Supported objects'
                             ' are BasicTools ConstantRectilinearMesh'
                             '  instances.')

    def _check_field_compatibility(self, gridname, field_shape):
        """Check if field dimension is compatible with the storing location."""
        group_type = self._get_group_type(gridname)
        if self._is_empty(gridname):
            # method create_image from array
            raise ValueError('{} is an empty grid. Use add_image, add_mesh,'
                             ' or add_imamge_from_field to initialize grid'
                             ''.format(gridname))
        elif group_type in SD_IMAGE_GROUPS.values():
            field_type, dimensionality = self._compatibility_with_image(
                gridname, field_shape)
        elif group_type in SD_MESH_GROUPS.values():
            field_type, dimensionality = self._compatibility_with_mesh(
                gridname, field_shape)
        else:
            raise tables.NodeError('location {} is not a grid.'
                                   ''.format(gridname))
        return field_type, dimensionality

    def _compatibility_with_mesh(self, meshname, field_shape):
        """Check if field has a number of values compatible with the mesh."""
        # Safety check: field must be a vector or a (Nvalues,Dim) array
        if len(field_shape) > 2:
            raise ValueError('Forbidden field shape. The field array must be'
                             ' of shape (Nvalues) or (Nvalues,Ndim). Received'
                             ' {}'.format(field_shape))
        node_field = False
        elem_field = False
        Nnodes = self.get_attribute('number_of_nodes', meshname)
        Nelem = np.sum(self.get_attribute('Number_of_elements', meshname))
        Nelem_bulk = np.sum(self.get_attribute('Number_of_bulk_elements',
                                               meshname))
        Nfield_values = field_shape[0]
        if len(field_shape) == 2:
            Field_dim = field_shape[1]
        else:
            Field_dim = 1
        if Nfield_values == Nnodes:
            node_field = True
            field_type='Nodal_field'
        elif Nfield_values == Nelem:
            elem_field = True
            field_type='Element_field'
        elif (Nfield_values % Nelem_bulk) == 0:
            elem_field = True
            field_type = 'IP_field'
        compatibility = node_field or elem_field
        if not(compatibility):
            raise ValueError('Field number of values ({}) is not conformant'
                             ' with mesh number of nodes ({}) or number of'
                             ' elements ({}).'
                             ''.format(Nfield_values, Nnodes, Nelem))
        if Field_dim not in XDMF_FIELD_TYPE:
            raise ValueError('Field dimensionnality `{}` is not know. '
                             'Supported dimensionnalities are Scalar (1),'
                             'Vector (3), Tensor6 (6), Tensor (9).'
                             'Maybe are you trying to add a 3D field into a'
                             '3D grid.')
        return field_type, XDMF_FIELD_TYPE[Field_dim]

    def _compatibility_with_image(self, imagename, field_shape):
        """Check if field has a number of values compatible with the image.

        Returns the type of field values (nodal or element field), and the
        dimensonality of the field in the XDMF convention (Scalar, Vector,
        Tensor6 or Tensor)
        """
        node_field = True
        elem_field = True
        image_node_dim = self.get_attribute('nodes_dimension', imagename)
        image_cell_dim = self.get_attribute('dimension', imagename)
        # Should never be equal but sanity check
        if np.all(image_node_dim == image_cell_dim):
            raise ValueError('Image group {} has identical node and cell'
                             ' dimensions. Please correct your image Group'
                             ' attributes.')
        for i in range(len(image_cell_dim)):
            if np.any(image_node_dim[i] != field_shape[i]):
                node_field = False
            if np.any(image_cell_dim[i] != field_shape[i]):
                elem_field = False
        if node_field:
            field_type='Nodal_field'
        elif elem_field:
            field_type='Element_field'
        compatibility = node_field or elem_field
        if not compatibility:
            raise ValueError('Field number of values ({}) is not conformant'
                             ' with image `{}` dimensions'
                             ''.format(field_shape, imagename))
        else:
            if len(field_shape) == len(image_node_dim):
                dimension = 1
            else:
                dimension = field_shape[-1]
        if dimension not in XDMF_FIELD_TYPE:
            raise ValueError('Field dimensionnality `{}` is not know. '
                             'Supported dimensionnalities are Scalar (1),'
                             'Vector (3), Tensor6 (6), Tensor (9).'
                             'Maybe are you trying to add a 3D field into a'
                             '3D grid.')
        return field_type, XDMF_FIELD_TYPE[dimension]

    def _mesh_field_padding(self, field, meshname):
        """Pad with zeros the mesh elem field to comply with size."""
        if self._is_image(meshname):
            return field, 'None', None
        Nelem_bulk = np.sum(self.get_attribute('Number_of_bulk_elements',
                                               meshname))
        Nelem_boundary = np.sum(self.get_attribute(
                                'Number_of_boundary_elements', meshname))
        padding = 'None'
        vis_field = None
        if field.shape[0] == Nelem_bulk:
            padding = 'bulk'
        elif field.shape[0] == Nelem_boundary:
            padding = 'boundary'
        elif (field.shape[0] % Nelem_bulk) == 0:
            # if the number of values  for the field is a multiplier of the
            # number of elements, it is considered that the array describes a
            # field integration point values and that it is stored as follows:
            # field[:,k] = [Val_elt1_IP1, Val_elt1_IP2, ...., Val_elt1_IPN,
            #               Val_elt2_IP1, ...., Val_eltN_IPN]
            padding = 'bulk_IP'
        if padding == 'None':
            pass
        elif padding == 'bulk':
            Nelem_boundary = np.sum(self.get_attribute(
                                'Number_of_boundary_elements',meshname))
            pad_array = np.zeros(shape=(Nelem_boundary, field.shape[1]))
            field = np.concatenate((field, pad_array), axis=0)
        elif padding == 'boundary':
            Nelem_bulk = np.sum(self.get_attribute(
                                    'Number_of_bulk_elements',meshname))
            pad_array = np.zeros(shape=(Nelem_bulk, field.shape[1]))
            field = np.concatenate((pad_array, field), axis=0)
        elif padding == 'bulk_IP':
            Nip_elt = field.shape[0] // Nelem_bulk
            new_shape = (Nelem_bulk,Nip_elt,*field.shape[1:])
            vis_field = field.reshape(new_shape)
            Nelem_boundary = np.sum(self.get_attribute(
                                'Number_of_boundary_elements',meshname))
            pad_array = np.zeros(shape=(Nelem_boundary, *vis_field.shape[1:]))
            vis_field = np.concatenate((vis_field, pad_array), axis=0)
        return field, padding, vis_field

    def _mesh_field_unpadding(self, field, parent_mesh, padding):
        """Remove zeros to return field to original shape, before padding."""
        Nelem_bulk = np.sum(self.get_attribute('Number_of_bulk_elements',
                                               parent_mesh))
        Nelem_boundary = np.sum(self.get_attribute(
                                'Number_of_boundary_elements', parent_mesh))
        if (padding == 'bulk') or (padding == 'bulk_IP'):
            field = field[:Nelem_bulk,:]
        elif padding == 'boundary':
            field = field[Nelem_boundary,:]
        elif padding == 'None':
            pass
        else:
            raise Warning('Cannot unpad the field, unknown padding type `{}`'
                          ''.format(padding))
        return field

    def _IP_field_for_visualisation(self, array, vis_type):
        # here it is supposed that the field has shape
        # [Nelem, Nip_per_elem, Ncomponent]
        if vis_type == 'Elt_max':
            array = np.max(array, axis=1)
        elif vis_type == 'Elt_mean':
            array = np.mean(array, axis=1)
        else:
            raise ValueError('Unkown integration point field visualisation'
                             ' convention. Possibilities are "Elt_max",'
                             ' "Elt_mean", "None"')
        return array

    def _add_mesh_geometry(self, mesh_object, mesh_group, replace,
                           bin_fields_from_sets):
        """Add Geometry data items of a mesh object to mesh group/xdmf."""
        self._verbose_print('Adding Geometry for mesh group {}'
                            ''.format(mesh_group._v_name))
        mesh_object.PrepareForOutput()
        # create Geometry group
        geo_group = self.add_group(groupname='Geometry',
                                   location=mesh_group._v_pathname,
                                   indexname=mesh_group._v_name+'_Geometry',
                                   replace=replace)
        # Add Nodes, NodesID and NodeTags
        self._add_mesh_nodes(mesh_object, mesh_group, geo_group, replace)
        # Add Elements, ElementsID and ElementTags
        self._add_mesh_elements(mesh_object, mesh_group, geo_group, replace)

    def _add_mesh_nodes(self, mesh_object, mesh_group, geo_group, replace):
        """Add Nodes, NodesID and NodeTags arrays in mesh geometry group."""
        # Add Nodes coordinates
        self._verbose_print('Creating Nodes data set in group {} in file {}'
                            ''.format(geo_group._v_pathname, self.h5_file))
        nodes_array = mesh_object.GetPosOfNodes()
        indexname = mesh_group._v_name+'_Nodes'
        Nodes = self.add_data_array(location=geo_group._v_pathname,
                                    name='Nodes', array=nodes_array,
                                    indexname=indexname)
        Node_attributes = {'number_of_nodes': mesh_object.nodes.shape[0],
                           'group_type': self._get_mesh_type(mesh_object),
                           'nodes_path':Nodes._v_pathname}
        self.add_attributes(Node_attributes, mesh_group._v_pathname)
        if isinstance(mesh_object, UnstructuredMesh):
            self._verbose_print('Creating Nodes ID data set in group {}'
                                ' in file {}'
                                ''.format(geo_group._v_pathname, self.h5_file))
            Nodes_ID = self.add_data_array(
                location=geo_group._v_pathname, name='Nodes_ID',
                array=mesh_object.originalIDNodes,
                indexname=mesh_group._v_name+'_Nodes_ID')
            Node_attributes = {'nodesID_path':Nodes_ID._v_pathname}
        self.add_attributes(Node_attributes, mesh_group._v_pathname)
        return

    def _add_mesh_elements(self, mesh_object, mesh_group, geo_group, replace):
        """Add Elements, ElementsID and ElementsTags in mesh geometry group."""
        # Determine Topology type
        if len(mesh_object.elements) > 1:
            topology_attributes = self._from_BT_mixed_topology(mesh_object)
        else:
            topology_attributes = self._from_BT_uniform_topology(mesh_object)
        # Add elements array
        self._add_topology(topology_attributes, mesh_object,mesh_group,
                           geo_group, replace)
        self.add_attributes(topology_attributes, mesh_group._v_pathname)
        return

    def _add_nodes_elements_tags(self,  mesh_object, mesh_group, replace,
                                 bin_fields_from_sets):
        """Add Node and ElemTags in mesh geometry group from mesh object."""
        geo_group = self.get_node(mesh_group._v_name+'_Geometry')
        # Add node tags
        self._add_mesh_nodes_tags(mesh_object, mesh_group, geo_group, replace,
                                  bin_fields_from_sets)
        # Add element tags
        self._add_mesh_elems_tags(mesh_object, mesh_group, geo_group, replace,
                                  bin_fields_from_sets)
        return

    def _add_mesh_nodes_tags(self, mesh_object, mesh_group, geo_group,
                             replace, bin_fields_from_sets):
        """Add NodeTags arrays in mesh geometry group from mesh object."""
        # create Noe tags group
        Ntags_group = self.add_group(groupname='NodeTags',
                                     location=geo_group._v_pathname,
                                     indexname=mesh_group._v_name+'_NodeTags',
                                     replace=replace)
        Node_tags_list = []
        for tag in mesh_object.nodesTags:
            # Add the node list of the tag in a data array
            name = tag.name
            node_list = mesh_object.nodesTags[tag.name].GetIds()
            if len(node_list) == 0:
                continue
            Node_tags_list.append(name)
            node = self.add_data_array(location=Ntags_group._v_pathname,
                                       name='NT_'+name, array=node_list,
                                       replace=replace)
            # remove from index : Nodesets may be too numerous and overload
            # content index --> actual choice is to remove them from index
            self._remove_from_index(node._v_pathname)
            if bin_fields_from_sets:
                # Add node tags as fields in the dataset and XDMF file
                data = np.zeros((mesh_object.GetNumberOfNodes(),1),
                                dtype=np.int8)
                data[node_list] = 1;
                node = self.add_field(mesh_group._v_pathname,
                                      fieldname='field_'+name,
                                      array=data, replace=replace,
                                      location=Ntags_group._v_pathname,
                                      complib='zlib', complevel=1,
                                      shuffle=True)
                # remove from index : Elsets may be too numerous and
                # overload content index --> actual choice is to remove
                # them from index
                self._remove_from_index(node._v_pathname)
        self.add_string_array(
            name='Node_tags_list', location=geo_group._v_pathname,
            indexname=mesh_group._v_name+'_NodeTagsList', data=Node_tags_list)
        return

    def _add_mesh_elems_tags(self, mesh_object, mesh_group, geo_group,
                             replace, bin_fields_from_sets):
        """Add ElementsTags arrays in mesh geometry group from mesh object."""
        # create an dictionnary for the offset to apply to elem local Ids
        # for each element type
        element_type = self.get_attribute('element_type', mesh_group._v_name)
        offsets = self.get_attribute('Elements_offset', mesh_group._v_name)
        offset_dic = {element_type[i]:offsets[i] for i in range(len(offsets))}
        # create Noe tags group
        Etags_group = self.add_group(groupname='ElementsTags',
                                     location=geo_group._v_pathname,
                                     indexname=mesh_group._v_name+'_ElemTags',
                                     replace=replace)
        Elem_tags_list = []
        Elem_tag_type_list = []
        for elem_type in mesh_object.elements:
            element_container = mesh_object.elements[elem_type]
            for tagname in element_container.tags.keys():
                name = tagname
                Elem_tags_list.append(name)
                Elem_tag_type_list.append(elem_type)
                elem_list = element_container.tags[tagname].GetIds()
                elem_list = elem_list + offset_dic[elem_type]
                if len(elem_list) == 0:
                    continue
                node = self.add_data_array(location=Etags_group._v_pathname,
                                           name='ET_'+name, array=elem_list,
                                           replace=replace)
                # remove from index : Elsets may be too numerous and overload
                # content index --> actual choice is to remove them from index
                self._remove_from_index(node._v_pathname)
                if bin_fields_from_sets:
                    # Add elem tags as fields in the dataset and XDMF file
                    elem_list_field = mesh_object.GetElementsInTag(tagname)
                    data = np.zeros((mesh_object.GetNumberOfElements(),1),
                                    dtype=np.int8)
                    data[elem_list_field] = 1;
                    node = self.add_field(mesh_group._v_pathname,
                                          fieldname='field_'+name,
                                          array=data, replace=replace,
                                          location=Etags_group._v_pathname,
                                          complib='zlib', complevel=1,
                                          shuffle=True)
                    # remove from index : Elsets may be too numerous and
                    # overload content index --> actual choice is to remove
                    # them from index
                    self._remove_from_index(node._v_pathname)
        self.add_string_array(
            name='Elem_tags_list', location=geo_group._v_pathname,
            indexname=mesh_group._v_name+'_ElTagsList', data=Elem_tags_list)
        self.add_string_array(
            name='Elem_tag_type_list', location=geo_group._v_pathname,
            indexname=mesh_group._v_name+'_ElTagsTypeList',
            data=Elem_tag_type_list)

    def _add_xdmf_node_element_set(self, subset_list_path, set_type='Cell',
                                   setname='', grid_name='',
                                   attributename=None):
        """Adds an xdmf set with an optional attribute to the xdmf file.

        For now, only Node and Cell Sets supporting scalar attributes
        are handled.
        """
        subset_list = self.get_node(subset_list_path)
        Xdmf_grid_node = self._find_xdmf_grid(grid_name)
        # Create xdmf Set node
        Set_xdmf = etree.Element(_tag='Set', Name=setname,
                                       SetType=set_type)
        # create Set Node/Cells subset
        Dimension = self._np_to_xdmf_str(subset_list.shape)
        NumberType = 'Int'
        Precision = str(subset_list.dtype).strip('int')
        Set_subset = etree.Element(_tag='DataItem', Format='HDF',
                                   Dimensions=Dimension,
                                   NumberType=NumberType,
                                   Precision=Precision)
        Set_subset.text = (self.h5_file + ':'
                           + self._name_or_node_to_path(subset_list_path))
        # add Subset Data Item to Set node
        Set_xdmf.append(Set_subset)
        # Create Attribute element if required
        if attributename is not None:
            Attribute_xdmf = etree.Element(_tag='Attribute',
                                           Name=attributename,
                                           AttributeType='Scalar',
                                           Center=set_type)
            # create data item element
            data = np.ones(subset_list.shape, dtype='int8')
            Dimension = self._np_to_xdmf_str(data.shape)
            NumberType = 'Int'
            Precision = str(data.dtype).strip('int')
            Attribute_data = etree.Element(_tag='DataItem', Format='XML',
                                           Dimensions=Dimension,
                                           NumberType=NumberType,
                                           Precision=Precision)
            Attribute_data.text = self._np_to_xdmf_str(data)
            # add data item to attribute
            Attribute_xdmf.append(Attribute_data)
            # add attribute to Set
            Set_xdmf.append(Attribute_xdmf)
        # add Set node to Grid node
        Xdmf_grid_node.append(Set_xdmf)
        return

    def _load_nodes_tags(self, meshname, mesh_object, as_numpy=True):
        """Add Node and ElemTags in mesh object from mesh geometry."""
        mesh_group = self.get_node(meshname)
        Ntags_group = self.get_node(mesh_group._v_name+'_NodeTags')
        if Ntags_group is not None:
            Ntag_list_indexname = mesh_group._v_name+'_NodeTagsList'
            Ntag_list = self.get_node(Ntag_list_indexname)
            for name in Ntag_list:
                tag_name = name.decode('utf-8')
                tag = mesh_object.GetNodalTag(tag_name)
                tag_path = os.path.join(Ntags_group._v_pathname,'NT_'+tag_name)
                tag.SetIds(self.get_node(tag_path, as_numpy))
        return mesh_object

    def _load_elements_tags(self, meshname, AllElements, as_numpy=True):
        """Add Node and ElemTags in mesh object from mesh group."""
        # get SD groups
        mesh_group = self.get_node(meshname)
        Etags_group = self.get_node(mesh_group._v_name+'_ElemTags')
        # create an dictionnary for the offset to apply to elem local Ids
        # for each element type
        element_type = self.get_attribute('element_type', mesh_group._v_name)
        offsets = self.get_attribute('Elements_offset', mesh_group._v_name)
        offset_dic = {element_type[i]:offsets[i] for i in range(len(offsets))}
        # load element tags
        if Etags_group is not None:
            Etag_list_indexname = mesh_group._v_name+'_ElTagsList'
            Etag_list = self.get_node(Etag_list_indexname)
            Etag_Etype_list_indexname = mesh_group._v_name+'_ElTagsTypeList'
            Etag_Etype_list = self.get_node(Etag_Etype_list_indexname)

            for i in range(len(Etag_list)):
                tag_name = Etag_list[i].decode('UTF-8')
                el_type = Etag_Etype_list[i].decode('UTF-8')
                elem_container = AllElements.GetElementsOfType(el_type)
                tag = elem_container.tags.CreateTag(tag_name,False)
                tag_path = os.path.join(Etags_group._v_pathname,'ET_'+tag_name)
                nodes_Ids = self.get_node(tag_path, as_numpy)
                ## Need to add local ids !! Substract the offset stored by
                ## get_mesh_elements
                tag.SetIds(nodes_Ids- offset_dic[el_type])
        return AllElements

    def _from_BT_mixed_topology(self, mesh_object):
        """Read mesh elements information/metadata from mesh_object."""
        import BasicTools.Containers.ElementNames as EN
        topology_attributes = {}
        topology_attributes['Topology'] = 'Mixed'
        element_type = []
        elements_offset = []
        Number_of_elements = []
        Number_of_bulk_elements = []
        Number_of_boundary_elements = []
        Xdmf_elements_code = []
        offset = 0
        # for each element type in the mesh_object, read type, number and
        # xdmf code for the elements
        # The process has 2 steps: 1 for bulk elements, then 1 for boundary
        # elements
        # First step for bulk elements
        for ntype, data in mesh_object.elements.items():
            if EN.dimension[ntype] == mesh_object.GetDimensionality():
                element_type.append(ntype)
                elements_offset.append(offset)
                offset += data.GetNumberOfElements()
                Number_of_elements.append(data.GetNumberOfElements())
                Xdmf_elements_code.append(XdmfNumber[ntype])
                Number_of_bulk_elements.append(data.GetNumberOfElements())
            elif EN.dimension[ntype] > mesh_object.GetDimensionality():
                raise ValueError('Elements dimensionality is higher than mesh'
                                 ' dimensionality.')
        # Second step for boundary elements
        for ntype, data in mesh_object.elements.items():
            if EN.dimension[ntype] < mesh_object.GetDimensionality():
                element_type.append(ntype)
                elements_offset.append(offset)
                offset += data.GetNumberOfElements()
                Number_of_elements.append(data.GetNumberOfElements())
                Xdmf_elements_code.append(XdmfNumber[ntype])
                Number_of_boundary_elements.append(data.GetNumberOfElements())
            elif EN.dimension[ntype] > mesh_object.GetDimensionality():
                raise ValueError('Elements dimensionality is higher than mesh'
                                 ' dimensionality.')
        # Return them in topology_attributes dic
        topology_attributes['element_type'] = element_type
        topology_attributes['Number_of_elements'] = np.array(
            Number_of_elements)
        topology_attributes['Elements_offset'] = np.array(
            elements_offset)
        topology_attributes['Number_of_bulk_elements'] = np.array(
            Number_of_bulk_elements)
        topology_attributes['Number_of_boundary_elements'] = np.array(
            Number_of_boundary_elements)
        topology_attributes['Xdmf_elements_code'] = Xdmf_elements_code
        return topology_attributes

    def _from_BT_uniform_topology(self, mesh_object):
        """Read mesh elements information/metadata from mesh_object."""
        topology_attributes = {}
        topology_attributes['Topology'] = 'Uniform'
        element_type =  mesh_object.elements.keys()[0]
        topology_attributes['element_type'] = [element_type]
        n_elements = mesh_object.GetNumberOfElements()
        topology_attributes['Number_of_elements'] = np.array([n_elements])
        topology_attributes['Number_of_bulk_elements'] = np.array([n_elements])
        topology_attributes['Number_of_boundary_elements'] = np.array([0])
        topology_attributes['Elements_offset'] = [0]
        topology_attributes['Xdmf_elements_code'] = [XdmfName[element_type]]
        return topology_attributes

    def _add_topology(self,topology_attributes, mesh_object, mesh_group,
                            geo_group, replace):
        """Add Elements array into mesh geometry group from mesh_object."""
        # Add Elements connectivity
        self._verbose_print('Creating Elements connectivity array in group {}'
                            ' in file {}'
                            ''.format(geo_group._v_pathname, self.h5_file))
        # Creating topology array
        data = np.empty((0,),dtype=np.int)
        for i in range(len(topology_attributes['element_type'])):
            element_type = topology_attributes['element_type'][i]
            elements = mesh_object.elements[element_type]
            data_tmp = elements.connectivity
            if topology_attributes['Topology'] == 'Mixed':
                # If mixed topology, add XDMF element type ID number
                # before Nodes ID
                xdmf_code = topology_attributes['Xdmf_elements_code'][i]
                type_col = np.ones(shape=(data_tmp.shape[0],1), dtype=np.int)
                if element_type == 'bar2':
                    data_tmp = np.concatenate((2*type_col, data_tmp),1)
                if element_type == 'point1':
                    data_tmp = np.concatenate((type_col, data_tmp),1)
                type_col = xdmf_code*type_col
                data_tmp = np.concatenate((type_col, data_tmp),1)
            data = np.concatenate((data, data_tmp.ravel()))
        # Add data array to HDF5 data set in mesh geometry node
        indexname = mesh_group._v_name+'_Elements'
        Elems = self.add_data_array(location=geo_group._v_pathname,
                                    name='Elements', array=data,
                                    indexname=indexname)
        self.add_attributes({'elements_path': Elems._v_pathname},
                            mesh_group._v_pathname)
        return

    def _add_image_to_xdmf(self, imagename, image_object):
        """Write grid geometry and topoly in xdmf tree/file."""
        # add 1 to each dimension to get grid dimension (from cell number to
        # point number --> XDMF indicates Grid points)
        image_type = self._get_image_type(image_object)
        # Get image dimension with reverted shape to compensate for Paraview
        # X,Y,Z indexing convention
        Dimension_tmp = image_object.GetDimensions()
        Spacing_tmp = image_object.GetSpacing()
        Origin_tmp = image_object.GetOrigin()
        if len(Dimension_tmp) == 2:
            Dimension_tmp = Dimension_tmp[[1,0]]
            Spacing_tmp = Spacing_tmp[[1,0]]
            Origin_tmp = Origin_tmp[[1,0]]
        elif len(Dimension_tmp) == 3:
            Dimension_tmp = Dimension_tmp[[2,1,0]]
            Spacing_tmp = Spacing_tmp[[2,1,0]]
            Origin_tmp = Origin_tmp[[2,1,0]]
        Dimension = self._np_to_xdmf_str(Dimension_tmp)
        Spacing = self._np_to_xdmf_str(Spacing_tmp)
        Origin = self._np_to_xdmf_str(Origin_tmp)
        Dimensionality = str(image_object.GetDimensionality())
        self._verbose_print('Updating xdmf tree...', line_break=False)
        # Creatge Grid element
        image_xdmf = etree.Element(_tag='Grid', Name=imagename,
                                   GridType='Uniform')
        # Create Topology element
        Topotype = XDMF_IMAGE_TOPOLOGY[image_type]
        topology_xdmf = etree.Element(_tag='Topology', TopologyType=Topotype,
                                      Dimensions=Dimension)
        # Create Geometry element
        Geotype = XDMF_IMAGE_GEOMETRY[image_type]
        geometry_xdmf = etree.Element(_tag='Geometry', Type=Geotype)
        origin_data = etree.Element(_tag='DataItem', Format='XML',
                                    Dimensions=Dimensionality)
        origin_data.text = Origin
        spacing_data = etree.Element(_tag='DataItem', Format='XML',
                                     Dimensions=Dimensionality)
        spacing_data.text = Spacing
        # Add nodes DataItem as childrens of node Geometry
        geometry_xdmf.append(origin_data)
        geometry_xdmf.append(spacing_data)
        # Add Geometry and Topology as childrens of Grid
        image_xdmf.append(topology_xdmf)
        image_xdmf.append(geometry_xdmf)
        # Add Grid to node Domain and get XDMF pathes
        self.xdmf_tree.getroot()[0].append(image_xdmf)
        return

    def _add_mesh_to_xdmf(self, mesh_group):
        """Write mesh grid element geometry and topoly in xdmf tree/file."""
        # get mesh group
        mesh_path = mesh_group._v_pathname
        # Creatge Grid element
        mesh_xdmf = etree.Element(_tag='Grid', Name=mesh_group._v_name,
                                  GridType='Uniform')
        # Create Geometry element
        mesh_type = self.get_attribute('group_type', mesh_path)
        if mesh_type == '2DMesh':
            geometry_xdmf = etree.Element(_tag='Geometry', Type='XY')
        elif mesh_type == '3DMesh':
            geometry_xdmf = etree.Element(_tag='Geometry', Type='XYZ')
        # Add geometry DataItem
        nodes = self.get_node(mesh_group._v_name+'_Nodes', as_numpy=False)
        Dim = self._np_to_xdmf_str(nodes.shape)
        geometry_data = etree.Element(_tag='DataItem', Format='HDF',
                                      Dimensions=Dim, NumberType='Float',
                                      Precision='64')
        geometry_data.text = self.h5_file + ':' + nodes._v_pathname
        # Add node DataItem as children of node Geometry
        geometry_xdmf.append(geometry_data)
        # Create Topology element
        Topology = self.get_attribute('Topology', mesh_path)
        if Topology == 'Uniform':
            Topotype = self.get_attribute('Xdmf_elements_code', mesh_path)[0]
        else:
            Topotype = 'Mixed'
        NElements_list = self.get_attribute('Number_of_elements', mesh_path)
        NElements = self._np_to_xdmf_str(np.sum(NElements_list))
        topology_xdmf = etree.Element(_tag='Topology', TopologyType=Topotype,
                                      NumberOfElements=NElements)
        # Create Topology DataItem
        elems =  self.get_node(mesh_group._v_name+'_Elements', as_numpy=False)
        Dim = self._np_to_xdmf_str(elems.shape)
        topology_data = etree.Element(_tag='DataItem', Format='HDF',
                                      Dimensions=Dim, NumberType='Int',
                                      Precision='64')
        topology_data.text = self.h5_file + ':' + elems._v_pathname
        # Add node DataItem as children of node Topology
        topology_xdmf.append(topology_data)
        # Add Geometry and Topology as childrens of Grid
        mesh_xdmf.append(geometry_xdmf)
        mesh_xdmf.append(topology_xdmf)
        # Add Grid to node Domain
        self.xdmf_tree.getroot()[0].append(mesh_xdmf)
        return

    def _append_field_index(self, gridname, fieldname):
        """Append field name to the field index of a grid group."""
        grid = self.get_node(gridname)
        index_path = os.path.join(grid._v_pathname,'Field_index')
        Field_index = self.get_node(index_path)
        if Field_index is None:
            Field_index = self.add_string_array(
                'Field_index', location=grid._v_pathname,
                indexname=grid._v_name+'Field_index')
        test_str = bytes(fieldname,'utf-8')
        if test_str not in Field_index:
            Field_index.append([fieldname])
        return

    def _transpose_field_comp(self, dimensionality, array):
        """Transpose fields components to comply with Paraview ordering."""
        # based on the conventions:
        # Tensor6 is passed as [xx,yy,zz,xy,yz,zx]
        # Tensor is passed as [xx,yy,zz,xy,yz,zx,yx,zy,xz]
        if dimensionality == 'Tensor6':
            transpose_indices = [0,3,5,1,4,2]
            transpose_back = [0,3,5,1,4,2]
        if dimensionality == 'Tensor':
            transpose_indices = [0,3,8,6,1,4,5,7,2]
            transpose_back = [0,4,8,1,5,6,3,7,2]
        return array[...,transpose_indices], transpose_back

    def _transpose_image_array(self, dimensionality, array):
        """Transpose the array X,Y,Z dimensions for XDMF conventions."""
        if dimensionality in ['Vector', 'Tensor6', 'Tensor']:
            # vector or tensor field
            if len(array.shape) == 3:
                # 2D image
                transpose_indices = [1,0,2]
            elif len(array.shape) == 4:
                # 3D image
                transpose_indices = [2,1,0,3]
        elif dimensionality == 'Scalar':
            # scalar field
            if len(array.shape) == 2:
                # 2D image
                transpose_indices = [1,0]
            elif len(array.shape) == 3:
                # 3D image
                transpose_indices = [2,1,0]
        else:
            raise ValueError('Unknown field dimensionality. Possible values'
                             ' are {}'.format(XDMF_FIELD_TYPE.values()))
        return array.transpose(transpose_indices), transpose_indices

    def _add_field_to_xdmf(self, fieldname, field):
        """Write field data as Grid Attribute in xdmf tree/file."""
        Node = self.get_node(fieldname)
        Grid_name = self.get_attribute('parent_grid_path', fieldname)
        Xdmf_grid_node = self._find_xdmf_grid(Grid_name)
        field_type = self.get_attribute('field_type', fieldname)
        if field_type == 'Nodal_field':
            Center_type = 'Node'
        elif (field_type == 'Element_field') or (field_type == 'IP_field'):
            Center_type = 'Cell'
        else:
            raise ValueError('unknown field type, should be `Nodal_field`'
                             ' or `Element_field`.')
        field_dimensionality = self.get_attribute('field_dimensionality',
                                                  fieldname)
        # create Attribute element
        Attribute_xdmf = etree.Element(_tag='Attribute', Name=Node._v_name,
                                       AttributeType=field_dimensionality,
                                       Center=Center_type)
        # Create data item element
        Dimension = self._np_to_xdmf_str(Node.shape)
        if (np.issubdtype(field.dtype, np.floating)):
            NumberType = 'Float'
            if (str(field.dtype) == 'float'):
                Precision = '32'
            else:
                Precision = '64'
        elif (np.issubdtype(field.dtype, np.integer)):
            NumberType = 'Int'
            Precision = str(field.dtype).strip('int')
        Attribute_data = etree.Element(_tag='DataItem', Format='HDF',
                                       Dimensions=Dimension,
                                       NumberType=NumberType,
                                       Precision=Precision)
        Attribute_data.text = (self.h5_file + ':'
                               + self._name_or_node_to_path(fieldname))
        # add data item to attribute
        Attribute_xdmf.append(Attribute_data)
        # add attribute to Grid
        Xdmf_grid_node.append(Attribute_xdmf)
        self.add_attributes({'xdmf_fieldname': Attribute_xdmf.get('Name')},
                            fieldname)
        return

    def _get_node_class(self, name):
        """Return Pytables Class type associated to the node name."""
        return self.get_attribute(attrname='CLASS', nodename=name)

    def _get_path_with_indexname(self, indexname):
        """Return node path from its indexname."""
        if indexname in self.content_index.keys():
            if isinstance(self.content_index[indexname], list):
                return self.content_index[indexname][0]
            else:
                return self.content_index[indexname]
        else:
            raise tables.NodeError('Index contains no item named {}'
                                   ''.format(indexname))
            return

    def _get_group_type(self, groupname):
        """Get SampleData HDF5 Group type (Mesh, Group or 3DImage)."""
        if groupname == '/':
            return 'GROUP'
        if self._is_group(groupname):
            grouptype = self.get_attribute(attrname='group_type',
                                           nodename=groupname)
            if grouptype is None:
                return 'GROUP'
            else:
                return grouptype
        else:
            return None

    def _get_image_type(self, image_object):
        try:
            return SD_IMAGE_GROUPS[image_object.GetDimensionality()]
        except:
            raise ValueError('Image dimension must correspond to a 2D or'
                             '3D image.')

    def _get_mesh_type(self, mesh_object):
        try:
            return SD_MESH_GROUPS[mesh_object.nodes.shape[1]]
        except:
            raise ValueError('Mesh dimension must correspond to a 2D or'
                             '3D mesh.')

    def _get_mesh_elements_offsets(self, meshname):
        types = self.get_attribute('element_type', meshname)
        offsets = self.get_attribute('Elements_offset', meshname)
        return {types[i]:offsets[i] for i in range(len(types))}

    def _get_parent_type(self, name):
        """Get the SampleData group type of the node parent group."""
        groupname = self._get_parent_name(name)
        return self._get_group_type(groupname)

    def _get_parent_name(self, name):
        """Get the name of the node parent group."""
        Node = self.get_node(name)
        Group = Node._g_getparent()
        return Group._v_name

    def _get_group_info(self, groupname, as_string=False):
        """Print a human readable information on the Pytables Group object."""
        s = ''
        Group = self.get_node(groupname)
        if Group is None:
            return f'No group named {groupname}'
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
        """Print a human readable information on the Pytables Group object."""
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
        s += str(' -- ')
        s += self.get_node_compression_info(nodename, as_string=True)
        size, unit = self.get_node_disk_size(nodename, print_flag=False)
        s += str(' -- Node size : {:9.3f} {}\n'.format(size, unit))
        s += str('----------------')
        if not(as_string):
            print(s)
            s = ''
        return s

    def _get_compression_opt(self, filters=None, **keywords):
        """Get inputed compression settings as `tables.Filters` instance."""
        # get pre-defined compression settings
        default_cp = False
        global_cp = False
        if 'default' in keywords:
            default_cp = keywords['default']
        else:
            # use global Filters as base settings if defined
            if hasattr(self, 'Filters'):
                global_cp = True
        # Apply pre-defined compression settings)
        if default_cp:
            Filters = self.set_default_compression()
        elif filters is not None:
            Filters = filters
        elif global_cp:
            Filters = self.Filters
        else:
            Filters = tables.Filters()
        # ------ read specified values of compression options
        #  These are prioritised over pre-defined settings when both are
        #  specified
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

    def _read_mesh_from_file(self, file=''):
        """Read a data array from a file, depending on the extension."""
        mesh_object = None
        # Get file extension
        _, tail = os.path.splitext(file)
        # HDF5 files
        if tail == '.geof':
            import BasicTools.IO.GeofReader as GR
            mesh_object = GR.ReadGeof(file)
        return mesh_object

    def _read_array_from_file(self, file=None, **keywords):
        """Read a data array from a file, depending on the extension."""
        array = None
        # Get file extension
        _, tail = os.path.splitext(file)
        # HDF5 files
        if tail == '.h5':
            with tables.File(file, mode='r') as f:
                if 'h5_path' in keywords:
                    h5_path = keywords['h5_path']
                array = f.get_node(where=h5_path).read()
        return array

    def _verbose_print(self, message, line_break=True):
        """Print message if verbose flag is `True`."""
        Msg = message
        if line_break:
            Msg = ('\n' + Msg)
        if self._verbose:
            print(Msg)
        return

    def _np_to_xdmf_str(self, array):
        Retstr =  str(array).strip('(').strip(')')
        Retstr =  str(Retstr).strip('[').strip(']')
        Retstr =  str(Retstr).replace(',', ' ')
        return Retstr
