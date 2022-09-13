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


# noinspection SpellCheckingInspection,PyProtectedMember
class SampleData:
    """Base class to store multi-modal datasets for material science.

    SampleData is a high level API designed to create and interact with
    complex datasets collecting all the data generated for a material sample
    by material scientists (from experiments, numerical simulation or data
    processing).

    Each dataset consist of a pair of files: one HDF5 file containing all
    data and metadata, and one XDMF data gathering metadata for all spatially
    origanized data (grids and fields). The class ensures creation and
    synchronization of both XDMF and HDF5 files.

    The HDF5 and XDMF data tree structure and content in both files are
    accessible through the `h5_dataset` and `xdmf_tree` class attributes,
    that are respectively instances of classes imported from the
    `Pytables <https://www.pytables.org/index.html>`_ and
    `lxml <https://lxml.de/>`_ packages.

    .. rubric:: SampleData Naming system

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

    .. rubric:: Data Compression

    HDF5 compression algorithm are available through the
    `Pytables <https://www.pytables.org/index.html>`_ package. SampleData
    offers an interface to it with the :func:`set_chunkshape_and_compression`
    method.

    .. rubric:: Arguments of the Class constructor

    :filename: `str`
        basename of HDF5/XDMF files to create/read. A file pair is created if
        the `filename` do not match any existing file.
    :sample_name: `str`, optional ('')
        name of the sample associated to data (metadata, dataset "title"). If
        the class is called to open a pre-existing `filename`, its sample name
        is not overwritten.
    :sample_description: `str`, optional ('')
        short description of the mechanical sample (material, type of
        tests.... -- metadata). If the class is called to open a pre-existing
        `filename`, its sample name is not overwritten.
    :verbose: `bool`, optional (False)
        set verbosity flag
    :overwrite_hdf5: `bool`, optional (False)
        set to `True` to overwrite existing HDF5/XDMF couple of files with the
        same `filename`.
    :autodelete: `bool`, optional (False)
        set to `True` to remove HDF5/XDMF files when deleting SampleData
        instance.
    :autorepack: `bool`, optional (False)
        if `True`, the HDF5 file is automatically repacked when deleting
        the SampleData instance, to recover the memory space freed up by data
        compression operations. See :func:`repack_h5file` for more details.

    .. rubric:: Class attributes

    :h5_file: basename of HDF5 file containing dataset (`str`)
    :h5_path: full path of the HDF5 dataset file
    :h5_dataset: :py:class:`tables.File` instance associated to the
        `h5_file`
    :xdmf_file: name of XDMF file associated with `h5_file` (`str`)
    :xdmf_path: full path of XDMF file (`str`)
    :xdmf_tree: :py:class:`lxml.etree` XML tree associated with `xdmf_file`
    :autodelete: autodelete flag (`bool`)
    :autorepack: autorepack flag (`bool`)
    :after_file_open_args: command arguments for `after_file_open` (dict)
    :content_index: Dictionnary of data items (nodes/groups)
        names and pathes in HDF5 dataset (`dic`)
    :aliases: Dictionnary of list of aliases for each item in
        content_index (`dic`)
"""
# TODO: Homogenize docstrings style

    def __init__(self, filename='sample_data', sample_name='',
                 sample_description=' ', verbose=False, overwrite_hdf5=False,
                 autodelete=False, autorepack=False,
                 after_file_open_args=dict()):
        """Sample Data constructor, see class documentation."""
        # get file directory and file name
        path_file = Path(filename).absolute()
        filename_tmp = path_file.stem
        file_dir = str(path_file.parent)

        self.h5_file = filename_tmp + '.h5'
        self.xdmf_file = filename_tmp + '.xdmf'
        self.file_dir = file_dir
        self.h5_path = os.path.join(self.file_dir, self.h5_file)
        self.xdmf_path = os.path.join(self.file_dir, self.xdmf_file)
        self._verbose = verbose
        self.autodelete = autodelete
        self.autorepack = autorepack
        if os.path.exists(self.h5_path) and overwrite_hdf5:
            self._verbose_print('-- File "{}" exists  and will be '
                                'overwritten'.format(self.h5_path))
            os.remove(self.h5_path)
            os.remove(self.xdmf_path)
        self._init_file_object(sample_name, sample_description)
        self._after_file_open(**after_file_open_args)
        self.sync()
        return

    def _after_file_open(self, **kwargs):
        """Initialization code to run after opening a Sample Data file.

        Empty method for this class. Use it for SampleData inherited classes,
        to create shortcut class attribute that are linked with hdf5 dataset
        elements (for instance, class attribute pointing towards a dataset
        structured table -- see `add_table` method and Microstructure class
        `grains` attribute)
        """
        return

    def __del__(self):
        """Sample Data destructor.

        Deletes SampleData instance and:
              - closes h5_file --> writes data structure into the .h5 file
              - writes the .xdmf file
        """
        self._verbose_print('Deleting DataSample object ')
        self.sync()
        if self.autorepack:
            self.repack_h5file()
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
        s += '\n'
        s += self.print_dataset_content(as_string=True, max_depth=3,
                                        short=True)
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

    def __getitem__(self, key):
        """Implement dictionnary like access to hdf5 dataset items."""
        # Return the object under the appropriate form
        if self._is_field(key):
            return self.get_field(key)
        elif self._is_array(key):
            return self.get_node(key, as_numpy=True)
        elif self._is_group(key):
            return self.get_node(key, as_numpy=False)

    def __getattribute__(self, name):
        """Implement attribute like access to hdf5 dataset items."""
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            if self._is_field(name):
                return self.get_field(name)
            elif self._is_array(name):
                return self.get_node(name, as_numpy=True)
            elif self._is_group(name):
                return self.get_node(name, as_numpy=False)
            else:
                raise AttributeError(f'{self.__class__} has no attribute'
                                     f' {name}')

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

    def print_dataset_content(self, as_string=False, max_depth=3,
                              to_file=None, short=False):
        """Print information on all nodes in the HDF5 file.

        :param bool as_string: If `True` solely returns string representation.
            If `False`, prints the string representation.
        :param int max_depth: Only prints data item whose depth is equal or
            less than this value. The depth is the number of parents a data
            item has. The root Group has thus a depth of 0, its children a
            depth of 1, the children of its children a depth of 2...
        :param str to_file: (optional) If not `None`, writes the dataset
            information to the provided text file name `to_file`. In that case,
            nothing is printed to the standard output.
        :param bool short: If `True`, return a short description of the
            dataset content, reduced to hdf5 tree structure and node memory
            sizes.
        :return str s: string representation of HDF5 nodes information
        """
        size, unit = self.get_file_disk_size(print_flag=False)
        s = f'Printing dataset content with max depth {max_depth}\n'
        if not short:
            s += ('\n****** DATA SET CONTENT ******\n -- File: {}\n '
                 '-- Size: {:9.3f} {}\n -- Data Model Class: {}\n'
                 ''.format(self.h5_file, size, unit, self.__class__.__name__))
            s += self.print_node_info('/', as_string=True)
            s += '\n************************************************\n\n'
        for node in self.h5_dataset.root:
            if node._v_depth > max_depth:
                continue
            if not(node._v_name == 'Index'):
                s += self.print_node_info(node._v_pathname, as_string=True,
                                          short=short)
                if self._is_group(node._v_pathname):
                    s += self.print_group_content(node._v_pathname,
                                                  recursive=True,
                                                  as_string=True,
                                                  max_depth=max_depth,
                                                  short=short)
                if not short:
                    s += ('\n**********************************'
                          '**************\n\n')
        if to_file:
            with open(to_file,'w') as f:
                f.write(s)
            return
        elif as_string:
            return s
        else:
            print(s)
            return

    def print_group_content(self, groupname, recursive=False, as_string=False,
                            max_depth=1000,  to_file=None, short=False):
        """Print information on all nodes in a HDF5 group.

        :param str groupname: Name, Path, Index name or Alias of the HDF5 group
        :param bool recursive: If `True`, print content of children groups
        :param bool as_string: If `True` solely returns string representation.
            If `False`, prints the string representation.
        :param int max_depth: Only prints data item whose depth is equal or
            less than this value. The depth is the number of parents a data
            item has. The root Group has thus a depth of 0, its children a
            depth of 1, the children of its children a depth of 2...
            Note that this depth is an absolute depth. Thus, if you want for
            instance to print the content of a group with depth 3, with 2
            levels of depth (its childrens and their childrens), you will need
            to specify a depth of 5 for this method.
        :param str to_file: (optional) If not `None`, writes the group
            contentto the provided text file name `to_file`. In that case,
            nothing is printed to the standard output.
        :param bool short: If `True`, return a short description of the
            dataset content, reduced to hdf5 tree structure and node memory
            sizes.
        :return str s: string representation of HDF5 nodes information
        """
        if short:
            s=''
        else:
            s = '\n****** Group {} CONTENT ******\n'.format(groupname)
        group = self.get_node(groupname)
        if group._v_depth > max_depth:
            return ''
        if group._v_nchildren == 0:
            return ''
        else:
            if not(as_string):
                print(s)
        for node in group._f_iter_nodes():
            s += self.print_node_info(node._v_pathname, as_string=True,
                                      short=short)
            if (self._is_group(node._v_pathname) and recursive):
                if ((node._v_name == 'ElementsTags') or
                   (node._v_name == 'NodeTags')):
                    # skip mesh Element and Node Tags group content
                    # that can be very large
                    continue
                s += self.print_group_content(node._v_pathname, recursive=True,
                                              as_string=True,
                                              max_depth=max_depth-1,
                                              short=short)
        if to_file:
            with open(to_file,'w') as f:
                f.write(s)
            return
        elif as_string:
            return s+'\n'
        else:
            print(s)
            return

    def print_node_info(self, nodename, as_string=False, short=False):
        """Print information on a node in the HDF5 tree.

        Prints node name, content, attributes, compression settings, path,
        childrens list if it is a group.

        :param str name: Name, Path, Index name or Alias of the HDF5 Node
        :param bool as_string: If `True` solely returns string representation.
            If `False`, prints the string representation.
        :param bool short: If `True`, return a short description of the
            node content, reduced to name, hdf5 type and node memory
            sizs.
        :return str s: string representation of HDF5 Node information
        """
        s = ''
        node_path = self._name_or_node_to_path(nodename)
        if self._is_array(node_path):
            s += self._get_array_node_info(nodename, as_string, short)
        else:
            s += self._get_group_info(nodename, as_string, short)
        if as_string:
            return s
        else:
            print(s)
            return

    def print_node_attributes(self, node_name, as_string=False):
        """Print the hdf5 attributes (metadata) of an array node.

        :param str node_name: Name, Path, Index name or Alias of the HDF5 Node
        :param bool as_string: If `True` solely returns string representation.
        If `False`, prints the string representation.
        :return str s: string representation of HDF5 Node compression settings
        """
        s = ''
        node = self.get_node(node_name)
        if node is None:
            return f'No group named {node_name}'
        s += str(f' -- {node._v_name} attributes : \n')
        for attr in node._v_attrs._v_attrnamesuser:
            value = node._v_attrs[attr]
            s += str('\t * {} : {}\n'.format(attr, value))
        if as_string:
            return s + '\n'
        else:
            print(s)
            return

    def print_node_compression_info(self, name, as_string=False):
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
            s += f'\n --- Chunkshape: {N.chunkshape}'
        else:
            s += '{} is not a data array node'.format(name)
        if not as_string:
            print(s)
            return
        return s+'\n'

    def print_data_arrays_info(self, as_string=False, to_file=None,
                               short=False):
        """Print information on all data array nodes in hdf5 file.

        Mesh node and element sets are excluded from the output due to
        their possibly very high number.

        :param bool as_string: If `True` solely returns string representation.
            If `False`, prints the string representation.
        :param str to_file: (optional) If not `None`, writes the dataset
            information to the provided text file name `to_file`. In that case,
            nothing is printed to the standard output.
        :param bool short: If `True`, return a short description of the
            dataset content, reduced to hdf5 tree structure and node memory
            sizes.
        :return str s: string representation of HDF5 nodes information
        """
        s = ''
        for node in self.h5_dataset:
            if node._v_name.startswith('ET_'):
                continue
            if self._is_array(node._v_name):
                s += self.print_node_info(node._v_name, as_string=True,
                                          short=short)
        if to_file:
            with open(to_file,'w') as f:
                f.write(s)
            return
        elif as_string:
            return s
        else:
            print(s)
            return

    def print_grids_info(self, as_string=False, to_file=None,
                               short=False):
        """Print information on all grid groups in hdf5 file.

        :param bool as_string: If `True` solely returns string representation.
            If `False`, prints the string representation.
        :param str to_file: (optional) If not `None`, writes the dataset
            information to the provided text file name `to_file`. In that case,
            nothing is printed to the standard output.
        :param bool short: If `True`, return a short description of the
            dataset content, reduced to hdf5 tree structure and node memory
            sizes.
        :return str s: string representation of HDF5 nodes information
        """
        s = ''
        for group in self.h5_dataset.walk_groups():
            if self._is_grid(group._v_name):
                s += self.print_node_info(group._v_name, as_string=True,
                                          short=short)
        if to_file:
            with open(to_file,'w') as f:
                f.write(s)
            return
        elif as_string:
            return s
        else:
            print(s)
            return

    def print_index(self, as_string=False, max_depth=3, local_root='/'):
        """Print a list of the data items in HDF5 file and their Index names.

        :param bool as_string: If `True` solely returns string representation.
            If `False`, prints the string representation.
        :param int max_depth: Only prints data item whose depth is equal or
            less than this value. The depth is the number of parents a data
            item has. The root Group has thus a depth of 0, its children a
            depth of 1, the children of its children a depth of 2...
        :param str local_root: prints only the Index for data items that are
            children of the provided local_root. The Name, Path, Indexname,
            or Alias of the local_root can be passed for this argument.
        :return str s: string representation of HDF5 nodes information if
            `as_string` is True.
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
        self._file_exist = True
        self._after_file_open()
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
                 bin_fields_from_sets=True, file=None,
                 compression_options=dict()):
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
        :param str location: Path, Name, Index Name or Alias of the parent group
            where the Mesh group is to be created
        :param str description: Description metadata for this mesh
        :param bool replace: remove Mesh group in the dataset with the same
            name/location if `True` and such group exists
        :param bool bin_fields_from_sets: If `True`, stores all Node and
            Element Sets in mesh_object as binary fields (1 on Set, 0 else)
        :param dict compression_options: Dictionary containing compression
            options items, see `set_chunkshape_and_compression` method for
            more details.
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
                           replace=replace,
                           compression_options=compression_options)
        for field_name, field in mesh_object.elemFields.items():
            self.add_field(gridname=mesh_group._v_pathname,
                           fieldname=field_name, array=field,
                           replace=replace,
                           compression_options=compression_options)
        return mesh_object

    def add_mesh_from_image(self, imagename, with_fields=True, ofTetras=False,
                            meshname='', indexname='', location='/',
                            description=' ', replace=False,
                            bin_fields_from_sets=True,
                            compression_options=dict()):
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
        :param dict compression_options: Dictionary containing compression
            options items, see `set_chunkshape_and_compression` method for
            more details.
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
                      replace, bin_fields_from_sets,
                      compression_options=compression_options)
        return

    def add_image(self, image_object=None, imagename='', indexname='',
                  location='/', description='', replace=False,
                  field_index_prefix='', compression_options=dict()):
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
        :param str field_index_prefix: the prefix to use to field stored in the
            image object.
        :param dict compression_options: Dictionary containing compression
            options items, see `set_chunkshape_and_compression` method for
            more details.
        """
        # Check if the image already exists as an empty image
        empty = False
        im_attrs = dict()
        old_descr = None
        try:
            gtype = self.get_attribute('group_type', indexname)
            if gtype == 'emptyImage':
                # we get the empty image attributes to transfer its metadata
                # to the new image group that will overwrite it
                empty = True
                im_attrs = self.get_dic_from_attributes(indexname)
                # remove old arguments for empty groups
                im_attrs.pop('empty')
                im_attrs.pop('group_type')
                old_descr = im_attrs.pop('description')
        except:
            pass
        if empty and (image_object is not None):
            replace = True
        # Create or fetch image group
        image_group = self.add_group(imagename, location, indexname, replace)
        # empty images creation
        if image_object is None:
            self.add_attributes({'empty': True, 'group_type': 'emptyImage'},
                                image_group._v_pathname)
            return
        else:
            self._check_image_object_support(image_object)
        # Add image grid to xdmf file
        self._add_image_to_xdmf(imagename, image_object)
        # store image metadata as HDF5 attributes
        image_type = self._get_image_type(image_object)
        image_nodes_dim = np.array(image_object.GetDimensions())
        image_cell_dim = image_nodes_dim - np.ones(image_nodes_dim.shape,
                                                   dtype=image_nodes_dim.dtype)
        if len(image_nodes_dim) == 2:
            image_xdmf_dim = image_nodes_dim[[1, 0]]
        elif len(image_nodes_dim) == 3:
            image_xdmf_dim = image_nodes_dim[[2, 1, 0]]
        # Add image attributes
        if (description == '') and (old_descr is not None):
            description = im_attrs['description']
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
        self.add_attributes(im_attrs, image_group._v_pathname)
        # Add fields if some are stored in the image object
        for field_name, field in image_object.nodeFields.items():
            self.add_field(gridname=image_group._v_pathname,
                           fieldname=field_name,
                           array=field,
                           indexname=field_index_prefix + field_name,
                           compression_options=compression_options)
        for field_name, field in image_object.elemFields.items():
            self.add_field(gridname=image_group._v_pathname,
                           fieldname=field_name,
                           array=field,
                           indexname=field_index_prefix + field_name,
                           compression_options=compression_options)
        return image_object

    def add_image_from_field(self, field_array, fieldname, imagename='',
                             indexname='', location='/', description=' ',
                             replace=False, origin=None, spacing=None,
                             is_scalar=True, is_elem_field=True,
                             compression_options=dict()):
        """Create a 2D/3D Image group in the dataset from a field data array.

        Construct an image object from the input field array. This array is
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
        :param str location: Path, Name, Index Name or Alias of the parent group
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
        :param bool is_elem_field: If `True` (default value), the array is
            considered as a pixel/voxel wise field value array. If `False`, the
            field is considered as a nodal value array.
        :param dict compression_options: Dictionary containing compression
            options items, see `set_chunkshape_and_compression` method for
            more details.

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
        if is_elem_field:
            field_dimensions = field_dimensions + np.ones((field_dim,))
        image_object = ConstantRectilinearMesh(dim=field_dim)
        image_object.SetDimensions(field_dimensions)
        image_object.SetOrigin(origin)
        image_object.SetSpacing(spacing)
        image_object.elemFields[fieldname] = field_array
        self.add_image(image_object, imagename=imagename,
                       indexname=indexname, location=location,
                       description=description, replace=replace,
                       compression_options=compression_options,
                       field_index_prefix=(imagename + '_'))
        return

    def add_grid_time(self, gridname, time_list):
        """Add a list of time values to a grid data group.

        If the grid has no time value, an xdmf grid temporal collection is
        created.

        :param str gridname: Path, name or indexname of the grid Group where
            to add time values.
        :param list(float) time_list: List of times to add to the grid. Can
            also be passed as a numpy array.
        """
        # WARNING : BUG, XDMF grids must be in increasing time order
        # not ensured
        # TODO: Create a method sort xdmf grids with time
        # if time_list is passed as a numpy array, transform it into a list
        if isinstance(time_list, np.ndarray):
            time_list = time_list.tolist()
        if isinstance(time_list, float):
            time_list = [time_list]
        if isinstance(time_list, int):
            time_list = [time_list]
        time_list = sorted(time_list)
        # get xdmf node of main grid
        xdmf_gridname = self.get_attribute('xdmf_gridname',gridname)
        grid = self._find_xdmf_grid(xdmf_gridname)
        # Find out if grid is a uniform grid
        grid_type = grid.get('GridType')
        if grid_type == 'Uniform':
            grid0 = grid
            # Default setting considers the already present grid node as the
            # first time value of the time serie --> first grid of the
            # collection. All attributes (fields) already added to this grid
            # will be considered as attribute value at first time value.
            grid0.set('Name', xdmf_gridname + '_T0')
            # Remove old grid from xdmf tree
            p = grid.getparent()
            p.remove(grid)
            # Create a new grid
            grid = etree.Element(_tag='Grid', Name=xdmf_gridname,
                                 GridType='Collection',
                                 CollectionType='Temporal')
            # Add time element to grid0
            time0 = etree.Element(_tag='Time', Value=f'{time_list[0]}')
            grid0.append(time0)
            # Append old grid to new grid collection
            grid.append(grid0)
            # Append grid collection to Domain
            p.append(grid)
            # Create a time_list with only first time value
            self.add_attributes({'time_list':[time_list.pop(0)]},gridname)
        elif grid_type == 'Collection':
            grid0 = grid.getchildren()[0]
        else:
            raise ValueError(f'Grid {gridname} type is not Uniform nor'
                             ' Collection.')
        # Get grid topology and Geometry nodes
        for ch in grid0.iterchildren():
            if ch.tag == 'Topology':
                topo = ch
            if ch.tag == 'Geometry':
                geo = ch
        # Get main grid time list
        time_list0 = self.get_attribute('time_list', gridname)
        if time_list0 is None:
            time_list0 = []
        index_count = len(time_list0)
        for T in time_list:
            if T not in time_list0:
                time_list0.append(T)
                # Create a new grid
                gridT_name = xdmf_gridname + f'_T{index_count}'
                gridT = etree.Element(_tag='Grid', Name=gridT_name,
                                      GridType='Uniform')
                # Add time element to grid
                timeT = etree.Element(_tag='Time', Value=f'{T}')
                gridT.append(timeT)
                # Append topology and geometry to grid
                gridT.append(topo.__copy__())
                gridT.append(geo.__copy__())
                index_count = index_count + 1
                # Append grid to grid collection
                grid.append(gridT)
        self.add_attributes({'time_list':time_list0},gridname)
        return

    def add_group(self, groupname, location='/', indexname='', replace=False):
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

    def add_field(self, gridname, fieldname, array=None, location=None,
                  indexname=None, chunkshape=None, replace=False,
                  visualisation_type='Elt_mean', compression_options=dict(),
                  time=None, bulk_padding=True):
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
        :param str location: Path, name or indexname of the Group in which the
            field array will be stored. This Group must be a children of the
            `gridname` Group. If not provided, the field is stored in the
            `gridname` Group.
        :param str indexname: Index name used to reference the field node
        :param tuple  chunkshape: The shape of the data chunk to be read or
            written in a single HDF5 I/O operation
        :param bool replace: remove 3DImage group in the dataset with the same
            name/location if `True` and such group exists
        :param str visualisation_type: Type of visualisation used to represent
            integration point fields with an element wise constant field.
            Possibilities are 'Elt_max' (maximum value per element), 'Elt_mean'
            (mean value per element), 'None' (no visualisation field).
            Default value is 'Elt_mean'
        :param dict compression_options: Dictionary containing compression
            options items, see `set_chunkshape_and_compression` method for
            more details.
        :param float time: Associate a time value for this field. IF a time
            value is provided, the suffix '_T{time_index}' is appended to
            the field_name and indexname
        :param bool bulk_padding: If adding a field on a mesh  that has as many
            bulk as boundary elements, forces field padding to `bulk` if True,
            or to `boundary` if false
        """
        self._verbose_print('Adding field `{}` into Grid `{}`'
                            ''.format(fieldname, gridname))
        # Fields can only be added to grid Groups --> sanity check
        if not(self._is_grid(gridname)):
            raise tables.NodeError('{} is not a grid, cannot add a field data'
                                   ' array in this group.'.format(gridname))
        # Get storage location for field data array
        if (location is None) and replace:
            # if replace, try to get the parent node of the possibly
            # existing node to replace
            node_field = self.get_node(fieldname)
            if node_field is not None:
                location = node_field._v_parent._v_pathname
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
        # Handle empty fields
        if array is None:
            node = self.add_data_array(array_location, fieldname, array,
                                       indexname, replace)
            # Add field path to grid node Field_list attribute
            self._append_field_index(gridname, indexname)
            gridpath = self._name_or_node_to_path(gridname)
            attribute_dic = {'parent_grid_path': gridpath,
                             'node_type': 'field_array'
                             }
            self.add_attributes(attribute_dic, nodename=indexname)
            return node
        # If needed, pad the field with 0s to comply with number of bulk and
        # boundary elements
        array, padding, vis_array = self._mesh_field_padding(array, gridname,
                                                             bulk_padding)
        # Check if the array shape is consistent with the grid geometry
        # and returns field dimension, xdmf Center attribute
        field_type, dimensionality = self._check_field_compatibility(
                                                        gridname, array.shape)

        # Apply indices transposition to assure consistency of the data
        # visualization in paraview with SampleData ordering and indexing
        # conventions
        if self._is_image(gridname):
            # indices transposition to ensure consistency between SampleData
            # and geometrical interpretation of coordinates in Paraview
            if self.get_attribute('group_type', gridname) == '2DImage':
                if len(array.shape) == 3:
                    array = np.squeeze(array)
            array, transpose_indices = self._transpose_image_array(
                dimensionality, array)
        if dimensionality in ['Tensor6', 'Tensor']:
            # indices transposition to ensure consistency between SampleData
            # components oredering convention and SampleData ordering
            # convention
            array, transpose_components = self._transpose_field_comp(
                dimensionality, array)

        # get indexname or create one
        if indexname is None:
            grid_path = self._name_or_node_to_path(gridname)
            grid_indexname = self.get_indexname_from_path(grid_path)
            indexname = grid_indexname+'_'+fieldname

        # If time value is provided, find out if a new temporal grid must be
        # added to the xdmf file
        time_gridname = None
        time_suffix = ''
        if time is not None:
            time_list = self.get_attribute('time_list', gridname)
            if time_list is None:
                time_list = [time]
                self.add_grid_time(gridname, time_list)
            else:
                if time not in time_list:
                    time_list = [time]
                    self.add_grid_time(gridname, time_list)
                    time_list = self.get_attribute('time_list', gridname)
            time_gridname = self._get_xdmf_time_grid_name(gridname,time)
            # get time suffix from gridname
            import re
            time_suffix = re.findall('_T\d+', time_gridname)[-1]
            # keep suffixless field_name as attribute name for the xdmf
            # time grid collection (same field name for each grid associated
            # to a different time step)
            time_serie_name = fieldname
            # Add time suffix to field indexname and name
            fieldname = fieldname + time_suffix
            indexname = indexname + time_suffix
        # Add data array into HDF5 dataset
        node = self.add_data_array(array_location, fieldname, array, indexname,
                                   chunkshape, replace,
                                   compression_options=compression_options)
        # Create attributes of the field node
        gridpath = self._name_or_node_to_path(gridname)
        xdmf_gname = self.get_attribute('xdmf_gridname', gridname)
        if time_gridname is not None:
            xdmf_gname = time_gridname
        attribute_dic = {'field_type': field_type,
                         'field_dimensionality': dimensionality,
                         'parent_grid_path': gridpath,
                         'xdmf_gridname': xdmf_gname, 'padding': padding,
                         'node_type':'field_array'
                         }
        if time is not None:
            attribute_dic['time'] = time
            attribute_dic['time_serie_name'] = time_serie_name
        if self._is_image(gridname):
            attribute_dic['transpose_indices'] = transpose_indices
        if dimensionality in ['Tensor6', 'Tensor']:
            attribute_dic['transpose_components'] = transpose_components
        # Add field for visualization of Integration Points mesh fields
        if (field_type == 'IP_field') and not (visualisation_type == 'None'):
            vis_array = self._IP_field_for_visualisation(vis_array,
                                                         visualisation_type)
            visname = fieldname+f'_{visualisation_type}'
            visindexname = indexname+f'_{visualisation_type}'
            if dimensionality in ['Tensor6', 'Tensor']:
                vis_array, _ = self._transpose_field_comp(
                    dimensionality, vis_array)
            node_vis = self.add_data_array(
                array_location, visname, vis_array, visindexname, chunkshape,
                replace, compression_options=compression_options)
            attribute_dic['visualisation_type'] = visualisation_type
            self.add_attributes(attribute_dic, nodename=visindexname)
            attribute_dic['visualisation_field_path'] = node_vis._v_pathname
        self.add_attributes(attribute_dic, nodename=indexname)
        # Add field description to XDMF file
        if (field_type == 'IP_field') and not (visualisation_type=='None'):
            self._add_field_to_xdmf(visindexname, vis_array)
        else:
            self._add_field_to_xdmf(indexname, array)
        # Add field path to grid node Field_list attribute
        self._append_field_index(gridname, indexname)
        return node

    def add_data_array(self, location, name, array=None, indexname=None,
                       chunkshape=None, replace=False,
                       compression_options=dict()):
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
        :param dict compression_options: Dictionary containing compression
            options items, see `set_chunkshape_and_compression` method for
            more details.

        """
        self._verbose_print('Adding array `{}` into Group `{}`'
                            ''.format(name, location))
        if array is None:
            empty = True
        else:
            empty=False
        # Safety checks
        saved_attrs = self._check_SD_array_init(name, location, replace, empty)
        # get location path
        location_path = self._name_or_node_to_path(location)
        # get compression options
        Filters = self._get_compression_opt(compression_options)
        # add to index
        if indexname is None:
            indexname = name
        self.add_to_index(indexname, '%s/%s' % (location_path, name))
        # Create dataset node to store array
        if empty:
            Node = self.h5_dataset.create_carray(
                    where=location_path, name=name, obj=np.array([0]),
                    title=indexname)
            self.add_attributes({'empty': True, 'node_type': 'data_array'},
                                Node._v_pathname)
        else:
            if 'normalization' in compression_options:
                optn = compression_options['normalization']
                array, norm_attributes = self._data_normalization(array, optn)
            Node = self.h5_dataset.create_carray(
                    where=location_path, name=name, filters=Filters,
                    obj=array, chunkshape=chunkshape,
                    title=indexname)
            self.add_attributes(saved_attrs, Node._v_pathname)
            self.add_attributes({'empty': False, 'node_type':'data_array'},
                                 Node._v_pathname)
            if 'normalization' in compression_options:
                if optn == 'standard_per_component':
                    mean_array = norm_attributes.pop('norm_mean_array')
                    std_array = norm_attributes.pop('norm_std_array')
                    Mean = self.h5_dataset.create_carray(
                        where=location_path, name=name+'_norm_mean',
                        filters=Filters, obj=mean_array, chunkshape=chunkshape)
                    Std = self.h5_dataset.create_carray(
                        where=location_path, name=name+'_norm_std',
                        filters=Filters, obj=std_array, chunkshape=chunkshape)
                    norm_attributes['norm_mean_array_path'] = Mean._v_pathname
                    norm_attributes['norm_std_array_path'] = Std._v_pathname
                self.add_attributes(norm_attributes, Node._v_pathname)
        return Node

    def add_table(self, location, name, description, indexname=None,
                  chunkshape=None, replace=False, data=None,
                  compression_options=dict()):
        """Add a structured storage table in HDF5 dataset.

        :param str location: Path where the array will be added in the dataset
        :param str name: Name of the array to create
        :param IsDescription description: Definition of the table rows, can be
            a numpy dtype.
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
        :param dict compression_options: Dictionary containing compression
            options items, see `set_chunkshape_and_compression` method for
            more details.
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
            table_path = '%s/%s' % (location_path, name)
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
        Filters = self._get_compression_opt(compression_options)
        self._verbose_print('-- Compression Options for dataset {}'
                            ''.format(name))
        # create table
        table = self.h5_dataset.create_table(where=location_path, name=name,
                                             description=description,
                                             filters=Filters,
                                             chunkshape=chunkshape)
        if data is not None:
            print(data.shape)
            table.append(data)
            print(table)
            table.flush()
        # add to index
        if indexname is None:
            warn_msg = (' (add_table) indexname not provided, '
                        ' the table name `{}` is used as index name '
                        ''.format(name))
            self._verbose_print(warn_msg)
            indexname = name
        self.add_to_index(indexname, table._v_pathname)
        self.add_attributes({'node_type':'structured_array'},
                            table._v_pathname)
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
                                 ' columns description input.\n'
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
            array_path = '%s/%s' % (location_path, name)
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
        # Determine if array is created empty
        if len(data) == 0:
            empty = True
        else:
            empty = False
            # add to index
        if indexname is None:
            warn_msg = (' (add_string_array) indexname not provided, '
                        ' the string array name `{}` is used as index name '
                        ''.format(name))
            self._verbose_print(warn_msg)
            indexname = name
        self.add_to_index(indexname, str_array._v_pathname)
        self.add_attributes({'node_type':'string_array', 'empty':empty},
                            str_array._v_pathname)
        return str_array

    def append_string_array(self, name, data=[]):
        """Append a list of strings to a string array node in the dataset.

        :param str name: Path, Indexname, Name or Alias of the string array
            to which append the list of strings.
        :param list(str) data: List of strings to append to the string array.
        """
        Sarray = self.get_node(name)
        if Sarray is None:
            raise tables.NodeError(f'No string array named {name} in the'
                                   ' dataset.')
        for string in data:
            a_str = bytes(string,'utf-8')
            Sarray.append([a_str])
        if len(Sarray) > 0 and self.get_attribute('empty', name):
            self.add_attributes({'empty': False}, name)
        return

    def append_table(self, name, data):
        """Append a numpy structured array to a table in the dataset.

        :param str name: Path, Indexname, Name or Alias of the string array
            to which append the list of strings.
        :param numpy.ndarray data: array to append to the table? Its `dtype`
            must match de table description.
        """
        table = self.get_node(name)
        table.append(data)
        table.flush()
        return

    def add_attributes(self, dic, nodename):
        """Add a dictionary entries as HDF5 Attributes to a Node or Group.

        :param dic dic: Python dictionary of items to store in HDF5 file as
            HDF5 Attributes
        :param str nodename: Path, Index name or Alias of the HDF5 node or
            group receiving the Attributes
        """
        node = self.get_node(nodename)
        for key, value in dic.items():
            node._v_attrs[key] = value
        return

    def add_alias(self, aliasname, path=None, indexname=None):
        """Add alias name to reference Node with input path or index name.

        :param str aliasname: name to add as alias to reference the node
        :param str path: Path of the node to reference with `aliasname`
        :param str indexname: indexname of the node to reference with
            `aliasname`
        """
        if (path is None) and (indexname is None):
            msg = ('(add_alias) None path nor indexname input. Alias'
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
            if indexname is None:
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
        is_present = self._is_in_index(indexname) or self._is_alias(indexname)
        if is_present:
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
        # TODO: move in Grid utils
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
            connectivity = self.get_mesh_elem_tag_connectivity(
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
        mesh_indexname = self.get_indexname_from_path(mesh_group._v_pathname)
        FIndex = mesh_indexname+'_Field_index'
        Field_index = self.get_node(FIndex)
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

    def get_mesh_node_tag(self, meshname, node_tag, as_numpy=True):
        """Returns the node IDs of a node tag of the mesh group.

        :param str meshname: Name, Path, Index name or Alias of the Mesh group
            in dataset
        :param str node_tag: name of the node tag whose IDs must be returned.
        :param bool as_numpy: if `True`, returns arrays in elements container
            as numpy array
        """
        # Add prefix to node tag name if needed
        if not node_tag.startswith('NT_'):
            node_tag_nodename = 'NT_' + node_tag
        # get mesh group
        mesh_group = self.get_node(meshname)
        # get path of Element tag
        NT_path = '%s/Geometry/NodeTags/%s' % (mesh_group._v_pathname, node_tag_nodename)
        # get element tag type
        tag = self.get_node(NT_path, as_numpy=as_numpy)
        return tag

    def get_mesh_node_tag_coordinates(self, meshname, node_tag):
        """Returns the node coordinates of a node tag of the mesh group.

        :param str meshname: Name, Path, Index name or Alias of the Mesh group
            in dataset
        :param str node_tag: name of the node tag whose IDs must be returned.
        :param bool as_numpy: if `True`, returns arrays in elements container
            as numpy array
        """
        tag = self.get_mesh_node_tag(meshname, node_tag, as_numpy=True)
        Nodes = self.get_mesh_nodes(meshname, as_numpy=True)
        return Nodes[tag,:]

    def get_mesh_xdmf_connectivity(self, meshname, as_numpy=True):
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
                if get_eltype_connectivity == element_type[i]:
                    return Elements.connectivity
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
        """Returns the elements IDs of an element tag of the mesh group.

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
            ET_path = '%s/Geometry/ElementsTags/%s' % (mesh_group._v_pathname, el_tag_nodename)
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
        indexname = self.get_indexname_from_path(mesh_group._v_pathname)
        Etag_list_indexname = indexname +'_ElTagsList'
        Etag_list = self.get_node(Etag_list_indexname)
        Etag_Etype_list_indexname = indexname +'_ElTagsTypeList'
        Etag_Etype_list = self.get_node(Etag_Etype_list_indexname)
        Tags_dict = {
            Etag_list[i].decode('utf-8'):Etag_Etype_list[i].decode('utf-8')
                     for i in range(len(Etag_list))}
        return Tags_dict

    def get_mesh_elem_tag_connectivity(self, meshname, element_tag):
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
        indexname = self.get_indexname_from_path(mesh_group._v_pathname)
        Ntag_list_indexname = indexname+'_NodeTagsList'
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
        spacing = self.get_attribute('spacing', imagename)
        origin = self.get_attribute('origin', imagename)
        # Create ConstantRectilinearMesh to serve as image_object
        image_object = ConstantRectilinearMesh(dim=len(dimensions))
        image_object.SetDimensions(dimensions)
        image_object.SetSpacing(spacing)
        image_object.SetOrigin(origin)
        # Get image fields
        image_group = self.get_node(imagename)
        FIndex_path = '%s/Field_index' % image_group._v_pathname
        Field_index = self.get_node(FIndex_path)
        if with_fields and (Field_index is not None):
            for fieldname in Field_index:
                name = fieldname.decode('utf-8')
                field_type = self.get_attribute('field_type', name)
                if field_type == 'Nodal_field':
                    data = self.get_field(field_name=name)
                    image_object.nodeFields[name] = data
                elif field_type == 'Element_field':
                    data = self.get_field(field_name=name)
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

    def get_table_description(self, tablename, as_dtype=False):
        """Get the description of the table as a description or Numpy dtype."""
        if as_dtype:
            return self[tablename].dtype
        return self.get_node(tablename).description

    def get_grid_field_list(self, gridname):
        """Return the list of fields stored on the grid.

        :param str gridname: Path, name or indexname of the grid Group on which
            the field will be added
        :return str Field_list: List of the name of the fields dataset stored
            in the hdf5 file and defined on the grid.
        """
        grid_group = self.get_node(gridname)
        FIndex_path = '%s/Field_index' % grid_group._v_pathname
        Field_index = self.get_node(FIndex_path)
        Field_list = []
        if Field_index is None:
            print(f'No Field_index node for grid {gridname}')
            return None
        for fieldname in Field_index:
            name = fieldname.decode('utf-8')
            Field_list.append(name)
        return Field_list

    def get_field(self, field_name, unpad_field=True,
                  get_visualisation_field=False):
        """Return a padded or unpadded field from a grid data group as array.

        Use this method to get a mesh element wise field in its original form,
        i.e. bulk element fields (defined on elements of the same dimensonality
        than the mesh) or a boundary field (defined on elements of a lower
        dimensionality than the mesh).

        :param str field_name: Name, Path, Index, Alias or Node of the field in
            dataset
        :param bool unpad_field: if `True` (default), remove the zeros added to
            to the field to comply with the mesh topology and return it with
            its original size (bulk or boundary field).
        """
        # Get field data array (or visualization field data array)
        field_type = self.get_attribute('field_type', field_name)
        if (field_type == 'IP_field') and get_visualisation_field:
            field_path = self.get_attribute('visualisation_field_path',
                                            field_name)
            field = self.get_node(field_path, as_numpy=True)
        else:
            field = self.get_node(field_name, as_numpy=True)
        # Handle array padding removal if needed
        padding = self.get_attribute('padding', field_name)
        parent_mesh = self.get_attribute('parent_grid_path', field_name)
        pad_field = (padding is not None) and unpad_field
        if (field_type == 'IP_field') and not get_visualisation_field:
            pad_field = False
        if pad_field:
            field = self._mesh_field_unpadding(field, parent_mesh, padding)
        # Handle field array dimensions to remove singleton dimension if field 
        # is a scalar mesh field
        if self._is_mesh(parent_mesh):
            field = np.squeeze(field)
        return field

    def get_node(self, name, as_numpy=False):
        """Return a HDF5 node in the dataset.

        The node is returned as a :py:class:`tables.Node`,
        :py:class:`tables.Group`  or a :py:class:`numpy.ndarray` depending
        on its nature, and the value of the `as_numpy` argument.

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
                if colname is not None:
                    node = node.col(colname)
            else:
                node = self.h5_dataset.get_node(node_path)
            if as_numpy and self._is_array(name) and (colname is None):
                # note : np.atleast_1d is used here to avoid returning a 0d
                # array when squeezing a scalar node
                node = np.atleast_1d(node.read())
                # Reverse data normalization if it has been applied
                # NOTE: it is very important to keep these lines before
                # those aiming at reversing transpositions for ordering
                # conventions as the arrays containing normalization parameters
                # comply to the in-memory ordering conventions
                norm = self.get_attribute('data_normalization', name)
                if norm == 'standard':
                    mu = self.get_attribute('normalization_mean', name)
                    std = self.get_attribute('normalization_std', name)
                    node = (node * std) + mu
                elif norm == 'standard_per_component':
                    mu = self.get_attribute('normalization_mean', name)
                    std = self.get_attribute('normalization_std', name)
                    for comp in range(node.shape[-1]):
                        node[..., comp] = (node[..., comp]*std[comp]) + mu[comp]
                # Reverse indices transpositions apply to ensure compatibility
                # between SampleData and Paraview ordering conventions
                transpose_indices = self.get_attribute('transpose_indices',
                                                       name)
                if transpose_indices is not None:
                    node = node.transpose(transpose_indices)
                transpose_components = self.get_attribute('transpose_components',
                                                       name)
                if transpose_components is not None:
                    node = node[..., transpose_components]
                node = np.atleast_1d(node)
        return node

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
        if data_path is None:
            self._verbose_print(' (get_attribute) neither indexname nor'
                                ' node_path passed, node return aborted')
        else:
            try:
                attribute = self.h5_dataset.get_node_attr(where=data_path,
                                                          attrname=attrname)
            except AttributeError:
                # self._verbose_print(' (get_attribute) node {} has no attribute'
                #                     ' `{}`'.format(nodename,attrname))
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
            while fsize / 1024 > 1:
                k = k + 1
                fsize = fsize / 1024
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
        #self.get_node('/')._v_attrs['sample_name'] = sample_name
        self.add_attributes({'sample_name': sample_name}, '/')

    def get_description(self, node='/'):
        """Get the string describing this node.

        By defaut the sample description is returned, from the root HDF5 Group.

        :param str node: the path or name of the node of interest.
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

        :param str image_group: Name, Path, Index name or Alias of the
            3DImage group
        :param np.array voxel_size: (dx, dy, dz) array of the voxel size in
            each dimension of the 3Dimage
        """
        old_spacing = self.get_attribute('spacing', image_group)
        if isinstance(voxel_size, float):
            voxel_size = np.ones(shape=(len(old_spacing),)) * voxel_size
        if len(old_spacing) != len(voxel_size):
            raise ValueError('Dimension mismatch between image group old'
                             f' grid spacing {old_spacing} and input'
                             f' new grid spacing {voxel_size}')
        self.add_attributes({'spacing': np.array(voxel_size)},
                            image_group)
        if len(voxel_size) == 2:
            VoxSize = voxel_size[[1, 0]]
        elif len(voxel_size) == 3:
            VoxSize = voxel_size[[2, 1, 0]]
        xdmf_grid = self._find_xdmf_grid(image_group)
        # find out if the grid has subgrids (temporal grid collection)
        if xdmf_grid.getchildren()[0].tag == 'Grid':
            # grid collection
            for grid in xdmf_grid:
                geo = self._find_xdmf_geometry(grid)
                spacing_node = geo.getchildren()[1]
                spacing_text = str(VoxSize).strip('[').strip(']').replace(',', ' ')
                spacing_node.text = spacing_text
        else:
            geo = self._find_xdmf_geometry(xdmf_grid)
            spacing_node = geo.getchildren()[1]
            spacing_text = str(VoxSize).strip('[').strip(']').replace(',', ' ')
            spacing_node.text = spacing_text
        self.sync()
        return

    def set_origin(self, image_group, origin):
        """Set origin coordinates for an HDF5/XDMF image data group.

        The origin corresponds to the first vertex of the first voxel, that is
        referenced by the [0,0,0] elements of arrays in the 3DImage group. The
        values are registered in the `origin` Attribute of the 3DImage group.

        :param str image_group: Name, Path, Index name or Alias of the
            3DImage group
        :param origin: (Ox, Oy, Oz) array of the coordinates in
            each dimension of the origin of this image group.
        """
        old_origin = self.get_attribute('origin', image_group)
        print(old_origin is None)
        if old_origin is not None and (len(old_origin) != len(origin)):
            raise ValueError('Dimension mismatch between image group origin'
                             f' {old_origin} and input new origin'
                             f' {origin}')
        self.add_attributes({'origin': origin}, image_group)
        if len(origin) == 2:
            Or = origin[[1, 0]]
        elif len(origin) == 3:
            Or = origin[[2, 1, 0]]
        xdmf_grid = self._find_xdmf_grid(image_group)
        # find out if the grid has subgrids (temporal grid collection)
        if xdmf_grid.getchildren()[0].tag == 'Grid':
            # grid collection
            for grid in xdmf_grid:
                geo = self._find_xdmf_geometry(grid)
                origin_node = geo.getchildren()[0]
                origin_text = str(Or).strip('[').strip(']').replace(',', ' ')
                origin_node.text = origin_text
        else:
            geo = self._find_xdmf_geometry(xdmf_grid)
            origin_node = geo.getchildren()[0]
            origin_text = str(Or).strip('[').strip(']').replace(',', ' ')
            origin_node.text = origin_text
        self.sync()
        return

    def set_tablecol(self, tablename, colname, column):
        """Store an array into a structured table column.

        If the column is not in the table description, a new field
        corresponding to the input column is added to the table description.

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
            raise ValueError('input column shape {} does not match the shape'
                             '{} of column {} in table {}'
                             ''.format(column.shape, col_shape, colname,
                                       tablename))
        table.modify_column(column=column, colname=colname)
        table.flush()
        return

    def set_nodes_compression_chunkshape(self, node_list=None, chunkshape=None,
                                         compression_options=dict()):
        """Set compression options for a list of nodes in the dataset.

        This methods sets the same set of compression options for a
        list of nodes in the dataset.

        :param list node_list: list of Name, Path, Index name or Alias of the
            HDF5 array nodes where to set the compression settings.
        :param tuple  chunkshape: The shape of the data chunk to be read or
            written in a single HDF5 I/O operation
        :param dict compression_options: Dictionary containing compression
            options items (keys are options names, values are )

        .. rubric:: Compression Options

        Compression settings can be passed through the `compression_options`
        dictionary as follows:
            compression_options[option_name] = option_value
        These options are the Pytables package `Filters
        <https://www.pytables.org/_modules/tables/filters.html#Filters>`_
        class constructor parameters (see `PyTables` documentation for details)
        The list of available compression options is provided here:

          * complevel: Compression level for data. Allowed range is 0-9.
            A value of 0 (the default) disables compression.
          * complib: Compression library to use. Possibilities are:
            zlib' (the default), 'lzo', 'bzip2' and 'blosc'.
          * shuffle:  Whether or not to use the *Shuffle* filter in the
            HDF5 library (may improve compression ratio).
          * bitshuffle: Whether or not to use the *BitShuffle* filter
            in the Blosc library (may improve compression ratio).
          * fletcher32: Whether or not to use the *Fletcher32* filter
            in the HDF5 library. This is used to add a checksum on each data
            chunk.
          * least_significant_digit:
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
            arguments, they are prioritised over the settings in the input
            Filter object.
        """
        if node_list is None:
            node_list = []
            for node in self.h5_dataset.root:
                if self._is_array(node):
                    node_list.append(node)
        for nodename in node_list:
            self.set_chunkshape_and_compression(nodename, chunkshape,
                                                compression_options)
        return

    def set_chunkshape_and_compression(self, nodename, chunkshape=None,
                                       compression_options=dict()):
        """Set the chunkshape and compression settings for a HDF5 array node.

        :param str nodename: Name, Path, Index name or Alias of the node
        :param tuple  chunkshape: The shape of the data chunk to be read or
            written in a single HDF5 I/O operation
        :param dict compression_options: Dictionary containing compression
            options items (keys are options names, values are )

        .. rubric:: Compression options

        Compression settings can be passed through the `compression_options`
        dictionary as follows:
            compression_options[option_name] = option_value
        See :func:`set_nodes_compression_chunkshape`, for the list of
        available compression options.

        .. important:: If the new compression settings reduce the size of the
            node in the dataset, the file size will not be changed. This is a
            standard behavior for HDF5 files, that preserves freed space in
            disk to add additional data in the future. If needed, use the
            :func:`repack_file` method to reduce file disk size after changing
            compression settings. This method is also called by the class
            instance destructor.

        .. note:: If compression settings are passed as additional keyword
            arguments, they are prioritised over the settings in the input
            Filter object.
        """
        if not self._is_array(nodename):
            msg = ('(set_chunkshape) Cannot set chunkshape or compression'
                   ' settings for a non array node')
            raise tables.NodeError(msg)
        # Get HDF5 node whose compression and chunkshape settings are to
        # be changed
        # chunkshape cannot be changed for a Pytables dataset node, and
        # changing compression settings can lead to errors or unexpected
        # behaviors
        # ==> New settings are set by reading rewriting data on the dataset
        # First get the hdf5 attributes of the target node
        attributes = self.get_dic_from_attributes(nodename)
        node_tmp = self.get_node(nodename)
        # Get node name, indexname and path
        nodename = node_tmp._v_name
        node_indexname = self.get_indexname_from_path(node_tmp._v_pathname)
        node_path = os.path.dirname(node_tmp._v_pathname)
        # Set new chunkshape if provided or get old one
        node_chunkshape = node_tmp.chunkshape
        if chunkshape is not None:
            node_chunkshape = chunkshape
        # Get node aliases
        if self.aliases.__contains__(node_indexname):
            node_aliases = self.aliases[node_indexname]
        else:
            node_aliases = []
        if self._is_table(nodename):
            # get stored array values
            array = node_tmp.read()
            description = node_tmp.description
            new_array = self.add_table(
                location=node_path, name=nodename, description=description,
                indexname=node_indexname, chunkshape=node_chunkshape,
                replace=True, data=array,
                compression_options=compression_options)
        elif self._is_field(nodename):
            array = self.get_field(nodename)
            parent_grid = self.get_attribute('parent_grid_path', nodename)
            visu_type = self.get_attribute('visualisation_type', nodename)
            if visu_type is None:
                visu_type = 'None'
            new_array = self.add_field(
                gridname=parent_grid, fieldname=nodename, array=array,
                indexname=node_indexname,
                chunkshape=node_chunkshape, replace=True,
                visualisation_type=visu_type,
                compression_options=compression_options)
        else:
            array = node_tmp.read()
            new_array = self.add_data_array(
                location=node_path, name=nodename, indexname=node_indexname,
                array=array, chunkshape=node_chunkshape,
                replace=True, compression_options=compression_options)
        for alias in node_aliases:
            self.add_alias(aliasname=alias, indexname=node_indexname)
        self.add_attributes(attributes, nodename)
        if self._verbose:
            self._verbose_print(self.print_node_compression_info(
                new_array._v_pathname))
        return

    def set_verbosity(self, verbosity=True):
        """Set the verbosity of the instance methods to input boolean."""
        self._verbose = verbosity
        return

    def remove_attribute(self, attrname, nodename):
        """Remove an attribute from a node in the dataset.

        :param str attrname: name of the attribute to remove
        :type attrname: str
        :param nodename: Name, Path or Index name of the node to modify
        :type nodename: str
        """
        node = self.get_node(nodename, as_numpy=False)
        node._v_attrs.__delitem__(attrname)
        return

    def remove_attributes(self, attr_list, nodename):
        """Remove an attribute from a node in the dataset.

        :param str attr_list: list of the names of the attribute to remove
        :type attr_list: list
        :param nodename: Name, Path or Index name of the node to modify
        :type nodename: str
        """
        for attr in attr_list:
            self.remove_attribute(attr, nodename)
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
        if isGroup and not recursive:
            msg = 'Node {} is a hdf5 group. '.format(node_path)
            msg += 'Use `recursive=True` keyword argument to remove it and its childrens.'
            self._verbose_print(msg)
            return

        # Remove visualisation field if node is a IP field node with an
        # additionnal visualisation values array
        IP_vis = self.get_attribute('visualisation_field_path', Node)
        if IP_vis is not None:
            IP = self.get_node(IP_vis)
            self.remove_node(IP)

        # Remove array used for per component data normalization.
        mean_path = self.get_attribute('norm_mean_array_path', Node)
        if mean_path is not None:
            mean = self.get_node(mean_path)
            self.remove_node(mean)
        std_path = self.get_attribute('norm_std_array_path', Node)
        if std_path is not None:
            Std = self.get_node(std_path)
            self.remove_node(Std)

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
            print('')
            self._remove_from_index(node_path=Node._v_pathname)
            # remove node in xdmf tree
            self._remove_from_xdmf(Node)
            Node.remove()
        # synchronize HDF5 and XDMF file with node removal
        self.sync()
        self._verbose_print('Node {} sucessfully removed'.format(name))
        return

    def resample_image_group(self, new_voxel_size, location='CellData',
                             new_location=None, in_place=False):
        """Resample a whole image group with a new spatial resolution.

        .. note::

          In the case where the spatial resolution if increased, the new cell
          data will have a surrounding layer of zeros (for the new cells centers
          located outside the original cells centers).

        :param float new_voxel_size: the new spatial resolution.
        :param str location: the location of the image group to process.
        :param str new_location: the name of the new location to store the
        resampled image group.
        :param bool in_place: if True, the actual image group will be replaced
        by the new resampled group.
        """

        # sanity check
        if not self._get_group_type(location) == '3DImage':
            print('works only on 3D images for now')
            return

        # work out each voxel coordinates
        dims = self.get_attribute('dimension', location)
        spacing = self.get_attribute('spacing', location)
        size = dims * spacing
        x = np.arange(0.5 * spacing[0], size[0], spacing[0])
        y = np.arange(0.5 * spacing[1], size[1], spacing[1])
        z = np.arange(0.5 * spacing[2], size[2], spacing[2])
        print(x.shape)

        # create the new coordinates
        new_spacing = np.array(3 * [new_voxel_size])
        x_new = np.arange(0.5 * new_spacing[0], size[0], new_voxel_size)
        y_new = np.arange(0.5 * new_spacing[1], size[1], new_voxel_size)
        z_new = np.arange(0.5 * new_spacing[2], size[2], new_voxel_size)
        print(x_new.shape)
        X_new, Y_new, Z_new = np.meshgrid(x_new,
                                          y_new,
                                          z_new,
                                          indexing='ij')
        # settings for the new group
        compression = self.default_compression_options
        if new_location is None:
            new_location = '%s_resampled' % location

        # now resample each field
        image_group = self.get_node(location)
        field_index_path = '%s/Field_index' % image_group._v_pathname
        field_list = self.get_node(field_index_path)
        from scipy.interpolate import RegularGridInterpolator
        for name in field_list:
            field_name = name.decode('utf-8')
            new_field_name = field_name
            if not in_place:
                new_field_name += '_resampled'
            print('+ resampling field %s' % field_name)
            field = self.get_field(field_name)
            if field.shape == (1,):
                print('skipping field %s' % field_name)
                continue
            # instanciate our interpolator
            resample = RegularGridInterpolator((x, y, z),
                                               field,
                                               method='nearest',
                                               bounds_error=False,
                                               fill_value=0)
            new_shape = list(X_new.shape)
            if field.ndim == 4:
                new_shape = new_shape.append(field.shape[3])
            new_field = resample(list(zip(X_new.ravel(),
                                          Y_new.ravel(),
                                          Z_new.ravel()))).reshape(new_shape).astype(field.dtype)
            # now add the resampled field to the new location
            if not self.__contains__(new_location):
                print('using add_image_from_field with %s' % new_field_name)
                self.add_image_from_field(field_array=new_field,
                                          fieldname=new_field_name,
                                          imagename=new_location,
                                          location='/',
                                          spacing=new_spacing,
                                          replace=True,
                                          compression_options=compression)
            else:
                print('using add_field with %s' % new_field_name)
                self.add_field(gridname=new_location,
                                 fieldname=new_field_name,
                                 array=new_field,
                                 location=new_location,
                                 replace=True,
                                 compression_options=compression)
        if in_place:
            self.remove_node(location, recursive=True)
            self.rename_node(new_location, location)
        return

    def repack_h5file(self):
        """Overwrite hdf5 file with a copy of itself to recover disk space.

        Manipulation to recover space leaved empty when removing data from
        the HDF5 tree or reducing a node space by changing its compression
        settings. This method is called also by the class destructor if the
        autorepack flag is `True`.
        """
        self.sync()
        head, tail = os.path.split(self.h5_path)
        tmp_file = os.path.join(head, 'tmp_' + tail)
        self.h5_dataset.copy_file(tmp_file)
        self.h5_dataset.close()
        shutil.move(tmp_file, self.h5_path)
        self.h5_dataset = tables.File(self.h5_path, mode='r+')
        self._file_exist = True
        self._after_file_open()
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
        """Create an element tag Id field on the input mesh.

        Creates a element wise field from the provided mesh,
        adding to each element the value of the Elset it belongs to.

        .. warning::

            - CAUTION : the method is designed to work with non intersecting
              element tags/sets. In this case, the produce field will indicate
              the value of the last elset containing it for each element.

        :param str meshname: Name, Path or index name of the mesh on which an
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
        n_elements = int(self.get_attribute('Number_of_elements', meshname))
        mesh = self.get_node(meshname)
        El_tag_path = '%s/Geometry/ElementsTags' % mesh._v_pathname
        ID_field = np.zeros((n_elements, 1), dtype=int)
        elem_tags = self.get_mesh_elem_tags_names(meshname)
        # if mesh is provided
        i = 0
        for set_name, set_type in elem_tags.items():
            elset_path = '%s/ET_%s' % (El_tag_path, set_name)
            element_ids = self.get_node(elset_path, as_numpy=True)
            if get_sets_IDs:
                set_ID = int(set_name.strip(tags_prefix))
            else:
                set_ID = i
                i += 1
            ID_field[element_ids] = set_ID
            if remove_elset_fields:
                field_path = '%s/field_%s' % (El_tag_path, set_name)
                self.remove_node(field_path)
        if store:
            if fieldname is None:
                fieldname = meshname + '_elset_ids'
            c_opt = {'complib': 'zlib', 'complevel': 1, 'shuffle': True}
            self.add_field(gridname=meshname, fieldname=fieldname,
                           array=ID_field, replace=True,
                           compression_options=c_opt)
        return ID_field

    # =========================================================================
    #  SampleData private methods
    # =========================================================================
    def _init_file_object(self, sample_name='', sample_description=''):
        """Initiate or create PyTable HDF5 file object."""
        try:
            self.h5_dataset = tables.File(self.h5_path, mode='r+')
            self._verbose_print('-- Opening file "{}" '.format(self.h5_file),
                                line_break=False)
            self._file_exist = True
        except IOError:
            self._file_exist = False
            self._verbose_print('-- File "{}" not found : file'
                                ' created'.format(self.h5_file),
                                line_break=True)
            self.h5_dataset = tables.File(self.h5_path, mode='a')
        self._init_xml_tree()
        # Generic Data Model initialization
        self._init_data_model()
        self._verbose_print('**** FILE CONTENT ****')
        self._verbose_print(SampleData.__repr__(self))
        if not self._file_exist:
            # add sample name and description specified at the creation
            self.set_sample_name(sample_name)
            self.set_description(sample_description)
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
        for level in range(max_path_level):
            self._verbose_print(f'Initializing level {level+1} of data model')
            for key, value in content_paths.items():
                if value.count('/') != level+1:
                    continue
                head, tail = os.path.split(value)
                # Find out if object is a description object
                is_descr = self._is_table_descr(content_type[key])
                if self.h5_dataset.__contains__(content_paths[key]):
                    if self._is_table(content_paths[key]):
                        msg = ('Updating table {}'.format(content_paths[key]))
                        self._verbose_print(msg)
                        self._update_table_columns(
                            table_name=content_paths[key],
                            description=content_type[key])
                    if self._is_empty(content_paths[key]):
                        self._verbose_print('Warning: node {} specified in the'
                                            ' minimal data model for this class'
                                            ' is empty'
                                            ''.format(content_paths[key]))
                    continue
                elif is_descr:
                    msg = ('Adding empty Table  {}'.format(content_paths[key]))
                    self.add_table(location=head, name=tail, indexname=key,
                                   description=content_type[key])
                elif content_type[key] == 'Group':
                    msg = f'Adding empty Group {content_paths[key]}'
                    self.add_group(groupname=tail, location=head,
                                   indexname=key, replace=False)
                elif content_type[key] in SD_IMAGE_GROUPS.values():
                    msg = f'Adding empty Image Group {content_paths[key]}'
                    self.add_image(imagename=tail, indexname=key, location=head)
                elif content_type[key] in SD_MESH_GROUPS.values():
                    msg = f'Adding empty Mesh Group {content_paths[key]}'
                    self.add_mesh(meshname=tail, indexname=key, location=head)
                elif content_type[key] == 'data_array':
                    msg = f'Adding empty data array  {content_paths[key]}'
                    self.add_data_array(location=head, name=tail,
                                        indexname=key)
                elif content_type[key] == 'field_array':
                    msg = (f'Adding empty field {content_paths[key]} to'
                           f' mesh group {head}')
                    print(msg)
                    self.add_field(gridname=head, fieldname=tail,
                                   indexname=key)
                elif content_type[key] == 'string_array':
                    msg = f'Adding empty string array {content_paths[key]}'
                    self.add_string_array(name=tail, location=head,
                                          indexname=key)
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

    def _check_SD_array_init(self, arrayname='', location='/', replace=False,
                             empty_input=False):
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
            array_path = '%s/%s' % (location_path, arrayname)
            if self.__contains__(array_path):
                empty = self.get_attribute('empty', array_path)
                if empty and (not empty_input):
                    # special case where we add data to a pre-existing
                    # node --> need to save the attributes and remove the
                    # empty array node
                    msg = ('Existing node {} will be overwritten to recreate '
                           'array/table'.format(array_path))
                    if not empty:
                        self._verbose_print(msg)
                    attrs = self.get_dic_from_attributes(array_path)
                    self.remove_node(array_path, recursive=True)
                    return attrs
                if replace:
                    msg = ('Existing node {} will be overwritten to recreate '
                           'array/table'.format(array_path))
                    if not empty:
                        self._verbose_print(msg)
                    self.remove_node(array_path, recursive=True)
                else:
                    msg = ('Array/table {} already exists. To overwrite, use '
                           'optional argument "replace=True"'
                           ''.format(array_path))
                    raise tables.NodeError(msg)
        return dict()

    def _init_SD_group(self, groupname='', location='/',
                       group_type='Group', replace=False):
        """Create or fetch a SampleData Group and returns it."""
        group = None
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
        # get rid of potential multiple / in path
        group_path = ('%s/%s' % (location, groupname)).replace('//', '/').replace('//', '/')
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
                       ' replace it by a new Group.'.format(groupname))
                raise tables.NodeError(msg)
        # create or fetch group
        if fetch_group:
            group = self.get_node(group_path)
            self._remove_from_index(group._v_pathname)
        else:
            self._verbose_print('Creating {} group `{}` in file {} at {}'
                                ''.format(group_type, groupname,
                                          self.h5_file, location))
            group = self.h5_dataset.create_group(where=location,
                                                 name=groupname,
                                                 title=groupname,
                                                 createparents=True)
            self.add_attributes(dic={'group_type': group_type}, nodename=group)
        return group

    @staticmethod
    def _merge_dtypes(dtype_1, dtype_2):
        """Merge 2 numpy.void dtypes to creates a new one."""
        descr = []
        for item in dtype_1.descr:
            if not(item in descr) and not(item[0] == ''):
                descr.append(item)
        for item in dtype_2.descr:
            if not(item in descr) and not(item[0] == ''):
                descr.append(item)
        return np.dtype(descr)

    def _update_table_columns(self, table_name, description):
        """Extends table with new fields in input Description."""
        table = self.get_node(table_name)
        current_desc = table.description
        current_dtype = tables.dtype_from_descr(table.description)
        if isinstance(description, np.dtype):
            desc_dtype = description
        else:
            desc_dtype = tables.dtype_from_descr(description)
        new_dtype = SampleData._merge_dtypes(current_dtype, desc_dtype)
        new_descr = tables.descr_from_dtype(new_dtype)[0]
        if current_dtype == new_dtype:
            self._verbose_print('Nothing to update for table `{}`'
                                ''.format(table_name))
            return
        self._verbose_print('Updating `{}` with fields {}'
                            ''.format(table_name, desc_dtype.fields))
        self._verbose_print('New table description is `{}`'
                            ''.format(new_dtype.fields))
        Nrows = table.nrows
        tab_name = table._v_name
        tab_indexname = self.get_indexname_from_path(table._v_pathname)
        tab_path = os.path.dirname(table._v_pathname)
        tab_chunkshape = table.chunkshape
        tab_filters = table.filters
        tab_c_opts = self._get_compression_opt_from_filter(tab_filters)
        # Create a new array with modified dtype
        data = np.array(np.zeros((Nrows,)), dtype=new_dtype)
        self._verbose_print(f'data is: {data}')
        # Get data from old table
        for key in current_desc._v_names:
            data[key] = self.get_tablecol(tablename=table_name,
                                          colname=key)
        # get table aliases
        if self.aliases.__contains__(tab_indexname):
            tab_aliases = self.aliases[tab_indexname]
        else:
            tab_aliases = []
        # remove old table
        tab_attrs = self.get_dic_from_attributes(tab_indexname)
        self.remove_node(tab_name)
        # create new table
        new_tab = self.add_table(location=tab_path, name=tab_name,
                                 description=new_descr,
                                 indexname=tab_indexname,
                                 chunkshape=tab_chunkshape, replace=True,
                                 data=data, compression_options=tab_c_opts)
        for alias in tab_aliases:
            self.add_alias(aliasname=alias, indexname=new_tab._v_pathname)
        self.add_attributes(tab_attrs, tab_indexname)
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
        try:
            if self._is_grid(nodename):
                xdmf_grid = self._find_xdmf_grid(nodename)
                p = xdmf_grid.getparent()
                p.remove(xdmf_grid)
            elif self._is_field(nodename):
                xdmf_field, xdmf_grid = self._find_xdmf_field(nodename)
                xdmf_grid.remove(xdmf_field)
        except ValueError:
            # node cannot be found
            pass
        return

    def _find_xdmf_grid(self, gridname):
        name = self.get_attribute('xdmf_gridname', gridname)
        if name is None:
            # in case we are looking for a temporal grid collection subgrid,
            # its name will be directly passed as argument to this method
            # but no node in the dataset will have this name.
            name = gridname
        for el in self.xdmf_tree.iterfind('.//Grid'):
            if el.get('Name') == name:
                return el
        return None

    def _get_xdmf_time_grid_name(self, gridname, time):
        name = self.get_attribute('xdmf_gridname', gridname)
        for el in self.xdmf_tree.iterfind('.//Grid'):
            if el.get('Name') == name:
                grid0 = el
        for ch_grid in grid0.iterchildren():
            for ch in ch_grid.iterchildren():
                if ch.tag == 'Time':
                    if float(ch.get('Value')) == time:
                        return ch_grid.get('Name')

    def _find_xdmf_geometry(self, xdmf_grid):
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
        try:
            for el in xdmf_grid:
                if el.tag == 'Grid':
                    for eel in el:
                        if eel.get('Name') == fieldname:
                            return eel, el
                else:
                    if el.get('Name') == fieldname:
                        return el, xdmf_grid
        except TypeError:
            msg = 'Warning: xdmf_grid not found for name %s' % fieldnodename
            print(msg)
            raise ValueError(msg)
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
        name_tmp = ('/%s' % name_or_node).replace('//', '/')
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

    def _is_table_descr(self, obj):
        """Find out if name or path references an array dataset."""
        is_descr=False
        try:
            is_descr = issubclass(obj, tables.IsDescription)
        except TypeError:
            is_descr=False
        if is_descr:
            return is_descr
        try:
            is_descr = (isinstance(obj, tables.IsDescription)
                        or isinstance(obj, np.dtype))
        except TypeError:
            is_descr=False
        return is_descr

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
        return test is not None

    def _is_in_index(self, name):
        return name in self.content_index

    def _is_alias(self, name):
        """Check if name is an HDF5 node alias."""
        Is_alias = False
        for item in self.aliases:
            if name in self.aliases[item]:
                Is_alias = True
                break
        return Is_alias

    def _is_alias_for(self, name):
        """Return the indexname for which input name is an alias."""
        Indexname = None
        for item in self.aliases:
            if name in self.aliases[item]:
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
                             ' or add_image_from_field to initialize grid'
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

    def _mesh_field_padding(self, field, meshname, bulk_padding):
        """Pad with zeros the mesh elem field to comply with size."""
        if self._is_image(meshname):
            return field, 'None', None
        Nelem_bulk = np.sum(self.get_attribute('Number_of_bulk_elements',
                                               meshname))
        Nelem_boundary = np.sum(self.get_attribute(
                                'Number_of_boundary_elements', meshname))
        if Nelem_bulk == Nelem_boundary:
            force_padding = True
        else:
            force_padding = False
        padding = 'None'
        vis_field = None
        # reshape field if it is a 1D array
        if len(field.shape) == 1:
            field = field.reshape((len(field),1))
        if field.shape[0] == Nelem_bulk:
            padding = 'bulk'
            if force_padding:
                if not bulk_padding:
                    padding = 'boundary'
        elif field.shape[0] == Nelem_boundary:
            padding = 'boundary'
        elif (field.shape[0] % Nelem_bulk) == 0:
            # if the number of values  for the field is a multiplier of the
            # number of elements, it is considered that the array describes a
            # field integration point values and that it is stored as follows:
            # field[:,k] = [Val_elt1_IP1, Val_elt1_IP2, ...., Val_elt1_IPN,
            #               Val_elt2_IP1, ...., Val_eltN_IPN]
            padding = 'bulk_IP'
            if force_padding:
                if not bulk_padding:
                    padding = 'boundary_IP'
        elif Nelem_boundary != 0:
            if (field.shape[0] % Nelem_boundary) == 0:
                padding = 'boundary_IP'
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
        elif padding == 'boundary_IP':
            Nip_elt = field.shape[0] // Nelem_boundary
            new_shape = (Nelem_boundary,Nip_elt,*field.shape[1:])
            vis_field = field.reshape(new_shape)
            Nelem_bulk = np.sum(self.get_attribute(
                                    'Number_of_bulk_elements',meshname))
            pad_array = np.zeros(shape=(Nelem_bulk, *vis_field.shape[1:]))
            vis_field = np.concatenate((pad_array, vis_field), axis=0)
        return field, padding, vis_field

    def _mesh_field_unpadding(self, field, parent_mesh, padding):
        """Remove zeros to return field to original shape, before padding."""
        Nelem_bulk = np.sum(self.get_attribute('Number_of_bulk_elements',
                                               parent_mesh))
        Nelem_boundary = np.sum(self.get_attribute(
                                'Number_of_boundary_elements', parent_mesh))
        if len(field.shape) == 1:
            L = field.shape[0]
            field = field.reshape((L,1))
        if (padding == 'bulk') or (padding == 'bulk_IP'):
            field = field[:Nelem_bulk,:]
        elif (padding == 'boundary') or (padding == 'boundary_IP'):
            field = field[Nelem_boundary,:]
        elif padding == 'None':
            pass
        else:
            raise Warning('Cannot unpad the field, unknown padding type `{}`'
                          ''.format(padding))
        return np.atleast_1d(field)

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
        # Get mesh group indexname
        indexname = self.get_indexname_from_path(mesh_group._v_pathname)
        # create Geometry group
        geo_group = self.add_group(groupname='Geometry',
                                   location=mesh_group._v_pathname,
                                   indexname=indexname+'_Geometry',
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
        # Get mesh group indexname
        indexname_tmp = self.get_indexname_from_path(mesh_group._v_pathname)
        indexname = indexname_tmp+'_Nodes'
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
                indexname=indexname_tmp+'_Nodes_ID')
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
        indexname = self.get_indexname_from_path(mesh_group._v_pathname)
        geo_group = self.get_node(indexname+'_Geometry')
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
        indexname = self.get_indexname_from_path(mesh_group._v_pathname)
        Ntags_group = self.add_group(groupname='NodeTags',
                                     location=geo_group._v_pathname,
                                     indexname=indexname+'_NodeTags',
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
                c_opt = {'complib':'zlib', 'complevel':1, 'shuffle':True}
                node = self.add_field(mesh_group._v_pathname,
                                      fieldname='field_'+name,
                                      array=data, replace=replace,
                                      location=Ntags_group._v_pathname,
                                      compression_options=c_opt)
                # remove from index : Elsets may be too numerous and
                # overload content index --> actual choice is to remove
                # them from index
                self._remove_from_index(node._v_pathname)
        self.add_string_array(
            name='Node_tags_list', location=geo_group._v_pathname,
            indexname=indexname+'_NodeTagsList', data=Node_tags_list)
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
        indexname = self.get_indexname_from_path(mesh_group._v_pathname)
        Etags_group = self.add_group(groupname='ElementsTags',
                                     location=geo_group._v_pathname,
                                     indexname=indexname+'_ElemTags',
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
                    c_opt = {'complib':'zlib', 'complevel':1, 'shuffle':True}
                    node = self.add_field(mesh_group._v_pathname,
                                          fieldname='field_'+name,
                                          array=data, replace=replace,
                                          location=Etags_group._v_pathname,
                                          compression_options=c_opt)
                    # remove from index : Elsets may be too numerous and
                    # overload content index --> actual choice is to remove
                    # them from index
                    self._remove_from_index(node._v_pathname)
        self.add_string_array(
            name='Elem_tags_list', location=geo_group._v_pathname,
            indexname=indexname+'_ElTagsList', data=Elem_tags_list)
        self.add_string_array(
            name='Elem_tag_type_list', location=geo_group._v_pathname,
            indexname=indexname+'_ElTagsTypeList',
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
        indexname = self.get_indexname_from_path(mesh_group._v_pathname)
        Ntags_group = self.get_node(indexname+'_NodeTags')
        if Ntags_group is not None:
            Ntag_list_indexname = indexname+'_NodeTagsList'
            Ntag_list = self.get_node(Ntag_list_indexname)
            for name in Ntag_list:
                tag_name = name.decode('utf-8')
                tag = mesh_object.GetNodalTag(tag_name)
                tag_path = '%s/NT_%s' % (Ntags_group._v_pathname, tag_name)
                tag.SetIds(self.get_node(tag_path, as_numpy))
        return mesh_object

    def _load_elements_tags(self, meshname, AllElements, as_numpy=True):
        """Add Node and ElemTags in mesh object from mesh group."""
        # get SD groups
        mesh_group = self.get_node(meshname)
        indexname = self.get_indexname_from_path(mesh_group._v_pathname)
        Etags_group = self.get_node(indexname+'_ElemTags')
        # create an dictionnary for the offset to apply to elem local Ids
        # for each element type
        element_type = self.get_attribute('element_type', mesh_group._v_name)
        offsets = self.get_attribute('Elements_offset', mesh_group._v_name)
        offset_dic = {element_type[i]:offsets[i] for i in range(len(offsets))}
        # load element tags
        if Etags_group is not None:
            mesh_idname = self.get_indexname_from_path(mesh_group._v_pathname)
            Etag_list_indexname = mesh_idname+'_ElTagsList'
            Etag_list = self.get_node(Etag_list_indexname)
            Etag_Etype_list_indexname = mesh_idname+'_ElTagsTypeList'
            Etag_Etype_list = self.get_node(Etag_Etype_list_indexname)

            for i in range(len(Etag_list)):
                tag_name = Etag_list[i].decode('UTF-8')
                el_type = Etag_Etype_list[i].decode('UTF-8')
                elem_container = AllElements.GetElementsOfType(el_type)
                tag = elem_container.tags.CreateTag(tag_name,False)
                tag_path = '%s/ET_%s' % (Etags_group._v_pathname, tag_name)
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
        # Get mesh group indexname
        indexname_tmp = self.get_indexname_from_path(mesh_group._v_pathname)
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
        indexname = indexname_tmp+'_Elements'
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
        mesh_indexname = self.get_indexname_from_path(mesh_group._v_pathname)
        nodes = self.get_node(mesh_indexname+'_Nodes', as_numpy=False)
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
        indexname = self.get_indexname_from_path(mesh_group._v_pathname)
        elems =  self.get_node(indexname+'_Elements', as_numpy=False)
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
        index_path = '%s/Field_index' % grid._v_pathname
        Field_index = self.get_node(index_path)
        if Field_index is None:
            grid_indexname = self.get_indexname_from_path(grid._v_pathname)
            Field_index = self.add_string_array(
                'Field_index', location=grid._v_pathname,
                indexname=grid_indexname+'_Field_index')
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
        """Write field data as Grid Attribute in xdmf tree/file.

        :param str fieldname: the string representing the field name.
        :param ndarray field: the field array.
        """
        Node = self.get_node(fieldname)
        Grid_name = self.get_attribute('xdmf_gridname', fieldname)
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
        # Get the time_serie field name if it exists time_serie_name
        time_serie_name = self.get_attribute('time_serie_name', fieldname)
        if time_serie_name is None:
            attr_name = Node._v_name
        else:
            attr_name = time_serie_name
        # create Attribute element
        Attribute_xdmf = etree.Element(_tag='Attribute', Name=attr_name,
                                       AttributeType=field_dimensionality,
                                       Center=Center_type)
        # Create data item element
        Dimension = self._np_to_xdmf_str(Node.shape)
        if np.issubdtype(field.dtype, np.floating):
            NumberType = 'Float'
            if str(field.dtype) == 'float':
                Precision = '32'
            else:
                Precision = '64'
        elif np.issubdtype(field.dtype, np.integer):
            NumberType = 'Int'
            Precision = str(field.dtype).strip('int')
        Attribute_data = etree.Element(_tag='DataItem', Format='HDF',
                                       Dimensions=Dimension,
                                       NumberType=NumberType,
                                       Precision=Precision)
        Attribute_data.text = (self.h5_file + ':'
                               + self._name_or_node_to_path(fieldname))
        # if data normalization is used, a intermediate DataItem is created
        # to apply a linear function to the dataitem
        norm = self.get_attribute('data_normalization', fieldname)
        if norm == 'standard':
            mu = self.get_attribute('normalization_mean', fieldname)
            std = self.get_attribute('normalization_std', fieldname)
            Func = f'{std}*($0) + {mu}'
            Function_data = etree.Element(_tag='DataItem',
                                          ItemType='Function',
                                          Function=Func,
                                          Dimensions=Dimension)
            # add data item to function data item
            Function_data.append(Attribute_data)
            # add data item to attribute
            Attribute_xdmf.append(Function_data)
        elif norm == 'standard_per_component':
            # Create dataitem for per component mean
            mean_path = self.get_attribute('norm_mean_array_path', fieldname)
            Attribute_mean = etree.Element(_tag='DataItem', Format='HDF',
                                                   Dimensions=Dimension,
                                                   NumberType=NumberType,
                                                   Precision=Precision)
            Attribute_mean.text = (self.h5_file + ':' + mean_path)
            # Create dataitem for per component std
            std_path = self.get_attribute('norm_std_array_path', fieldname)
            Attribute_std = etree.Element(_tag='DataItem', Format='HDF',
                                                   Dimensions=Dimension,
                                                   NumberType=NumberType,
                                                   Precision=Precision)
            Attribute_std.text = (self.h5_file + ':' + std_path)
            # Create dataitem for Function
            Func = '($2*$0) + $1'
            Function_data = etree.Element(_tag='DataItem',
                                          ItemType='Function',
                                          Function=Func,
                                          Dimensions=Dimension)
            # add data items to function data item
            Function_data.append(Attribute_data)
            Function_data.append(Attribute_mean)
            Function_data.append(Attribute_std)
            # add function data item to attribute
            Attribute_xdmf.append(Function_data)
        else:
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

    def _get_group_info(self, groupname, as_string=False, short=False):
        """Print a human readable information on the Pytables Group object."""
        s = ''
        Group = self.get_node(groupname)
        if Group is None:
            return f'No group named {groupname}'
        gname = Group._v_name
        if short:
            s = ('  '*Group._v_depth)+'|--'
            gtype = self.get_attribute('group_type', Group)
            s += f'GROUP {gname}: {Group._v_pathname} ({gtype}) \n'
            return s
        s += str('\n GROUP {}\n'.format(gname))
        s += str('=====================\n')
        gparent_name = Group._v_parent._v_name
        s += str(' -- Parent Group : {}\n'.format(gparent_name))
        s += str(' -- Group attributes : \n')
        for attr in Group._v_attrs._v_attrnamesuser:
            value = Group._v_attrs[attr]
            s += str('\t * {} : {}\n'.format(attr, value))
        s += str(' -- Childrens : ')
        for child in Group._v_children:
            s += str('{}, '.format(child))
        s += str('\n----------------')
        if not(as_string):
            print(s)
            s = ''
        return s

    def _get_array_node_info(self, nodename, as_string=False, short=False):
        """Print a human readable information on the Pytables Group object."""
        Node = self.get_node(nodename)
        if Node is None:
            return f'No group named {nodename}'
        size, unit = self.get_node_disk_size(nodename, print_flag=False)
        if short:
            s = ('  '*Node._v_depth)+' --'
            ntype = self.get_attribute('node_type', Node)
            if self._is_empty(Node):
                empty_s = ' - empty'
            else:
                empty_s= ''
            s += f'NODE {Node._v_name}: {Node._v_pathname}'
            s += f' ({ntype}{empty_s}) ({size:9.3f} {unit})'
            return s+'\n'
        s = ''
        s += str('\n NODE: {}\n'.format(Node._v_pathname))
        s += str('====================\n')
        nparent_name = Node._v_parent._v_name
        s += str(' -- Parent Group : {}\n'.format(nparent_name))
        s += str(' -- Node name : {}\n'.format(Node._v_name))
        s += self.print_node_attributes(Node, as_string=True)
        s += str(' -- content : {}\n'.format(str(Node)))
        if self._is_table(nodename):
            s += ' -- table description : \n'
            s += repr(Node.description)+'\n'
        s += str(' -- ')
        s += self.print_node_compression_info(nodename, as_string=True)
        s += str(' -- Node memory size : {:9.3f} {}\n'.format(size, unit))
        s += str('----------------\n')
        if not(as_string):
            print(s)
            s = ''
        return s

    def _get_compression_opt(self, compression_opts=dict()):
        """Get input compression settings as `tables.Filters` instance."""
        Filters = tables.Filters()
        # ------ read compression options in dict
        for option in compression_opts:
            if (option == 'complib'):
                Filters.complib = compression_opts[option]
            elif (option == 'complevel'):
                Filters.complevel = compression_opts[option]
            elif (option == 'shuffle'):
                Filters.shuffle = compression_opts[option]
            elif (option == 'bitshuffle'):
                Filters.bitshuffle = compression_opts[option]
            elif (option == 'checksum'):
                Filters.fletcher32 = compression_opts[option]
            elif (option == 'least_significant_digit'):
                Filters.least_significant_digit = compression_opts[option]
        return Filters

    def _get_compression_opt_from_filter(self, Filters):
        """Get input compression settings as `tables.Filters` instance."""
        c_opts = {}
        c_opts['complib'] = Filters.complib
        c_opts['complevel'] = Filters.complevel
        c_opts['shuffle'] = Filters.shuffle
        c_opts['bitshuffle'] = Filters.bitshuffle
        c_opts['least_significant_digit'] = Filters.least_significant_digit
        c_opts['bitshuffle'] = Filters.bitshuffle
        return c_opts

    @staticmethod
    def _data_normalization(array, normalization):
        """Apply normalization to data array to improve compression ratios"""
        if normalization == 'standard':
            mu = np.mean(array)
            std= np.std(array)
            array = (array - mu) / std
            normalization_attributes = {'data_normalization':normalization,
                                        'normalization_mean':mu,
                                        'normalization_std':std}
        if normalization == 'standard_per_component':
            # Apply component per component
            mu = np.zeros((array.shape[-1]))
            std = np.zeros((array.shape[-1]))
            mu_array = np.zeros((array.shape))
            std_array = np.zeros((array.shape))
            for comp in range(array.shape[-1]):
                mu[comp] = np.mean(array[..., comp])
                std[comp] = np.std(array[..., comp])
                array[..., comp] = (array[..., comp] - mu[comp]) / std[comp]
                mu_array[..., comp] = mu[comp]
                std_array[..., comp] = std[comp]
            # Create normalization attributes
            normalization_attributes = {'data_normalization':normalization,
                                        'normalization_mean':mu,
                                        'normalization_std':std,
                                        'norm_mean_array':mu_array,
                                        'norm_std_array':std_array}
        return array, normalization_attributes

    def _read_mesh_from_file(self, file=''):
        """Read a data array from a file, depending on the extension."""
        mesh_object = None
        from BasicTools.IO.UniversalReader import ReadMesh
        mesh_object = ReadMesh(file)
        return mesh_object

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
