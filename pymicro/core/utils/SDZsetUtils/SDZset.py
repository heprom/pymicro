#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SD Zset Module: base class to create SampleData / Zset bindings class
"""

## Imports
import os
import shutil
import numpy as np

from subprocess import run
from pathlib import Path
from string import Template

import BasicTools.IO.UtReader as UR

from pymicro.core.utils.templateUtils import ScriptTemplate
from pymicro.core.samples import SampleData

# SD Zset mesher class
class SDZset():
    """Base class implementing a Zset / SampleData interface.

    This class is an inteface between the SampleData data platform class and
    the Zset software (developed at Centre des Matériaux, Mines Paris).
    It provides a high-level interface to use Zset to process data stored in
    SampleData Mesh groups, and store the results in the same SampleData
    instance/file.

    The use of this class methods requires a Zset compatible environnement.
    That means that the Zset software must be available in your PATH
    environnement variable at least.
    """

    def __init__(self, data=None, sd_datafile=None,
                 inp_filename=Path('.').absolute() / 'script.inp',
                 inputmesh=None, outputmesh=None,
                 input_meshfile=Path('.').absolute() / 'input.geof',
                 output_meshfile=None,
                 verbose=False, autodelete=False, data_autodelete=False):
        """Zset SampleData interface constructor.

        :param data: SampleData object to store input and output data for/from
            Zset. If `None`, the class will open a SampleData instance from a
            datafile path. Defaults to None
        :type data: :py:class:`pymicro.core.samples.SampleData` object, optional
        :param sd_datafile:  Path of the hdf5/xdmf data file to open as the
            SampleData instance containing mesh tools input data. Defaults to
            None
        :type sd_datafile: str, optional
        :param inp_filename: Name of the .inp script file associated to the
            class instance. Zset commands passed to the class instance are
            written in this file. Defaults to `./script.inp`
        :type inp_filename: str, optional
        :param inputmesh: Name, Path, Indexname or Alias of the mesh group
            in the SampleData instance to use as Zset inputmesh. Defaults to
            None
        :type inputmesh: str, optional
        :param outputmesh:  Name, Path, Indexname or Alias of the mesh group
            in the SampleData instance to store Zset output mesh (for Zset
            meshers). Defaults to None
        :type outputmesh: str, optional
        :param input_meshfile: Name of .geof mesh file to use as Zset input.
            If `inputmesh` is `None`, the file must exist and be a valid .geof
            meshfile for Zset. If `inputmesh` is not `None`, then the mesh data
            refered by `mesh_name` in the SampleData instance will be written
            as a .geof mesh file `meshfilename.geof`.
            Defaults to `./input.geof`
        :type input_meshfile: str, optional
        :param output_meshfile: Name of the mesh .geof file to use as output
            when using Zset meshers. Defaults to `./output.geof`
        :type output_meshfile: str, optional
        :param verbose: verbosity flag, defaults to False
        :type verbose: bool, optional
        :param autodelete: If `True`, removes all temporary files, script files
            and Zset output files (~ Zclean) when deleting the class isntance.
            Defaults to False
        :type autodelete: bool, optional
        :param data_autodelete: If `True`, set `autodelete` flag to `True` on
            SampleData instance. Defaults to False
        :type data_autodelete: bool, optional
        """
        # Initialize SampleData instance
        self.set_data(data=data, datafile=sd_datafile,
                      data_autodelete=data_autodelete)
        # Init inp file content and associated ScriptTemplate file instance
        self.inp_lines = []
        self.script_args_list = []
        self.Script = ScriptTemplate(template_file='tmp_file.inp',
                                     script_command='Zrun',
                                     autodelete=autodelete)
        # Set default values for input output files
        self.set_inp_filename(filename=inp_filename)
        self.set_inputmesh(meshname=inputmesh, meshfilename=input_meshfile)
        self.set_outputmesh(meshname=outputmesh, meshfilename=output_meshfile)
        # init line position counter for the script file writer
        self._current_position = 0
        self._init_script_content()
        # Init flags
        self._verbose = verbose
        self.autodelete = autodelete
        return

    def __del__(self):
        """SDZset destructor."""
        if self.data is not None:
            del self.data
        if self.autodelete:
            self.clean_output_files()
            self.clean_script_files()
            if hasattr(self, 'data_inputmesh'):
                self.clean_input_mesh_file()
        return

    def __repr__(self):
        """String representation of the class."""
        classname = str(self.__class__).split('.')[-1][:-2]
        s  = '\n {} Class instance: \n\n'.format(classname)
        if self.data is not None:
            s += ('---- Associated SampleData dataset:\n\t\t{}\n'
                  ''.format(self.data.h5_file))
            s += ('\t\tdataset name:\t{}\n'
                  ''.format(self.data.get_sample_name()))
            s += ('\t\tdataset description:\t{}\n'
                  ''.format(self.data.get_description()))
            if hasattr(self, 'data_inputmesh'):
                path = self.data._name_or_node_to_path(self.data_inputmesh)
                s += (f'\n---- Input Mesh data path: {path}\n')
            if hasattr(self, 'data_outputmesh'):
                path = self.data._name_or_node_to_path(self.data_outputmesh)
                s += (f'\n---- Output Mesh data path: {path}\n')
        else:
            s += '---- Associated SampleData dataset:\n\t\tNone\n'
        s += '\n---- Mesh input file: {}\n'.format(str(self.input_meshfile))
        s += '\n---- Mesh output file: {}\n'.format(str(self.output_meshfile))
        s += '\n---- Inp script template filename: {}\n'.format(
                                                        str(self.inp_script))
        s += '\n---- Inp commands argument values:\n'
        for key,value in self.Script.args.items():
            s += '\t\t{:30} --> {} \n'.format(key,value)
        s += '\n---- Autodelete flag:\n\t\t{}\n'.format(self.autodelete)
        return s

    def set_data(self, data=None, datafile=None, data_autodelete=False):
        """Set the SampleData instance to use to manipulate mesh data.

        :param SampleData data: SampleData instance to use
        :param str datafile: SampleData file to open as a the SampleData
            instance to use, if `data` argument is `None`. If datafile does not
            exist, the SampleData instance is created.
        """
        if data is None:
            if datafile is None:
                self.data = None
            else:
                self.data = SampleData(filename=datafile)
                if self.autodelete:
                    self.data.autodelete = data_autodelete
        elif (data is not None):
            self.data = data
        return

    def set_inputmesh(self, meshname=None, meshfilename=None,
                      fields_to_transfer=None):
        """Set the inputmesh file or path into the SampleData instance.

        :param str meshname: Name, Path, Indexname or Alias of the mesh group
            to use as input mesh for Zset in the SampleData instance.
        :param str meshfilename: Name of the mesh file to use as input mesh
            for the mesher. If `mesh_name` is `None`, the file must exist and be
            a valid .geof meshfile for Zset. If `mesh_name` is not `None`, then
            the mesh data refered by `mesh_name` in the SampleData instance will
            be written as a .geof mesh file `meshfilename.geof`
        """
        self.fields_to_transfer = []
        if meshname is not None:
            self.data_inputmesh = meshname
        if meshfilename is not None:
            p = Path(meshfilename).absolute()
            self.input_meshfile = p.parent / f'{p.stem}.geof'
            self.set_script_args(input_meshfile=str(self.input_meshfile))
        if fields_to_transfer is not None:
            self.fields_to_transfer = fields_to_transfer
        return

    def set_outputmesh(self, meshname=None, meshfilename=None):
        """Set output mesh file or data path into the SampleData instance.

        :param str meshname: Name, Path, Indexname or Alias of the mesh group
            to use as output data group to store the mesh processed by a
            Zset mesher in the SampleData instance.
            If `mesh_name` is `None`, the mesher will only output a .geof file,
            with name `meshfilename`.
        :param str meshfilename: Name of the mesh .geof file to use as output.
        """
        if meshname is not None:
            self.data_outputmesh = meshname
        if meshfilename is not None:
            p = Path(meshfilename).absolute()
            self.output_meshfile = p.parent / f'{p.stem}.geof'
            self.set_script_args(output_meshfile=str(self.output_meshfile))
        return

    def set_inp_filename(self, filename=None):
        """Set the name of the .inp file and template object to run Zset."""
        if filename is not None:
            p = Path(filename).absolute()
            self.inp_script = p.parent / f'{p.stem}.inp'
            self.inp_template = p.parent / f'{p.stem}_tmp.inp'
            self.Script.set_template_filename(self.inp_script)
        return

    def set_workdir(self, workdir=None):
        """Set the path of the work directory to write and run Zset script."""
        self.Script.set_work_directory(workdir)
        return

    def set_script_args(self, args_dict={}, **keywords):
        """Set the value of the dict inputed command arguments values.

        The values must be passed as entries of the form
        'argument_value_template_name':argument_value. They can be passed all
        at once in a dict, or one by one as keyword arguments.

        The template names of command arguments in the current script can be
        printed using the `print_inp_template_content` method.

        :param dict args_dict: values of Zset command arguments. The dict
            must contain entries of the form
            'argument_value_template_name':argument_value.
        """
        self.Script.set_arguments(args_dict, **keywords)

    def clean_script_files(self, remove_template=True):
        """Remove the template and .inp script files created by the class.

        :param bool remove_template: If `True` (default), removes the inp
            template file. If `False`, removes only the script file built from
            the template file.
        """
        # Remove last script file
        self.Script.clean_script_file()
        # Remove template file
        if remove_template and self.inp_script.exists():
            print('Removing {} ...'.format(str(self.inp_script)))
            os.remove(self.inp_script)
        return

    def clean_output_files(self, problem_name=None, clean_Zset_output=True):
        """Remove all Zset output files and output .geof file if possible."""
        if problem_name is None:
            problem_name = f'{self.inp_script.stem}_tmp'
        if clean_Zset_output:
            run(args=['Zclean',problem_name])
        mesh_path = Path(problem_name).with_suffix('.geof')
        if mesh_path.exists():
            os.remove(mesh_path)
        if hasattr(self, 'output_meshfile'):
            if self.output_meshfile.exists():
                print('Removing {} ...'.format(str(self.output_meshfile)))
                os.remove(self.output_meshfile)
        return

    def clean_input_mesh_file(self):
        """Remove Zset inpput .geof file if possible."""
        if self.input_meshfile.exists():
            print('Removing {} ...'.format(str(self.input_meshfile)))
            os.remove(self.input_meshfile)
        return

    def print_inp_template_content(self):
        """Print the current Zset script template content to write as .inp.

        This methods prints all Zset commands prescribed by this class
        instance, without the value of the command arguments specified, as if
        it was a .inp template file.
        """
        print('Content of the .inp template:\n-----------------------\n')
        for line in self.inp_lines:
            print(line, end='')
        return

    def print_inp_content(self):
        """Print the current Zset script content.

        This methods prints all Zset commands prescribed by this class
        instance, with the prescribed command arguments, as if it was the
        a .inp file.
        """
        print('\nContent of the .inp script:\n-----------------------\n')
        self._check_arguments_list()
        for line in self.inp_lines:
            line_tmp = Template(line)
            print(line_tmp.substitute(self.Script.args), end='')
        return

    def print_Zset_msg(self):
        """Print the content of the Zset .msg output log file."""
        # The .inp file template is transformed into a _tmp.inp script file
        # and hence, the .msg file has the same base_name {inp}_tmp.msg
        msg_file = self.inp_script.parent / f'{self.inp_script.stem}_tmp.msg'
        print('\n============================================')
        print('Content of the {} file'.format(str(msg_file)))
        with open(msg_file,'r') as f:
            print(f.read())
        return

    def load_input_mesh(self, meshname=None, mesh_location='/',replace=False,
                        mesh_indexname='', bin_fields_from_sets=True):
        """Load into SampleData instance the inputmesh .geof file data.

        :param str meshname: Name of mesh data group to create to load file
            content into SampleData instance
        :param str mesh_location: Path of the meshgroup to create into the
            SampleData instance.
        :param str mesh_indexname: Indexname for the mesh group to create.
        :param bool replace: If `True` overwrites the meshgroup if needed.
        :param bool bin_fields_from_sets: If `True`, creates a binary field for
            each node/element set defined in the .geof file.
        :return mesh: BasicTools Unstructured Mesh object containing mesh data
            from the .geof file
        """
        self.load_geof_mesh(self.input_meshfile, meshname=meshname,
                            mesh_location=mesh_location,
                            mesh_indexname=mesh_indexname, replace=replace,
                            bin_fields_from_sets=bin_fields_from_sets)
        return

    def load_geof_mesh(self, filename=None, meshname=None, mesh_location='/',
                       mesh_indexname='', replace=False,
                       bin_fields_from_sets=True):
        """Read and load into SampleData instance mesh data from a .geof file.

        :param str filename: Relative or absolute path of .geof file to load.
        :param str meshname: Name of mesh data group to create to load file
            content into SampleData instance
        :param str mesh_location: Path of the meshgroup to create into the
            SampleData instance.
        :param str mesh_indexname: Indexname for the mesh group to create.
        :param bool replace: If `True` overwrites the meshgroup if needed.
        :param bool bin_fields_from_sets: If `True`, creates a binary field for
            each node/element set defined in the .geof file.
        :return mesh: BasicTools Unstructured Mesh object containing mesh data
            from the .geof file
        """
        import BasicTools.IO.GeofReader as GR
        p = Path(filename).absolute()
        mesh = GR.ReadGeof(str(p))
        if (self.data is not None) and (meshname is not None):
            self.data.add_mesh(mesh_object=mesh, meshname=meshname,
                               indexname=mesh_indexname, replace=replace,
                               location=mesh_location,
                               bin_fields_from_sets=bin_fields_from_sets)
        return mesh

    def load_FEA_metadata(self, ut_file=None, load_in_data=False):
        """If Zset script output exists, read fem analysis metadata.

        :param ut_file: .ut file from which the metadata must be loaded.
            Defaults to None
        :type ut_file: str, optional
        :param load_in_data: If `True`, load the metadata as output Mesh Group
            HDF5 attributes in the SampleData instance. Defaults to False
        :type load_in_data: bool, optional
        """
        if ut_file is None:
            ut_file = str(self.inp_template.with_suffix('.ut'))
        SDZset._clean_comments(ut_file)
        self.metadata = UR.ReadUTMetaData(ut_file)
        self.metadata['Sequence'] = self.metadata["time"][:, 4]
        if load_in_data:
            self.data.add_attributes(self.metadata, self.data_inputmesh)
        return

    def load_field_sequence(self, field_sequence, sequence_name, suffix=None,
                            replace=False):
        """Load a field sequence into the SampleData instance.

        :param field_sequence: dict of the form {fielname: field array} of
            FEM fields from one sequence of a Zset FEA calculation. See
            `read_output_fields` for more details.
        :type field_sequence: TYPE
        :param sequence_name: Name of the HDF5 group used to store the sequence
            fields. If the group does not exist, it is created.
        :type sequence_name: str
        :param suffix: suffix to add to the fields names. If `None` (default),
            suffix is '_sequence_name'.
        :type suffix: TYPE, optional
        """
        # create a Group to store the sequence in the Mesh group
        if not self.data.__contains__(sequence_name):
            self.data.add_group(groupname=sequence_name,
                                location=self.data_inputmesh, replace=replace)
        # store fields in the group for the mesh group
        for key, value in field_sequence.items():
            if suffix is None:
                name = key + '_' + sequence_name
            else:
                name = key + suffix
            self.data.add_field(gridname=self.data_inputmesh, fieldname=name,
                                array=value, location=sequence_name,
                                replace=replace)
        return

    def load_displacement_field_sequence(self, ut_file=None,
                                        sequence_list=None, storage_group=None,
                                        field_basename='U', storage_mesh=None):
        """Load displacement as vector field in SampleData instance from .ut.

        :param sequence_list: Load selected only displacement field in output
            sequence (see .ut file, output metadata). Defaults to None (all
            output sequence elements are loaded)
        :type sequence_list: list[int], optional
        :param storage_group: HDF5 group to store the displacement field
            arrays in SampleData group. Defaults to None: in this case they
            are stored in the data_inputmesh group.
        :type storage_group: str, optional
        :param field_basename: Basename to store the displacement fields in the
            SampleData instance. Defaults to 'U' (for a sequence: U_1, U_2....)
        :type field_basename: str, optional
        :param storage_mesh: SampleData Mesh group in which the fields will be
            loaded (Name, path, indexname or alias of the group).
            Defaults to None.
        :type storage_mesh: str, optional
        """
        # read the U1, U2, U3 fields
        fields_list = ['U1','U2','U3']
        Nodal_fields, _ = self.read_output_fields(ut_file, fields_list,
                                                  sequence_list)
        # define SampleData mesh group to store displacement fields
        if storage_mesh is None:
            storage_mesh = self.data_inputmesh
        # define HDF5 storage group
        if (storage_group is None) and (hasattr(self, 'data_inputmesh')):
            storage_group = self.data._name_or_node_to_path(
                                                           self.data_inputmesh)
        # load the outputs into the SampleData instance
        for i in range(len(Nodal_fields)):
            fieldname = f'{field_basename}_{i+1}'
            field_dic = Nodal_fields[i]
            if not self.data.__contains__(storage_group):
                self.data.add_group(storage_group, self.data_inputmesh,
                                    replace=False)
            U = np.zeros((len(field_dic['U1']),3),
                            dtype=field_dic['U1'].dtype)
            U[:,0] = field_dic['U1']
            U[:,1] = field_dic['U2']
            U[:,2] = field_dic['U3']
            self.data.add_field(gridname=storage_mesh, location=storage_group,
                                fieldname=fieldname, array=U, replace=False)
        return

    def load_strain_field_sequence(self, ut_file=None, sequence_list=None,
                                   storage_group=None, field_basename='E',
                                   storage_mesh=None, replace=False):
        """Load deformation as tensor field in SampleData instance from .ut.

        :param sequence_list: Load selected only displacement field in output
            sequence (see .ut file, output metadata). Defaults to None (all
            output sequence elements are loaded)
        :type sequence_list: list[int], optional
        :param storage_group: HDF5 group to store the displacement field
            arrays in SampleData group. Defaults to None: in this case they
            are stored in the data_inputmesh group.
        :type storage_group: str, optional
        :param field_basename: Basename to store the displacement fields in the
            SampleData instance. Defaults to 'E' (for a sequence: E_1, E_2....)
        :type field_basename: str, optional
        :param storage_mesh: SampleData Mesh group in which the fields will be
            loaded (Name, path, indexname or alias of the group).
            Defaults to None.
        :type storage_mesh: str, optional
        """
        # read the U1, U2, U3 fields
        fields_list = ['eto11','eto22','eto33', 'eto12', 'eto23', 'eto31']
        Nodal_fields, _ = self.read_output_fields(ut_file, fields_list,
                                                  sequence_list)
        # define SampleData mesh group to store displacement fields
        if storage_mesh is None:
            storage_mesh = self.data_inputmesh
        # define HDF5 storage group
        if (storage_group is None) and (hasattr(self, 'data_inputmesh')):
            storage_group = self.data._name_or_node_to_path(
                                                           self.data_inputmesh)
        # load the outputs into the SampleData instance
        for i in range(len(Nodal_fields)):
            fieldname = f'{field_basename}_{i+1}'
            field_dic = Nodal_fields[i]
            if not self.data.__contains__(storage_group):
                self.data.add_group(storage_group, self.data_inputmesh,
                                    replace=False)
            E = np.zeros((len(field_dic['eto11']),6),
                            dtype=field_dic['eto11'].dtype)
            E[:,0] = field_dic['eto11']
            E[:,1] = field_dic['eto22']
            E[:,2] = field_dic['eto33']
            E[:,3] = field_dic['eto12']
            E[:,4] = field_dic['eto23']
            E[:,5] = field_dic['eto31']
            self.data.add_field(gridname=storage_mesh, location=storage_group,
                                fieldname=fieldname, array=E, replace=replace)
        return

    def load_stress_field_sequence(self, ut_file=None, sequence_list=None,
                                   storage_group=None, field_basename='Sig',
                                   storage_mesh=None, replace=False):
        """Load stress as tensor field in SampleData instance from .ut.

        :param sequence_list: Load selected only displacement field in output
            sequence (see .ut file, output metadata). Defaults to None (all
            output sequence elements are loaded)
        :type sequence_list: list[int], optional
        :param storage_group: HDF5 group to store the displacement field
            arrays in SampleData group. Defaults to None: in this case they
            are stored in the data_inputmesh group.
        :type storage_group: str, optional
        :param field_basename: Basename to store the displacement fields in the
            SampleData instance. Defaults to 'Sig'
            (for a sequence: Sig_1, Sig_2....)
        :type field_basename: str, optional
        :param storage_mesh: SampleData Mesh group in which the fields will be
            loaded (Name, path, indexname or alias of the group).
            Defaults to None.
        :type storage_mesh: str, optional
        """
        # read the U1, U2, U3 fields
        fields_list = ['sig11','sig22','sig33', 'sig12', 'sig23', 'sig31']
        Nodal_fields, _ = self.read_output_fields(ut_file, fields_list,
                                                  sequence_list)
        # define SampleData mesh group to store displacement fields
        if storage_mesh is None:
            storage_mesh = self.data_inputmesh
        # define HDF5 storage group
        if (storage_group is None) and (hasattr(self, 'data_inputmesh')):
            storage_group = self.data._name_or_node_to_path(
                                                           self.data_inputmesh)
        # load the outputs into the SampleData instance
        for i in range(len(Nodal_fields)):
            fieldname = f'{field_basename}_{i+1}'
            field_dic = Nodal_fields[i]
            if not self.data.__contains__(storage_group):
                self.data.add_group(storage_group, self.data_inputmesh,
                                    replace=False)
            S = np.zeros((len(field_dic['sig11']),6),
                            dtype=field_dic['sig11'].dtype)
            S[:,0] = field_dic['sig11']
            S[:,1] = field_dic['sig22']
            S[:,2] = field_dic['sig33']
            S[:,3] = field_dic['sig12']
            S[:,4] = field_dic['sig23']
            S[:,5] = field_dic['sig31']
            self.data.add_field(gridname=storage_mesh, location=storage_group,
                                fieldname=fieldname, array=S, replace=replace)
        return

    def load_output_fields(self, ut_file=None, field_list=None,
                           sequence_list=None,
                           sequence_basename='output_fields',
                           is_time_sequence=True, fields_suffix='',
                           replace=False):
        """Load fields from Zset FEA output into the SampleData instance.

        :param field_list: List of variables names corresponding to the fields
            to read from the FEA output. Defaults to None: read all fields.
        :type field_list: list[str], optional
        :param sequence_list: List of output sequences to read. Defaults to None:
            read all sequences.
        :type sequence_list: list[int], optional
        :raises RuntimeError: raised if the FEA has not been runed.
        :raises ValueError: raised if fields not in FEA output are required
        :return Nodal_field_sequence: List of dict{field_name: field array}, one
            element for each sequence, for the node fields.
        :rtype: list
        :return Integ_field_sequence: List of dict{field_name: field array}, one
            element for each sequence, for the integration points fields.
        :rtype: list
        """
        # read the output fields
        Nodal_fields, Integ_fields = self.read_output_fields(ut_file,
                                                    field_list, sequence_list)
        # load the outputs into the SampleData instance
        for i in range(len(Nodal_fields)):
            if len(Nodal_fields) == 1:
                suffix = f'{fields_suffix}'
            else:
                suffix = f'{fields_suffix}_{i}'
            if is_time_sequence:
                sequence_group = sequence_basename+f'time_{i}'
            else:
                sequence_group = sequence_basename
            N_fields = self._gather_tensor_components(Nodal_fields[i])
            I_fields = self._gather_tensor_components(Integ_fields[i])
            self.load_field_sequence(N_fields, sequence_group, suffix, replace)
            self.load_field_sequence(I_fields, sequence_group, suffix, replace)
        return

    def read_output_fields(self, ut_file=None, field_list=None,
                           sequence_list=None):
        """Read node or integration point fields from Zset FEA output.

        :param ut_file: Name of the Zset output from which the fields must be
            loaded. Defaults to None: read all fields.
        :type ut_file: str, optional
        :param field_list: List of variables names corresponding to the fields
            to read from the FEA output. Defaults to None: read all fields.
        :type field_list: list[str], optional
        :param sequence_list: List of output sequences to read. Defaults to None:
            read all sequences.
        :type sequence_list: list[int], optional
        :raises RuntimeError: raised if the FEA has not been runed.
        :raises ValueError: raised if fields not in FEA output are required
        :return Nodal_field_sequence: List of dict{field_name: field array}, one
            element for each sequence, for the node fields.
        :rtype: list
        :return Integ_field_sequence: List of dict{field_name: field array}, one
            element for each sequence, for the integration points fields.
        :rtype: list
        """
        if ut_file is None:
            ut_file = str(self.inp_template.with_suffix('.ut'))
        # verify FEA output presence
        if not self._check_fea_output_presence(Path(ut_file)):
            raise RuntimeError('No finite element analysis output found for'
                               f' the class inp script {self.inp_template}.'
                               ' Cannot load results.')
        # get output file
        SDZset._clean_comments(ut_file)
        # load FEA calc metadata
        self.load_FEA_metadata(ut_file)
        metadata_field_list = [*self.metadata['node']]
        metadata_field_list.extend(self.metadata['integ'])
        if field_list is None:
            field_list = metadata_field_list
        else:
            # check if required fields are in fea metadata
            if not set(field_list) <= set(metadata_field_list):
                raise ValueError(f'Inputed field list ({field_list}) is'
                                 f' not contained into FEA output field'
                                 f' list ({metadata_field_list})')
        # load nodal fields for each sequence and for each variable
        Nodal_field_sequence = []
        Integ_field_sequence = []
        for t in self.metadata['Sequence']:
            if sequence_list is not None:
                if t not in sequence_list:
                    continue
            nodal_fields_dic = {}
            integ_fields_dic = {}
            for fieldname in field_list:
                if fieldname in self.metadata['node']:
                    nodal_fields_dic[fieldname] = UR.ReadFieldFromUt(
                        fileName=ut_file, fieldname=fieldname, time=t,
                        atIntegrationPoints=False)
                elif fieldname in self.metadata['integ']:
                    integ_fields_dic[fieldname] = UR.ReadFieldFromUt(
                        fileName=ut_file, fieldname=fieldname, time=t,
                        atIntegrationPoints=True)
            Nodal_field_sequence.append(nodal_fields_dic)
            Integ_field_sequence.append(integ_fields_dic)
        return Nodal_field_sequence, Integ_field_sequence

    def write_input_mesh_to_geof(self, with_tags=True):
        """Write the input data from SampleData instance to .geof file.

        :param bool with_tags: If `True`, writes the element/node sets stored
            in SampleData instance into .geof file.
        """
        if self.input_meshfile is None:
            raise Warning('Cannot write input mesh to geof as `input_meshfile`'
                          ' Mesher attribute is `None`.')
        if self.data_inputmesh is None:
            raise Warning('Cannot write input mesh to geof as `data_inputmesh`'
                          ' Mesher attribute is `None`.')
        self.write_mesh_to_geof(filename=self.input_meshfile,
                                meshname=self.data_inputmesh,
                                with_tags=with_tags)
        return

    def write_output_mesh_to_geof(self, with_tags=True):
        """Write .geof out data from a Zset mesher to SampleData instance.

        :param bool with_tags: If `True`, writes the element/node sets stored
            in SampleData instance into .geof file.
        """
        if self.output_meshfile is None:
            raise Warning('Cannot write input mesh to geof as `output_meshfile`'
                          ' Mesher attribute is `None`.')
        if self.data_outputmesh is None:
            raise Warning('Cannot write input mesh to geof as `data_outputmesh`'
                          ' Mesher attribute is `None`.')
        self.write_mesh_to_geof(filename=self.output_meshfile,
                                meshname=self.data_outputmesh,
                                with_tags=with_tags)
        return

    def write_output_from_SDmesh(self, meshname=None, problem_name=None,
                               fields_sequences=dict(), time_sequence=None):
        """Write a Zset output from SampleData mesh and fields.

        This method writes a

        :param meshname: SampleData mesh group to write as Zset output.
            If `None` (default), uses the `data_inputmesh` class attribute.
        :type meshname: str, optional
        :param problem_name: Name of the Zset problem to write, i.e. base_name
            for the .ut, .node, .integ files.
            If `None` (default), uses the `inp_script` class attribute base_name
        :type problem_name: TYPE, optional
        :param fields_sequences: field values to write as Zset output.
        :type fields_sequences: TYPE, optional
        :param time_sequence: DESCRIPTION, defaults to None
        :type time_sequence: TYPE, optional
        :raises ValueError: DESCRIPTION
        """
        # get problem name
        if problem_name is not None:
            ut_file = Path(problem_name).absolute()
        else:
            ut_file = str(self.inp_script.with_suffix('.ut'))
        # get mesh_name
        if meshname is None:
            meshname = self.data_inputmesh
        # Create time sequence metadata
        Tseq, Nsequence = self._build_timeseq(time_sequence)
        # get fields to write
        node_fields, integ_fields = self._build_field_dict(fields_sequences,
                                                           Nsequence)
        # Write output
        self._write_Zset_output(ut_file, meshname, node_fields, integ_fields,
                                Tseq)
        return

    def write_output_from_SDimage(self, imagename=None, problem_name=None,
                                  fields_sequences=dict(), time_sequence=None):
        """Write a Zset output from SampleData mesh and fields.

        This method writes a

        :param mesh_name: SampleData mesh group to write as Zset output.
            If `None` (default), uses the `data_inputmesh` class attribute.
        :type mesh_name: str, optional
        :param problem_name: Name of the Zset problem to write, i.e. base_name
            for the .ut, .node, .integ files.
            If `None` (default), uses the `inp_script` class attribute base_name
        :type problem_name: TYPE, optional
        :param fields_sequences: field values to write as Zset output.
        :type fields_sequences: TYPE, optional
        :param time_sequence: DESCRIPTION, defaults to None
        :type time_sequence: TYPE, optional
        :raises ValueError: DESCRIPTION
        """
        # get problem name
        if problem_name is not None:
            ut_file = Path(problem_name).absolute().with_suffix('.ut')
        else:
            ut_file = self.inp_script.with_suffix('.ut')
        # get mesh_name
        if imagename is None:
            imagename = self.data_inputmesh
        # Create time sequence metadata
        Tseq, Nsequence = self._build_timeseq(time_sequence)
        # get fields to write
        node_fields, integ_fields = self._build_field_dict(fields_sequences,
                                                           Nsequence)
        # Write mesh
        self._write_SDimage_as_mesh(imagename, ut_file)
        # Write .ut file
        self._write_ut_file(ut_file, node_fields, integ_fields, Tseq)
        # Write .integ file
        if len(integ_fields) > 0:
            self._write_integ_file(ut_file, integ_fields, Tseq)
        # Write .node file
        if len(node_fields) > 0:
            self._write_node_file(ut_file, node_fields, Tseq)
        return

    def write_mesh_to_geof(self, filename=None, meshname=None, with_tags=True):
        """Write data from a SampleData isntance mesh group as a .geof file.

        :param str filename: Relative or absolute path of .geof file to write.
        :param str meshname: Name, Path, Indexname or Alias of mesh data group
            to write as a .geof file
        :param bool with_tags: If `True`, writes the element/node sets stored
            in SampleData instance into .geof file.
        """
        if self.data is None:
            raise Warning('Mesher has None SampleData instance.'
                          'Cannot write any mesh data to .geof file.')
            return
        import BasicTools.IO.GeofWriter as GW
        p = Path(filename).absolute()
        mesh = self.data.get_mesh(meshname=meshname, with_tags=with_tags,
                                  with_fields=False, as_numpy=True)
        mesh.PrepareForOutput()
        OW = GW.GeofWriter()
        OW.Open(str(p))
        OW.Write(mesh)
        OW.Close()
        return

    def write_inp_template(self, inp_filename=None):
        """Write a .inp Zset script template with current commands.

        This methods writes a .inp file containing all commands prescribed
        by this class instance, without the value of the command
        arguments specified. This template file is the one used by the
        TemplateScript class in the `self.Script` attribute, to write
        specific .inp meshers with specific command argument values.

        :param str inp_filename: Name/Path of the .inp file to write
        """
        if inp_filename is not None:
            self.set_inp_filename(inp_filename)
            self.Script.set_script_filename()
        with open(self.inp_script,'w') as f:
            f.writelines(self.inp_lines)
        return

    def write_inp(self, inp_filename=None):
        """Write a .inp Zset file current mesher commands and arg values.

        This methods writes a .inp file containing all commands and command
        argument values prescribed by this class instance. It should be
        used individually to control manuallt the content of the .inp file.
        This method is automatically called by the `run_script` method and its
        execution is thus not needed to run the Zset script.

        :param str inp_filename: Name/Path of the .inp file to write
        """
        self.write_inp_template(inp_filename)
        self.Script.createScript()
        return

    def run_inp(self, inp_filename=None, workdir=None, print_output=False):
        """Run the .inp Zset script with current commands and arguments.

        This methods writes and runs a Zset script that will execute all
        commands prescribed by the class instance. First the method
        writes the inp script template and the input .geof file from the
        SampleData instance mesh data if the `data_inputmesh` and the `data`
        attributes are set to a valid value. The results are loaded in
        the SampleData instance if the `data` attribute exists.
        """
        self.write_inp_template(inp_filename)
        if hasattr(self, 'data_inputmesh'):
            self.write_input_mesh_to_geof()
        self._check_arguments_list()
        script_output = self.Script.runScript(workdir, append_filename=True,
                                              print_output=print_output)
        return script_output

    def reinitialize_inp_commands(self):
        """Removes all added Zset commands to produce virgin script file."""
        self.inp_lines = []
        self.script_args_list = []
        self.Script.args = {}
        self.set_script_args(input_meshfile=str(self.input_meshfile))
        self.script_args_list = ['input_meshfile']
        if self.output_meshfile is not None:
            self.set_script_args(output_meshfile=str(self.output_meshfile))
            self.script_args_list.append('output_meshfile')
        self._current_position = 0
        self._init_script_content()


    def _init_script_content(self):
        """Create mesher minimal text content.

            Redefine this method in each derived class dedicated to a specific
            type of Zset script with adapter **** lines (Mesher, calc....)
        """
        lines=['****return']
        self._current_position = self._add_inp_lines(lines,0) - 1
        return

    def _set_position_at_command(self, command='', after=True):
        """Set file position after/before passed Zset stared command."""
        count = 0
        for line in self.inp_lines:
            if line.strip().startswith(command.strip()):
                self._current_position = count + after*1
                return
            count += 1
        return

    def _add_inp_lines(self, lines, position):
        """Add a line to the .inp file content."""
        pos = position
        for line in lines:
            self.inp_lines.insert(pos,line.replace('\n','')+'\n')
            pos = pos+1
        return pos

    def _add_command_options(self, lines, options_dict=dict(), level='   *'):
        """Add to lines options passed as kwargs and track args list."""
        for key,value in options_dict.items():
            temp = self._find_template_string(str(value))
            if temp:
                self.script_args_list.append(temp)
            line = level+f'{key} {value}'
            if key[0:4] == 'func':
                line = line+';'
            lines.append(line)
        return lines

    def _check_fea_output_presence(self, output_file=None):
        """Verify that Zset has produced a finite element analysis output."""
        if output_file is None:
            output_file = self.inp_template.with_suffix('.ut')
        if output_file.exists():
            return True
        else:
            return False

    def _check_arguments_list(self):
        """Verify that each template argument has been provided a value."""
        args_not_set = []
        args_none = []
        for arg in self.script_args_list:
            if arg not in self.Script.args.keys():
                args_not_set.append(arg)
            elif self.Script.args[arg] is None:
                args_none.append(arg)
        if len(args_none) > 0:
            raise Warning('Script command arguments `{}` value(s) are `None`.'
                          ' Use `set_script_args` method to assign a '
                          'value if needed.'.format(args_none))
        if len(args_not_set) > 0:
            raise ValueError('Script command arguments `{}` value(s) missing.'
                             ' Use `set_script_args` method to assign a '
                             'value'.format(args_not_set))
        return

    def _clean_ut_file_comments(self):
        SDZset._clean_comments(str(self.inp_template.with_suffix('.ut')))
        return

    @staticmethod
    def _clean_comments(filename):
        new_lines = []
        with open(filename,'r') as f:
            for line in f.readlines():
                if not line.startswith('%'):
                    new_lines.append(line)
        with open(filename, 'w') as f:
            f.writelines(new_lines)
        return

    def _get_fields_to_transfer(self):
        self.fields_storage = {}
        for fieldname in self.fields_to_transfer:
            field  = self.data.get_field(fieldname, unpad_field=True)
            self.fields_storage[fieldname] = field
        return

    def _load_fields_to_transfer(self):
        for fieldname, field in self.fields_storage.items():
            self.data.add_field(gridname=self.data_outputmesh,
                                fieldname=fieldname, array=field)
        return

    def _build_timeseq(self, time_sequence):
        if time_sequence is not None:
            Nsequence = len(time_sequence)
            tseq = time_sequence
        else:
            Nsequence = 1
            tseq = 1.
        Tseq = np.ones(shape=(Nsequence, 5))
        Tseq[:,0] = np.arange(Nsequence, dtype=int) + 1.
        Tseq[:,4] = tseq
        return Tseq, Nsequence


    def _build_field_dict(self, fields_sequences, Nsequence):
        import collections
        node_fields = collections.OrderedDict()
        integ_fields = collections.OrderedDict()
        for field_Zset_name, SD_field_sequence in fields_sequences.items():
            # loop over all sampledata field datasets that constitute the
            # Zset field sequence
            seq_idx = 0
            if len(SD_field_sequence) != Nsequence:
                raise ValueError(f'Sequence for field {field_Zset_name} is a '
                                 ' list of {len(SD_field_sequence)} field data'
                                 ' but the provided time sequence has '
                                 '{Nsequence} values.')
            for SD_field in SD_field_sequence:
                # get field of the sequence (nodal or integration point)
                ftype = self.data.get_attribute('field_type', SD_field)
                field  = self.data.get_field(SD_field)
                parent_grid = self.data.get_attribute('parent_grid_path',
                                                      SD_field)
                if self.data._is_image(parent_grid) and (len(field.shape) == 3):
                    field = field.reshape((np.product(field.shape[:]),1))
                else:
                    field = field.reshape((np.product(field.shape[0:-1]),
                                           field.shape[-1]))
                # loop over all components of the field
                for k in range(field.shape[-1]):
                    vect_comp = ['1','2','3']
                    stens_comp = ['11','22','33','12','23','31']
                    tens_comp = ['11','22','33','12','21','23','32','13','31']
                    # name of field component for Zset output
                    if field.shape[-1] == 1:
                        s = f'{field_Zset_name}'
                    elif field.shape[-1] == 3:
                        s = f'{field_Zset_name}{vect_comp[k]}'
                    elif field.shape[-1] == 6:
                        s = f'{field_Zset_name}{stens_comp[k]}'
                    elif field.shape[-1] == 9:
                        s = f'{field_Zset_name}{tens_comp[k]}'
                    else:
                        s = f'{field_Zset_name}{k+1}'
                    if ftype == 'Nodal_field':
                        # initialize if needed
                        if seq_idx == 0:
                            shp = (field.shape[0],Nsequence)
                            node_fields[s] = np.empty(shp)
                        # fill value of field
                        node_fields[s][:,seq_idx] = field[:,k]
                    elif (ftype == 'Element_field') or (ftype == 'IP_field'):
                        # TODO: readapt when IP fields are handled by SampleData
                        # initialize if needed
                        if seq_idx == 0:
                            shp = (field.shape[0],Nsequence)
                            integ_fields[s] = np.empty(shp)
                        # fill value of field
                        integ_fields[s][:,seq_idx] = field[:,k]
                seq_idx += 1
        return node_fields, integ_fields

    @staticmethod
    def _gather_tensor_components(field_dict):
        # import re to use regexps for finding names of fields
        import re

        tmp_dict = field_dict.copy()
        # find out base_name of tensor fields
        Tensor_list = re.findall('\\b[a-zA-z]*\d\d\\b', str(field_dict.keys()))
        for i in range(len(Tensor_list)):
            Tensor_list[i] = Tensor_list[i][:-2]
        # get a list of tensor names without duplicates
        Tensor_list = list(set(Tensor_list))

        # find out base_name of vector fields
        Vector_list = re.findall('\\b[a-zA-z]*\d\\b', str(field_dict.keys()))
        for i in range(len(Vector_list)):
            Vector_list[i] = Vector_list[i][:-1]
        # get a list of vector names without duplicates
        Vector_list = list(set(Vector_list))

        # initialize output field dictionnary
        Tens_fields = {}
        for tensor in Tensor_list:
            components = re.findall(f'\\b{tensor}\d\d\\b',
                                    str(field_dict.keys()))
            if len(components) == 6:
                # symetric tensor
                comp_shape = field_dict[tensor+'11'].shape[0]
                try:
                    Tens_fields[tensor] = np.zeros(shape=(comp_shape,6))
                    Tens_fields[tensor][:,0] = tmp_dict.pop(tensor+'11')
                    Tens_fields[tensor][:,1] = tmp_dict.pop(tensor+'22')
                    Tens_fields[tensor][:,2] = tmp_dict.pop(tensor+'33')
                    Tens_fields[tensor][:,3] = tmp_dict.pop(tensor+'12')
                    Tens_fields[tensor][:,4] = tmp_dict.pop(tensor+'23')
                    Tens_fields[tensor][:,5] = tmp_dict.pop(tensor+'31')
                except:
                    continue
            elif len(components) == 9:
                # not symetric tensor
                comp_shape = field_dict[tensor+'11'].shape[0]
                try:
                    Tens_fields[tensor] = np.zeros(shape=(comp_shape,9))
                    comp_shape = field_dict[tensor+'11'].shape[0]
                    Tens_fields[tensor] = np.zeros(shape=(comp_shape,6))
                    Tens_fields[tensor][:,0] = tmp_dict.pop(tensor+'11')
                    Tens_fields[tensor][:,1] = tmp_dict.pop(tensor+'22')
                    Tens_fields[tensor][:,2] = tmp_dict.pop(tensor+'33')
                    Tens_fields[tensor][:,3] = tmp_dict.pop(tensor+'12')
                    Tens_fields[tensor][:,4] = tmp_dict.pop(tensor+'21')
                    Tens_fields[tensor][:,5] = tmp_dict.pop(tensor+'23')
                    Tens_fields[tensor][:,6] = tmp_dict.pop(tensor+'32')
                    Tens_fields[tensor][:,7] = tmp_dict.pop(tensor+'13')
                    Tens_fields[tensor][:,8] = tmp_dict.pop(tensor+'31')
                except:
                    continue
            else:
                # not a tensor
                continue
        for vector in Vector_list:
            components = re.findall(f'\\b{vector}\d\\b',
                                    str(field_dict.keys()))
            if len(components) == 3:
                # symetric tensor
                comp_shape = field_dict[vector+'1'].shape[0]
                try:
                    Tens_fields[vector] = np.zeros(shape=(comp_shape,3))
                    Tens_fields[vector][:,0] = tmp_dict.pop(vector+'1')
                    Tens_fields[vector][:,1] = tmp_dict.pop(vector+'2')
                    Tens_fields[vector][:,2] = tmp_dict.pop(vector+'3')
                except:
                    continue
            else:
                # not a tensor
                continue
        Tens_fields.update(tmp_dict)
        return Tens_fields

    def _write_SDimage_as_mesh(self, imagename, ut_file):
        # imports
        import BasicTools.IO.GeofWriter as GW
        # write image as ConstantRectilinearmesh .geof with BasicTools
        meshfile = ut_file.with_suffix('.geof')
        imCRM = self.data.get_image(imagename)
        Wr = GW.GeofWriter()
        Wr.Open(meshfile)
        Wr.Write(imCRM)
        Wr.Close()
        # Change elements to reduced integration elements
        with open(meshfile,'r') as f:
            lines = []
            for line in f.readlines():
                lines.append(line.replace('c3d8', 'c3d8r'))
        with open(meshfile,'w') as f:
            f.writelines(lines)
        return

    def _write_Zset_output(self, ut_file, meshname, node_fields, integ_fields,
                           Tseq):
        import BasicTools.IO.UtWriter as UW
        # create and set Writer
        UtW = UW.UtWriter()
        UtW.SetFileName(ut_file)
        UtW.AttachMesh(self.data.get_mesh(meshname))
        UtW.AttachData(data_node=node_fields, data_integ=integ_fields)
        UtW.AttachSequence(Tseq)
        # Write files
        UtW.WriteFiles(writeGeof=True)
        return

    def _write_ut_file(self, ut_file, nodal_fields, integ_fields, Tseq):
        #
        meshfile = ut_file.with_suffix('.geof')
        # extract names of field to write
        nodal_field_names = ''
        for name in nodal_fields.keys():
            nodal_field_names += f'{name} '
        integ_field_names = ''
        for name in integ_fields.keys():
            integ_field_names += f'{name} '
        # prepare lines to write in .ut
        Ut_file_lines = [
        f'**meshfile {meshfile}',
        f'**node {nodal_field_names}',
        f'**integ {integ_field_names}',
         '**element'
            ]
        for i in range(Tseq.shape[0]):
            l = Tseq[i,:]
            line = f'{int(l[0])} {int(l[1])} {int(l[2])} {int(l[3])} {l[4]}'
            Ut_file_lines.append(line)
        # Write .ut file
        with open(ut_file,'w') as f:
            f.writelines(s + '\n' for s in Ut_file_lines)
        return

    def _write_node_file(self, ut_file, node_fields, Tseq):
        # dimensions of integ data vector to build
        Ntime = Tseq.shape[0]
        Nnodes = node_fields[list(node_fields.keys())[0]].shape[0]
        Nvar = len(node_fields)
        # Initialize data vector
        node_data = np.empty(shape=(Ntime*Nnodes*Nvar))
        node_names = list(node_fields.keys())
        # prepare vector to write in binary .node file
        # loop over time steps
        for t in range(Ntime):
            # loop over variables
            for i in range(len(node_names)):
                # concatenate variables as columns
                count0 = t*(Nvar*Nnodes) + i*Nnodes
                count1 = t*(Nvar*Nnodes) + (i+1)*Nnodes
                node_data[count0:count1] = node_fields[node_names[i]][:].ravel()
        # write node file
        nodefile = ut_file.with_suffix('.node')
        with open(nodefile,'w') as f:
            node_data.astype(np.float32).byteswap().tofile(f)
        return

    def _write_integ_file(self, ut_file, integ_fields, Tseq):
        # WARNING: works only for image fields, i.e. fields of hexaedral
        # elements with reduced intergration = fields of regular meshes of
        # 'c3d8r' elements
        # dimensions of integ data vector to build
        Ntime = Tseq.shape[0]
        Nip = integ_fields[list(integ_fields.keys())[0]].shape[0]
        Nvar = len(integ_fields)
        # Initialize data vector
        integ_data = np.empty(shape=(Ntime*Nip*Nvar))
        integ_names = list(integ_fields.keys())
        # prepare vector to write in binary .integ file
        # loop over time steps
        for t in range(Ntime):
            # loop over variables
            loc_integ_data = np.empty(shape=(Nip,Nvar))
            for i in range(len(integ_names)):
                # concatenate variables as columns
                loc_integ_data[:,i] = integ_fields[integ_names[i]][:].ravel()
            # reshape integ_data to have the values of all variables at one
            # element that are cosnecutive in the array
            integ_data[t*Nip*Nvar:(t+1)*Nip*Nvar] = (
                                           loc_integ_data.reshape((Nip*Nvar,)))
        # write integ file
        integfile = ut_file.with_suffix('.integ')
        with open(integfile,'w') as f:
            integ_data.astype(np.float32).byteswap().tofile(f)
        return


    def _add_templates_to_args(self, input_list):
        """Search input_list values for templates and add them to script args.
        """
        for item in input_list:
            temp = self._find_template_string(str(item))
            if temp:
                self.script_args_list.append(temp)
        return

    @staticmethod
    def _find_template_string(string):
        if (string.find('${') > -1):
            return string[string.find('${')+2:string.find('}')]
        else:
            return 0
