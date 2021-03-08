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
            refered by `meshname` in the SampleData instance will be written
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
            self.clean_mesher_files()
            if hasattr(self, 'data_inputmesh'):
                self.clean_input_mesh_file()
        return
    
    def __repr__(self):
        """String representation of the class."""
        s  = '\n {} Class instance: \n\n'.format(self.__name__)
        if self.data is not None:
            s += ('---- Associated SampleData dataset:\n\t\t{}\n'
                  ''.format(self.data.h5_file))
            s += ('\t\tdataset name:\t{}\n'
                  ''.format(self.data.get_sample_name()))
            s += ('\t\tdataset description:\t{}\n'
                  ''.format(self.data.get_description()))
            if hasattr(self, 'data_inputmesh'):
                s += ('---- Input Mesh data path:\n\t\t{}\n'
                      ''.format(self.data_inputmesh))
            if hasattr(self, 'data_outputmesh'):
                s += ('---- Output Mesh data path:\n\t\t{}\n'
                      ''.format(self.data_outputmesh))
        else:
            s += '---- Associated SampleData dataset:\n\t\tNone\n'
        s += '---- Mesh input file:\n\t\t{}\n'.format(str(self.input_meshfile))
        s += '---- Mesh output file:\n\t\t{}\n'.format(str(self.output_meshfile))
        s += '---- Inp script template filename:\n\t\t{}\n'.format(str(self.inp_mesher))
        s += '---- Inp commands argument values:\n'
        for key,value in self.Script.args.items():
            s += '\t\t{:30} --> {} \n'.format(key,value)
        s += '---- Autodelete flag:\n\t\t{}\n'.format(self.autodelete)
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
            for the mesher. If `meshname` is `None`, the file must exist and be
            a valid .geof meshfile for Zset. If `meshname` is not `None`, then
            the mesh data refered by `meshname` in the SampleData instance will
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
            If `meshname` is `None`, the mesher will only output a .geof file,
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
    
    def clean_output_files(self, clean_output_mesh=True):
        """Remove all Zset output files and output .geof file if possible."""
        if clean_output_mesh:
            run(args=['Zclean',f'{self.inp_script.stem}_tmp'])
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
        for line in self.inp_lines:
            line_tmp = Template(line)
            print(line_tmp.substitute(self.Script.args), end='')
        return
    
    def print_Zset_msg(self):
        """Print the content of the Zset .msg output log file."""
        # The .inp file template is transformed into a _tmp.inp script file
        # and hence, the .msg file has the same basename {inp}_tmp.msg
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
    
    def _add_inp_lines(self, lines, position):
        """Add a line to the .inp file content."""
        pos = position
        for line in lines:
            self.inp_lines.insert(pos,line.replace('\n','')+'\n')
            pos = pos+1
        return pos
    
    def _add_command_options(self, lines, **keywords):
        """Add to lines options passed as kwargs and track args list.""" 
        for key,value in keywords.items():
            temp = self._find_template_string(value)
            if temp:
                self.script_args_list.append(temp)
            line = f'   *{key} {value}'
            if key[0:4] == 'func':
                line = line+';'
            lines.append(line)   
        return lines
    
    def _check_arguments_list(self):
        """Verify that each template argument has been provided a value."""
        for arg in self.script_args_list:
            if arg not in self.Script.args.keys():
                raise ValueError('Script command argument `{}` value missing.'
                                 ' Use `set_script_args` method to assign a '
                                 'value'.format(arg))
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
    
    @staticmethod
    def _find_template_string(string):
        if (string.find('${') > -1):
            return string[string.find('${')+2:string.find('}')]
        else:
            return 0
