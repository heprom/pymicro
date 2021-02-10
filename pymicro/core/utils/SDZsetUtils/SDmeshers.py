#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SD meshers module to allow mesher tools & SampleData instances interactions


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
class SDZsetMesher():
    """A Class to use Zset meshers to process SampleData meshes.

    This class is an inteface between the SampleData data platform class and
    the Zset software (developed at Centre des Matériaux, Mines Paris)
    meshing tools. It is designed to use Zset meshers to process data stored in
    SampleData Mesh groups, and store the results in the same SampleData
    instance/file.

    The use of this class methods requires a Zset compatible environnement.
    That means that the Zset software must be available in your PATH
    environnement variable at least.

    .. rubric:: PASSING MESHER COMMANDS OPTIONS AND ARGUMENTS
            
    Some class methods are directly bound to some Zset mesher commands. In
    practice, they write the commands into the class instance mesher, and
    handle its argument values through string templates. They can accept
    Zset command options as method keyword arguments. Each command will be
    added to the mesher as a line of the form:
        *command keyword['command']
            
    The value of the keyword argument (keyword['command']) must be a string
    that may contain a string template (an expression between ${..}). In
    this case, the template is automatically detected and handled by the
    ScriptTemplate attribute of the Mesher. The value of the template
    expression can be prescribed with the `set_mesher_args` method.
    
    No additional documentation on the command options is provided here.
    Details are available in the Zset user manual (accessible through
    the shell command line `Zman user`).   

    .. rubric:: CONSTRUCTOR PARAMETERS

    :data: SampleData object
        SampleData object containing the input data for the meshing operations.
        If `None`, the class will open a SampleData instance from a datafile
        path.
    :sd_datafile: str, optional
        Path of the hdf5/xdmf data file to open as the SampleData instance
        containing mesh tools input data.
    :verbose: `bool`, optional (False)
        verbosity flag
    """

    # BASIC IDEA
    #   the class fills mesher_lines with the meshing operations to realize,
    #   then writes the inp tmp file, write the .geof inptu file,
    #   run the mesher with Zset, and store data from output.geof (intelligent
    #   loading --> do not load mesh a second time)
    #
    #   the mesher commands are written as string lines in the attribute
    #   mesher_lines, with add_line. They are stored somewhere as string
    #   templates. Fill templates values here or with ScriptTemplate ?
    #
    #   First glance --> best with ScriptTemplate => need to store args dict
    #   as a class attribute (self.mesher_args ?). Think of a mechanisms to
    #   launch parametric studies
    #
    # TODO : save mesh fields before applying mesh operations, and adding
    #        fields to output_mesh when input and output mesh are equal

    def __init__(self, data=None, sd_datafile=None, inp_filename=None,
                 inputmesh=None, input_meshfile=None, outputmesh=None,
                 output_meshfile=None, verbose=False, autodelete=False):
        """SDZsetMesher class constructor."""
        # Initialize SampleData instance
        self.set_data(data=data, datafile=sd_datafile)
        # Set default values for mesher input output data
        self.input_meshfile = ( Path('.').absolute() / 'input.geof')
        self.output_meshfile = ( Path('.').absolute() / 'output.geof')
        self.inp_mesher = ( Path('.').absolute() / 'mesher.inp')
        # Set inputed values if passed for mesher input output data
        self.set_inputmesh(meshname=inputmesh, meshfilename=input_meshfile)
        self.set_outputmesh(meshname=outputmesh, meshfilename=output_meshfile)
        self.set_inp_filename(filename=inp_filename)
        # Init mesher content
        self.mesher_lines = []
        self.Script = ScriptTemplate(template_file=self.inp_mesher,
                                     script_command='Zrun', 
                                     autodelete=autodelete)
        self.mesher_args_list = ['input_meshfile','output_meshfile']
        self.set_mesher_args(input_meshfile=str(self.input_meshfile))
        self.set_mesher_args(output_meshfile=str(self.output_meshfile))
        self._current_position = 0
        self._init_mesher_content()
        # Other flags
        self._verbose = verbose
        self.autodelete = autodelete
        return

    def __del__(self):
        """SDZsetMesher destructor."""
        if self.data is not None and self._del_data:
            del self.data
        if self.autodelete:
            self.clean_output_files()
            self.clean_mesher_files()
            if hasattr(self, 'data_inputmesh'):
                self.clean_input_mesh_file()
        return
    
    def __repr__(self):
        """String representation of the class."""
        s  = '\nSampleData--Zset Mesher Class instance: \n\n'
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
        s += '---- Mesher template filename:\n\t\t{}\n'.format(str(self.inp_mesher))
        s += '---- Mesher commands argument values:\n'
        for key,value in self.Script.args.items():
            s += '\t\t{:30} --> {} \n'.format(key,value)
        s += '---- Mesher autodelete flag:\n\t\t{}\n'.format(self.autodelete)
        return s
    
    def set_data(self, data=None, datafile=None):
        """Set the SampleData instance to use to manipulate mesh data.
        
        :param SampleData data: SampleData instance to use 
        :param str datafile: SampleData file to open as a the SampleData
            instance to use, if `data` argument is `None`. If datafile does not
            exist, the SampleData instance is created.            
        """
        self._del_data = False
        if data is None:
            if datafile is None:
                self.data = None
            else:
                self.data = SampleData(filename=datafile)
                if self.autodelete:
                    self.data.autodelete = True
                    self.del_data = True
        elif (data is not None):
            self.data = data
        return

    def set_inputmesh(self, meshname=None, meshfilename=None,
                      fields_to_transfer=None):
        """Set the inputmesh file or path into the SampleData instance.

        :param str meshname: Name, Path, Indexname or Alias of the mesh group
            to use as input mesh for the mesher in the SampleData instance.
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
            self.set_mesher_args(input_meshfile=str(self.input_meshfile))
        if fields_to_transfer is not None:
            self.fields_to_transfer = fields_to_transfer
        return
    
    def set_outputmesh(self, meshname=None, meshfilename=None):
        """Set output mesh file or data path into the SampleData instance.

        :param str meshname: Name, Path, Indexname or Alias of the mesh group
            to use as output data group to store the mesh processed by the
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
            self.set_mesher_args(output_meshfile=str(self.output_meshfile))
        return
    
    def set_inp_filename(self, filename=None):
        """Set the name of the .inp mesher file used to run the Zset mesher."""
        if filename is not None:
            p = Path(filename).absolute()
            self.inp_mesher = p.parent / f'{p.stem}.inp'
            self.Script.set_template_filename(self.inp_mesher)
        return
    
    def set_workdir(self, workdir=None):
        """Set the path of the work directory to write and run mesher."""
        self.Script.set_work_directory(workdir)
        return
    
    def set_mesher_args(self, args_dict={}, **keywords):
        """Set the value of the inputed mesher command arguments values.
        
        The values must be passed as entries of the form
        'argument_value_template_name':argument_value. They can be passed all
        at once in a dict, or one by one as keyword arguments.
        
        The template names of command arguments in the current mesher can be
        printed using the `print_mesher_template_content` method.
        
        :param dict args_dict: values of mesher command arguments. The dict
            must contain entries of the form
            'argument_value_template_name':argument_value.
        """
        self.Script.set_arguments(args_dict, **keywords)
    
    def clean_mesher_files(self, remove_template=True):
        """Remove the template and .inp script files created by the class.
        
        :param bool remove_template: If `True` (default), removes the mesher
            template file. If `False`, removes only the script file built from
            the template file.
        """
        # Remove last script file
        self.Script.clean_script_file()
        # Remove template file
        if remove_template and self.inp_mesher.exists():
            print('Removing {} ...'.format(str(self.inp_mesher)))
            os.remove(self.inp_mesher)
        return
    
    def clean_output_files(self, clean_output_mesh=True):
        """Remove all Zset output files and output .geof file if possible."""
        if clean_output_mesh:
            run(args=['Zclean',f'{self.inp_mesher.stem}_tmp'])
        if self.output_meshfile.exists():
            print('Removing {} ...'.format(str(self.output_meshfile)))
            os.remove(self.output_meshfile)
        return
    
    def clean_input_mesh_file(self):
        """Remove all Zset output files and output .geof file if possible."""
        if self.input_meshfile.exists():
            print('Removing {} ...'.format(str(self.input_meshfile)))
            os.remove(self.input_meshfile)
        return

    def print_mesher_template_content(self):
        """Print the current Zset mesher template content to write as .inp.
        
        This methods prints all commands prescribed by this mesher class
        instance, without the value of the command arguments specified, as if
        it was a .inp template file.
        """
        print('Content of the mesher template:\n-----------------------\n')
        for line in self.mesher_lines:
            print(line, end='')
        return

    def print_mesher_content(self):
        """Print the current Zset mesher content to write as .inp.
        
        This methods prints all commands prescribed by this mesher class
        instance, with the prescribed command arguments, as if it was the
        a .inp file.
        """
        print('\nContent of the mesher:\n-----------------------\n')
        for line in self.mesher_lines:
            line_tmp = Template(line)
            print(line_tmp.substitute(self.Script.args), end='')
        return
    
    def print_mesher_msg(self):
        """Print the content of the Zset .msg output log file."""
        # The .inp file template is transformed into a _tmp.inp script file
        # and hence, the .msg file has the same basename {inp}_tmp.msg
        msg_file = self.inp_mesher.parent / f'{self.inp_mesher.stem}_tmp.msg'
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
        """Write the input data from SampleData instance to .geof file.
        
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
        OW = GW.GeofWriter()
        OW.Open(str(p))
        OW.Write(mesh)
        OW.Close()
        return 
                               
    
    def write_mesher_template(self, mesher_filename=None):
        """Write a .inp Zset mesher with current mesher commands template.
        
        This methods writes a .inp mesher containing all commands prescribed
        by this mesher class instance, without the value of the command
        arguments specified. This template file is the one used by the
        TemplateScript class in the `self.Script` attribute, to write
        specific .inp meshers with specific command argument values.
        
        :param str mesher_filename: Name/Path of the .inp file to write
        """
        if mesher_filename is not None:
            self.set_inp_filename(mesher_filename)
            self.Script.set_script_filename()
        with open(self.inp_mesher,'w') as f:
            f.writelines(self.mesher_lines)
        return 

    def write_mesher(self, mesher_filename=None):
        """Write a .inp Zset mesher file current mesher commands and args.
        
        This methods writes a .inp mesher containing all commands and command
        argument values prescribed by this mesher class instance. It should be
        used individually to control manuallt the content of the .inp file. 
        This method is automatically called by the `run_mesher` method and its
        execution is thus not needed to run the Zset mesher.
        
        :param str mesher_filename: Name/Path of the .inp file to write
        """
        self.write_mesher_template(mesher_filename)
        self.Script.createScript()
        return 

    def run_mesher(self, mesher_filename=None, workdir=None,
                   print_output=False, mesh_location='/', load_sets=True):
        """Run the .inp Zset mesher with current commands and arguments.
        
        This methods writes and runs a Zset mesher that will execute all
        mesh commands prescribed by the class instance. First the method 
        writes the mesher script template and the input .geof file from the
        SampleData instance mesh data if the `data_inputmesh` attribute the
        `data` attributes are set to a valid value. The results are loaded in
        the SampleData instance if the `data_outputmesh` attribute is set.
        """
        self.write_mesher_template(mesher_filename)
        if hasattr(self, 'data_inputmesh'):
            self.write_input_mesh_to_geof()
        self._check_arguments_list()
        mesher_output = self.Script.runScript(workdir, append_filename=True,
                                              print_output=print_output)
        if hasattr(self, 'data_outputmesh'):
            self._get_fields_to_transfer()
            self.load_geof_mesh(filename=self.output_meshfile,
                                meshname=self.data_outputmesh,
                                replace=True, bin_fields_from_sets=load_sets)
            self._load_fields_to_transfer()
        return mesher_output
    
    def create_XYZ_min_max_nodesets(self, margin=None, relative_margin=0.01,
                                    Exterior_surf_set=None):
        """Create nodesets with extremal values of each XYZ coordinate.
        
        :param float margin: Define the maximal distance to the max/min value
            of nodes coordinate in each direction that is used to defined the
            nsets. For instance, The Xmin and Xmax elsets will be defined by
            x < min(Xnodes) + margin and x > max(Xnodes) - margin. If `None` is
            passed (default), the relative margin is used.
        :param float relative_margin: Same as `margin`, but defines the margin
            as a fraction of the smallest extent of the mesh in the 3
            dimensions. For isntance, if the smallest extent of the mesh is in
            the X direction, margin = relative_margin*(Xmax - Xmin)
        """
        # Get the input mesh
        if hasattr(self, 'data_inputmesh'):
            input_mesh = self.data_inputmesh
        else:
            input_mesh = 'input_mesh'
        if not self.data.__contains__(input_mesh):
            self.load_input_mesh(meshname=input_mesh, 
                                 bin_fields_from_sets=False)
        # Get nodes and find nsets bounds
        Nodes = self.data.get_mesh_nodes(meshname=input_mesh, as_numpy=True)
        if margin is None:
            margin = relative_margin*np.min(Nodes.max(0) - Nodes.min(0))
        Bmin = Nodes.min(0) + margin*np.ones(shape=(3,))
        Bmax = Nodes.max(0) - margin*np.ones(shape=(3,))
        # Add to mesh template the bounding planes nset template
        self._add_bounding_planes_nset_template()
        self.set_mesher_args(Xbmin=Bmin[0], Xbmax=Bmax[0], Ybmin=Bmin[1],
                             Ybmax=Bmax[1], Zbmin=Bmin[2], Zbmax=Bmax[2])
        if Exterior_surf_set is not None:
            boundary_nsets= 'Xmin Xmax Ymin Ymax Zmin Zmax'
            self.create_point_set(set_name=Exterior_surf_set,
                                  use_nset=boundary_nsets)
        return
    
    def create_element_set(self, set_name='my_set', **keywords):
        """Write a Zset mesher node set or boundary creation command.
        
        This method write a **elset command block in the mesher. It
        can accept as additional keyword arguments the command options for the
        **elset command (see Zset user manual and class docstring).
        
        :param str set_name: Name of the eset to create (can contain a
              string template)
        """
        # find out if the set_name is a script template element
        temp = self._find_template_string(set_name)
        if temp:
            self.mesher_args_list.append(temp)
        # creates line for mesher point set definition command base line
        lines = [f'  **elset {set_name}']   
        # Add command options
        self._add_command_options(lines, **keywords)  
        # write command in mesher
        self._current_position = self._add_mesher_lines(lines,
                                                        self._current_position)     
        return
    
    def create_boundary_node_set(self, setname='my_set',
                                 used_elsets=None,
                                 create_nodeset=False, remove_bset=False):
        """Write a **unshared_faces command to create bset from elset.
        
        If used without `used_elsets`, create a boundary set of the mesh
        exterior surface or a nodeset of the mesh exterior nodes. If some
        mesh elsets are passed into the 
        
        :param str node_setname:  Name of the nset/bset to create (can contain
              a string template)
        :param str used_elsets: List of elsets to use to construct the
            boundary.
        :param bool create_nodeset: If `True`, create a nset with the same
            name. If `False`, only a bset is created.
        """
        temp = self._find_template_string(setname)
        if temp:
            self.mesher_args_list.append(temp)
        # creates line for mesher point set definition command base line
        lines = [f'  **unshared_faces {setname}']
        if used_elsets is not None:
            temp = self._find_template_string(used_elsets)
            if temp:
                self.mesher_args_list.append(temp)
            lines.append(f'   *elsets {used_elsets}')
        # write command in mesher 
        self._current_position = self._add_mesher_lines(lines,
                                                        self._current_position)
        # add nset creation if required
        if create_nodeset:
            self.create_point_set(set_name=setname, use_bset=setname)
            if remove_bset:
                self.remove_set(bsets=setname)
        return
    
    def create_boundary_set(self, set_name='my_set', **keywords):
        """Write a Zset mesher node set or boundary creation command.
        
        This method write a **nset or **bset command block in the mesher. It
        can accept as additional keyword arguments the command options for the
        **nset and **bset commands (see Zset user manual and class docstring).
        
        :param str set_name: Name of the nset/bset to create (can contain a
              string template)
        :param bool is_bset: If `True`, create a bset definition command. If
            `False`, create a nset definition command.
        """
        # find out if the set_name is a script template element
        temp = self._find_template_string(set_name)
        if temp:
            self.mesher_args_list.append(temp)
        # creates line for mesher boundary set definition command base line
        lines = [f'  **bset {set_name}']
        # Add command options
        self._add_command_options(lines, **keywords)   
        # write command in mesher 
        self._current_position = self._add_mesher_lines(lines,
                                                        self._current_position)
        return
    
    def create_point_set(self, set_name='my_set', **keywords):
        """Write a Zset mesher node set or boundary creation command.
        
        This method write a **nset or **bset command block in the mesher. It
        can accept as additional keyword arguments the command options for the
        **nset and **bset commands (see Zset user manual and class docstring).
        
        :param str set_name: Name of the nset/bset to create (can contain a
              string template)
        :param bool is_bset: If `True`, create a bset definition command. If
            `False`, create a nset definition command.
        """
        # find out if the set_name is a script template element
        temp = self._find_template_string(set_name)
        if temp:
            self.mesher_args_list.append(temp)
        # creates line for mesher point set definition command base line
        lines = [f'  **nset {set_name}']
        # Add command options
        self._add_command_options(lines, **keywords)   
        # write command in mesher 
        self._current_position = self._add_mesher_lines(lines,
                                                        self._current_position)
        return
    
    def create_nset_intersection(self, set_name, nsets):
        """Write a Zset mesher nset intersection command.
        
        :param str set_name: Name of the intersection nset to create
        :param str nsets: String containing all the names of the nsets to
            intersect.
        """
        # find out if the set_name is a script template element
        temp = self._find_template_string(set_name)
        if temp:
            self.mesher_args_list.append(temp)
        # creates line for mesher nset intersection command base line
        lines = [ '  **nset_intersection',
                 f'   *nsets {nsets}',
                 f'   *intersection_name {set_name}']
        # write command in mesher 
        self._current_position = self._add_mesher_lines(lines,
                                                        self._current_position)
        return
    
    def create_nset_difference(self, set_name, nset1, nset2):
        """Write commands to compute the difference of two nsets.
        
        The difference nset contains the nodes in nset1 that are not in nset2.
        
        :param str set_name: Name of the intersection nset to create
        :param str nset1: First nset to use.
        :param str nset2: Second nset to use, to substract from nset1
        """
        # find out if the set_names are a script template element
        temp = self._find_template_string(set_name)
        if temp:
            self.mesher_args_list.append(temp)
        temp = self._find_template_string(nset1)
        if temp:
            self.mesher_args_list.append(temp)
        temp = self._find_template_string(nset2)
        if temp:
            self.mesher_args_list.append(temp)
        # First create intersection of the nsets
        randint = np.random.randint(0,1000)
        tmp_intersection_name = 'tmp_intersection_'+str(randint)
        nset_names = nset1+' '+nset2
        self.create_nset_intersection(tmp_intersection_name, nset_names)
        # Create the new elset from nset1
        self.create_point_set(set_name, function='1', use_nset=nset1)
        # Now substract intersection from nset1
        self.remove_nodes_from_sets(set_name, tmp_intersection_name)
        # Finally, remove tmp intersection nset
        self.remove_set(nsets=tmp_intersection_name)
        return
    
    def create_all_nodes_nset(self, set_name='All_nodes'):
        """Create a nset containing all mesh nodes."""
        self.create_point_set(set_name=set_name, function='1')
        return  
    
    def create_interior_nodes_nset(self, set_name='interior_nodes'):
        """Create a nset will all nodes except those on the mesh ext. surf."""
        self.create_all_nodes_nset(set_name)
        self.create_boundary_node_set(setname='tmp_exterior', 
                                      create_nodeset=True, remove_bset=True)
        self.remove_nodes_from_sets(set_name, 'tmp_exterior')
        self.remove_set(nsets='tmp_exterior')
        return
    
    def create_mesh_from_elsets(self, elsets):
        """Create output mesh with only input mesh elements in elsets.
        
        :param str elsets: list of elset to use a new mesh
        """
        # find out if the set_names are a script template element
        temp = self._find_template_string(elsets)
        if temp:
            self.mesher_args_list.append(temp)
        # create the reunion of elsets 
        self.create_element_set(set_name='new_mesh_all_elements',
                                add_elset=elsets)
        # create the elset to remove
        self.create_element_set(set_name='cut_from_mesh',
                                not_in_elset='new_mesh_all_elements')
        # remove elements in elset
        self.delete_element_sets(elsets='cut_from_mesh')
        # remove created tmp elset
        self.remove_set(elsets='new_mesh_all_elements')
        return
    
    def delete_element_sets(self, elsets):
        """Zset command to remove input list of element sets from mesh."""
        # find out if the sets_names are a script template element
        temp = self._find_template_string(elsets)
        if temp:
            self.mesher_args_list.append(temp)
        lines = [ f'  **delete_elset {elsets}']
        # write command in mesher 
        self._current_position = self._add_mesher_lines(lines,
                                                        self._current_position)
        return
    
    def remove_nodes_from_sets(self, target_nset, nsets_to_remove):
        """Write a Zset mesher command to remove nodes from nset.
        
        :param str target_nset: Name of the nset from which the nodes must be
            removed.
        :param str nset_to_remove: Name of the nset whose content must be
            removed from `target_nset`
        """
        # find out if the sets_names are a script template element
        temp = self._find_template_string(target_nset)
        if temp:
            self.mesher_args_list.append(temp)
        temp = self._find_template_string(nsets_to_remove)
        if temp:
            self.mesher_args_list.append(temp)
        lines = [ '  **remove_nodes_from_nset',
                 f'   *nset_name {target_nset}',
                 f'   *nsets_to_remove {nsets_to_remove}']
        # write command in mesher 
        self._current_position = self._add_mesher_lines(lines,
                                                        self._current_position)
        return
        
    
    def remove_set(self, **keywords):
        """Write a Zset mesher nset/elset/bset removal command.
        
        This method write a **remove_set command block in the mesher. It
        can accept keyword arguments the command options for the
        **remove_set  commands (see Zset user manual and class docstring). 
        """
        lines = ['  **remove_set']
        # Add command options
        self._add_command_options(lines, **keywords)   
        # write command in mesher 
        self._current_position = self._add_mesher_lines(lines,
                                                        self._current_position)
        return 

    def _init_mesher_content(self):
        """Create mesher minimal text content."""
        lines=['****mesher',
               ' ***mesh ${output_meshfile}',
               '  **open ${input_meshfile}',
               '****return']
        self._current_position = self._add_mesher_lines(lines,0) - 1
        return
    
    def _add_bounding_planes_nset_template(self):
        self.create_point_set(set_name='Xmin', function='(x < ${Xbmin})')
        self.create_point_set(set_name='Xmax', function='(x > ${Xbmax})')
        self.create_point_set(set_name='Ymin', function='(y < ${Ybmin})')
        self.create_point_set(set_name='Ymax', function='(y > ${Ybmax})')
        self.create_point_set(set_name='Zmin', function='(z < ${Zbmin})')
        self.create_point_set(set_name='Zmax', function='(z > ${Zbmax})')
        return
    
    def _add_mesher_lines(self, lines, position):
        """Add a line to the .inp mesher file content."""
        pos = position
        for line in lines:
            self.mesher_lines.insert(pos,line.replace('\n','')+'\n')
            pos = pos+1
        return pos
    
    def _add_command_options(self, lines, **keywords):
        """Add to lines options passed as kwargs and track args list.""" 
        for key,value in keywords.items():
            temp = self._find_template_string(value)
            if temp:
                self.mesher_args_list.append(temp)
            line = f'   *{key} {value}'
            if key[0:4] == 'func':
                line = line+';'
            lines.append(line)   
        return lines
    
    
    def _check_arguments_list(self):
        """Verify that each template argument has been provided a value."""
        for arg in self.mesher_args_list:
            if arg not in self.Script.args.keys():
                raise ValueError('Mesher command argument `{}` value missing.'
                                 ' Use `set_mesher_args` method to assign a '
                                 'value'.format(arg))
        return
    
    @staticmethod
    def _find_template_string(string):
        if (string.find('${') > -1):
            return string[string.find('${')+2:string.find('}')]
        else:
            return 0
        
    def _get_fields_to_transfer(self):
        self.fields_storage = {}
        for fieldname in self.fields_to_transfer:
            self.fields_storage[fieldname] = self.data.get_node(fieldname, 
                                                                as_numpy=True)
        return     
    
    def _load_fields_to_transfer(self):
        for fieldname, field in self.fields_storage.items():
            self.data.add_field(gridname=self.data_outputmesh,
                                fieldname=fieldname, array=field)
        return
            

# SD Image mesher class
class SDImageMesher():
    """Class to mesh SampleData Image fields and store in SD mesh groups.

    This class is an inteface between the SampleData data platform class and
    the mesh tools used and/or developed at Centre des Matériaux, Mines Paris,
    by Franck Nguyen.

    The use of this class methods requires a Zset compatible environnement.
    That means that the Zset software must be available in your PATH
    environnement variable at least. It also relies on the MATLAB
    software, and henceforth a MATLAB compatible envrionnement.

    .. rubric:: CONSTRUCTOR PARAMETERS

    :data: SampleData object
        SampleData object containing the input data for the meshing operations.
        If `None`, the class will open a SampleData instance from a datafile
        path.
    :sd_datafile: str, optional ('')
        Path of the hdf5/xdmf data file to open as the SampleData instance
        containing mesh tools input data.
    :verbose: `bool`, optional (False)
        verbosity flag
    """

    def __init__(self, data=None, sd_datafile='', verbose=False):
        """SDImageMesher class constructor."""
        if data is None:
            self.data = SampleData(filename=sd_datafile, verbose=verbose)
        else:
            self.data = data
        self._verbose = verbose
        return

    def multi_phase_mesher(self, multiphase_image_name='', meshname='',
                           indexname='', location='', load_surface_mesh=False,
                           bin_fields_from_sets=True, replace=False,
                           **keywords):
        """Create a conformal mesh from a multiphase image.

        A Matlab multiphase mesher is called to create a conformal mesh of a
        multiphase image: a voxelized/pixelized field of integers identifying
        the different phases of a microstructure. Then, the mesh is stored in
        the Mesher SampleData instance at the desired location with the
        desired meshname and Indexname.

        The meshing procedure involves the construction of a surface mesh that
        is conformant with the phase boundaries in the image. The space
        between boundary elements is then filled with tetrahedra to construct
        a volumic mesh. This intermediate surface mesh can be store into the
        SampleData instance if required.

        The mesher path must be correctly set in the `global_variables.py`
        file, as well as the definition and path of the Matlab command. The
        multiphase mesher is a Matlab program that has been developed by
        Franck Nguyen (Centre des Matériaux). It also relies on the Zset
        and the MeshGems-Tetra softwares.

        Mesh parameters can be passed as optional keyword arguments (see
        MeshGems - Tetra software User Manual for more details.)

        :param str multiphase_image_name: Path, Name, Index Name or Alias of
            the multiphase image field to mesh in the SampleData instance.
        :param str meshname: name used to create the Mesh group in dataset
        :param indexname: Index name used to reference the Mesh group
        :param str location: Path, Name, Index Name or Alias of the parent
            group where the Mesh group is to be created
        :param bool load_surface_mesh: If `True`, load the intermediate
            surface mesh in the dataset.
        :param bool bin_fields_from_sets: If `True`, stores all Node and
            Element Sets in mesh_object as binary fields (1 on Set, 0 else)
        :param bool replace: if `True`, overwrites pre-existing Mesh group
            with the same `meshname` to add the new mesh.
        :param float MEM: (optional) memory (in Mb) to reserve for the mesh
            construction
        :param float HGRAD: Controls the ratio between the size of elements
            close to surfaces and elements in the bulk of the mesh.
        :param float HMIN: Minimal size for the mesh edges
        :param float HMAX: Maximal size for the mesh edges
        :param float HAUSD: Hausdorff distance between the initially produced
            surface mesh and the volume mesh created by MeshGems from the
            surface mesh.
        """
        # Defaults mesh parameters
        default_params = {'MEM':50,'HGRAD':1.5,'HMIN':5,'HMAX':50, 'HAUSD':3,
                          'ANG':50}
        # Local Imports
        from pymicro.core.utils.SDUtilsGlobals import MATLAB, MATLAB_OPTS
        from pymicro.core.utils.SDUtilsGlobals import (MESHER_TEMPLATE,
                                                       MESHER_TMP)
        # get data and output pathes
        DATA_PATH = self.data._name_or_node_to_path(multiphase_image_name)
        DATA_DIR, _ = os.path.split(self.data.h5_path)
        OUT_DIR = os.path.join(DATA_DIR, 'Tmp/')
        # create temp directory for mesh files
        if not os.path.exists(OUT_DIR): os.mkdir(OUT_DIR)
        # perpare mesher script
        mesher = ScriptTemplate(template_file=MESHER_TEMPLATE,
                                script_file = MESHER_TMP, autodelete=True,
                                script_command=MATLAB)
        # set command line options
        matlab_command = '"'+"run('" + MESHER_TMP + "');exit;"+'"'
        mesher.set_script_command_options([MATLAB_OPTS, matlab_command])
        # set mesher script parameters
        mesher_arguments = {'DATA_PATH':DATA_PATH,'OUT_DIR':OUT_DIR,
                            'DATA_H5FILE': self.data.h5_path}
        mesher_arguments.update(default_params)
        mesher.set_arguments(mesher_arguments, **keywords)
        mesher.createScript(filename=MESHER_TMP)
        # launch mesher
        CWD = os.getcwd()
        self.data.sync() # flushes H5 dataset
        mesher.runScript(workdir=OUT_DIR, append_filename=False)
        os.chdir(CWD)
        # Add mesh to SD instance
        out_file = os.path.join(OUT_DIR,'Tmp_mesh_vor_tetra.geof')
        self.data.add_mesh(file=out_file, meshname=meshname, replace=replace,
                           indexname=indexname, location=location,
                           bin_fields_from_sets=bin_fields_from_sets)
        # Add surface mesh if required
        if load_surface_mesh:
            out_file = os.path.join(OUT_DIR,'Tmp_mesh_vor.geof')
            self.data.add_mesh(file=out_file, meshname=meshname+'_surface',
                               location=location, replace=replace,
                               bin_fields_from_sets=bin_fields_from_sets)
        # Remove tmp mesh files
        shutil.rmtree(OUT_DIR)
        return

    def morphological_image_cleaner(self, target_image_field='',
                                    clean_fieldname='', indexname='',
                                    replace=False, **keywords):
        """Apply a morphological cleaning treatment to a multiphase image.

        A Matlab morphological cleaner is called to smooth the morphology of
        the different phases of a multiphase image: a voxelized/pixelized
        field of integers identifying the different phases of a
        microstructure.

        This cleaning treatment is typically used to improve the quality of a
        mesh produced from the multiphase image, or improved image based
        mechanical modelisation techniques results, such as FFT-based
        computational homogenization solvers.

        The cleaner path must be correctly set in the `global_variables.py`
        file, as well as the definition and path of the Matlab command. The
        multiphase cleaner is a Matlab program that has been developed by
        Franck Nguyen (Centre des Matériaux).

        :param str target_image_field: Path, Name, Index Name or Alias of
            the multiphase image field to clean in the SampleData instance
        :param str clean_fieldname: name used to add the morphologically
            cleaned field to the image group
        :param indexname: Index name used to reference the Mesh group
        :param bool replace: If `True`, overwrite any preexisting field node
            with the name `clean_fieldname` in the image group with the
            morphologically cleaned field.
        """
        # Local Imports
        from pymicro.core.utils.SDUtilsGlobals import MATLAB, MATLAB_OPTS
        from pymicro.core.utils.SDUtilsGlobals import (CLEANER_TEMPLATE,
                                                       CLEANER_TMP)
        self.data.sync()
        # get data and output pathes
        imagename = self.data._get_parent_name(target_image_field)
        DATA_DIR, _ = os.path.split(self.data.h5_path)
        DATA_PATH = self.data._name_or_node_to_path(target_image_field)
        OUT_FILE = os.path.join(DATA_DIR, 'Clean_image.mat')
        # perpare cleaner script
        cleaner = ScriptTemplate(template_file=CLEANER_TEMPLATE,
                                 script_file = CLEANER_TMP, autodelete=True,
                                 script_command=MATLAB)
        # set command line options
        matlab_command = '"'+"run('" + CLEANER_TMP + "');exit;"+'"'
        cleaner.set_script_command_options([MATLAB_OPTS, matlab_command])
        # set mesher script parameters and create script file
        cleaner_arguments = {'DATA_PATH':DATA_PATH,'OUT_FILE':OUT_FILE,
                            'DATA_H5FILE': self.data.h5_path}
        cleaner.set_arguments(cleaner_arguments)
        cleaner.createScript(filename=CLEANER_TMP)
        # launch cleaner
        CWD = os.getcwd()
        self.data.sync() # flushes H5 dataset
        cleaner.runScript(append_filename=False)
        os.chdir(CWD)
        # Add image to SD instance
        from scipy.io import loadmat
        mat_dic = loadmat(OUT_FILE)
        image = mat_dic['mat3D_clean']
        self.data.add_field(gridname=imagename, fieldname=clean_fieldname,
                            array=image, replace=replace, **keywords)
        # Remove tmp mesh files
        os.remove(OUT_FILE)
        return
