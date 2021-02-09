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
    # TODO Meshers
    #   mesher 

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
        if data is None:
            if datafile is None:
                self.data = None
            else:
                self.data = SampleData(filename=datafile)
        elif (data is not None):
            self.data = data
        return

    def set_inputmesh(self, meshname=None, meshfilename=None):
        """Set the inputmesh file or path into the SampleData instance.

        :param str meshname: Name, Path, Indexname or Alias of the mesh group
            to use as input mesh for the mesher in the SampleData instance.
        :param str meshfilename: Name of the mesh file to use as input mesh
            for the mesher. If `meshname` is `None`, the file must exist and be
            a valid .geof meshfile for Zset. If `meshname` is not `None`, then
            the mesh data refered by `meshname` in the SampleData instance will
            be written as a .geof mesh file `meshfilename.geof`
        """
        if meshname is not None:
            self.data_inputmesh = meshname
        if meshfilename is not None:
            p = Path(meshfilename).absolute()
            self.input_meshfile = p.parent / f'{p.stem}.geof'
            self.set_mesher_args(input_meshfile=str(self.input_meshfile))
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
            self.load_geof_mesh(filename=self.output_meshfile,
                                meshname=self.data_outputmesh,
                                replace=True, bin_fields_from_sets=load_sets)
        return mesher_output
    
    def create_XYZ_min_max_nodesets(self, margin=None, relative_margin=0.01):
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
        return
    
    def create_nodeset_with_function(self, nset_name='my_nset', function='1;',
                                     nset_template= None, func_template=False):
        """Add a command to create a nodeset with a function to the mesher.
        
        :param str nset_name: Name of the nodeset to create 
        :param str function: Function used to construct the node set, for
            instance `(x > 0.5)` (see Zset user manual for more details)
        :param str nset_template: If not `None`, this string is used as a
            nset name template in the mesher. Its value has to be set with the
            `set_mesher_args` method. The string must contain a template 
            motif of the form ${template_str} 
        :param str func_template: If not `None`, this string is used as a
            function template in the mesher. Its value has to be set with the
            `set_mesher_args` method. The string must contain a template 
            motif of the form ${template_str} 
        """
        if nset_template is not None:
            nset_name = nset_template
            temp = self._find_template_string(nset_template)
            self.mesher_args_list.append(temp)
        if func_template:
            function = func_template
            temp = self._find_template_string(func_template)
            self.mesher_args_list.append(temp)
        lines = [f'  **nset {nset_name}',
                 f'   *function {function};'] 
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
        self.create_nodeset_with_function('Xmin', 
                                          func_template='(x < ${Xbmin})')
        self.create_nodeset_with_function('Xmax', 
                                          func_template='(x > ${Xbmax})')
        self.create_nodeset_with_function('Ymin', 
                                          func_template='(y < ${Ybmin})')
        self.create_nodeset_with_function('Ymax', 
                                          func_template='(y > ${Ybmax})')
        self.create_nodeset_with_function('Zmin', 
                                          func_template='(z < ${Zbmin})')
        self.create_nodeset_with_function('Zmax', 
                                          func_template='(z > ${Zbmax})')
        return
    
    def _add_mesher_lines(self, lines, position):
        """Add a line to the .inp mesher file content."""
        pos = position
        for line in lines:
            self.mesher_lines.insert(pos,line.replace('\n','')+'\n')
            pos = pos+1
        return pos
    
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
        return string[string.find('${')+2:string.find('}')]

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
