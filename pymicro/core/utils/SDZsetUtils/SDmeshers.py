#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SD meshers module to allow mesher tools & SampleData instances interactions


"""

## Imports
import os
import shutil
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
    #   mesher find external surfaces
    # TODO DataLoaders
    #   write geof from data, load geof in data
    # TODO safety
    #   keep a mesher arguments list. Check if args contains all elements in
    #   mesher argument list.

    def __init__(self, data=None, sd_datafile=None, inp_filename=None,
                 inputmesh=None, input_meshfile=None, outputmesh=None,
                 output_meshfile=None, verbose=False):
        """SDZsetMesher class constructor."""
        # Initialize SampleData instance
        if data is None:
            if sd_datafile is None:
                self.data =None
            else:
                self.data = SampleData(filename=sd_datafile, verbose=verbose)
        elif (data is not None):
            self.data = data
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
                                     autodelete=True)
        self.set_mesher_args(input_meshfile=str(self.input_meshfile))
        self.set_mesher_args(output_meshfile=str(self.output_meshfile))
        self._current_position = 0
        self._init_mesher_content()
        # Other flags
        self._verbose = verbose
        return

    def __del__(self):
        """SDZsetMesher destructor."""
        if hasattr(self, 'data'):
            del self.data
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

    def print_mesher_template_content(self):
        """Print the current Zset mesher template content to write as .inp.
        
        This methods prints all commands prescribed by this mesher class
        instance, without the value of the command arguments specified, as if
        it was a .inp template file.
        """
        print('Content of the mesher:\n-----------------------\n\n')
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
                   print_output=False):
        """Run the .inp Zset mesher with current commands and arguments.
        
        This methods writes and runs a Zset mesher that will execute all
        mesh commands prescribed by the class instance.
        """
        self.write_mesher_template(mesher_filename)
        mesher_output = self.Script.runScript(workdir, append_filename=True,
                                              print_output=print_output)
        return mesher_output

    def _init_mesher_content(self):
        """Create mesher minimal text content."""
        lines=['****mesher',
               ' ***mesh ${output_meshfile}',
               '  **open ${input_meshfile}',
               '****return']
        self._add_mesher_lines(lines,0)
        self._current_position = 4
        return
    
    def _add_mesher_lines(self, lines, position):
        """Add a line to the .inp mesher file content."""
        pos = position
        for line in lines:
            self.mesher_lines.insert(pos,line.replace('\n','')+'\n')
            pos = pos+1
        return pos

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
