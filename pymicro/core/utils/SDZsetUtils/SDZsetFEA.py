#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SD Zset Fea module for FEM soft Zset & SampleData instances interactions

"""

## Imports
import numpy as np
from pathlib import Path
from pymicro.core.utils.SDZsetUtils.SDZset import SDZset


# SD Zset Finite element analysis class
class SDZsetFEA(SDZset):
    """A Class to use Zset meshers to process SampleData meshes.

    This class is an inteface between the SampleData data platform class and
    the Zset software (developed at Centre des Matériaux, Mines Paris)
    finite element tools. It is designed to use Zset to run finite element
    analysis and/or process FEM fields from data in SampleData Mesh groups, and
    store the results in the same SampleData instance/file, in the same or
    another Mesh group. 

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
        """SDZsetFEA class constructor.
        
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
        super(SDZsetFEA, self).__init__(data, sd_datafile, inp_filename,
                                        inputmesh, outputmesh,
                                        input_meshfile, output_meshfile,
                                        verbose, autodelete, data_autodelete)
        self.set_calculation_type()
        self.set_material_file()
        return
    
    def add_linear_elastic_material_block(self, volumic_mass=None,
                                       young_modulus=None, poisson_ratio=None):
        """Add a linear elastic material behavior to the FEA.
        
        :param volumic_mass: volumic mass of the material, defaults to None
        :type volumic_mass: float, optional
        :param young_modulus: Young modulus of the material, defaults to None
        :type young_modulus: float, optional
        :param poisson_ratio: Poisson's ratio of the material, defaults to None
        :type poisson_ratio: float, optional
        """
        self._set_position_at_command(command='****return')
        lines=[' ***behavior linear_elastic',
               '  **coefficient',
               '    masvol ${masvol}',
               ' **elasticity isotropic',
               '   young ${young}',
               '   poisson ${poisson}',
               ' ***return']
        self.set_script_args(masvol=volumic_mass, young=young_modulus,
                             poisson=poisson_ratio)
        self.script_args_list.extend(['masvol', 'young','poisson'])
        self._add_inp_lines(lines, self._current_position)
        return
    
    def add_resolution(self, resolution_type='newton', use_lumped_mass=False):
        """Add a ***resolution procedure to the inp script.
        """
        # add resolution procedure
        self._set_position_at_command(command='****return', after=False)
        lines=[' ***resolution ${resolution_type}']
        self.script_args_list.append('resolution_type')
        self.set_script_args(resolution_type=resolution_type)
        if use_lumped_mass:
            lines.append('  **use_lumped_mass')
        self._current_position = self._add_inp_lines(lines,
                                                     self._current_position)
        return
    
    def add_sequence_or_cycle(self, N_sequences=1, sequence_options=dict(),
                              is_cycle=False):
        """Add a sequence command to ***resolution procedure.
        
        :param N_sequences: Number of sequences, defaults to 1
        :type N_sequences: int, optional
        :param sequence_options: Dictionary of '*' options to add to the
            **sequence command block in .inp script --> `*dict_key dict_entry` 
            Default is an empty dict.
        :type sequence_options: TYPE, optional
        :param is_cycle: if `True`, add a **cycle command. If `False`, add a
            **sequence command.
        """
        # add the **sequence command after the ***resolution procedure
        self._set_position_at_command(command='***resolution')
        # create lines
        if is_cycle:
            lines=['  **cycle ${N_sequences}']
        else:
            lines=['  **sequence ${N_sequences}']
        # add script arguments
        self.script_args_list.append('N_sequences')
        self.set_script_args(N_sequences=N_sequences)
        # add command options
        self._add_command_options(lines, sequence_options)
        self._current_position = self._add_inp_lines(lines, 
                                                     self._current_position)
        return
    
    def set_skip_cycle(self, skip_type='', skip_options=dict()):
        """Add skip_cycle command to ***resolution procedure.
        
        :param skip_type: type of extrapolation to speed up cyclic calculations
            Defaults to ''
        :type skip_type: str, optional
        :param sequence_options: Dictionary of '*' options to add to the
            **skip_cycle command block in .inp script --> `*dict_key dict_entry` 
            Default is an empty dict.
        :type sequence_options: dict, optional
        """
        # add resolution procedure
        self._set_position_at_command(command='****return')
        lines=[' ***skip_cycle {skip_type}']
        self.script_args_list.append('skip_type')
        # add command options
        self._add_command_options(lines, skip_options)
        self._current_position = self._add_inp_lines(lines, 
                                                     self._current_position)
        return
    
    def add_init_d_dof(self, ratio=None, sequence=None):
        """Add proportional initialization of dofs command to ***resolution.
        
        :param ratio:  scaling factor for proportional initialization. 
            Defaults to None (Zset default is 1)
        :type ratio: float, optional
        :param sequence: If `True`, activate procedure only within each
            sequence. Defaults to None
        :type sequence: bool, optional
        """
        # add the **sequence command after the ***resolution procedure
        self._set_position_at_command(command='***resolution')
        # create command line
        c_line = '  **init_d_dof '
        if ratio is not None:
            c_line = c_line + f' ratio {ratio}'
            self._add_templates_to_args([str(ratio)])
        if sequence:
            c_line = c_line + ' sequence'
        self._current_position = self._add_inp_lines([c_line], 
                                                     self._current_position)
        return
    
    def add_max_divergence(self, value):
        """Add max_divergence procedure to ***resolution.
        
        :param value: maximum allowable ratio of two successive convergence
            ratios
        :type value: float 
        """
        # add the **sequence command after the ***resolution procedure
        self._set_position_at_command(command='***resolution')
        # create command line
        c_line = '  **max_divergence {value} '
        self._current_position = self._add_inp_lines([c_line], 
                                                     self._current_position)
        return
    
    def add_automatic_time(self, autotime_type='standard', vars_inc=dict(),
                           global_iterations=None,  mandatory=False, 
                           autotime_options=dict()):
        """Add automatic substepping command block to ***resolution procedure.
        
        :param Autotime_type: Type of autotime, defaults to ''
        :type Autotime_type: TYPE, optional
        :param vars_inc: Dictionary whose key are variables names and
            values are the max. increments allowed for this variables for one
            increment. If the increment surpasses this value, time substepping
            is applied. Defaults to empty dict()
        :type vars_inc: dict, optional
        :param global_iterations: If provided, set a max. iterations number for
            each time increment. If more iterations are required, substepping
            is enforced. Defaults to None
        :type global_iterations: int, optional
        :param mandatory: If `True` set the mandatory option for the variables
            increment values. In this case, time substepping is applied until
            the specified variable increments are under the specified values
            Defaults to False
        :type mandatory: bool, optional
        :param autotime_options: Dictionary of '*' options to add to the
            **autotime_options command block in .inp script 
            --> `*dict_key dict_entry`. Default is an empty dict.
        :type autotime_options: dict, optional
        """
        # add the **sequence command after the ***resolution procedure
        self._set_position_at_command(command='***resolution')
        # create command line
        c_line = f'  **automatic_time {autotime_type}'
        self._add_templates_to_args([autotime_type])
        for key, value in vars_inc:
            c_line = c_line + f' {key} {value}'
            self._add_templates_to_args([str(key),str(value)])
        if global_iterations is not None:
            c_line = c_line + f' global {global_iterations}'
            self._add_templates_to_args([str(global_iterations)])
        if mandatory:
            c_line = c_line + ' mandatory'
        # create lines
        lines=[c_line]
        # add command options
        self._add_command_options(lines, autotime_options)
        self._current_position = self._add_inp_lines(lines, 
                                                     self._current_position)
        return
    
    def add_boundary_condition(self, bc_type, set_names, dof_names, values,
                               tables=None):
        """Add a boundary condition block to the Zset FEA script.
        
        The methods adds a block of the form::
            
            **bc_type
              set_names[0] dof_names[0] values[0] tables[0]
              set_names[1] dof_names[1] values[1] tables[1]
              ...          ...          ...       ...
              
        The length of the input arguments sets_names, dof_names, values,
        tables must be equal. 
        
        .. warning::
            No safety checks are conducted on the value of the arguments. Make
            sure to use existing sets, dofs, and boundary condition types, to
            avoid errors when running Zset.
        
        For more details on boundary condition blocks, see Zman user section
        3.39. 
        
        :param bc_type: type of boundary condition block to add,
            defaults to None
        :type bc_type: str
        :param set_names: list of sets (nsets, elsets or bsets) for which the
            boundary conditions are prescribed.
        :type sets_names: list[str] 
        :param dof_names:  list of degrees of fredom for which the
            boundary conditions are prescribed for the sets in `set_names`.
        :type dof_names: list[str] 
        :param values: list of the values prescribed for the dof specified in
            `dof_names`.
        :type values: list[str] 
        :param tables: list of the loading tables associated to the values
            prescribed in `values`. Defaults to None
        :type tables: list[str], optional
        """
        # safety checks on input arguments
        if ((len(set_names) != len(dof_names))
            or (len(set_names) != len(values))
            or (len(dof_names) != len(values))):
                raise ValueError('`set_names`, `dof_names` and `values` must'
                                 ' have the same lengths.')
        if tables is None:
            # if tables is None, create a list of empty strings of the right
            # size
            tables = []
            for i in range(len(set_names)):
                tables.append('')
        else:
            if len(tables) != len(set_names):
                raise ValueError('`set_names`, `dof_names`, `values` and'
                                 ' `tables` must have the same lengths.')
        # set position in ***bc block
        self._set_position_at_command(command='***bc')
        # create the boundary condition block
        lines = [f'  **{bc_type}']
        for i in range(len(set_names)):
            # find any eventual string template
            self._add_templates_to_args([set_names[i], dof_names[i], values[i],
                                         tables[i]])
            line = '    {:25} {:8} {:10} {:30}'.format(set_names[i],
                                            dof_names[i], values[i], tables[i])
            lines.append(line)
        # add boundary condition block to ***bc block
        self._add_inp_lines(lines, self._current_position)
        return       
    
    def add_submodel(self, global_problem, dofs, driven_nsets,
                     sub_format=None):
        """Add a submodel boundary condition for the FEM calculation.
        
        This boundary condition imposes degrees of freedom located at boundary
        nodes in a submodel to values computed from a larger problem, already
        solved (master problem).
        
        :param global_problem: .ut file of the Zset master problem 
        :type global_problem: str
        :param dofs: List of the names of the degree of fredom to drive on the
            submodel with their values computed on the master problem.
        :type dofs: list[str]
        :param driven_nsets: name of a nodeset of the submodel whose nodes are
            all driven by the master problem values, for the specified degrees
            of freedom
        :type driven_nsets: TYPE
        :param sub_format: DESCRIPTION, defaults to None
        :type sub_format: TYPE, optional
        """
        # add the **sequence command after the ***resolution procedure
        self._set_position_at_command(command='***bc')
        # create command line
        lines = ['  **submodel']
        if sub_format is not None:
            lines.append(f'   *format {sub_format}')
            self._add_templates_to_args([str(sub_format)])
        lines.append(f'   *global_problem {global_problem}')
        line = '   *dofs '
        for item in dofs:
            line = line + ' ' + item + ' '
        lines.append(line)
        lines.append(f'   *driven_nsets {driven_nsets}')
        # add command options
        self._current_position = self._add_inp_lines(lines, 
                                                     self._current_position)
        return
    
    def add_post_processing(self, PostProcessing):
        """Add a post processing block to the inp script.
        
        :param SDZsetPostProcessing: Class instance handling the post
            processing block to add to the inp file.
        :type SDZsetPostProcessing: SDZsetPostProcessing
        """
        if not isinstance(PostProcessing, SDZsetPostProcessing):
            raise Warning('Input is not a SDZsetPostProcessing class instance.'
                          ' Cannot add the post processing block.')
            return
        # add post processing block at the end of inp script
        pos = len(self.inp_lines)
        self._add_inp_lines(PostProcessing.inp_lines, pos)
        # update script arguments 
        self.Script.args.update(PostProcessing.Script.args)
        # update script arguments lists
        self.script_args_list.extend(PostProcessing.script_args_list)
        return
    
    def set_fixed_set_bc(self, nset_list=None, bset_list=None):
        """Add a U=0 boundary condition for the inputed nsets or bsets.
        
        :param nset_list: List of nsets to which the BC applies, defaults to None
        :type nset_list: list[str], optional
        :param bset_list: List of bsets to which the BC applies, defaults to None
        :type bset_list: list[str], optional
       """
        if nset_list is not None:
            set_names = []
            dof_names = []
            values = []
            for item in nset_list:
                set_names.append(item)
                set_names.append(item)
                set_names.append(item)
                dof_names.append('U1')
                dof_names.append('U2')
                dof_names.append('U3')
                values.append('0.')
                values.append('0.')
                values.append('0.')
            self.add_boundary_condition('impose_nodal_dof', set_names,
                                        dof_names, values)
        if bset_list is not None:
            set_names = []
            dof_names = []
            values = []
            for item in nset_list:
                set_names.append(item)
                set_names.append(item)
                set_names.append(item)
                dof_names.append('U1')
                dof_names.append('U2')
                dof_names.append('U3')
                values.append('0.')
                values.append('0.')
                values.append('0.')
            self.add_boundary_condition('impose nodal dof density', set_names,
                                        dof_names, values)
        return

    def set_eigenmode_lanczos_calculation(self, N_modes, F_max, Niter='',
                                          Nb_sub=''):
        """Add a computation of eigenmodes with lanczos method.
        
        :param N_modes: Number of eigen frequencies to compute
        :type N_modes: int or str
        :param F_max: Maximal value for the eigen frequencies to compute
        :type F_max: float or str
        :param Niter:  number of iteration for the eigen value search by QR
            decomposition. Defaults to ''
        :type Niter: int, optional
        :param Nb_sub:  size of the Krylov subspace. Default is 2nb_freq if
            nb_freq< 8, else it is 2nb_freq+8. Defaults to ''
        :type Nb_sub: int, optional
        """
        # set calculation type to eigen modes
        self.set_calculation_type('eigen')
        # add eigen block at the end of ****calc block
        self._set_position_at_command('****return', after=False)
        # add eigen block
        lines = [ ' ***eigen lanczos',
                 f'    {N_modes} {F_max} {Niter} {Nb_sub}']
        self._add_templates_to_args([N_modes, F_max, Niter, Nb_sub])
        self._add_inp_lines(lines, self._current_position)
        return        

    def set_calculation_type(self, calculation_type=None):
        """Set the finite element calculation type.
        
        :param calculation_type: one of the following 'mechanical', 'eigen',
            'dynamic', 'explicit_mechanical', 'thermal_steady_state',
            'thermal_transient', 'diffusion', 'weak_coupling'. If `None`, the
            calculation will be run with the 'mechanical' type (see Zman user).
            Defaults to `None`.
        :type calculation_type: str, optional
        """
        types = ['mechanical', 'eigen', 'dynamic', 'explicit_mechanical',
                 'thermal_steady_state', 'thermal_transient', 'diffusion',
                 'weak_coupling', None]
        if calculation_type not in types:
            raise ValueError("Calculation type must one of the following:\n"
                             "'mechanical', 'eigen', 'dynamic',"
                             "'explicit_mechanical', 'thermal_steady_state',"
                             "'thermal_transient', 'diffusion',"
                             " 'weak_coupling', `None` . Got {}"
                             "".format(calculation_type))
        self.set_script_args(calculation_type=calculation_type)
        
    def set_material_file(self, material_file=None):
        """Set name of file containing Zset material behavior block.
        
        :param material_file: name of the file. If `None` (default), the
            material block is written in the same file as the script, in the
            class inp file.
        """
        if material_file is None:
            mat_file = f'{self.inp_script.stem}_tmp.inp'
            self.set_script_args(material_file=mat_file)
        else:
            mat_file = Path(material_file).name
            self.set_script_args(material_file=mat_file)
        return
            
    def load_eigenmodes(self, store_group='eigenmodes'):
        """Load eigenmodes into the SampleData outputmesh group.
        
        :param store_group: Name of the HDF5 group in which to store the
            eigenmode fields in the SampleData instance.
            Defaults to 'eigenmodes'.
        :type store_group: str, optional
        """
        # read the U1, U2, U3 fields
        fields_list = ['U1','U2','U3']
        Node_fields, _ = self.read_output_fields(field_list=fields_list)
        # create a group to store the eigen modes
        if not self.data.__contains__(store_group):
            self.data.add_group(store_group, self.data_inputmesh,
                                replace=False)
        # add the modes vector fields in teh group and the input mesh group
        count = 1
        for field_dic in Node_fields:
            Mode = np.zeros((len(field_dic['U1']),3),
                            dtype=field_dic['U1'].dtype)
            Mode[:,0] = field_dic['U1']
            Mode[:,1] = field_dic['U2']
            Mode[:,2] = field_dic['U3']
            Mode_str = f'eigenmode_{count}'
            self.data.add_field(gridname=self.data_inputmesh, 
                                location=store_group, fieldname=Mode_str, 
                                array=Mode, replace=True)
            count += 1
        return

    def _init_script_content(self):
        """Create mesher minimal text content."""
        lines=['****calcul ${calculation_type}',
               ' ***mesh',
               '  **file ${input_meshfile}',
               ' ***material',
               '   *file ${material_file}',
               ' ***bc',
               '****return']
        self.script_args_list.append('calculation_type')
        self.script_args_list.append('input_meshfile')
        self.script_args_list.append('material_file')
        self._current_position = self._add_inp_lines(lines,0) - 1
        return
 

# SD Zset Field transfer class
class SDZsetFieldsTransfer(SDZset):
    """Class to use Zset field transfer operators with Sampledata fields.

    This class is an inteface between the SampleData data platform class and
    the Zset software (developed at Centre des Matériaux, Mines Paris)
    finite element tools. It is designed to use Zset to run finite element
    analysis and/or process FEM fields from data in SampleData Mesh groups, and
    store the results in the same SampleData instance/file, in the same or
    another Mesh group. 

    The use of this class methods requires a Zset compatible environnement.
    That means that the Zset software must be available in your PATH
    environnement variable at least. 
    
    """
    
    def __init__(self, problem_name, transfer_type='nodal',
                 data=None, sd_datafile=None,
                 inp_filename=Path('.').absolute() / 'transfer.inp',
                 inputmesh=None, outputmesh=None, 
                 input_meshfile=Path('.').absolute() / 'input.geof', 
                 output_meshfile=Path('.').absolute() / 'output.geof', 
                 verbose=False, autodelete=False, data_autodelete=False):
        """SDZsetFEA class constructor.
        
        :param problem_name: Base name of the Zset files where the fields to
            transfer are stored (base_name.ut or base_name.integ ....).
        :type problem name: str
        :param transfer_type: Type of field transfer to apply. 'nodal': node
            field transfer, 'integ': integration points field transfer.
            Defaults to 'nodal'
        :type transfer_type name: str, optional
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
        :param input_meshfile: Name of .geof mesh on which the fields must
            be transfered. If `inputmesh` is `None`, the file must exist and
            be a valid .geof meshfile for Zset. If `inputmesh` is not `None`,
            then the mesh data refered by `mesh_name` in the SampleData instance
            will be written as a .geof mesh file `meshfilename.geof`.
            Defaults to `./input.geof`
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
        super(SDZsetFieldsTransfer, self).__init__(data, 
            sd_datafile, inp_filename, inputmesh, outputmesh, input_meshfile,
            output_meshfile, verbose, autodelete, data_autodelete)
        self.set_problem_name(problem_name)
        self.set_transfer_type(transfer_type)
        return  
    
    def clean_output_files(self, clean_Zset_output=True):
        """Remove all Zset output files and output .geof file if possible."""
        super(SDZsetFieldsTransfer, self).clean_output_files(
                                           clean_Zset_output=clean_Zset_output)      
        from subprocess import run
        if clean_Zset_output:
            run(args=['Zclean','output'])
        return
     

    def set_problem_name(self, problem_name):
        """Set the finite element calculation type.
        
        :param problem_name: Name of base_name of the Zset .ut file where the
            fields to transfer metadata are stored.
        :type problem name: str
        """
        p = Path(problem_name).absolute()
        self.problem_name = p.parent / f'{p.stem}.ut'
        self.set_script_args(problem_name=str(self.problem_name))
        return

    def set_transfer_type(self, transfer_type='nodal'):
        """Set the finite element calculation type.
        
        :param transfer_type: Type of field transfer to apply. 'nodal': node
            field transfer, 'integ': integration points field transfer.
            Defaults to 'nodal'
        :type transfer_type name: str, optional
        """
        if transfer_type not in ['nodal', 'integ']:
            raise ValueError('The only field transfer types accepted are'
                             f'"node" or "integ", got {transfer_type}')
        self.transfer_type = transfer_type
        self.set_script_args(transfer_type=transfer_type)
        return
    
    def run_inp(self, inp_filename=None, workdir=None, print_output=False):
        """Run the .inp Zset script with current commands and arguments.
                
        This methods writes and runs a Zset script that will execute all
        commands prescribed by the class instance. First the method 
        writes the inp script template and the input .geof file from the
        SampleData instance mesh data if the `data_inputmesh` and the `data`
        attributes are set to a valid value. The results are loaded in
        the SampleData instance if the `data` attribute exists. In addition,
        this method writes if needed the problem and output mesh files, and
        verifies that the problem files containing the fields to transfer
        exists.
        """
        # check problem existence
        presence = self._check_problem()
        if not presence:
            # TODO: implement construction of .ut problem from Sampledata
            #       mesh groups
            raise RuntimeError('Cannot transfer fields from problem '
                               f'{self.problem_name}: files not found')
        # write .inp script template
        self.write_inp_template(inp_filename)
        # if needed, write the input and output meshes
        if hasattr(self, 'data_outputmesh'):
            self.write_output_mesh_to_geof()
        self._check_arguments_list()
        script_output = self.Script.runScript(workdir, append_filename=True,
                                              print_output=print_output)
        self._correct_output_meshfile_name()
        return script_output
    
    def load_transfered_displacement(self, sequence_list=None,
                                     storage_group=None, field_basename='U'):
        """Load projected displacement as vector field on target mesh group.
        
        :param ut_file: Name of the Zset output from which the fields must be
            loaded. Defaults to None: read all fields.
        :type ut_file: str, optional
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
        :type field_basename: TYPE, optional
        """
        output_file = str(self.output_meshfile.with_suffix('.ut'))
        if storage_group is None:
            storage_group = self.data_outputmesh
        self.load_displacement_field_sequence(
            ut_file=output_file, sequence_list=sequence_list,
            storage_group=storage_group, field_basename=field_basename, 
            storage_mesh=self.data_outputmesh)
        return
    
    def _correct_output_meshfile_name(self):
        with open(self.output_meshfile.with_suffix('.ut'),'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            if line.strip().startswith('**meshfile'):
                new_lines.append(f'**meshfile {self.output_meshfile}\n')
            else:
                new_lines.append(line)
        with open(self.output_meshfile.with_suffix('.ut'),'w') as f:
            f.writelines(new_lines)
        return
        
    
    def _check_problem(self):
        """Verify that Zset has produced a finite element analysis output."""
        presence = self._check_fea_output_presence(self.problem_name)
        if presence:
            self.load_FEA_metadata(ut_file=str(self.problem_name))
            meshfile = Path(self.metadata['meshFile'])
            if meshfile.exists():
                return True
            else:
                return False
        else:
            return False

    def _init_script_content(self):
        """Create mesher minimal text content."""
        lines=['****transfer_fields',
               ' ***new_mesh',
               '  **name ${output_meshfile}',
               ' ***old_mesh',
               '  **name ${problem_name}',
               ' ***${transfer_type}_values',
               '****return']
        self.script_args_list.append('output_meshfile')
        self.script_args_list.append('problem_name')
        self.script_args_list.append('transfer_type')
        self._current_position = self._add_inp_lines(lines,0) - 1
        return
 

# SD Zset Postprocessing class
class SDZsetPostProcessing(SDZset):
    """Class to use Zset postprocessing operators on Sampledata fields.

    This class is an inteface between the SampleData data platform class and
    the Zset software (developed at Centre des Matériaux, Mines Paris)
    finite element tools. It is designed to use Zset to run finite element
    analysis and/or process FEM fields from data in SampleData Mesh groups, and
    store the results in the same SampleData instance/file, in the same or
    another Mesh group. 

    The use of this class methods requires a Zset compatible environnement.
    That means that the Zset software must be available in your PATH
    environnement variable at least. 
    
    Convention::
        
        The Zset ouput data used as input for the post processing must exist,
        and the associated .geof file must as well.
                
    """
    
    def __init__(self, data=None, sd_datafile=None,
                 inp_filename=Path('.').absolute() / 'post_processing.inp',
                 inputmesh=None, outputmesh=None, 
                 input_meshfile=None, output_meshfile=None,
                 verbose=False, autodelete=False, data_autodelete=False):
        """SDZsetFEA class constructor.
        
        :type transfer_type name: str, optional
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
        :param input_meshfile: Name of .geof mesh on which the fields must
            be transfered. If `inputmesh` is `None`, the file must exist and
            be a valid .geof meshfile for Zset. If `inputmesh` is not `None`,
            then the mesh data refered by `mesh_name` in the SampleData instance
            will be written as a .geof mesh file `meshfilename.geof`.
            Defaults to `./input.geof`
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
        # init post proc block count
        self._pp_count = 0
        # parent class init
        super(SDZsetPostProcessing, self).__init__(data, 
            sd_datafile, inp_filename, inputmesh, outputmesh, input_meshfile,
            output_meshfile, verbose, autodelete, data_autodelete)
        return   

    def add_post_proc_bloc(self, precision=None, prefix=None,
                           suppress_p=False):
        """Add a post processing block to theend of the inp script file."""
        lines=['****post_processing',
               '****return']
        pos = len(self.inp_lines)
        # add lines and set position inside post processing block
        self._current_position = self._add_inp_lines(lines,pos) - 1
        self._pp_count += 1
        if precision is not None:
            self.set_precision(precision)
        if prefix is not None:
            self.set_post_file_prefix(prefix)
        if suppress_p:
            lines = [' ***suppress_p_on_post_files']
            self._current_position = self._add_inp_lines(lines, 
                                                        self._current_position)
        return
    
    def add_data_source_type(self, source_type='Z7', filename=''):
        """Add a ***data_source command to the current post_processing.
        
        :param source_type: type of external source of data to post process,
            defaults to 'Z7'
        :type source_type: str, optional
        :param filename: name of the file containing the external source,
            defaults to ''
        :type filename: str, optional
        """
        lines = [f' ***data_source {source_type}',
                 f'  **open {filename}']
        self._current_position = self._add_inp_lines(lines, 
                                                      self._current_position)
        return
    
    def add_data_output(self, output_type='Z7', pb_name='', elset_name=None):
        """Add a ***data_output command to the current post_processing.
        
        :param source_type: type of external source of data to post process,
            defaults to 'Z7'
        :type source_type: str, optional
        :param pb_name:  output file will be this name appended with a dot
            suffix of the file type. Defaults to ''
        :type pb_name: str, optional
        :param elset_name: elset to use as submesh for the output of post
            processing. Default to None
        :type elset_name: str, optional
        """
        lines = [f' ***data_output {output_type}',
                 f'  **problem_name {pb_name}',
                 f'   *elset {elset_name}']
        self._current_position = self._add_inp_lines(lines, 
                                                      self._current_position)
        return
    
    def add_local_post_processing(self, post_proc_opts=dict(),
                                  process_list=None):
        # create block
        lines = [' ***local_post_processing']
        # add options 
        lines = self._add_command_options(lines, post_proc_opts, level='  **')
        # add process blocks
        if process_list is not None:
            for process in process_list:
                lines.append(self.add_process(process_opts=process))
        # add lines
        self._current_position = self._add_inp_lines(lines, 
                                                      self._current_position)
        return
    
    def add_global_post_processing(self, post_proc_opts=dict(),
                                  process_list=None):
        # create block
        lines = [' ***global_post_processing']
        # add options 
        lines = self._add_command_options(lines, post_proc_opts, level='  **')
        # add process blocks
        if process_list is not None:
            for process in process_list:
                lines.extend(self.add_process(process_opts=process))
        # add lines
        self._current_position = self._add_inp_lines(lines, 
                                                      self._current_position)
        return
    
    def add_process(self, process_opts):
        """Add a process block to the current post processing block.
        
        :param process_opts: Dictionnary of options describing the process
            block. Must contain 'process_type'. Can containe a 'material_file'
            key to specify the material file to set for this process block.
            All other entries are treated as '*' options for the process block
            and written in it as '   *{key} value' lines.
        :type process_opts: dict
        """
        if 'process_type' in process_opts:
            p_type = process_opts.pop('process_type')
        else:
            raise ValueError('No "process_type" key in the process_opts dict.')
        if 'material_file' in process_opts:
            matfile = process_opts.pop('material_file')
            lines = [f'  **material_file {matfile}',
                     f'  **process {p_type}']
        else:
            lines = [f'  **process {p_type}']
        lines = self._add_command_options(lines, process_opts)
        return lines
    
    def add_results_management(self, add_list=None):
        """Add a ****result_management block on top of current script.
        
        :param add_opts: Dictionnary of options describing the process
            block. Must contain 'process_type'. Can containe a 'material_file'
            key to specify the material file to set for this process block.
            All other entries are treated as '*' options for the process block
            and written in it as '   *{key} value' lines.
        :type process_opts: dict
        """
        # create block
        lines = ['****result_management']
        # add process blocks
        if add_list is not None:
            for add in add_list:
                lines.append(self.add_to_result_management(add_opts=add))
        lines.append('****return')
        # add lines
        self._current_position = self._add_inp_lines(lines, 
                                                      self._current_position)
        return
    
    def add_to_result_management(self, add_opts):
        """Add a ***add block to the current ****result_management.
        
        :param add_opts: Dictionnary of options describing the add
            block. Must contain 'filename'. All other entries are treated as
            '*' options for the process block and written in it as '   *{key}
            value' lines.
        :type process_opts: dict
        """
        if 'filename' in add_opts:
            fname = add_opts.pop('filename')
        else:
            raise ValueError('No "filename" key in the add_opts dict.')
        lines = [f' ***add {fname}']
        lines = self._add_command_options(lines, add_opts)
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
            ut_file = str(self.inp_template.with_suffix('.utp'))
            
        Nodal_field_sequence, Integ_field_sequence = (
            super(SDZsetPostProcessing, self).read_output_fields(ut_file,
                    field_list, sequence_list)
            )
        return Nodal_field_sequence, Integ_field_sequence
    
    def run_inp(self, inp_filename=None, workdir=None, print_output=False):
        """Run the .inp Zset script with current commands and arguments.
                
        This methods writes and runs a Zset script that will execute all
        commands prescribed by the class instance. First the method 
        writes the inp script template and the input .geof file from the
        SampleData instance mesh data if the `data_inputmesh` and the `data`
        attributes are set to a valid value. The results are loaded in
        the SampleData instance if the `data` attribute exists. In addition,
        this method writes if needed the problem and output mesh files, and
        verifies that the problem files containing the fields to transfer
        exists.
        """
        # Add post processing command option 
        if self._pp_count == 1:
            self.Script.set_script_command_options(['-pp'])
        elif self._pp_count > 1:
            command_opts = [f'-N {self._pp_count}',' -pp']
            self.Script.set_script_command_options(command_opts)
        # Write inp script template
        self.write_inp_template(inp_filename)
        # Check that script template arguments are well defined
        self._check_arguments_list()
        # Run Script
        script_output = self.Script.runScript(workdir, append_filename=True,
                                              print_output=print_output)
        return script_output

    def set_precision(self, precision):
        """Set the float precision of the post processing output.
        
        :param precision: precision for formatted output of real values
        :type precision: float
        """
        lines = [f' ***precision {precision}']
        self._current_position = self._add_inp_lines(lines, 
                                                      self._current_position)
        return
    
    def set_post_file_prefix(self, prefix='post'):
        """Set output files probel name."""
        lines = [f' ***post_file_prefix {prefix}']
        self._current_position = self._add_inp_lines(lines, 
                                                      self._current_position)
        return

    def _init_script_content(self):
        """Create inp minimal text content."""
        self.add_post_proc_bloc()
        return

