#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SD Zset Fea module for FEM soft Zset & SampleData instances interactions

"""

## Imports
import numpy as np
from pathlib import Path
from pymicro.core.utils.SDZsetUtils.SDZset import SDZset


# SD Zset mesher class
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

    .. rubric:: PASSING ZSET COMMANDS OPTIONS AND ARGUMENTS
            
    Some class methods are directly bound to some Zset finite element analysis
    commands. They write the command into the class mesher file, and handle its
    argument values through string templates. They can accept Zset command
    options as method keyword arguments. Each command will be added to the
    mesher as a line of the form:
        *command keyword['command']
            
    The value of the keyword argument (keyword['command']) must be a string
    that may contain a string template (an expression between ${..}). In
    this case, the template is automatically detected and handled by the
    ScriptTemplate attribute of the Mesher. The value of the template
    expression can be prescribed with the `set_mesher_args` method.
    
    No additional documentation on the command options is provided here.
    Details are available in the Zset user manual (accessible through
    the shell command line `Zman user`).   
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
            
    def load_eigenmodes(self, store_group='eigenmodes'):
        # read the U1, U2, U3 fields
        fields_list = ['U1','U2','U3']
        Node_fields, _ = self.read_output_fields(fields_list)
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
    
    