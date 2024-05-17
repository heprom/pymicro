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
from pymicro.core.utils.SDZsetUtils.SDZset import SDZset


# SD Zset mesher class
class SDZsetMesher(SDZset):
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

    Some class methods are directly bound to some Zset mesher commands. They
    write the command into the class mesher file, and handle its argument
    values through string templates. They can accept Zset command options as
    method keyword arguments. Each command will be added to the mesher as a
    line of the form:
        *command keyword['command']

    The value of the keyword argument (keyword['command']) must be a string
    that may contain a string template (an expression between ${..}). In
    this case, the template is automatically detected and handled by the
    ScriptTemplate attribute of the Mesher. The value of the template
    expression can be prescribed with the `set_script_args` method.

    No additional documentation on the command options is provided here.
    Details are available in the Zset user manual (accessible through
    the shell command line `Zman user`).
    """

    def __init__(self, data=None, sd_datafile=None,
                 inp_filename=Path('.').absolute() / 'script.inp',
                 inputmesh=None, outputmesh=None,
                 input_meshfile=Path('.').absolute() / 'input.geof',
                 output_meshfile=Path('.').absolute() / 'output.geof',
                 verbose=False, autodelete=False, data_autodelete=False):
        """SDZsetMesher class constructor.

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
        super(SDZsetMesher, self).__init__(data, sd_datafile, inp_filename,
                                           inputmesh, outputmesh,
                                           input_meshfile, output_meshfile,
                                           verbose, autodelete, data_autodelete)
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
        mesher_output= super(SDZsetMesher, self).run_inp(mesher_filename,
                                                         workdir, print_output)
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
        self.set_script_args(Xbmin=Bmin[0], Xbmax=Bmax[0], Ybmin=Bmin[1],
                             Ybmax=Bmax[1], Zbmin=Bmin[2], Zbmax=Bmax[2])
        if Exterior_surf_set is not None:
            boundary_nsets= 'Xmin Xmax Ymin Ymax Zmin Zmax'
            self.create_point_set(set_name=Exterior_surf_set,
                                  options={'use_nset':boundary_nsets})
        return

    def create_element_set(self, set_name='my_set', options=dict()):
        """Write a Zset mesher node set or boundary creation command.

        This method write a **elset command block in the mesher. Each item
        of the option dict argument will be added as a [*key value] line to the
        inp **elset block. The command block will look like::

            **elset {set_name}
             *options_key1 options_value1
             *options_key2 options_value2
             *...          ...

        :param str set_name: Name of the eset to create (can contain a
              string template)
        :param dict options: Dictionary of **elset command options. Each item
            will be added as a [*key value] line to the inp **elset block.
        """
        # find out if the set_name is a script template element
        self._add_templates_to_args([set_name])
        # creates line for mesher point set definition command base line
        lines = [f'  **elset {set_name}']
        # Add command options
        self._add_command_options(lines, options)
        # write command in mesher
        self._current_position = self._add_inp_lines(lines,
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
            self.script_args_list.append(temp)
        # creates line for mesher point set definition command base line
        lines = [f'  **unshared_faces {setname}']
        if used_elsets is not None:
            self._add_templates_to_args(used_elsets)
            lines.append(f'   *elsets {used_elsets}')
        # write command in mesher
        self._current_position = self._add_inp_lines(lines,
                                                        self._current_position)
        # add nset creation if required
        if create_nodeset:
            self.create_point_set(set_name=setname,
                                  options={'use_bset':setname})
            if remove_bset:
                self.remove_set(options={'bsets':setname})
        return

    def create_boundary_set(self, set_name='my_set', options=dict()):
        """Write a Zset mesher boundary set creation command.

        This method write a  **bset command block in the mesher. Each
        item of the option dict argument will be added as a [*key value] line
        to the inp **. The command block will look like::

            **bset {set_name}
             *options_key1 options_value1
             *options_key2 options_value2
             *...          ...

        :param str set_name: Name of the bset to create (can contain a
              string template)
        :param bool is_bset: If `True`, create a bset definition command. If
            `False`, create a nset definition command.
        :param dict options: Dictionary of **elset command options. Each item
            will be added as a [*key value] line to the inp **elset block.
        """
        # find out if the set_name is a script template element
        self._add_templates_to_args([set_name])
        # creates line for mesher boundary set definition command base line
        lines = [f'  **bset {set_name}']
        # Add command options
        self._add_command_options(lines, options)
        # write command in mesher
        self._current_position = self._add_inp_lines(lines,
                                                        self._current_position)
        return

    def create_point_set(self, set_name='my_set', options=dict()):
        """Write a Zset mesher node set creation command.

        This method write a **nset command block in the mesher. Each
        item of the option dict argument will be added as a [*key value] line
        to the inp **. The command block will look like::

            **nset {set_name}
             *options_key1 options_value1
             *options_key2 options_value2
             *...          ...

        :param str set_name: Name of the nset/bset to create (can contain a
              string template)
        :param dict options: Dictionary of **elset command options. Each item
            will be added as a [*key value] line to the inp **elset block.
        """
        # find out if the set_name is a script template element
        self._add_templates_to_args([set_name])
        # creates line for mesher point set definition command base line
        lines = [f'  **nset {set_name}']
        # Add command options
        self._add_command_options(lines, options)
        # write command in mesher
        self._current_position = self._add_inp_lines(lines,
                                                        self._current_position)
        return

    def create_nset_intersection(self, set_name, nsets):
        """Write a Zset mesher nset intersection command.

        :param str set_name: Name of the intersection nset to create
        :param str nsets: String containing all the names of the nsets to
            intersect.
        """
        # find out if the set_name is a script template element
        self._add_templates_to_args([set_name])
        # creates line for mesher nset intersection command base line
        lines = [ '  **nset_intersection',
                 f'   *nsets {nsets}',
                 f'   *intersection_name {set_name}']
        # write command in mesher
        self._current_position = self._add_inp_lines(lines,
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
        self._add_templates_to_args([set_name, nset1, nset2])
        # First create intersection of the nsets
        randint = np.random.randint(0,1000)
        tmp_intersection_name = 'tmp_intersection_'+str(randint)
        nset_names = nset1+' '+nset2
        self.create_nset_intersection(tmp_intersection_name, nset_names)
        # Create the new elset from nset1
        self.create_point_set(set_name, options={'function':'1',
                                                 'use_nset':nset1})
        # Now substract intersection from nset1
        self.remove_nodes_from_sets(set_name, tmp_intersection_name)
        # Finally, remove tmp intersection nset
        self.remove_set(options={'nsets':tmp_intersection_name})
        return

    def create_all_nodes_nset(self, set_name='All_nodes'):
        """Create a nset containing all mesh nodes."""
        self.create_point_set(set_name=set_name, options={'function':'1'})
        return

    def create_interior_nodes_nset(self, set_name='interior_nodes'):
        """Create a nset will all nodes except those on the mesh ext. surf."""
        self.create_all_nodes_nset(set_name)
        self.create_boundary_node_set(setname='tmp_exterior',
                                      create_nodeset=True, remove_bset=True)
        self.remove_nodes_from_sets(set_name, 'tmp_exterior')
        self.remove_set(options={'nsets':'tmp_exterior'})
        return

    def create_mesh_from_elsets(self, elsets):
        """Create output mesh with only input mesh elements in elsets.

        :param str elsets: list of elset to use a new mesh
        """
        # find out if the set_names are a script template element
        self._add_templates_to_args(elsets)
        # create the reunion of elsets
        self.create_element_set(set_name='new_mesh_all_elements',
                                options={'add_elset':elsets})
        # create the elset to remove
        self.create_element_set(set_name='cut_from_mesh',
                                options={'not_in_elset':'new_mesh_all_elements'})
        # remove elements in elset
        self.delete_element_sets(elsets='cut_from_mesh')
        # remove created tmp elset
        self.remove_set(options={'elsets':'new_mesh_all_elements'})
        return

    def deform_mesh(self, deform_map, input_problem, magnitude=1.,
                    mesh_format='Z7'):
        """Add a **deform_mesh command to the mesher.

        **deform_mesh applies to a mesh a displacement field obtained from
        another Zset calculation to produced a new mesh, whose nodes have been
        displaced with this field multiplied by a scale factor.

        :param deform_map: sequence number of the displacement field to use in
            Zset output to deform the mesh
        :type deform_map: int
        :param input_problem: name of the .ut file of the Zset output
            containing the displacement field to deform the mesh
        :type input_problem: str
        :param magnitude: scale factor applied to the displacement field used
            to deform the mesh, defaults to 1.
        :type magnitude: float, optional
        :param mesh_format: Format of the deformed mesh output.
            Defaults to 'Z7'
        :type mesh_format: str, optional
        """
        ut_file = Path(input_problem).absolute().with_suffix('.ut')
        # find out if the sets_names are a script template element
        lines = [ '  **deform_mesh',
                 f'   *map {deform_map}',
                 f'   *input_problem {ut_file}',
                 f'   *magnitude {magnitude}',
                 f'   *format {mesh_format}']
        # write command in mesher
        self._current_position = self._add_inp_lines(lines,
                                                        self._current_position)
        return

    def delete_element_sets(self, elsets):
        """Zset command to remove input list of element sets from mesh."""
        # find out if the sets_names are a script template element
        self._add_templates_to_args(elsets)
        lines = [ f'  **delete_elset {elsets}']
        # write command in mesher
        self._current_position = self._add_inp_lines(lines,
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
        self._add_templates_to_args([target_nset, nsets_to_remove])
        lines = [ '  **remove_nodes_from_nset',
                 f'   *nset_name {target_nset}',
                 f'   *nsets_to_remove {nsets_to_remove}']
        # write command in mesher
        self._current_position = self._add_inp_lines(lines,
                                                        self._current_position)
        return

    def remove_set(self, options=dict()):
        """Write a Zset mesher nset/elset/bset removal command.

        This method write a **remove_set command block in the mesher. Each
        item of the option dict argument will be added as a [*key value] line
        to the inp **. The command block will look like::

            **remove_set
             *options_key1 options_value1
             *options_key2 options_value2
             *...          ...

        :param dict options: Dictionary of **elset command options. Each item
            will be added as a [*key value] line to the inp **elset block.
        """
        lines = ['  **remove_set']
        # Add command options
        self._add_command_options(lines, options)
        # write command in mesher
        self._current_position = self._add_inp_lines(lines,
                                                        self._current_position)
        return

    def _init_script_content(self):
        """Create mesher minimal text content."""
        lines=['****mesher',
               ' ***mesh ${output_meshfile}',
               '  **open ${input_meshfile}',
               '****return']
        self.script_args_list.append('output_meshfile')
        self.script_args_list.append('input_meshfile')
        self._current_position = self._add_inp_lines(lines,0) - 1
        return

    def _add_bounding_planes_nset_template(self):
        self.create_point_set(set_name='Xmin',
                              options={'function':'(x < ${Xbmin})'})
        self.create_point_set(set_name='Xmax',
                              options={'function':'(x > ${Xbmax})'})
        self.create_point_set(set_name='Ymin',
                              options={'function':'(y < ${Ybmin})'})
        self.create_point_set(set_name='Ymax',
                              options={'function':'(y > ${Ybmax})'})
        self.create_point_set(set_name='Zmin',
                              options={'function':'(z < ${Zbmin})'})
        self.create_point_set(set_name='Zmax',
                              options={'function':'(z > ${Zbmax})'})
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
                           elset_id_field=True,
                           bin_fields_from_sets=True, replace=False,
                           mesher_opts=dict(), print_output=False):
        """Create a conformal mesh from a multiphase image.

        A Matlab multiphase mesher is called to create a conformal mesh of a
        multiphase image: a voxelized/pixelized field of integers identifying
        the different phases of a microstructure. Then, the mesh is stored in
        the Mesher SampleData instance at the desired location with the
        desired mesh_name and Indexname.

        Depending on the dimensionality of the image (2D or 3D), a 2D or 3D
        mesher is called.

        .. IMPORTANT::

            The multiphase meshers are Matlab programs that have been developed
            by Franck Nguyen (Centre des Matériaux). They also rely on the Zset
            and the MeshGems-Tetra softwares. Those softwares must be available
            in the environement in which the mesher is called by this method.
            Hence, to use this method, make sure that:
                * 'matlab' and 'Zrun' are available commands in your shell
                  environment in which you run your Python code.


        Mesh parameters can be passed as optional keyword arguments (see
        MeshGems - Tetra software User Manual for more details.)

        .. Warning::

            The meshing program systematically ignores the phase with an ID 0
            in the multiphase image. The mesher option `MESH_ID0` (see below)
            the method adds allows to get around this behavior. If it is set
            to `true`, a +1 offset is added to all phase IDs and all phases
            are meshed. The resulting mesh and elsets numbers will be
            consistent in this case with the values in the image (+ 1 offset).

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
            with the same `mesh_name` to add the new mesh.
        :param dict mesher_opt: Optional dictionary of mesher options to
            specify various mesh caracteristics. Possible keys:values opts are:
                * 'MEM': float memory (in Mb) to reserve for the mesh
                         construction, only for 3D meshes
                * 'HGRAD': Controls the ratio between the size of elements
                           close to the boundaries and elements in the bulk of
                           the mesh, only for 3D meshes
                * 'HAUSD': Hausdorff distance between the initially produced
                           surface mesh and the volume mesh created by
                           MeshGems from the surface mesh, only for 3D Meshes
                * 'LS': Line sampling ratio. This ratio is the average number
                        of points used to discretized a boundary in the image.
                        The higher 'LS', the better the curvature of boundaries
                        will be rendered in the mesh. Increases the cost of the
                        meshing procedure, only for 2D meshes
                * 'HMIN': Minimal size for the mesh edges, for both 2D and 3D
                * 'HMAX': Maximal size for the mesh edges, for both 2D and 3D
                * 'MESH_ID0': If `false`, the phase with ID0 in the multiphase
                           image is not meshed. If `true`, and if the image
                           contains a ID 0 phase, a +1 offset is added to all
                           phase IDs and all phases are meshed.
                           WARNING : this value is written in a Matlab script,
                           it must be `true` or `false` and not `True` or
                           `False` (Python boolean values)
        """
        # Find out image dimensionality
        ImName = self.data.get_attribute('parent_grid_path',
                                         multiphase_image_name)
        ImType = self.data._get_group_type(ImName)
        if ImType == '2DImage':
            self.multi_phase_mesher2D(multiphase_image_name, meshname,
                                      indexname, location,
                                      bin_fields_from_sets, elset_id_field,
                                      replace, mesher_opts, print_output)
        elif ImType == '3DImage':
            self.multi_phase_mesher3D(multiphase_image_name, meshname,
                                      indexname, location, load_surface_mesh,
                                      elset_id_field,
                                      bin_fields_from_sets, replace,
                                      mesher_opts, print_output)
        else:
            raise ValueError('Could not find an appropriate parent Image Group'
                             f' for node {multiphase_image_name}')
        return

    def multi_phase_mesher3D(self, multiphase_image_name='', meshname='',
                            indexname='', location='', load_surface_mesh=False,
                            elset_id_field=True,
                            bin_fields_from_sets=False, replace=False,
                            mesher_opts=dict(), print_output=False):
        """Create a conformal mesh from a multiphase 3D image.

        A Matlab multiphase mesher is called to create a conformal mesh of a
        multiphase image: a voxelized/pixelized field of integers identifying
        the different phases of a microstructure. Then, the mesh is stored in
        the Mesher SampleData instance at the desired location with the
        desired mesh_name and Indexname.

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

        .. Warning::

            The meshing program systematically ignores the phase with an ID 0
            in the multiphase image. The mesher option `MESH_ID0` (see below)
            the method adds allows to get around this behavior. If it is set
            to `true`, a +1 offset is added to all phase IDs and all phases
            are meshed. The resulting mesh and elsets numbers will be
            consistent in this case with the values in the image (+ 1 offset).

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
            with the same `mesh_name` to add the new mesh.
        :param dict mesher_opt: Optional dictionary of mesher options to
            specify various mesh caracteristics. Possible keys:values opts are:
                * 'MEM': float memory (in Mb) to reserve for the mesh
                         construction
                * 'HGRAD': Controls the ratio between the size of elements
                           close to the boundaries and elements in the bulk of
                           the mesh.
                * 'HMIN': Minimal size for the mesh edges
                * 'HMAX': Maximal size for the mesh edges
                * 'HAUSD': Hausdorff distance between the initially produced
                           surface mesh and the volume mesh created by
                           MeshGems from the surface mesh.
                * 'MESH_ID0': If `false`, the phase with ID0 in the multiphase
                           image is not meshed. If `true`, and if the image
                           contains a ID 0 phase, a +1 offset is added to all
                           phase IDs and all phases are meshed.
                           WARNING : this value is written in a Matlab script,
                           it must be `true` or `false` and not `True` or
                           `False` (Python boolean values)
        """
        # Defaults mesh parameters
        default_params = {'MEM':50,'HGRAD':1.5,'HMIN':5,'HMAX':50, 'HAUSD':3,
                          'ANG':50, 'MESH_ID0':'false'}
        # Local Imports
        from pymicro.core.utils.SDUtilsGlobals import MATLAB, MATLAB_OPTS
        from pymicro.core.utils.SDUtilsGlobals import (MESHER3D_TEMPLATE,
                                                       MESHER3D_TMP,
                                                       mesher3D_file_dir)
        # get data and output pathes
        DATA_PATH = self.data._name_or_node_to_path(multiphase_image_name)
        DATA_DIR, _ = os.path.split(self.data.h5_path)
        OUT_DIR = os.path.join(DATA_DIR, 'Tmp/')
        # create temp directory for mesh files
        if not os.path.exists(OUT_DIR): os.mkdir(OUT_DIR)
        # perpare mesher script
        mesher = ScriptTemplate(template_file=MESHER3D_TEMPLATE,
                                script_file = MESHER3D_TMP, autodelete=True,
                                script_command=MATLAB)
        # set command line options
        matlab_command = '"'+"run('" + MESHER3D_TMP + "');exit;"+'"'
        mesher.set_script_command_options([MATLAB_OPTS, matlab_command])
        # set mesher script parameters
        mesher_arguments = {'DATA_PATH':DATA_PATH,'OUT_DIR':OUT_DIR,
                            'DATA_H5FILE': self.data.h5_path,
                            'SRC_PATH': mesher3D_file_dir}
        mesher_arguments.update(default_params)
        mesher_arguments.update(mesher_opts)
        mesher.set_arguments(mesher_arguments)
        mesher.createScript(filename=MESHER3D_TMP)
        # launch mesher
        CWD = os.getcwd()
        self.data.sync() # flushes H5 dataset
        mesher.runScript(workdir=OUT_DIR, append_filename=False,
                         print_output=print_output)
        os.chdir(CWD)
        # Add mesh to SD instance
        out_file = os.path.join(OUT_DIR,'Tmp_mesh_vor_tetra_p.geof')
        self.data.add_mesh(file=out_file, meshname=meshname, replace=replace,
                           indexname=indexname, location=location,
                           bin_fields_from_sets=bin_fields_from_sets)
        # Add surface mesh if required
        if load_surface_mesh:
            out_file = os.path.join(OUT_DIR,'Tmp_mesh_vor.geof')
            self.data.add_mesh(file=out_file, meshname=meshname+'_surface',
                               location=location, replace=replace,
                               bin_fields_from_sets=bin_fields_from_sets)
        if elset_id_field:
            self.data.create_elset_ids_field(mesh_name=meshname)
        # Remove tmp mesh files
        shutil.rmtree(OUT_DIR)
        # Resize mesh to Image domain
        image_group = self.data.get_attribute('parent_grid_path',
                                              multiphase_image_name)
        Im_range = (self.data.get_attribute('dimension',image_group)
                    * self.data.get_attribute('spacing',image_group))
        Im_origin = self.data.get_attribute('origin',image_group)
        # get mesh nodes and rescale mesh
        from pymicro.core.utils.SDGridUtils import SDMeshTools
        SDMeshTools.rescale_and_translate_mesh(
            data=self.data, meshname=meshname, mesh_new_origin=Im_origin,
            mesh_new_size=Im_range)
        if load_surface_mesh:
            SDMeshTools.rescale_and_translate_mesh(
                data=self.Data, meshname=meshname+'_surface',
                mesh_new_origin=Im_origin, mesh_new_size=Im_range)
        return

    def multi_phase_mesher2D(self, multiphase_image_name='', meshname='',
                            indexname='', location='',
                            bin_fields_from_sets=False,
                            elset_id_field=True, replace=False,
                            mesher_opts=dict(), print_output=False):
        """Create a conformal mesh from a 2D multiphase image.

        A Matlab multiphase mesher is called to create a conformal mesh of a
        multiphase image: a voxelized/pixelized field of integers identifying
        the different phases of a microstructure. Then, the mesh is stored in
        the Mesher SampleData instance at the desired location with the
        desired mesh_name and Indexname.

        The meshing procedure involves the detection and discretization of the
        boundaries in the 2D image. The space between boundary elements is then
        filled with elements to construct a surface mesh.

        The mesher path must be correctly set in the `global_variables.py`
        file, as well as the definition and path of the Matlab command. The
        multiphase mesher is a Matlab program that has been developed by
        Franck Nguyen (Centre des Matériaux). It also relies on the Zset
        and the MeshGems-Tetra softwares.

        Mesh parameters can be passed as optional keyword arguments (see
        MeshGems - Tetra software User Manual for more details.)

        .. Warning::

            The meshing program systematically ignores the phase with an ID 0
            in the multiphase image. The mesher option `MESH_ID0` (see below)
            the method adds allows to get around this behavior. If it is set
            to `true`, a +1 offset is added to all phase IDs and all phases
            are meshed. The resulting mesh and elsets numbers will be
            consistent in this case with the values in the image (+ 1 offset).

        :param str multiphase_image_name: Path, Name, Index Name or Alias of
            the multiphase image field to mesh in the SampleData instance.
        :param str meshname: name used to create the Mesh group in dataset
        :param indexname: Index name used to reference the Mesh group
        :param str location: Path, Name, Index Name or Alias of the parent
            group where the Mesh group is to be created
        :param bool bin_fields_from_sets: If `True`, stores all Node and
            Element Sets in mesh_object as binary fields (1 on Set, 0 else)
        :param bool replace: if `True`, overwrites pre-existing Mesh group
            with the same `mesh_name` to add the new mesh.
        :param dict mesher_opt: Optional dictionary of mesher options to
            specify various mesh caracteristics. Possible keys:values opts are:
                * 'LS': Line sampling ratio. This ratio is the average number
                        of points used to discretized a boundary in the image.
                        The higher 'LS', the better the curvature of boundaries
                        will be rendered in the mesh. Increases the cost of the
                        meshing procedure.
                * 'HMIN': Minimal size for the mesh edges
                * 'HMAX': Maximal size for the mesh edges
                * 'MESH_ID0': If `false`, the phase with ID0 in the multiphase
                           image is not meshed. If `true`, and if the image
                           contains a ID 0 phase, a +1 offset is added to all
                           phase IDs and all phases are meshed.
                           WARNING : this value is written in a Matlab script,
                           it must be `true` or `false` and not `True` or
                           `False` (Python boolean values)
        """
        # Local Imports
        from pymicro.core.utils.SDUtilsGlobals import MATLAB, MATLAB_OPTS
        from pymicro.core.utils.SDUtilsGlobals import (
            MESHER2D_TEMPLATE, MESHER2D_TMP, MESHER2D_LIBS, MESHER2D_ENV,
                                     mesher2D_file_dir)
        # Set environnement variable
        os.environ["PRG_ZEB"] = MESHER2D_ENV
        # Defaults mesh parameters
        default_params = {'LS':5,'HMIN':5,'HMAX':100, 'MESH_ID0':'false'}
        #
        # get data and output pathes
        DATA_PATH = self.data._name_or_node_to_path(multiphase_image_name)
        DATA_DIR, _ = os.path.split(self.data.h5_path)
        OUT_DIR = os.path.join(DATA_DIR, 'Tmp/')
        # create temp directory for mesh files
        if not os.path.exists(OUT_DIR): os.mkdir(OUT_DIR)
        # perpare mesher script
        mesher = ScriptTemplate(template_file=MESHER2D_TEMPLATE,
                                script_file = MESHER2D_TMP, autodelete=True,
                                script_command=MATLAB)
        # set command line options
        matlab_command = '"'+"run('" + MESHER2D_TMP + "');exit;"+'"'
        mesher.set_script_command_options([MATLAB_OPTS, matlab_command])
        # set mesher script parameters
        mesher_arguments = {'DATA_PATH':DATA_PATH,'OUT_DIR':OUT_DIR,
                            'DATA_H5FILE': self.data.h5_path,
                            'SRC_PATH': mesher2D_file_dir}
        mesher_arguments.update(default_params)
        mesher_arguments.update(mesher_opts)
        mesher.set_arguments(mesher_arguments)
        mesher.createScript(filename=MESHER2D_TMP)
        # launch mesher
        CWD = os.getcwd()
        self.data.sync() # flushes H5 dataset
        mesher.runScript(workdir=OUT_DIR, append_filename=False,
                         print_output=print_output)
        os.chdir(CWD)
        # Add mesh to SD instance
        out_file = os.path.join(OUT_DIR,'Tmp_mesh_fuse_remesh.geof')
        self.data.add_mesh(file=out_file, meshname=meshname, replace=replace,
                            indexname=indexname, location=location,
                            bin_fields_from_sets=bin_fields_from_sets)
        if elset_id_field:
            self.data.create_elset_ids_field(mesh_name=meshname)
        # Remove tmp mesh files
        shutil.rmtree(OUT_DIR)
        # Resize mesh to Image domain
        image_group = self.data.get_attribute('parent_grid_path',
                                              multiphase_image_name)
        Im_dim = self.data.get_attribute('dimension',image_group)[[1,0]]
        Im_sp = self.data.get_attribute('spacing',image_group)[[1,0]]
        Im_range =  Im_dim*Im_sp
        Im_origin = self.data.get_attribute('origin',image_group)
        # get mesh nodes and rescale mesh
        from pymicro.core.utils.SDGridUtils import SDMeshTools
        SDMeshTools.rescale_and_translate_mesh(
            data=self.data, meshname=meshname, mesh_new_origin=Im_origin,
            mesh_new_size=Im_range)
        return

    def morphological_image_cleaner(self, target_image_field='',
                                    clean_fieldname='', indexname='',
                                    replace=False, print_output=False,
                                    **keywords):
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
                                                       CLEANER_TMP,
                                                       mesher3D_file_dir)
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
                            'DATA_H5FILE': self.data.h5_path,
                            'SRC_PATH': mesher3D_file_dir}
        cleaner.set_arguments(cleaner_arguments)
        cleaner.createScript(filename=CLEANER_TMP)
        # launch cleaner
        CWD = os.getcwd()
        self.data.sync() # flushes H5 dataset
        cleaner.runScript(append_filename=False, print_output=print_output)
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
