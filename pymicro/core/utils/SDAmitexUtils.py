#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SD Amitex Module: module handling interactions between Amitex_fftp
   simulations and SampleData datasets.
"""

## Imports
import os
import numpy as np
import vtk
import re

from pathlib import Path

# Lists of fields for Numpy dtypes mapping Amitex standard output files content
Amitex_std_finite_strain = [('time', np.double ),
                           ('sigma', np.double, (6,)),
                           ('grad_u', np.double, (9,)),
                           ('sigma_rms', np.double, (6,)),
                           ('grad_u_rms', np.double, (9,))]

                           # ('boussinesq', np.double, (9,)),
                           # ('GL_strain', np.double, (6,)),
                           # ('boussinesq_rms', np.double, (9,)),
                           # ('GL_strain_rms', np.double, (6,)),

Amitex_std_hpp = [('time', np.double),
                  ('sigma', np.double, (6,)),
                  ('epsilon', np.double, (6,)),
                  ('sigma_rms', np.double, (6,)),
                  ('epsilon_rms', np.double, (6,))]

# SD Amitex interface class
class SDAmitexIO():
    """Base class implementing a Amitex FFTP / SampleData interface.

    This class is an inteface between the SampleData data platform class and
    the Amitex_fftp software (developed at the CEA Saclay).
    It provides a high-level interface to create Amitex_fftp input files
    and read Amitex_fftp output files from/to SampleData datasets.

    """

    @staticmethod
    def load_std(std_path, start=None, step=None, stop=None, int_var_names={}):
        """Return content of a .m/z/std file  as Numpy structured array.

        Allow to read a standard output file of Amitex_fftp: .std, .mstd
        or .zstd. The file can be read with a specific start, step and stop
        slicing.

        :param std_path: name/path of the .std file.
        :type std_path: Path(pathlib) object or string
        :param start: starting row index to read output
        :type start: int
        :param step: spacing between rows to read
        :type step: int
        :param stop: index of the last row to read
        :type stop: int
        :param dict int_var_names: a dictionary containing the name of the
        internal variable.
        :return: results, Numpy structured array containing the output values
            in .std file: the array fields are 'time', 'sigma' (Cauchy stress),
            'epsilon' (small strains tensor), 'sigma_rms' and 'epsilon_rms'
            root mean square of tensors over the unit cell, 'N_iterations'
            number of iterations of the FFT algorithm to reach convergence at
            each increment.
        :rtype: numpy structured array
        """
        finite_strain=False
        std_lines = []
        p = Path(std_path).absolute()
        # get pattern to find out which internal variables values are present
        # (for .zstd only)
        pattern = re.compile('variable interne \d+')
        idx_pattern = re.compile('\d+e')
        varInt_names = dict()
        # read txt content of .std or .mstd or .zstd file
        with open(p,'r') as f:
            l = f.readline()
            while l:
                if not l.startswith('#'):
                    A = np.array(l.split()).astype(np.double)
                    std_lines.append(A)
                else:
                    if 'xx,yy,zz,xy,xz,yz,yx,zx,zy' in l:
                        finite_strain = True
                    if 'variable interne' in l:
                        suffix = ''
                        if 'ecart type' in l:
                            suffix = '_std'
                        varInt_number = int(pattern.findall(l)[0].split()[2])
                        varInt_index = int(idx_pattern.findall(l)[0][:-1])
                        if varInt_number in int_var_names:
                            name = f'{int_var_names[varInt_number]}' + suffix
                            varInt_names[name] = varInt_index - 1
                        else:
                            name = f'varInt_{varInt_number}'+suffix
                            varInt_names[name] = varInt_index - 1
                l = f.readline()
        # create Numpy structured array
        if finite_strain:
            dtype_description = Amitex_std_finite_strain
        else:
            dtype_description = Amitex_std_hpp
        for key in varInt_names:
            dtype_description.append((key, np.double, (1,)))
        dt = np.dtype(dtype_description)
        # fill results array for each time step
        n_rows = len(std_lines)
        results = np.empty(shape=(n_rows,), dtype=dt)
        if finite_strain:
            for t in range(n_rows):
                # load standard outputs
                results[t]['time'] = std_lines[t][0]
                results[t]['sigma'] = std_lines[t][[1, 2, 3, 4, 6, 5]]
                # results[t]['boussinesq'] = std_lines[t][[7,8,9,10,14,13,11,15,
                #                                          12]]
                # results[t]['GL_strain'] = std_lines[t][[16,17,18,19,21,20]]
                results[t]['grad_u'] = std_lines[t][[22, 23, 24, 25, 29,
                                                     28, 26, 30, 27]]
                results[t]['sigma_rms'] = std_lines[t][[31, 32, 33, 34, 36, 35]]
                # results[t]['boussinesq_rms'] = std_lines[t][[37,38,39,40,44,43,
                #                                              41,45,42]]
                # results[t]['GL_strain_rms'] = std_lines[t][[46,47,48,49,51,50]]
                results[t]['grad_u_rms'] = std_lines[t][[52, 53, 54, 55, 59,
                                                         58, 56, 60, 57]]
                for key, value in varInt_names.items():
                    results[t][key] = std_lines[t][value]
        else:
            for t in range(n_rows):
                results[t]['time'] = std_lines[t][0]
                results[t]['sigma'][0:3] = std_lines[t][[1,2,3]]
                results[t]['sigma'][3:6] = 0.5*std_lines[t][[4,6,5]]
                results[t]['epsilon'][0:3] = std_lines[t][[7,8,9]]
                results[t]['epsilon'][3:6] = 0.5*std_lines[t][[10,12,11]]
                results[t]['sigma_rms'][0:3] = std_lines[t][[13,14,15]]
                results[t]['sigma_rms'][3:6] = 0.5*std_lines[t][[16,18,17]]
                results[t]['epsilon_rms'][0:3] = std_lines[t][[19,20,21]]
                results[t]['epsilon_rms'][3:6] = 0.5*std_lines[t][[22,24,23]]
                for key, value in varInt_names.items():
                    results[t][key] = std_lines[t][value]

        if start is None:
            start = 0
        if stop is None:
            stop = n_rows
        if step is None:
            step = 1
        return results[start:stop:step]

    @staticmethod
    def load_amitex_output_fields(vtk_basename, grip_size=0, ext_size=0,
                                  grip_dim=2, boussinesq_stress=False,
                                  Int_var_names=dict()):
        """Return stress/strain fields as numpy tensors from Amitex vtk output.

        This method loads stress, strain and internal variables output fields
        from Amitex fft simulation outputs. It loads the Cauchy Stress and
        infinitesimal strain tensor for simulations conducted with the
        infinitesimal strain hypothesis. It loads, by default, the Cauchy
        Stress and the Green-Lagrange strain tensor for a finite strain
        simulation. If requested, the Boussinesq (PKI) stress tensor can be
        output.

        :param vtk_basename: Basename of vtk stress/Strain fields to load.
            Fields names are outputed by Amitex with the following structure:
            'base_name' + '_field_component' + '_increment' + '.vtk'.
        :type vtk_basename: str
        :param int grip_size: Width in voxels of the material layer used in
            simulation unit cell for tension grips
        :param int ext_size: Width in voxels of the void material layer used to
            simulate free surfaces.
        :param int grip_dim: Dimension along which the tension test has been
            simulated (0:x, 1:y, 2:z)
        :param bool boussinesq_stress: if `True`, return the Boussinesq stress
            tensor fields for finite strain simulations instead of the Cauchy
            stress tensor.
        :return Stress: Cauchy stress tensors dict read from output or
            Boussinesq stress tensor if finite strain and requested.
        :rtype: Dict of Numpy arrays for each increment
            {Incr(int):[Nx,Ny,Nz,6 or 9]}
        :return Strain: Strain tensors dict read from output (infinitesimal
            strain or Green-Lagrange strain tensor for finite strain
            simulations).
        :rtype: Dict of Numpy arrays for each increment
            {Incr(int):[Nx,Ny,Nz,6 or 9]}
        :return VarInt: Internal variables fields dict. One subdict for each
            material with internal variable field outputs, which has one
            subdict for each output increment.
        :rtype: Dict of Numpy arrays for each increment
            {material(int):{Incr(int):{VarInt_number(int):[Nx,Ny,Nz,1]}}}
        """
        # initialize finite strain flag to False
        finite_strain = False
        # Check if strain outputs exist
        vtk_path = Path(vtk_basename).absolute()
        # Get all names of vtk files in the directory and associeted increments
        pattern = re.compile(vtk_path.stem+'_def\d?_\d+.vtk')
        incr_pattern = re.compile('\d+.vtk')
        comp_pattern = re.compile('def\d')
        fs_pattern = re.compile('def9')
        eps_files = []
        eps_incr = []
        for filepath in os.listdir(vtk_path.parent):
            if pattern.match(filepath):
                fileP = vtk_path.parent / filepath
                eps_files.append(str(fileP))
                incr = int(incr_pattern.findall(filepath)[0].strip('.vtk'))
                if incr is None:
                    raise ValueError('At least one Amitex_fftp .vtk file in '
                                     'the directory has no increment number in'
                                     ' its name.')
                fs_list = fs_pattern.findall(filepath)
                if len(fs_list) > 0:
                    finite_strain = True
                eps_incr.append(incr)
        # Get first value shape to initialize output and
        # find output_slice
        if len(eps_files) > 0:
            eps_tmp = SDAmitexIO.read_vtk_legacy(eps_files[0])
            Sl = SDAmitexIO.get_amitex_tension_test_relevant_slice(
                init_shape=eps_tmp.shape, grip_size=grip_size, grip_dim=grip_dim,
                ext_size=ext_size)
            if finite_strain:
                eps_shape = (Sl[0,1] - Sl[0,0], Sl[1,1] - Sl[1,0],
                             Sl[2,1] - Sl[2,0], 9)
            else:
                eps_shape = (Sl[0,1] - Sl[0,0], Sl[1,1] - Sl[1,0],
                             Sl[2,1] - Sl[2,0], 6)
        Increments_eps = np.unique(np.array(eps_incr))
        # Initialize strain dict
        Strain_dict = {}
        for incr in Increments_eps:
            Strain_dict[incr] = np.zeros(shape=eps_shape, dtype=np.double)
        # Fill strain dict with output
        for file in eps_files:
            eps_tmp = SDAmitexIO.read_vtk_legacy(file, Sl)
            increment = int(incr_pattern.findall(file)[0].strip('.vtk'))
            comp_list = comp_pattern.findall(file)
            if len(comp_list) == 0:
                # all components are within the same vtk file
                Strain_dict[increment] = eps_tmp
            elif len(comp_list) == 1:
                # Component is in a specific vtk file
                component = int(comp_list[0].strip('def')) - 1
                # change component to comply to pymicro convention
                component = SDAmitexIO.get_sd_comp_from_amitex_comp(
                                                      component, finite_strain)
                Strain_dict[increment][...,component] = eps_tmp
            else:
                raise ValueError(f' Vtk file {file} name has an invalid'
                                 ' component value (must be one digit).')
        # Same for stress fields
        # Get all names of vtk files in the directory and associeted increments
        pattern = re.compile(vtk_path.stem+'_sig\d?_\d+.vtk')
        incr_pattern = re.compile('\d+.vtk')
        comp_pattern = re.compile('sig\d')
        sig_files = []
        sig_incr = []
        for filepath in os.listdir(vtk_path.parent):
            if pattern.match(filepath):
                fileP = vtk_path.parent / filepath
                sig_files.append(str(fileP))
                incr = int(incr_pattern.findall(filepath)[0].strip('.vtk'))
                if incr is None:
                    raise ValueError('At least one Amitex_fftp .vtk file in '
                                     'the directory has no increment number in'
                                     ' its name.')
                sig_incr.append(incr)
        # Get first value shape to initialize output and
        # find output_slice
        if len(sig_files) > 0:
            sig_tmp = SDAmitexIO.read_vtk_legacy(sig_files[0])
            Sl = SDAmitexIO.get_amitex_tension_test_relevant_slice(
                init_shape=sig_tmp.shape, grip_size=grip_size, grip_dim=grip_dim,
                ext_size=ext_size)
            if finite_strain and boussinesq_stress:
                sig_shape = (Sl[0,1] - Sl[0,0], Sl[1,1] - Sl[1,0],
                             Sl[2,1] - Sl[2,0], 9)
            else:
                sig_shape = (Sl[0,1] - Sl[0,0], Sl[1,1] - Sl[1,0],
                             Sl[2,1] - Sl[2,0], 6)
        Increments_sig = np.unique(np.array(sig_incr))
        # Initialize stress dict
        Stress_dict = {}
        for incr in Increments_sig:
            Stress_dict[incr] = np.zeros(shape=sig_shape, dtype=np.double)
        # Fill Stress dict with output
        for file in sig_files:
            sig_tmp = SDAmitexIO.read_vtk_legacy(file, Sl)
            increment = incr = int(incr_pattern.findall(file)[0].strip('.vtk'))
            comp_list = comp_pattern.findall(file)
            if len(comp_list) == 0:
                Stress_dict[increment] = sig_tmp
            elif len(comp_list) == 1:
                # Component is in a specific vtk file
                component = int(comp_list[0].strip('sig')) - 1
                # change component to comply to pymicro convention
                component = SDAmitexIO.get_sd_comp_from_amitex_comp(
                                component, finite_strain and boussinesq_stress)
                Stress_dict[increment][...,component] = sig_tmp
            else:
                raise ValueError(f' Vtk file {file} name has an invalid'
                                 ' component value (must be one digit).')
        # Same for internal variables fields
        # Get all names of vtk files in the directory and associeted increments
        pattern = re.compile(vtk_path.stem+'_M\d_varInt\d+_\d+.vtk')
        incr_pattern = re.compile('\d+.vtk')
        comp_pattern = re.compile('varInt\d+')
        material_pattern = re.compile('_M\d+')
        varI_files = []
        varI_incr = []
        varI_mat = []
        for filepath in os.listdir(vtk_path.parent):
            if pattern.match(filepath):
                fileP = vtk_path.parent / filepath
                varI_files.append(str(fileP))
                incr = int(incr_pattern.findall(filepath)[0].strip('.vtk'))
                if incr is None:
                    raise ValueError('At least one Amitex_fftp .vtk file in '
                                     'the directory has no increment number in'
                                     ' its name.')
                mat = int(material_pattern.findall(filepath)[0].strip('_M'))
                varI_incr.append(incr)
                if not varI_mat.__contains__(mat):
                    varI_mat.append(mat)
        # Get first value shape to initialize output and
        # find output_slice
        if len(varI_files) > 0:
            varI_tmp = SDAmitexIO.read_vtk_legacy(varI_files[0])
            Sl = SDAmitexIO.get_amitex_tension_test_relevant_slice(
                init_shape=varI_tmp.shape, grip_size=grip_size,
                grip_dim=grip_dim, ext_size=ext_size)
        Increments_vI = np.unique(np.array(varI_incr))
        # Initialize internal variables dict
        VarInt_dict = {}
        for mat in varI_mat:
            VarInt_dict[mat] = {}
        # Fill dict with output
        for file in varI_files:
            varInt_tmp = SDAmitexIO.read_vtk_legacy(file, Sl)
            increment = incr = int(incr_pattern.findall(file)[0].strip('.vtk'))
            comp_list = comp_pattern.findall(file)
            component = int(comp_list[0].strip('varInt'))
            mat = int(material_pattern.findall(file)[0].strip('_M'))
            if not VarInt_dict[mat].__contains__(increment):
                VarInt_dict[mat][increment] = {}
            VarInt_dict[mat][increment][component] = varInt_tmp
        # return non empty increment list
        if len(Increments_eps) > 0:
            Increments = Increments_eps
        elif len(Increments_sig) > 0:
            Increments = Increments_sig
        elif len(Increments_vI) > 0:
            Increments = Increments_vI
        else:
            Increments = []
        Increments = sorted(Increments)
        return Stress_dict, Strain_dict, VarInt_dict, Increments

    @staticmethod
    def read_vtk_legacy(vtk_path, output_slice=None):
        """Read a Amitex_fftp vtk output and return the fields stored in it.

        :param vtk_path: name/path of the .vtk file.
        :type vtk_path: string
        :param output_slice: Interesting slice of data to get. Used if simulation
            has been carried with additional materials for tension grips and
            sample exterior. The slice can be generated with the method
            '_get_amitex_tension_test_relevant_slice'
        :type output_slice: numpy array (3,2)
        :return: Return a dict. of the amitex_fftp output fields stored in
            the vtk. file, whose keys are the field names.
        :rtype: dict( 'field_name':np.double array)
        """
        # local imports
        from vtk.util import numpy_support
        # set file path
        p = Path(vtk_path).absolute().with_suffix('.vtk')
        # init vtk reader
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(str(p))
        reader.Update()
        # read raw data
        Array = reader.GetOutput().GetCellData().GetArray(0)
        spacing = reader.GetOutput().GetSpacing()
        dim = reader.GetOutput().GetDimensions()
        output_shape = tuple([i-1 for i in dim])
        data = numpy_support.vtk_to_numpy(Array)
        data = data.reshape(output_shape, order='F')
        # get usefull slice
        if output_slice is not None:
            data = data[output_slice[0,0]:output_slice[0,1],
                        output_slice[1,0]:output_slice[1,1],
                        output_slice[2,0]:output_slice[2,1],...]
        return data, spacing
    
    def write_vtk_legacy(array, vtk_path='output', spacing=1,
                         array_name="mat_id"):
        """ Writes a vtk legacy file from a numpy array

        :param array: Numpy array to write as binary image
        :type array: numpy array 
        :param output_name: path of the vtk file to write
        :type output_name: string

        """
        from vtk.util import numpy_support
        
        # transform data to vtk data array
        vtk_data_array = numpy_support.numpy_to_vtk(np.ravel(array, order='F'),
                                                    deep=1)
        vtk_data_array.SetName(array_name)
        
        # Init VTK regular grid 
        if len(spacing) == 1:
            voxel_size = np.array([spacing, spacing, spacing])
        else:
            voxel_size = spacing
            
        grid = vtk.vtkImageData()
        size = array.shape
        grid.SetExtent(0, size[0], 0, size[1], 0, size[2])
        grid.GetCellData().SetScalars(vtk_data_array)
        grid.SetSpacing(voxel_size[0], voxel_size[1], voxel_size[2])
        
        # Init vtk writer and write file
        writer = vtk.vtkStructuredPointsWriter()
        writer.SetFileName('%s.vtk' % vtk_path)
        writer.SetFileTypeToBinary()
        writer.SetInputData(grid)
        writer.Write()
        
        print(f'File {vtk_path}.vtk written as VTK legacy file')
        return

    @staticmethod
    def get_amitex_tension_test_relevant_slice(init_shape, grip_size=1,
                                                grip_dim=2, ext_size=1):
        """Return indices of material unit cell in amitex tension results.

        :param int grip_size: Width in voxels of the material layer used in
            simulation unit cell for tension grips
        :param int grip_dim: Dimension along which the tension test has been
            simulated (0:x, 1:y, 2:z)
        :param int ext_size: Width in voxels of the void material layer used to
            simulate free surfaces.
        """
        ext_indices = np.setdiff1d([0,1,2], grip_dim)
        Mat_slice = np.empty(shape=(3,2), dtype=int)
        # slice for grip
        Mat_slice[grip_dim,0] = grip_size
        Mat_slice[grip_dim,1] = init_shape[grip_dim] - grip_size
        # slice for exterior
        Mat_slice[ext_indices[0],0] = ext_size
        Mat_slice[ext_indices[0],1] = init_shape[ext_indices[0]] - ext_size
        Mat_slice[ext_indices[1],0] = ext_size
        Mat_slice[ext_indices[1],1] = init_shape[ext_indices[1]] - ext_size
        return Mat_slice

    @staticmethod
    def get_sd_comp_from_amitex_comp(amitex_comp, finite_strain=False):
        """Return the Amitex component index value in SampleData convention."""
        indices_small_strain = [0,1,2,3,5,4]
        indices_finite_strain = [0,1,2,3,7,6,4,8,5]
        if finite_strain:
            return indices_finite_strain[amitex_comp]
        else:
            return indices_small_strain[amitex_comp]