#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SD Amitex Module: module handling interactions between Amitex_fftp
   simulations and SampleData datasets.
"""

## Imports
import os
import numpy as np
import vtk

from pathlib import Path

# SD Amitex interface class
class SDAmitexIO():
    """Base class implementing a Amitex FFTP / SampleData interface.

    This class is an inteface between the SampleData data platform class and
    the Amitex_fftp software (developed at the CEA Saclay).
    It provides a high-level interface to create Amitex_fftp input files
    and read Amitex_fftp output files from/to SampleData datasets.

    """

    @staticmethod
    def load_std(std_path):
        """Read content of a .std file and returns as Numpy structured array.

        This method must be transfered to a new subpackage SDAmitex_utils

        :param std_path: name/path of the .std file.
        :type std_path: Path(pathlib) object or string
        :return: Results, Numpy structured array containing the output values
            in .std file: the array fields are 'time', 'sigma' (Cauchy stress),
            'epsilon' (small strains tensor), 'sigma_rms' and 'epsilon_rms'
            root mean square of tensors over the unit cell, 'N_iterations'
            number of iterations of the FFT algorithm to reach convergence at
            each increment.
        :rtype:
        """
        # TODO: implement reading results with start:step:stop
        finite_strain=False
        std_lines = []
        p = Path(std_path).absolute()
        # read txt content of .std file
        with open(p,'r') as f:
            l = f.readline()
            while l:
                if not l.startswith('#'):
                    A = np.array(l.split()).astype(np.double)
                    std_lines.append(A)
                else:
                    if 'xx,yy,zz,xy,xz,yz,yx,zx,zy' in l:
                        finite_strain = True
                l = f.readline()
        # create Numpy structured array
        if finite_strain:
            dt = np.dtype([('time', np.double, (1,)),
                           ('sigma', np.double, (6,)),
                           ('boussinesq', np.double, (9,)),
                           ('GL_strain', np.double, (6,)),
                           ('grad_u', np.double, (9,)),
                           ('sigma_rms', np.double, (6,)),
                           ('boussinesq_rms', np.double, (9,)),
                           ('GL_strain_rms', np.double, (6,)),
                           ('grad_u_rms', np.double, (9,)),
                           ('N_iterations', np.double, (1,))])
        else:
            dt = np.dtype([('time', np.double, (1,)),
                           ('sigma', np.double, (6,)),
                           ('epsilon', np.double, (6,)),
                           ('sigma_rms', np.double, (6,)),
                           ('epsilon_rms', np.double, (6,)),
                           ('N_iterations', np.double, (1,))])
        # fill results array for each time step
        N_times = len(std_lines)
        Results = np.empty(shape=(N_times,), dtype=dt)
        if finite_strain:
            for t in range(N_times):
                Results[t]['time'] = std_lines[t][0]
                Results[t]['sigma'] = std_lines[t][[1,2,3,4,6,5]]
                Results[t]['boussinesq'] = std_lines[t][[7,8,9,10,14,13,11,15,
                                                         12]]
                Results[t]['GL_strain'] = std_lines[t][[16,17,18,19,21,20]]
                Results[t]['grad_u'] = std_lines[t][[22,23,24,25,29,28,26,30,
                                                     27]]
                Results[t]['sigma_rms'] = std_lines[t][[31,32,33,34,36,35]]
                Results[t]['boussinesq_rms'] = std_lines[t][[37,38,39,40,44,43,
                                                             41,45,42]]
                Results[t]['GL_strain_rms'] = std_lines[t][[46,47,48,49,51,50]]
                Results[t]['grad_u_rms'] = std_lines[t][[52,53,54,55,59,58,56,
                                                         60,57]]
                Results[t]['N_iterations'] = std_lines[t][-1]
        else:
            for t in range(N_times):
                Results[t]['time'] = std_lines[t][0]
                Results[t]['sigma'] = std_lines[t][[1,2,3,4,6,5]]
                Results[t]['epsilon'] = std_lines[t][[7,8,9,10,12,11]]
                Results[t]['sigma_rms'] = std_lines[t][[13,14,15,16,18,17]]
                Results[t]['epsilon_rms'] = std_lines[t][[19,20,21,22,24,23]]
                Results[t]['N_iterations'] = std_lines[t][-1]
        return Results

    @staticmethod
    def load_amitex_stress_strain(vtk_basename, grip_size=0, ext_size=0,
                                   grip_dim=2):
        """Return stress/strain fields as numpy tensors from Amitex vtk output.

        This method must be transfered to a new subpackage SDAmitex_utils

        :param vtk_basename: Basename of vtk stress/Strain fields to load.
            Fields names are outputed by Amitex with the following structure:
            'basename' + '_field_component' + '_increment' + '.vtk'.
        :type vtk_basename: str
        :param int grip_size: Width in voxels of the material layer used in
            simulation unit cell for tension grips
        :param int ext_size: Width in voxels of the void material layer used to
            simulate free surfaces.
        :param int grip_dim: Dimension along which the tension test has been
            simulated (0:x, 1:y, 2:z)
        :return Stress: Stress tensors dict read from output.
        :rtype: Dict of Numpy arrays for each increment
            {Incr(int):[Nx,Ny,Nz,6]}
        :return Strain: Strain tensors dict read from output.
        :rtype: Dict of Numpy arrays for each increment
            {Incr(int):[Nx,Ny,Nz,6]}
        """
        # local imports
        import re
        # Check if stress outputs exist
        vtk_path = Path(vtk_basename).absolute()
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
        # Get first value to initialize Stress output and find output_slice
        sig_tmp = SDAmitexIO.read_vtk_legacy(sig_files[0])
        Sl = SDAmitexIO.get_amitex_tension_test_relevant_slice(
            init_shape=sig_tmp.shape, grip_size=grip_size, grip_dim=grip_dim,
            ext_size=ext_size)
        # TODO: adapt if loading a non symmetric Stress tensor (finite strains)
        sig_shape = (Sl[0,1] - Sl[0,0], Sl[1,1] - Sl[1,0], Sl[2,1] - Sl[2,0],
                     6)
        Increments = np.unique(np.array(sig_incr))
        # Initialize stress dict
        Stress_dict = {}
        for incr in Increments:
            Stress_dict[incr] = np.zeros(shape=sig_shape, dtype=np.double)
        # Fill Stress dict with output
        for file in sig_files:
            sig_tmp = SDAmitexIO.read_vtk_legacy(file, Sl)
            increment = incr = int(incr_pattern.findall(file)[0].strip('.vtk'))
            comp_list = comp_pattern.findall(file)
            if len(comp_list) == 0:
                Stress_dict[increment] = sig_tmp
            elif len(comp_list) == 1:
                component = int(comp_list[0].strip('sig')) - 1
                # change component to comply to pymicro Voigt convention
                if component == 3:
                    component = 5
                elif component == 5:
                    component = 3
                Stress_dict[increment][...,component] = sig_tmp
            else:
                raise ValueError(f' Vtk file {file} name has an invalid'
                                 ' component value (must be one digit).')
        # Same for strain fields
        # Get all names of vtk files in the directory and associeted increments
        pattern = re.compile(vtk_path.stem+'_def\d?_\d+.vtk')
        incr_pattern = re.compile('\d+.vtk')
        comp_pattern = re.compile('def\d')
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
                eps_incr.append(incr)
        Increments = np.unique(np.array(sig_incr))
        # Initialize stress dict
        Strain_dict = {}
        for incr in Increments:
            Strain_dict[incr] = np.zeros(shape=sig_shape, dtype=np.double)
        # Fill Stress dict with output
        for file in eps_files:
            eps_tmp = SDAmitexIO.read_vtk_legacy(file, Sl)
            increment = incr = int(incr_pattern.findall(file)[0].strip('.vtk'))
            comp_list = comp_pattern.findall(file)
            if len(comp_list) == 0:
                Strain_dict[increment] = eps_tmp
            elif len(comp_list) == 1:
                component = int(comp_list[0].strip('def')) - 1
                # change component to comply to pymicro Voigt convention
                if component == 3:
                    component = 5
                elif component == 5:
                    component = 3
                Strain_dict[increment][...,component] = eps_tmp
            else:
                raise ValueError(f' Vtk file {file} name has an invalid'
                                 ' component value (must be one digit).')
        return Stress_dict, Strain_dict

    @staticmethod
    def read_vtk_legacy(vtk_path, output_slice=None):
        """Read a Amitex_fftp vtk output and return the fields stored in it.

        This method must be transfered to a new subpackage SDAmitex_utils

        :param vtk_path: name/path of the .vtk file.
        :type vtk_path: string
        :param output_slice: Interesting slice of data to get. Used if simulation
            has been carried with additional materials for tension grips and
            sample exterior. The slice can be generated with the method
            '_get_amitex_tension_test_relevant_slice'
        :type output_slice: numpy array (3,2)
        :return: Return a dict. of the amitex_fftp output fields stored in
            the vtk. file, whose keys are the field names.
        :rtype: dict( 'fieldname':np.double array)
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
        # spacing = reader.GetOutput().GetSpacing()
        dim = reader.GetOutput().GetDimensions()
        output_shape = tuple([i-1 for i in dim])
        data = numpy_support.vtk_to_numpy(Array)
        data = data.reshape(output_shape, order='F')
        # get usefull slice
        if output_slice is not None:
            data = data[output_slice[0,0]:output_slice[0,1],
                        output_slice[1,0]:output_slice[1,1],
                        output_slice[2,0]:output_slice[2,1],...]
        return data

    @staticmethod
    def get_amitex_tension_test_relevant_slice(init_shape, grip_size=1,
                                                grip_dim=2, ext_size=1):
        """Return indices of material unit cell in amitex tension results.

        This method must be transfered to a new subpackage SDAmitex_utils

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

