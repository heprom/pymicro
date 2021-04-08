#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module to hold global variables for the SD utils.

"""

# Matlab software : alias or path to the Matlab software executable file
MATLAB = 'matlab'
MATLAB_OPTS = '-nodisplay -nosplash -nodesktop -r '
# F. Nugyen multiphase mesher and image cleaner
#  --> make sure that the mesher_file_dir contains all the matlab code required
#      to run the mesher, and that the mesher template contains a addpath
#      matlab command poiting towards the directory
# 3D mesher
mesher3D_file_dir = '/home/users02/amarano/Travail/Simulation/Multiphase_mesher/'
MESHER3D_TEMPLATE = mesher3D_file_dir+'multi_phase_mesh.m'
MESHER3D_TMP = mesher3D_file_dir+'Mesher_tmp.m'

# 2D Mesher
mesher2D_file_dir = '/home/users02/amarano/Travail/Simulation/Multiphase_mesher_2D/'
MESHER2D_TEMPLATE = mesher2D_file_dir+'multi_phase_mesh2D.m'
MESHER2D_TMP = mesher2D_file_dir+'Mesher_tmp.m'
MESHER2D_LIBS = [mesher2D_file_dir+'libZRemove_elem.so',
               mesher2D_file_dir+'libZSelect.so']
MESHER2D_ENV = '/home/users02/amarano/Travail/Simulation/Multiphase_mesher_2D/PROG_CPP'

# Morpho cleaner
CLEANER_TEMPLATE = mesher3D_file_dir+'morphological_image_cleaner.m'
CLEANER_TMP = mesher3D_file_dir+'Cleaner_tmp.m'