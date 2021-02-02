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
mesher_file_dir = '/home/users02/amarano/Travail/Simulation/Multiphase_mesher/'
MESHER_TEMPLATE = mesher_file_dir+'multi_phase_mesh.m'
MESHER_TMP = mesher_file_dir+'Mesher_tmp.m'
CLEANER_TEMPLATE = mesher_file_dir+'morphological_image_cleaner.m'
CLEANER_TMP = mesher_file_dir+'Cleaner_tmp.m'