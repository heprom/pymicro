#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module to hold global variables for the `core` package.
"""

# Module global variables for xdmf compatibility
XDMF_FIELD_TYPE = {1: 'Scalar', 2: 'Vector', 3: 'Vector', 6: 'Tensor6',
                   9: 'Tensor'}
XDMF_IMAGE_TOPOLOGY = {'3DImage': '3DCoRectMesh', '2DImage': '2DCoRectMesh'}
XDMF_IMAGE_GEOMETRY = {'3DImage': 'ORIGIN_DXDYDZ', '2DImage': 'ORIGIN_DXDY'}

# Samples Group types global variables
SD_GROUP_TYPES = ['Group', '2DImage', '3DImage', '2DMesh', '3DMesh']
# name of all group names to store grid data in a SampleData instance
SD_GRID_GROUPS = ['2DImage', '3DImage', '2DMesh', '3DMesh', 'emptyImage',
                  'emptyMesh']
# name of image groups for various dimensionalities
# -1 is for the generic name of the group type
SD_IMAGE_GROUPS = {2:'2DImage', 3:'3DImage', 0:'emptyImage', -1:'Image'}
# name of image groups for various dimensionalities
# -1 is for the generic name of the group type
SD_MESH_GROUPS = {2:'2DMesh', 3:'3DMesh', 0:'emptyMesh', -1:'Mesh'}
# hdf5 node names reserved for meshes arrays
SD_RESERVED_NAMES = ['Nodes', 'Elements']

# usefull lists to parse keyword arguments
COMPRESSION_KEYS = ['complib', 'complevel', 'shuffle', 'bitshuffle',
                    'checksum', 'least_significant_digit',
                    'default_compression']

#### External software commands and pathes
# TODO : externalize in utils.SDZsetUtils
# Matlab software : alias or path to the Matlab software executable file
MATLAB = 'matlab'
MATLAB_OPTS = '-nodisplay -nosplash -nodesktop -r '
# F. Nugyen multiphase mesher
#  --> make sure that the mesher_file_dir contains all the matlab code required
#      to run the mesher, and that the mesher template contains a addpath
#      matlab command poiting towards the directory
mesher_file_dir = '/home/users02/amarano/Travail/Simulation/Multiphase_mesher/'
MESHER_TEMPLATE = mesher_file_dir+'multi_phase_mesh.m'
MESHER_TMP = mesher_file_dir+'Mesher_tmp.m'
CLEANER_TEMPLATE = mesher_file_dir+'morphological_image_cleaner.m'
CLEANER_TMP = mesher_file_dir+'Cleaner_tmp.m'