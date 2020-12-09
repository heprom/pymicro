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