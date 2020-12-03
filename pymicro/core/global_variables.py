#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module to hold global variables for the `core` package.
"""

# Module global variables for xdmf compatibility
XDMF_FIELD_TYPE = {1: 'Scalar', 2: 'Vector', 3: 'Vector', 6: 'Tensor6',
                   9: 'Tensor'}
XDMF_CENTERS = {'3DImage': 'Cell', '2DImage': 'Cell', 'Mesh': 'Node'}
XDMF_IMAGE_TOPOLOGY = {'3DImage': '3DCoRectMesh', '2DImage': '2DCoRectMesh'}
XDMF_IMAGE_GEOMETRY = {'3DImage': 'ORIGIN_DXDYDZ', '2DImage': 'ORIGIN_DXDY'}

# Samples Group types global variables
SD_GROUP_TYPES = ['Group', '2DImage', '3DImage', 'Mesh']
SD_GRID_GROUPS = ['2DImage', '3DImage', 'Mesh']
SD_IMAGE_GROUPS = ['2DImage', '3DImage']
SD_RESERVED_NAMES = ['Nodes', 'Elements'] # hdf5 node names reserved for meshes

#

# usefull lists to parse keyword arguments
COMPRESSION_KEYS = ['complib', 'complevel', 'shuffle', 'bitshuffle',
                    'checksum', 'least_significant_digit',
                    'default_compression']