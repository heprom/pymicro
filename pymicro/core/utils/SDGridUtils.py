#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SD Grid utils Module: utility methods for handling operations on Grids:
      - geometric operations
      - boolean operations
      - field transfer
"""

import numpy as np

# SD Amitex interface class
class SDMeshTools():
    """Mesh operations toolbox class.

    This class is designed to interact with SampleData Mesh data groups.
    Methods designed to compute a specific geometric or boolean operation
    on mesh groups, or to create specific element/node sets, should be
    implemented here.
    """

    @staticmethod
    def rescale_and_translate_mesh(data, meshname,
                                   mesh_new_origin=np.array([0.,0.,0.]),
                                   mesh_new_size=np.array([1.,1.,1.])):
        """Recompute mesh nodes coordinates to match new range and origin.

        The new mesh nodes coordinates are computed using a translation and
        a dilatation per dimension in order to make the node
        with the lowest coordinates coincide with the inputed new_origin,
        and to occupy an area of the inputed mesh size.

        The new coordinates will be computed as follows:
            translation:
            New_coordI = Old_coord - Old_coord.min + New_origin_coord
            dilatation
            New_coordI = CoordI * ( New_coord_range / Old_coord_range)



        :param data: SampleData instance containing the mesh to rescale.
        :type data: Sampledata instance
        :param meshname: Name, Indexname, Path or Alias for the mesh data
            group to rescale in the SampleData instance
        :type meshname: str
        :param mesh_new_origin: New coordinates of the lowest point in the
            mesh to rescale.
        :type mesh_new_origin: numpy.array (3,), default to [0.,0.,0.]
        :param mesh_new_size: New size of the mesh domain for each dimension.
        :type mesh_new_size: numpy.array (3,), default to [1.,1.,1.]
        """
        # get mesh nodes as hdf5 node (directly in memory data)
        # and rescale mesh
        nodes = data.get_mesh_nodes(meshname, as_numpy=False)
        # Translate mesh to origin 0,0,0
        nodes[:,0] = nodes[:,0] - nodes[:,0].min()
        nodes[:,1] = nodes[:,1] - nodes[:,1].min()
        if nodes.shape[1] == 3:
            nodes[:,2] = nodes[:,2] - nodes[:,2].min()
        # Get mesh new X,Y and Z sizes
        X_new_size = mesh_new_size[0]
        Y_new_size = mesh_new_size[1]
        if nodes.shape[1] == 3:
            Z_new_size = mesh_new_size[2]
        # Compute old mesh corrdinate range for each dimension
        X_size = nodes[:,0].max() - nodes[:,0].min()
        Y_size = nodes[:,1].max() - nodes[:,1].min()
        if nodes.shape[1] == 3:
            Z_size = nodes[:,2].max() - nodes[:,2].min()
        # Dilate mesh along each dimension to new mesh size
        nodes[:,0] = (nodes[:,0]/X_size)*X_new_size
        nodes[:,1] = (nodes[:,1]/Y_size)*Y_new_size
        if nodes.shape[1] == 3:
            nodes[:,2] = (nodes[:,2]/Z_size)*Z_new_size
        # Translate mesh
        nodes[:,0] = nodes[:,0] + mesh_new_origin[0]
        nodes[:,1] = nodes[:,1] + mesh_new_origin[1]
        if nodes.shape[1] == 3:
            nodes[:,2] = nodes[:,2] + mesh_new_origin[2]
        # flush new nodes in memory
        nodes.flush()
        return

