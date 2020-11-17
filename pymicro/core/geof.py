#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
     A package to interact with .geof file (mesh description files for FE soft
     Zset)

     Read and write usual commands used to describe FE meshes in .geof files
     (geometry, nodes, elements, sets....)

     IMPORTANT : Node ID convention
           Nodes are identified by their position in the array containing them
           within this module, hence with Python indexing (starting from 0)
           In geof files, indexing start from 1 (offset of 1 between the two)

Created on Mon Apr 27 13:55:54 2020
@author: amarano  --> aldo.marano@mines-paristech.fr
"""
import numpy as np
import warnings


# error class
class GeofReadError(Exception):
    """Exception raised when reading data from .geof file fails.

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message):
        self.message = message


def read_geof_nodes(file_name):
    """
        Return the coordinates of the nodes from a mesh defined in a .geof file

        Nodes coordinates are defined in the ***geometry/**node command of the
        .geof file

        Argument
        ---------
            file_name (str) : relative or full path name for .geof file

        Return
        --------
            Vertexes np.array(double) [Nvertexes,Dim] : coordinates of the mesh nodes
                                Nvertexes : number of nodes
                                      Dim : dimension of the mesh (1D, 2D, 3D)
    """

    with open(file_name, 'r') as f:
        print('\n...  Reading nodes defined in {} ...'.format(file_name))
        # find geometry command
        line = f.readline().strip()
        while line != '***geometry':
            line = f.readline().strip()
            if line == '':
                message = 'command ***geometry not found in {}'.format(
                        file_name)
                raise GeofReadError(message)
        # find node command
        while line != '**node':
            line = f.readline().strip()
            if line == '':
                message = 'command **node not found in {}'.format(file_name)
                raise GeofReadError(message)
        # get the dimension of the nodes coordinates table
        line = f.readline().strip()
        N_nodes, Dim = np.fromstring(line, dtype='int64', sep=' ')

        # initialize Nodes coordinates array
        Vertexes = np.zeros((N_nodes, Dim), dtype='double')

        for i in range(N_nodes):
            line = f.readline().strip()
            # first value in the line is the node id_number --> skipped
            Vertexes[i, :] = np.fromstring(line, dtype='double', sep=' ')[1:]

        # print informations about the read nodes
        print('\t {} nodes of dimension {} have been found in {}  \n'.format(
                N_nodes, Dim, file_name))

        return Vertexes


def read_geof_elements(file_name):
    """
        Return the connectivity arrays describing the elements forming the mesh
        defined in a .geof file

        Elements are defined in the ***geometry/**element command of the
        .geof file

        The function returns a list of connectivity matrices, one for each type
        of finite element found in the .geof file.

        Argument
        ---------
            file_name (str) : relative or full path name for .geof file

        Return
        --------
            - Element_list : list
                   contains one string indicating each element type found in
                   in the mesh description file (ex 's3d3, c3d4 ....' )

            - Element_connectivity : list of np.arrays.int64
                   For each element type in Element_list, contains the
                   connectivity matrix for all the elements of the considered
                   type. Each np.array has the shape (Nelements, Nvertexes)
                   where Nvertexes denotes the number of vertexes for the
                   considered element type (ex. 4 for tetraedra 'c3d4')
    """

    with open(file_name, 'r') as f:
        print('\n...  Reading elements defined in {} ...'.format(file_name))
        # find geometry command
        line = f.readline().strip()
        while line != '***geometry':
            line = f.readline().strip()
            if line == '':
                message = 'command ***geometry not found in {}'.format(
                        file_name)
                raise GeofReadError(message)
        # find element command
        while line != '**element':
            line = f.readline().strip()
            if line == '':
                message = 'command **element not found in {}'.format(file_name)
                raise GeofReadError(message)

        # get the dimension of the nodes coordinates table
        line = f.readline().strip()
        N_elem = np.fromstring(line, dtype='int64', sep=' ')[0]
        # initialize Nodes coordinates array
        Elements = []
        Elements_Id = []
        Element_type = []

        for i in range(N_elem):
            line = f.readline().split()
            # first value in the line is the element id_number --> skipped
            # second value in line is the string code for the element type
            # rest of the line is the element vertexes id_numbers
            Element_type.extend([line[1]])
            Elements_Id.extend([np.array(line[0]).astype('int64')])
            Elements.extend([np.array(line[2:]).astype('int64')])

        # identify the set of element type read
        Element_set = set(Element_type)

        # sort connectivity matrix by element type
        Element_list = []
        Element_connectivity = []

        # loop over all elements type in the set
        for x in Element_set:
            Element_list.append(x)
            count_elem = Element_type.count(x)

            # Assemble connectivity matrix of all elements of the type x in one
            # numpy array
            j = 0
            Temp_connectivity = np.zeros([count_elem, len(Elements[
                    Element_type.index(x)])]).astype('int64')

            for i in range(N_elem):
                if Element_type[i] == x:
                    Temp_connectivity[j, :] = Elements[i] - np.ones(
                            Elements[i].shape)
                    j = j+1

            Element_connectivity.append(Temp_connectivity)

            # print number of element found for each element type
            print('\t {} elements of type {} have been found in {} \n'.format(
                    count_elem, x, file_name))

        return Element_list, np.array(Elements_Id), Element_connectivity


def read_geof_nset(file_name):
    """
        Return the node sets defined for the mesh described in a .geof file

        Elements are defined in the ***group/**nset section of the
        .geof file

        Argument
        ---------
            file_name (str) : relative or full path name for .geof file

        Return
        --------
            - Nsets  : python dictionnary
                    dict keys are nset names and associated values are
                    np.arrays.int64 containing the associated nodes_id lists

    """

    with open(file_name, 'r') as f:
        print('\n...  Reading nsets defined in {} ...'.format(file_name))
        # find group command
        line = f.readline().strip()
        while (line != '***group'):
            line = f.readline().strip()
            if line == '':
                message = 'command ***group not found in {}'.format(file_name)
                raise GeofReadError(message)

#         initialize empty dictionnary
        Nsets = {}

        # initialize detection flag for ***return comand
        return_flag = False

        # loop to read all nset defined in .geof file
        while(True):
            # if line defines a nset : read
            if line[0:6] == '**nset':
                # The command line should be **nset nset_name.
                # The second element of line is thus the name of the nset.
                tmp_name = line.split()[1]
                # Initialize temp nset array.
                Nset_tmp = np.array([], dtype='int64')
                # while loop to read element id_numbers
                while(True):
                    line = f.readline().strip()
                    # Next line start with a new command or is end of file,
                    # break iterations.
                    if (line[0] == '*') or (line == ''):
                        break
                    # store nodes Id indicated on the present line
                    Nset_tmp = np.append(Nset_tmp, np.array(line.split()))

                # add nset to dictionnary
                Nsets[tmp_name] = Nset_tmp
            else:
                line = f.readline().strip()

            # end of nset read

            # break if end of file
            if (line == ''):
                break

            # break if ***return command is found
            if (line == '***return'):
                return_flag = True

        # print found elements sets
        for x in Nsets:
            print('\t Found nset "{}" with {} nodes'.format(x, len(Nsets[x])))
            print('\t\t first node : {} , last node : {} \n'.format(
                    Nsets[x][0], Nsets[x][-1]))

        if not(return_flag):
            warnings.warn('***return node not found for ***group command \
                          in {}'.format(file_name))

        return Nsets
