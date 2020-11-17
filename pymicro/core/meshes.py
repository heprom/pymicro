#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mesh data handling package: utility package for SampleData class

@author: amarano
"""
import numpy as np
import h5py as h5
import pymicro.core.geof as geof
# =============================================================================
#  Utility classes
#   TODO : update documentation
#   TODO : Integrate with Pymicro and Basic Tools codes for mesh handling
#   TODO : Mesh/Image Objects and Readers verbosity
#   TODO : Crop mesh_object method
#   TODO : Construct Mesh_reader from SampleData
# =============================================================================
class MeshObject():
    """ Base class to store mesh data

        Attributes:
            - nodes (np.Array float64, shape (Nnodes, mesh_dim= 1 or 2 or 3) )
                coordinates of mesh nodes

            - element_topology (list of [str])
                List of element types in the mesh.
                Each element is another list containing
                  [(str) 'element_topology', element_degree (int)]
                Element type is consistent with XDMF format

            - element_ID (list of [ np.array int64 shape (Nelem, 1) ])
                List of elements ID_numbers arrays in the mesh.
                Used for element sets definition
                One array for each element type in the mesh

            - element_connectivity (list of [np.array int64 shape (Nelem, NVertex)])
                List of numpy Integer arrays of element node connectivity list.
                      1 row = 1 element of element If_numbers forming the element
                One array for each element type in the mesh

            - fields (list)
                contains a list of fields defined on the mesh. Each field is
                represented by a dictionnary : {field_name:field_values}
                field_values is a np.array containing the field nodel values
                corresponding to the nodes in MeshObject.nodes
    """

    def __init__(self,
                 **keywords):
        """ init Mesh Object with provided data or empty attributes

            Attributes:
                - grid_type (str)
                    Type of grid. Supported : 'Uniform', 'Collection'

                - topology_type (str)
                    type of grid elements (only for Uniform grids)

                - **keywords
                    Not implemented yet. Intended to pass on Nodes and
                    Element arrays, Node and Element Sets Dictionnaries etc...

        """

        # mesh heavy data defaut emtpy instantiation
        self.nodes = np.array([], dtype='float64')
        self.element_topology = []
        self.element_Id = []
        self.element_connectivity = []

        # mesh sets defaut emtpy instantiation
        self.node_sets = {}

        # mesh fields defaut empty instantiation
        self.fields = {}

        return

    def add_nodes(self,
                  Nodes):
        """ Adds Nodes coordinates to the mesh

            Arguments:
                - Nodes (np.array - double - shape : NNodes, Dim)
                    coordinates arrays of the Nodes to add to the MeshObject
                    one line is one vertex
        """

        if (len(self.nodes) == 0):
            self.nodes = Nodes
        #            print('\t \t  Added {} nodes to the mesh'.format(len(Nodes)))
        #            print('\t \t \t First node  : {}'.format(
        #                    Nodes[0,:]))
        #            print('\t \t \t Last node  : {}'.format(
        #                    Nodes[-1,:]))
        else:
            if (self.nodes.shape[1] == Nodes.shape[1]):
                self.nodes = np.append(self.nodes, Nodes)
            #                print('\t \t Added {} nodes to the mesh'.format(len(Nodes)))
            #                print('\t \t \t First node  : {}'.format(
            #                        Nodes[0,:]))
            #                print('\t \t \t Last node   : {}'.format(
            #                        Nodes[-1,:]))
            else:
                raise ValueError(''' shape of added Nodes array {} do not
                                 match MeshObject.nodes shape {}
                                 '''.format(Nodes.shape, self.nodes.shape))

        return

    def add_elements(self,
                     Elements_topology,
                     Elements_Id,
                     Elements_connectivity):
        """ Add a set of element to the MeshObject

            Arguments:
                    - Elements_type list : [(str) , int]
                        List describing the type of elements in the element set
                        added to the MeshObject, according to XDMF format
                        elements naming conventions
                        string : element topology, int : element degree

                    - Elements_Id (np.array - int64 - shape : Nelements)
                        Integer array containing the Id numbe of the elements in
                        the element set to add to the MeshObject

                    - Elements_connectivity (np.array - int64 -
                                              shape : Nelements, NNodes)
                        Integer array containing the ocnnectivity matrix for the
                        element set to add to the MeshObject
                        Each line contains the Nodes Id forming the element

            NOTE : the correspondance between Id and connectivit arrays is
                   line by line. Elements_Id[K] is the ID of the element
                   composed of the Nodes Elements_connectivity[K,:]

        """
        self.element_topology.append(Elements_topology)
        self.element_Id.append(Elements_Id)
        self.element_connectivity.append(Elements_connectivity)
        #        print('\t \t Added {} {} elements of degree {} to the mesh'.format(
        #                Elements_Id.size,Elements_topology[0],Elements_topology[1]))
        #        print('\t \t \t First element nodes  : {}'.format(
        #                Elements_connectivity[0,:]))
        #        print('\t \t \t Last  element nodes  : {}'.format(
        #                Elements_connectivity[-1,:]))
        return

    def add_sets(self,
                 Node_sets={}):
        """ Adds nodes/elements sets to the MeshObject

            Arguments:
                - Node_sets (dict :  'nset name':np.array.int64)
                  dictionnary linking node set names (keys) to integer arrays
                  of the node set verteces ID numbers (values)
                  default : empty dict

        """
        self.node_sets.update(Node_sets)
        # print('\t Added the foolowing node sets to the mesh: ')

    #        for nset in Node_sets:
    #            print('\t\t - {}'.format(nset))
    #        return

    def add_field(self,
                  Field,
                  Field_name):
        """ add field nodal values to the MeshObject

            Arguments:
                - Field (np.array shape : Nnodes x Ncomponents)
                  ex: 3D vector field shape is Nnodes x 3
                  Array containing the nodal values of the field, defined on the
                  nodes coordinates contained in self.nodes

                - Field_name (str)
                    Name of the field

            Fields are added to the mesh object as a dict :
                field_name:field_values
        """

        # verify that number of field nodes and self.nodes are coherent
        if (Field.shape[0] != self.nodes.shape[0]):
            raise ValueError(''' The nodal value array given for the field
                              contains {} nodal values, which do not match the
                              number of nodes in the mesh object ({} nodes)
                             '''.format(Field.shape[0], self.nodes.shape[0]))

        # in case of a two dimensional array : reshape to 3dim array with
        # singleton last dimension
        if (Field.ndim < 2):
            Field = Field.reshape([Field.shape[0], 1])
        self.fields[Field_name]= Field
        #        print('\t field {} of dimension {} added to the mesh'.format(Field_name,
        #              Field.shape))
        return


class MeshReader():
    """ Class to read mesh data from a file into a mesh object

        Contains a mesh object following hdf5/xdfm format conventions

        Currently supported file formats
            .geof

        Attributes:
            - filename (str)
                name/path to file containing the mesh data

            - Geof_to_Xdmf_topology (dictionnary)
                translation table between element names in .geof and xdmf format

            - Mesh (MeshObject)
                mesh object class instance containing mesh data read from file

    """

    # dict for Geof to xdmf element topology naming convention
    Geof_to_Xdmf_topology = {
        'c2d3r': ['Triangle', 1],
        'c2d3': ['Triangle', 1],
        'c2d4r': ['Quadrilateral', 1],
        'c2d4': ['Quadrilateral', 1],
        'c2d6r': ['Tri_6', 2],
        'c2d6': ['Tri_6', 2],
        'c2d8r': ['Quad_8', 2],
        'c2d8': ['Quad_8', 2],
        's3d3r': ['Triangle', 1],
        's3d3': ['Triangle', 1],
        's2d4r': ['Quadrilateral', 1],
        's2d4': ['Quadrilateral', 1],
        's2d6r': ['Tri_6', 2],
        's2d6': ['Tri_6', 2],
        's2d8r': ['Quad_8', 2],
        's2d8': ['Quad_8', 2],
        'c3d4r': ['Tetrahedron', 1],
        'c3d4': ['Tetrahedron', 1],
        'c3d10r': ['Tetrahedron', 2],
        'c3d10': ['Tetrahedron', 2],
        'c3d6r': ['Wedge', 1],
        'c3d6': ['Wedge', 1],
        'c3d15r': ['Wedge_15', 1],
        'c3d15': ['Wedge_15', 1],
        'c3d8r': ['Hexahedron', 1],
        'c3d8': ['Hexahedron', 1],
        'c3d20r': ['Hex_20', 2],
        'c3d20': ['Hex_20', 2]
    }

    def __init__(self,
                 mesh_filename,
                 **keywords
                 ):
        """ init MeshReader with filename and reading options

            Attributes:
                - mesh_filename (str)
                    path to the mesh file to be read by the MeshReader. Extension
                    must be a known format

                - keywords
                    allow to pass options required to read data for some mesh
                    formats. See below

                Handled mesh formats:
                    - .geof
                    -----------
                        --> only geometrical mesh description data
                        --> all information is in file, no keywords required

                    - .mat
                    -----------
                        --> geometrical and/or field data
                        --> h5 file with unknown hierarchy, must be passed as
                            __init__method keywords for nodes, elements and
                            fields location in hdf5 structure

                                Keywords required :
                                    * matlab_mesh_nodes : Hdf5 node storing the mesh
                                                  nodes coordinates (str)

                                    * matlab_mesh_elements : Hdf5 node storing the mesh
                                                    element connectivity array
                                                    (str)

                                    * matlab_mesh_element_type (list)
                                            two elements [0] string indicating
                                            elements topology
                                            [1] int indicating elements degree
                                            See Geof_to_Xdmf_topology Dict in
                                            the class for a comprehensive list

                                    * matlab_variables :  list of (str) indicating the
                                                   nodes storing the various
                                                   fields to read

                                    * matlab_mesh_transpose : True if data need
                                        to be transposed
                    - .vtk
                    ----------

        """
        self.filename = mesh_filename
        self.get_mesh_type()

        # empty Matlab read keywords init
        self.matlab_mesh_nodes = ''
        self.matlab_mesh_elements = ''
        self.matlab_mesh_element_type = ''
        self.matlab_variables = []
        self.matlab_mesh_transpose = False

        # instantiate mesh object
        self.mesh = MeshObject()

        # get mesh reading keywords
        for words in keywords:
            if (words == 'matlab_mesh_nodes'):
                self.matlab_mesh_nodes = keywords[words]
            elif (words == 'matlab_mesh_elements'):
                self.matlab_mesh_elements = keywords[words]
            elif (words == 'matlab_variables'):
                self.matlab_variables = keywords[words]
            elif (words == 'matlab_mesh_element_type'):
                self.matlab_mesh_element_type = keywords[words]
            elif (words == 'matlab_mesh_transpose'):
                self.matlab_mesh_transpose = True

        # get mesh data
        self.read_mesh()

        return

    #############################

    def get_mesh_type(self):
        """ Find out mesh file format type from mesh file extension"""

        ext = self.filename[self.filename.rfind('.'):]

        #        print(' MeshReader : reading extension of file {} to find file format'.format(self.filename))
        if (ext == '.mat'):
            self.file_type = 'matlab'
        elif (ext == '.geof'):
            self.file_type = 'geof'
        elif (ext == '.vtk'):
            self.file_type = 'vtk'
        else:
            print(' MeshReader : File format not handled for file {}'.format(self.filename))
        # print(' MeshReader : file format --> {}'.format(self.file_type))
        return

    #############################

    def read_mesh(self):
        """ Call the appropriate read method for the considered mesh format """

        if (self.file_type == 'matlab'):
            self.read_mesh_matlab()
        elif (self.file_type == 'geof'):
            self.read_geof_mesh()
        elif (self.file_type == 'vtk'):
            msg = (" vtk reading not implemented yet for MeshReader, no data has been read ")
            raise ValueError(msg)
        return

    def read_mesh_matlab(self):
        """ Read .mat formatted mesh data """

        # verification of Matlab read keywords
        if (self.matlab_mesh_nodes == ''):
            raise ValueError(''' Matlab mesh nodes hdf5 location not specified,
                                 got empty location ''')

        if (self.matlab_mesh_elements == ''):
            raise ValueError(''' Matlab mesh elements hdf5 location not
                                 specified, got empty location ''')

        if (self.matlab_mesh_element_type == []):
            raise ValueError(''' Matlab mesh element type not specified,
                                 got empty list''')

        #        print(' MeshReader : reading mesh in {} ....'.format(self.filename))
        # open .mat file as hdf5 file
        matfile_data = h5.File(self.filename, 'r')

        # read mesh nodes
        #        print(' \t reading mesh nodes in h5 node {} ....'.format(
        #                self.matlab_mesh_nodes))
        Nodes = np.array(matfile_data[self.matlab_mesh_nodes][()]).astype('double')
        if (self.matlab_mesh_transpose):
            Nodes = Nodes.transpose()
        self.mesh.add_nodes(Nodes)

        # read mesh elements
        #        print(' \t reading mesh elements in h5 node {} ....'.format(
        #                self.matlab_mesh_elements))
        Connectivity = np.array(matfile_data[self.matlab_mesh_elements][()]).astype('int64')
        # correction accounting for different indexing convention in Matlab
        Connectivity = Connectivity - np.ones(Connectivity.shape, dtype='int64')
        if (self.matlab_mesh_transpose):
            Connectivity = Connectivity.transpose()
        Nelements = Connectivity.shape[0]
        self.mesh.add_elements(self.matlab_mesh_element_type, np.arange(Nelements),
                               Connectivity)

        # read fields
        for fieldloc in self.matlab_variables:
            fieldname = fieldloc[fieldloc.rindex('/') + 1:]
            field = matfile_data[fieldloc][()]
            #            print(' \t reading field in h5 node {} ....'.format(fieldloc))
            self.mesh.add_field(field, fieldname)
        # close .mat file
        matfile_data.close()
        return

    def read_geof_mesh(self):
        """ Read .geof formatted meshes with .geof package support """

        #        print(' MeshReader : reading geof mesh  in {} ....'.format(self.filename))
        # read mesh nodes
        self.mesh.add_nodes(geof.read_geof_nodes(self.filename))

        # read mesh elements and translate type into xdmf format
        TopologyType, Elements_Id, Element_connectivity = \
            geof.read_geof_elements(self.filename)
        for j in range(len(TopologyType)):
            self.mesh.add_elements(Elements_topology= \
                                       self.Geof_to_Xdmf_topology[TopologyType[j]],
                                   Elements_Id=Elements_Id[j],
                                   Elements_connectivity=Element_connectivity[j])

        # read nsets
        self.mesh.add_sets(Node_sets=geof.read_geof_nset(self.filename))
        return