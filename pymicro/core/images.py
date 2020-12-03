#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image data handling package: utility package for SampleData class

@author: amarano
"""
import numpy as np
import h5py as h5

# =============================================================================
#  Utility classes
#   TODO : update documentation
#   TODO : Mesh/Image Objects and Readers verbosity
#   TODO : Crop image_object method
#   TODO : Construct Image_reader from SampleData
# =============================================================================

class ImageObject():
    """ Base class to store image data

        Attributes:
            - dimension [Nx,Ny,Nz] int64
                number of voxels for each dimension of the 3D image

            - origin [Ox, Oy, Oz] np.double
                coordinates of the origin (center of the [0,0,0] voxel)

            - spacing [Dx, Dy, Dz] np.double
                voxel size in each dimension

                 DEFAUT DIMENSIONS : voxels size [1,1,1] and origin at
                 [0.5, 0.5, 0.5] (first voxel center)
                 --> the first node of the first voxel is
                 the point with coordinates [0 0 0]

            - fields (list)
                contains a list of fields defined on the mesh. Each field is
                represented by a dictionnary : {field_name:field_values}
                field_values is a np.array containing the field nodel values
                corresponding to the nodes in MeshObject.nodes

    """

    def __init__(self,
                 dimension=[1, 1, 1],
                 origin=np.array([0., 0., 0.], dtype='double'),
                 spacing=np.array([1, 1, 1], dtype='double'),
                 **keywords):
        """ init Image Object with provided data or empty attributes

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
        self.dimension = dimension
        self.origin = origin
        self.spacing = spacing

        # mesh fields defaut empty instantiation
        self.fields = {}
        return

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
        # if (Field.shape != self.dimension):
        #     raise ValueError(''' The array given for the field shape is {}
        #                        which does not match ImageObject dimension : {}
        #                      '''.format(Field.shape, self.dimension))
        self.fields[Field_name] = Field
        return


class ImageReader():
    """ Class to read 3D image data from a file into an ImageObject

        Contains an ImageObject following hdf5/xdfm format conventions

        Currently supported file formats
            .mat (Matlab)

        Attributes:
            - filename (str)
                name/path to file containing the mesh data

            - Mesh (ImageObject)
                ImageObject class instance containing image data from file

    """

    def __init__(self,
                 image_filename,
                 **keywords):
        """ init ImageReader with filename and reading options

            Attributes:
                - mesh_filename (str)
                    path to the file to be read by the ImageReader.
                    Extension must be a known format

                - keywords
                    allow to pass options required to read data for some mesh
                    formats. See below

                Handled image formats:
                    - .mat
                    -----------
                        --> h5 file with unknown hierarchy, must be passed as
                            __init__method keywords for fields location in
                            hdf5 structure

                                Keywords required :
                                    * matlab_variables :  list of (str) indicating the
                                                   nodes storing the various
                                                   fields to read

                Other keywords
                ------------------

                        * dimension : [Nx,Ny,Nz]
                            number of voxels in each dimension

                        * spacing : [Dx,Dy,Dz]
                           size of voxels in each direction

                        * origin : [Ox, Oy, Oz]
                            coordinates of the center of the first
                            voxel in the image
        """

        self.filename = image_filename
        self.get_image_type()

        # empty Matlab read keywords init
        self.matlab_variables = []

        # instantiate Image Object
        self.image = ImageObject()

        # get image reading keywords
        for words in keywords:
            if (words == 'dimension'):
                self.image.dimension = keywords[words]
            elif (words == 'origin'):
                self.image.origin = keywords[words]
            elif (words == 'spacing'):
                self.image.spacing = keywords[words]
            elif (words == 'matlab_variables'):
                self.matlab_variables = keywords[words]
            elif (words == 'matlab_field_names'):
                self.matlab_field_names = keywords[words]

        # read image
        self.read_image()

        return

    def get_image_type(self):
        """ Find out mesh file format type from mesh file extension"""

        ext = self.filename[self.filename.rfind('.'):]

        #        print(' ImageReader : reading extension of file {} to find file format'.format(self.filename))
        if (ext == '.mat'):
            self.file_type = 'matlab'
        elif (ext == '.vtk'):
            self.file_type = 'vtk'
        else:
            print(' ImageReader : File format not handled for file {}'.format(self.filename))
        # print(' ImageReader : file format --> {}'.format(self.file_type))
        return

    def read_image(self):
        """ Call the appropriate read method for the considered 3D image format """
        if (self.file_type == 'matlab'):
            self.read_image_matlab()
        elif (self.file_type == 'vtk'):
            msg = (" vtk reading not implemented yet for ImageReader, no data has been read ")
            raise ValueError(msg)
        return

    def read_image_matlab(self):
        """ Read .mat formatted 3D image data """

        #        print(' ImageReader : reading 3D image data in {} ....'.
        #              format(self.filename))
        # open .mat file as hdf5 file
        matfile_data = h5.File(self.filename, 'r')

        # get image dimension
        if (self.image.dimension == [1, 1, 1]):
            Dim = matfile_data[self.matlab_variables[0]][()].shape
            self.image.dimension = Dim
        else:
            Dim = self.image.dimension

        # read fields
        for fieldloc, fieldname in zip(self.matlab_variables, self.matlab_field_names):
            # fieldname = fieldloc[fieldloc.rindex('/')+1:]
            field = matfile_data[fieldloc][()]
            #            print(' \t reading field in h5 node {} ....'.format(fieldloc))

            # check dimension
            loc_dim = field.shape
            if (loc_dim != Dim):
                raise ValueError(''' Dimension of the added field {} : {}, do not
                                 match image dimension {}
                                 '''.format(fieldname, loc_dim, Dim))
            self.image.add_field(field, fieldname)

        # close .mat file
        matfile_data.close()

        return