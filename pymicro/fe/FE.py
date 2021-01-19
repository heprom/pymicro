"""
The FE module allows some simple operations on FE calculations.
Import/Export are partly supported, in particular from/to the vtk
file format.
In addition utilities are available to:
* read a .ut file to query the fields stored in .integ and .node files
and also the list of the cards available
* retreive the field (or single value in result file)
integ file is easy to read: cards -> elements -> fields > integ points
"""
import os
import sys
import vtk
import numpy
import math


class FE_Calc():
    '''This class is used to manipulate a finite element calculation.'''

    def __init__(self, prefix='mesh', wdir='.'):
        '''
        initialize with a name. If not specified, the name will be 'mesh'.

        NB: right now the class is Zset oriented but could be more general.
        one could read a calculation from abaqus, zset or even craft his own.
        '''
        self._name = prefix
        self._wdir = wdir
        self.U = None
        self.nodal_vars = []
        self.nodal_fields = []
        self.integ_vars = []
        self.integ_fields = []
        self.times = []
        self._mesh = FE_Mesh()

    def __repr__(self):
        ''' Gives a string representation of the zset_mesh instance.'''
        out = '%s FE calcultion\n' % (self.__class__.__name__)
        out += 'working directory = %s\n' % self._wdir
        out += 'node vars = %s\n' % self.nodal_vars.__repr__()
        out += 'integ vars = %s\n' % self.integ_vars.__repr__()
        # out += 'times = %s\n' % self.times.__repr__()
        out += 'mesh: %s' % self._mesh.__repr__()
        return out

    def read_ut(self):
        ut = open(os.path.join(self._wdir, self._name + '.ut'))
        reading_cards = False
        for line in ut:
            if reading_cards:
                time = numpy.float(line.split()[4])
                print('reading card, time=', time)
                self.avail_times.append(time)
            elif line.startswith('**meshfile'):
                meshfile = line.split()[1]
                print('meshfile is', meshfile)
            elif line.startswith('**node'):
                self.avail_node_vars = line.split()[1:]
                print('node variables are', self.nodal_vars)
            elif line.startswith('**integ'):
                self.avail_integ_vars = line.split()[1:]
                print('integ variables are', self.integ_vars)
            elif line.startswith('**element'):
                reading_cards = True
                self.avail_times = []
        ut.close()
        mesh = FE_Mesh.load_from_geof(os.path.join(self._wdir, meshfile))
        self._mesh = mesh  # must be a FE_Mesh instance

    def get_name(self):
        return self._name

    def set_mesh(self, mesh):
        self._mesh = mesh

    def add_integ_field(self, field_name, field):
        '''Add integ field to the list.
        
        :param str field_name: The field name.
        :param ndarray field: The field data.
        '''
        self.integ_vars.append(field_name)
        self.integ_fields.append(field)

    def add_nodal_field(self, field_name, field):
        '''Add nodal field to the list.
        
        :param str field_name: The field name.
        :param ndarray field: The field data.
        '''
        self.nodal_vars.append(field_name)
        self.nodal_fields.append(field)

    def read_ip_values(self, card, field, el_rank, verbose=False):
        '''Read the values of the given element for the specified field
        at integration point and for the given card.

        An array with all integration point values is returned.
        '''
        integ = open(os.path.join(self._wdir, self._name + '.integ'), 'rb')
        offset = (card - 1) * len(self.avail_integ_vars) * self._mesh.get_number_of_gauss_points() * 4
        # increase offset to look at the element
        for i, el in enumerate(self._mesh._elements[:el_rank]):
            nip = el.get_number_of_gauss_points()
            # integ data is stored as Big Endian 32 bit floats (4 bytes)
            bits_to_read = nip * len(self.avail_integ_vars) * 4
            offset += bits_to_read
        # now retrieve the float32 values for this element
        integ.seek(offset)
        el = self._mesh._elements[el_rank]
        nip = el.get_number_of_gauss_points()
        # ip_values = numpy.empty(el.get_number_of_gauss_points(), dtype=np.float32)
        bits_to_read = nip * len(self.avail_integ_vars) * 4
        dt = numpy.dtype('>f4')
        float_data = numpy.fromstring(integ.read(bits_to_read), dt).astype(numpy.float32)
        element_data = numpy.reshape(float_data, (len(self.avail_integ_vars), nip), order='C')
        integ.close()
        return element_data[self.avail_integ_vars.index(field), :]

    def read_integ(self, card, field, verbose=False):
        '''Read field data in a Z-set integ file.
        
        Read the specified field at integration point and for the given
        card.

        :param int card: the card number (starts at 1).
        :param str field: the string describing the field to read.
        :param bool verbose: flag to activate verbose mode.
        :returns: returns the integ field as a 1D numpy array.

        .. note::

          Integ file can be read as: cards -> elements -> fields > integ points
          the card parameter numbering starts at 1 (just as in the .ut file).

        .. warn::
        
          VTK cells can only handle one value per cell so for each cell, all integration point values are averaged.

        '''
        integ = open(os.path.join(self._wdir, self._name + '.integ'), 'rb')
        offset = (card - 1) * len(self.avail_integ_vars) * self._mesh.get_number_of_gauss_points() * 4
        if verbose:
            print('reading field in integ file with offset', offset)
            print('quering field %s which has number %d' % (field, self.avail_integ_vars.index(field)))
        component = numpy.empty(self._mesh.get_number_of_elements())
        integ.seek(offset)
        for i, el in enumerate(self._mesh._elements):
            nip = el.get_number_of_gauss_points()
            # integ data is stored as Big Endian 32 bit floats (4 bytes)
            bits_to_read = nip * len(self.avail_integ_vars) * 4
            dt = numpy.dtype('>f4')
            float_data = numpy.fromstring(integ.read(bits_to_read), dt).astype(numpy.float32)
            element_data = numpy.reshape(float_data, (len(self.avail_integ_vars), nip), order='C')
            component[i] = numpy.mean(element_data[self.avail_integ_vars.index(field), :])
        integ.close()
        return component

    def read_nodal(self, card, field, verbose=False):
        '''Read field data in a Z-set node file.

        Read the specified nodal field for the given card (ie time increment).

        :param int card: the card number (starts at 1).
        :param str field: the string describing the field to read.
        :param bool verbose: flag to activate verbose mode.
        :returns: returns the nodal field as a 1D numpy array.

        .. note::

          Node file can be read as: cards -> fields > nodes
          the card parameter numbering starts at 1 (just as in the .ut file).
        '''
        nodal = open(os.path.join(self._wdir, self._name + '.node'), 'rb')
        index_field = self.avail_node_vars.index(field)
        offset = ((card - 1) * len(self.avail_node_vars) + index_field) * self._mesh.get_number_of_nodes() * 4
        if verbose:
            print('reading field in node file with offset', offset)
            print('quering field %s which has number %d' % (field, self.avail_node_vars.index(field)))
        #nodal_field = numpy.empty(self._mesh.get_number_of_nodes())
        nodal.seek(offset)
        bits_to_read = self._mesh.get_number_of_nodes() * 4
        dt = numpy.dtype('>f4')
        nodal_field = numpy.fromstring(nodal.read(bits_to_read), dt).astype(numpy.float32)
        nodal.close()
        if verbose:
            print('read nodal field %s, range is [%.4f - %.4f]' % (field, nodal_field.min(), nodal_field.max()))
        return nodal_field

    def read_displacement_field(self, card, field_names=['U1', 'U2', 'U3']):
        from vtk.util import numpy_support
        dim = len(field_names)  # should be 2 or 3
        vtk_data_array = vtk.vtkFloatArray()
        vtk_data_array.SetName('U')
        vtk_data_array.SetNumberOfComponents(dim)
        vtk_data_array.SetNumberOfTuples(self._mesh.get_number_of_nodes())
        for i in range(dim):
            # read the individual displacement components
            print('reading displacement component %s' % field_names[i])
            Ui = self.read_nodal(card, field_names[i], verbose=False)
            vtk_data_array.CopyComponent(i, numpy_support.numpy_to_vtk(numpy.ravel(Ui, order='F'), deep=1), 0)
        self.U = vtk_data_array
        return vtk_data_array

    def build_vtk(self):
        print('building vtk stuff for FE_calc')
        vtk_mesh = self._mesh.build_vtk()
        # also store some meta data
        #model = vtk.vtkModelMetadata()
        #model.SetTitle(self._name)
        #model.Pack(vtk_mesh)
        #print('grid has meta data ?', vtk.vtkModelMetadata.HasMetadata(vtk_mesh))
        from vtk.util import numpy_support
        # add the displacement field if present
        if self.U:
            vtk_mesh.GetPointData().AddArray(self.U)
        # one point data array for each nodal field
        for i, field_name in enumerate(self.nodal_vars):
            print('adding nodal field', field_name)
            vtk_data_array = numpy_support.numpy_to_vtk(self.nodal_fields[i], deep=1)
            vtk_data_array.SetName(field_name)
            vtk_mesh.GetPointData().AddArray(vtk_data_array)
        # add one cell data array for each field
        for i, field_name in enumerate(self.integ_vars):
            print('adding integ field', field_name)
            vtk_data_array = numpy_support.numpy_to_vtk(self.integ_fields[i], deep=1)
            vtk_data_array.SetName(field_name)
            vtk_mesh.GetCellData().AddArray(vtk_data_array)
        if len(self.integ_vars) > 0:
            vtk_mesh.GetCellData().SetActiveScalars(self.integ_vars[0])
        return vtk_mesh

    @staticmethod
    def make_vtu(inp):
        calc = FE_Calc(prefix=inp)
        calc.read_ut()
        # now output a .vtu file
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(calc.get_name() + '.vtu')
        if vtk.vtkVersion().GetVTKMajorVersion() > 5:
            writer.SetInputData(calc.build_vtk())
        else:
            writer.SetInput(calc.build_vtk())
        writer.Write()


class FE_Mesh():
    '''This class is used to represent a finite element mesh.'''

    def __init__(self, dim=3):
        '''
        Create an empty mesh.
        '''
        self._dim = dim
        self._nodes = []
        self._elements = []
        self._nip = 0
        self._elsets = [self._elements]
        self._elset_names = ['ALL_ELEMENT']
        self._lisets = []
        self._liset_names = []

    def __repr__(self):
        ''' Gives a string representation of the FE_Mesh instance.'''
        out = '%s mesh\n' % self.__class__.__name__
        out += 'dimension = %d\n' % self._dim
        out += 'nb of nodes = %d\n' % self.get_number_of_nodes()
        out += 'nb of elements = %d\n' % self.get_number_of_elements()
        out += 'list of elsets:' + self._elset_names.__repr__()
        if len(self._lisets) > 0:
            out += 'list of lisets:' + self._liset_names.__repr__()
        return out

    def locate_element(self, elid):
        """Locate an element given its id.

        :param int elid: the element id.
        :return: The `FE_Element` instance.
        """
        # first try rank elid -1
        if self._elements[elid - 1].give_id() == elid:
            return self._elements[elid - 1]
        else:
            for el in self._elements:
                if el.give_id() == elid:
                    return el
        raise ValueError('element %d not found in this mesh' % elid)

    def delete_elset(self, elset_name):
        """Method to delete elements belonging to an elset.

        After deleting all the element, the ranks are updated.

        :param str elset_name: the name of the elset to delete.
        """
        print('deleting elset %s' % elset_name)
        index = self._elset_names.index(elset_name)
        elid_to_del = self._elsets[index]
        for elid in elid_to_del:
            el = self.locate_element(elid)
            self._elements.remove(el)
        print('%d elements deleted' % len(elid_to_del))
        print('updating ranks')
        for i, element in enumerate(self._elements):
            element._rank = i
        self._elsets.pop(index)
        self._elset_names.pop(index)
        for elset in self._elsets:
            elset = list(set(elset) - set(elid_to_del))
        print(self._elset_names)

    def delete_orphan_nodes(self):
        """Detect and delete node not belonging to any element."""
        # build a list of all used node ids
        node_ids = []
        [node_ids.extend([node.give_id() for node in element._nodelist]) for element in self._elements]
        node_ids = numpy.unique(node_ids)
        removed_node_nb = len(self._nodes)
        # keep only nodes with id in the list
        self._nodes = [node for node in self._nodes if node.give_id() in node_ids]
        removed_node_nb -= len(self._nodes)
        # recompute node rank
        self.compute_id_to_rank(nodes=True)
        print('%d nodes were removed' % removed_node_nb)

    def compute_elset_center_of_mass(self, elset_id=0):
        """Method to compute the center of mass of a given elset.

        :param int elset_id: the id of the elset to consider.
        :return: the center of mass of this elset.
        """
        print('computing center of mass for elset %s' % self._elset_names[elset_id])
        id_to_rank = self.compute_id_to_rank(nodes=False)
        com = 0.
        for i in range(len(self._elsets[elset_id])):
            com += self._elements[id_to_rank[self._elsets[elset_id][i]]].get_center_of_mass()
        com /= len(self._elsets[elset_id])
        return com

    @staticmethod
    def make_vtu(path, add_elset_id_field=False, elset_prefix='_ELSET'):
        '''Convert a mesh to vtk format.
        
        This method reads the mesh and then write the corresponding .vtu file.
        The file will have the same name as the input mesh, with the extension changed to vtu.
        Only .geof and .mesh file are currently supported.

        :param str path: path to the mesh file.
        :param bool add_elset_id_field: flag to add a field representing the elset id.
        :param str elset_prefix: prefix to use if computing the leset id field.
        '''
        if path.endswith('.geof'):
            m = FE_Mesh.load_from_geof(path)
        elif path.endswith('.mesh'):
            m = FE_Mesh.load_from_mesh(path)
        else:
            print('unsupported mesh format (must be .geof or .mesh): %s' % path)
            return
        vtk_mesh = m.build_vtk()
        if add_elset_id_field:
            from vtk.util import numpy_support
            id_field = m.compute_elset_id_field(elset_prefix)
            print('adding field %s' % 'elset_id')
            vtk_data_array = numpy_support.numpy_to_vtk(id_field, deep=1)
            vtk_data_array.SetName('elset_id')
            vtk_mesh.GetCellData().AddArray(vtk_data_array)
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(path[:-5] + '.vtu')
        if vtk.vtkVersion().GetVTKMajorVersion() > 5:
            writer.SetInputData(vtk_mesh)
        else:
            writer.SetInput(vtk_mesh)
        writer.Write()

    def get_number_of_nodes(self):
        '''Return the total number of nodes in the mesh.'''
        return len(self._nodes)

    def get_number_of_elements(self):
        '''Return the total number of elements in the mesh.'''
        return len(self._elements)

    def update_number_of_gauss_points(self):
        '''Compute the total number of integration points within the mesh. '''
        nip = 0
        for el in self._elements:
            nip += el.get_number_of_gauss_points()
        self._nip = nip

    def get_number_of_gauss_points(self):
        '''Return the total number of integration point in the mesh.
        
        .. note::
        
           If you made some changes to the mesh, you should call `update_number_of_gauss_points`
           before reading that value.
        '''
        return self._nip

    def get_bounds(self):
        """Return the min and max bounds of the mesh."""
        xmin = min([node._x for node in self._nodes])
        ymin = min([node._y for node in self._nodes])
        zmin = min([node._z for node in self._nodes])
        xmax = max([node._x for node in self._nodes])
        ymax = max([node._y for node in self._nodes])
        zmax = max([node._z for node in self._nodes])
        return xmin, ymin, zmin, xmax, ymax, zmax

    @staticmethod
    def load_from_mesh(mesh_path, verbose=False):
        '''
        Creates a mesh instance from an ascii .mesh file (INRIA file format).

        :param str mesh_path: path to the file to read.
        :param bool verbose: flag to activate verbose mode.
        :returns: a FE_Mesh instance.
        '''
        mesh = open(mesh_path, 'r')
        # look for Dimension
        while (True):
            line = mesh.readline().strip()  # get read of unnecessary spaces
            if line.startswith('Dimension'):
                break
        dim = int(mesh.readline())
        print('dimension is %d' % dim)
        fe_mesh = FE_Mesh(dim=dim)
        # look for Vertices
        while True:
            line = mesh.readline().strip()  # get read of unnecessary spaces
            if line.startswith('Vertices'):
                break
        nv = int(mesh.readline())
        percent = math.ceil(nv / 100.)
        print('loading nodes: %2d %%' % (0 / percent))
        assert (dim == 3)  # assume 3 dimensional mesh
        for i in range(nv):
            if i % percent == 0:
                print('\b\b\b\b%2d %%' % (i / percent))
            [x, y, z, t] = mesh.readline().split()
            node = FE_Node(int(i + 1))
            node._x = float(x)
            node._y = float(y)
            node._z = float(z)
            node._rank = i
            if verbose:
                print('adding node', node)
            fe_mesh._nodes.append(node)
        # look for Triangles
        while True:
            line = mesh.readline().strip()  # get read of unnecessary spaces
            if line.startswith('Triangles'):
                break
        nt = int(mesh.readline())
        percent = math.ceil(nt / 100.)
        print('building triangles: %2d %%' % (0 / percent))
        for i in range(nt):
            if i % percent == 0:
                print('\b\b\b\b%2d %%' % (i / percent))
            line = mesh.readline()
            tokens = line.split()
            el_id = i + 1
            el_type = 's3d3'
            el_node_nb = 3
            element = FE_Element(el_id, el_type)
            element._rank = i
            for n in range(el_node_nb):
                # here rank is always id - 1
                element._nodelist.append(fe_mesh._nodes[int(tokens[n]) - 1])
            if verbose:
                print('adding element', element)
            fe_mesh._elements.append(element)
        mesh.close()
        return fe_mesh

    @staticmethod
    def load_from_geof(geof_path, verbose=False):
        '''
        Creates a mesh instance from a Z-set ascii geof file (binary is not supported).
        
        :param str geof_path: path to the file to read.
        :param bool verbose: flag to activate verbose mode.
        :returns: a FE_Mesh instance.
        '''
        geof = open(geof_path, 'r')
        geof.readline()
        # look for **node
        while True:
            line = geof.readline().strip()  # get read of unnecessary spaces
            if line.startswith('**node'):
                break
        [snv, sdim] = geof.readline().split()
        nv = int(snv)
        dim = int(sdim)
        fe_mesh = FE_Mesh(dim=dim)
        for i in range(nv):
            if dim == 2:
                [id, x, y] = geof.readline().split()
            else:
                [id, x, y, z] = geof.readline().split()
            node = FE_Node(int(id))
            node._x = float(x)
            node._y = float(y)
            if dim == 3:
                node._z = float(z)
            node._rank = i
            if verbose:
                print('adding node', node)
            fe_mesh._nodes.append(node)
        # build a rank <-> id table
        id_to_rank = fe_mesh.compute_id_to_rank(nodes=True)
        '''
        print('id_to_rank table size is %d' % max_node_id)
        id_to_rank = numpy.zeros(1+max_node_id, dtype=int)
        for node in fe_mesh._nodes:
          id_to_rank[node._id] = node._rank
        '''
        # look for **element
        while True:
            line = geof.readline().strip()
            if line.startswith('**element'):
                break
        ne = int(geof.readline())
        percent = math.ceil(ne / 100.)
        print('building elements: %2d %%' % (0 / percent))
        for i in range(ne):
            if i % percent == 0:
                print('\b\b\b\b%2d %%' % (i / percent))
            line = geof.readline()
            tokens = line.split()
            el_id = int(tokens[0])
            el_type = tokens[1]
            el_node_nb = int(el_type[3:].split('_')[0].split('r')[0])
            if (el_type not in ['c2d3', 's3d3', 'c3d4', 'c3d6', 'c3d20',
                                'c3d20r', 'c3d15', 'c3d13', 'c3d10', 'c3d10_4',
                                'c3d8', 'c3d8r', 'c2d4', 'c2d8', 'c2d8r']):
                print('error, element type %s is not supported yet' % el_type)
                continue
            element = FE_Element(el_id, el_type)
            element._rank = i
            for n in range(el_node_nb):
                element._nodelist.append(fe_mesh._nodes[id_to_rank[int(tokens[n + 2])]])
            if verbose:
                print('adding element', element)
            fe_mesh._elements.append(element)
        # look for ***group
        while True:
            line = geof.readline().strip()
            print(line)
            if line.startswith('***group'):
                break
        # look for ***return
        line = geof.readline()
        while True:
            if line.startswith('**elset'):
                elset_name = line.split()[1]
                if elset_name == 'ALL_ELEMENT':
                    line = geof.readline()
                    continue  # already stored as the first elset
                new_elset = []
                while (True):
                    line = geof.readline()
                    if line.startswith('*'):
                        break  # escape if entering anoter group
                    for elid in line.split():
                        new_elset.append(int(elid))
                if fe_mesh._elset_names.count(elset_name) == 0:
                    fe_mesh._elset_names.append(elset_name)
                    print('adding new elset: %s' % elset_name)
                    fe_mesh._elsets.append(new_elset)
                else:
                    index = fe_mesh._elset_names.index(elset_name)
                    print('appending element ids to elset ' + elset_name)
                    for el_id in new_elset:
                        fe_mesh._elsets[index].append(el_id)
                print('nb of elsets currently in mesh:', len(fe_mesh._elsets))
            elif line.startswith('**liset'):
                liset_name = line.split()[1]
                new_liset = []
                while True:
                    line = geof.readline()
                    print(len(line), line == '\n', line)
                    if line.startswith('*') or line == ('\n'):
                        break  # escape if entering anoter group
                    tokens = line.split()
                    if tokens[0] == 'line':
                        new_liset.append([int(tokens[1]), int(tokens[2])])
                    elif tokens[0] == 'quad':
                        new_liset.append([int(tokens[1]), int(tokens[3])])
                if fe_mesh._liset_names.count(liset_name) == 0:
                    fe_mesh._liset_names.append(liset_name)
                    print('adding new liset: %s' % liset_name)
                    fe_mesh._lisets.append(new_liset)
            if line.startswith('***return'):
                break
            if not line.startswith('**elset'):
                line = geof.readline()
        fe_mesh.update_number_of_gauss_points()
        geof.close()
        return fe_mesh

    def save_to_geof(self, file_name):
        """Function to save a FE_Mesh instance to Z-set geof ascii format.

        :param str file_name: the name of the file to use.
        """
        if not file_name.endswith('.geof'):
            file_name += '.geof'
        print('saving mesh to file %s' % file_name)
        f = open(file_name, 'w')
        # write nodes
        f.write('***geometry\n')
        f.write('**node\n')
        f.write('%d %d\n' % (len(self._nodes), self._dim))
        for node in self._nodes:
            f.write('%d %f %f %f\n' % (node.give_id(), node._x, node._y, node._z))
        f.write('**element\n')
        f.write('%d\n' % len(self._elements))
        # write elements
        for element in self._elements:
            s = '%d %s' % (element.give_id(), element._type)
            for node in element._nodelist:
                s += ' %d' % node.give_id()
            s += '\n'
            f.write(s)
        # write elsets
        f.write('***group\n')
        for i in range(len(self._elset_names)):
            if self._elset_names[i] == 'ALL_ELEMENT':
                # skip this one
                continue
            elset = self._elsets[i]
            f.write('**elset %s\n' % self._elset_names[i])
            for el_id in elset:
                f.write(' %d' % el_id)
            f.write('\n')
        f.write('***return\n')
        f.close()

    def translate_mesh(self, T):
        """Translate a mesh by modifying the nodes coordinates."""
        assert len(T) == self._dim
        print('translating mesh')
        for node in self._nodes:
            node._x += T[0]
            node._y += T[1]
            if self._dim == 3:
                node._z += T[2]

    def rotate_mesh(self, R):
        """Rotate a mesh by transforming the nodes coordinates using a rotation matrix."""
        assert R.shape[0] == R.shape[1]
        assert R.shape[0] == self._dim
        print('rotating mesh')
        for node in self._nodes:
            new_position = numpy.dot(R, [node._x, node._y, node._z])
            node._x = new_position[0]
            node._y = new_position[1]
            node._z = new_position[2]

    def compute_id_to_rank(self, nodes=True):
        if nodes:
            the_list = self._nodes
        else:
            the_list = self._elements
        max_id = 0
        for thing in the_list:
            if thing._id > max_id:
                max_id = thing._id
        id_to_rank = numpy.zeros(1 + max_id, dtype=int)
        if max_id > 10 ** 8:
            print('maximum id is %d, consider renumbering your mesh entities' % max_id)
            sys.exit(1)
        for thing in the_list:
            id_to_rank[thing._id] = thing._rank
        return id_to_rank

    def compute_elset_id_field(self, elset_prefix=None, use_name_as_id=False):
        '''Compute a new field showing to which elset the element
        belongs. Note this suppose elsets are mutually exclusive (except
        for the very first one ALL_ELEMENT which is disregarded here).
        
        :param str elset_prefix: a prefix to filter the elsets to consider.
        :param bool use_name_as_id: a flag to use the elset name with the prefix removed as id.
        :returns: the elset id field as a numpy array.
        '''
        if elset_prefix:
            elset_list = list(filter(lambda k: elset_prefix in k, self._elset_names))
        else:
            elset_list = self._elset_names[1:]
        # figure out the depth of the field to use
        names = [s for s in self._elset_names if elset_prefix in s]
        names.sort()
        max_id = int(names[-1].split(elset_prefix)[1])
        if max_id > 255:
            print('warning, max id larger than 255, using a uint16 field')
            elset_id = numpy.zeros(self.get_number_of_elements(), dtype=numpy.uint16)
        else:
            elset_id = numpy.zeros(self.get_number_of_elements(), dtype=numpy.uint8)
        id_to_rank = self.compute_id_to_rank(nodes=False)
        for elset_name in elset_list:
            try:
                j = self._elset_names.index(elset_name)
                print('j=%d, elset name=%s' % (j, elset_name))
            except ValueError:
                print('elset %s not found in mesh, skipping' % elset_name)
            for el_id in self._elsets[j]:
                if use_name_as_id:
                    this_id = elset_name.split(elset_prefix)[1]
                    elset_id[id_to_rank[el_id]] = int(this_id)
                else:
                    elset_id[id_to_rank[el_id]] = j
        return elset_id

    def compute_grain_id_field(self, grain_prefix='grain_'):
        '''Compute a new field composed by the grain ids.'''
        grain_ids = self.compute_elset_id_field(elset_prefix=grain_prefix, use_name_as_id=True)
        if numpy.max(grain_ids) < 1:
            print('Warning, no grain found, verify the grain prefix...')
        return grain_ids

    @staticmethod
    def to_vtk_element_type(el_type):
        if el_type == 'c3d4':
            return vtk.VTK_TETRA  # 10
        if el_type.startswith('c3d10'):
            return vtk.VTK_QUADRATIC_TETRA  # 24
        if el_type.startswith('c3d8'):
            return vtk.VTK_HEXAHEDRON  # 12
        if el_type.startswith('c2d3') or el_type.startswith('s3d3'):
            return vtk.VTK_TRIANGLE  # 5
        if el_type.startswith('c2d4') or el_type.startswith('s3d4'):
            return vtk.VTK_QUAD  # 9
        if el_type.startswith('c3d6'):
            return vtk.VTK_WEDGE  # 13
        if el_type.startswith('c3d15'):
            return vtk.VTK_QUADRATIC_WEDGE  # 26
        if el_type.startswith('c3d13'):
            return vtk.VTK_QUADRATIC_PYRAMID  # 27
        if el_type.startswith('c3d20'):
            return vtk.VTK_QUADRATIC_HEXAHEDRON  # 25
        if el_type.startswith('c2d8') or el_type.startswith('s3d8'):
            return vtk.VTK_QUADRATIC_QUAD  # 1

    def build_vtk(self):
        '''Build a vtkUnstructuredGrid instance corresponding to the mesh.
        
        This method creates a new vtkUnstructuredGrid object, set the nodes and the elements.
        
        :returns: the vtkUnstructuredGrid object.
        '''
        print('building vtk stuff for FE_Mesh')
        vtk_mesh = vtk.vtkUnstructuredGrid()
        # take care of nodes
        nodes = vtk.vtkPoints()
        nodes.SetNumberOfPoints(self.get_number_of_nodes());
        for i in range(self.get_number_of_nodes()):
            (x, y, z) = self._nodes[i]._x, self._nodes[i]._y, self._nodes[i]._z
            nodes.InsertPoint(i, x, y, z)  # here i == self._nodes[i].give_rank()
        vtk_mesh.SetPoints(nodes)
        # take care of elements
        for i in range(self.get_number_of_elements()):
            el = self._elements[i]
            vtk_type = FE_Mesh.to_vtk_element_type(el._type)
            # allocate memory for this element type
            # vtk_mesh.Allocate(vtk_type, numpy.shape(el_list)[0])
            if el._type in ['c2d3', 's3d3', 'c3d4', 'c3d6', 'c3d8',
                            'c3d8r', 'c3d13']:
                Ids = vtk.vtkIdList()
                for j in range(len(el._nodelist)):
                    Ids.InsertNextId(el._nodelist[j].give_rank())
                vtk_mesh.InsertNextCell(vtk_type, Ids)
            elif el._type.startswith('c3d10'):
                Ids = vtk.vtkIdList()
                Ids.InsertNextId(el._nodelist[0].give_rank())
                Ids.InsertNextId(el._nodelist[2].give_rank())
                Ids.InsertNextId(el._nodelist[1].give_rank())
                Ids.InsertNextId(el._nodelist[9].give_rank())
                Ids.InsertNextId(el._nodelist[5].give_rank())
                Ids.InsertNextId(el._nodelist[4].give_rank())
                Ids.InsertNextId(el._nodelist[3].give_rank())
                Ids.InsertNextId(el._nodelist[6].give_rank())
                Ids.InsertNextId(el._nodelist[8].give_rank())
                Ids.InsertNextId(el._nodelist[7].give_rank())
                vtk_mesh.InsertNextCell(vtk_type, Ids)
            elif el._type.startswith('c3d15'):
                Ids = vtk.vtkIdList()
                Ids.InsertNextId(el._nodelist[0].give_rank())
                Ids.InsertNextId(el._nodelist[2].give_rank())
                Ids.InsertNextId(el._nodelist[4].give_rank())
                Ids.InsertNextId(el._nodelist[9].give_rank())
                Ids.InsertNextId(el._nodelist[11].give_rank())
                Ids.InsertNextId(el._nodelist[13].give_rank())
                Ids.InsertNextId(el._nodelist[1].give_rank())
                Ids.InsertNextId(el._nodelist[3].give_rank())
                Ids.InsertNextId(el._nodelist[5].give_rank())
                Ids.InsertNextId(el._nodelist[10].give_rank())
                Ids.InsertNextId(el._nodelist[12].give_rank())
                Ids.InsertNextId(el._nodelist[14].give_rank())
                Ids.InsertNextId(el._nodelist[6].give_rank())
                Ids.InsertNextId(el._nodelist[7].give_rank())
                Ids.InsertNextId(el._nodelist[8].give_rank())
                vtk_mesh.InsertNextCell(vtk_type, Ids)
            elif el._type.startswith('c3d20'):
                Ids = vtk.vtkIdList()
                Ids.InsertNextId(el._nodelist[0].give_rank())
                Ids.InsertNextId(el._nodelist[6].give_rank())
                Ids.InsertNextId(el._nodelist[4].give_rank())
                Ids.InsertNextId(el._nodelist[2].give_rank())
                Ids.InsertNextId(el._nodelist[12].give_rank())
                Ids.InsertNextId(el._nodelist[18].give_rank())
                Ids.InsertNextId(el._nodelist[16].give_rank())
                Ids.InsertNextId(el._nodelist[14].give_rank())
                Ids.InsertNextId(el._nodelist[7].give_rank())
                Ids.InsertNextId(el._nodelist[5].give_rank())
                Ids.InsertNextId(el._nodelist[3].give_rank())
                Ids.InsertNextId(el._nodelist[1].give_rank())
                Ids.InsertNextId(el._nodelist[19].give_rank())
                Ids.InsertNextId(el._nodelist[17].give_rank())
                Ids.InsertNextId(el._nodelist[15].give_rank())
                Ids.InsertNextId(el._nodelist[13].give_rank())
                Ids.InsertNextId(el._nodelist[8].give_rank())
                Ids.InsertNextId(el._nodelist[11].give_rank())
                Ids.InsertNextId(el._nodelist[10].give_rank())
                Ids.InsertNextId(el._nodelist[9].give_rank())
                vtk_mesh.InsertNextCell(vtk_type, Ids)
            elif el._type.startswith('c2d4') or el._type.startswith('s3d4'):
                Ids = vtk.vtkIdList()
                for j in range(len(el._nodelist)):
                    Ids.InsertNextId(el._nodelist[j].give_rank())
                vtk_mesh.InsertNextCell(vtk_type, Ids)
            elif el._type.startswith('c2d8') or el._type.startswith('s3d8'):
                Ids = vtk.vtkIdList()
                Ids.InsertNextId(el._nodelist[0].give_rank())
                Ids.InsertNextId(el._nodelist[2].give_rank())
                Ids.InsertNextId(el._nodelist[4].give_rank())
                Ids.InsertNextId(el._nodelist[6].give_rank())
                Ids.InsertNextId(el._nodelist[1].give_rank())
                Ids.InsertNextId(el._nodelist[3].give_rank())
                Ids.InsertNextId(el._nodelist[5].give_rank())
                Ids.InsertNextId(el._nodelist[7].give_rank())
                vtk_mesh.InsertNextCell(vtk_type, Ids)
        return vtk_mesh

    def build_vtk_for_lisets(self):
        print('building vtk stuff for FE_Mesh')
        vtk_mesh = vtk.vtkUnstructuredGrid()
        # take care of nodes
        nodes = vtk.vtkPoints()
        nodes.SetNumberOfPoints(self.get_number_of_nodes());
        for i in range(self.get_number_of_nodes()):
            (x, y, z) = self._nodes[i]._x, self._nodes[i]._y, self._nodes[i]._z
            nodes.InsertPoint(i, x, y, z)  # here i == self._nodes[i].give_rank()
        vtk_mesh.SetPoints(nodes)
        print('%d lisets to add to the grid' % len(self._lisets))
        for i, liset in enumerate(self._lisets):
            print('%04d adding liset %s' % (i, self._liset_names[i]))
            for line_segment in liset:
                Ids = vtk.vtkIdList()
                [node1_id, node2_id] = line_segment
                Ids.InsertNextId(node1_id - 1)  # self._nodes[node1_id].give_rank())
                Ids.InsertNextId(node2_id - 1)  # self._nodes[node2_id].give_rank())
                vtk_mesh.InsertNextCell(4, Ids)
        return vtk_mesh


class FE_Node():
    '''This class is used to represent a finite element node.'''

    def __init__(self, id):
        '''
        Create an empty node at the origin.
        '''
        self._id = id
        self._rank = None
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0

    def __repr__(self):
        ''' Gives a string representation of the node.'''
        out = '%s id = %d, rank = %d\n' % (self.__class__.__name__, self._id, self._rank)
        out += 'position = (%.3f, %.3f, %.3f)' % (self._x, self._y, self._z)
        return out

    def give_id(self):
        return self._id

    def give_rank(self):
        return self._rank

    def set_rank(self, r):
        self._rank = r


class FE_Element():
    '''This class is used to represent a finite element.'''

    def __init__(self, id, el_type):
        '''
        Create an empty element (no nodes).
        '''
        self._id = id
        self._rank = None
        self._type = el_type
        self._nodelist = []

    def __repr__(self):
        ''' Gives a string representation of the element.'''
        out = '%s element\n' % self.__class__.__name__
        out += 'type: %s\n' % self._type
        out += 'node id list = [ '
        for node in self._nodelist:
            out += '%d ' % node.give_id()
        out += ']'
        return out

    def get_number_of_gauss_points(self):
        '''Returns the total number of integration points within this element.
           see zUtilityMesh/Declare_geometries.c in the Z-set code
        '''
        if self._type in ['c2d3']:
            return 4
        if self._type in ['c3d4', 'c3d6', 'c2d4', 'c2d8', 'c3d10_4']:
            return 4
        elif self._type in ['c3d8', 'c3d20r']:
            return 8
        elif self._type in ['c3d20', 'c3d13']:
            return 27
        elif self._type in ['c3d15']:
            return 18
        elif self._type in ['c3d15r', 's3d3']:
            return 6
        elif self._type in ['c3d10', 'c3d13r']:
            return 5
        elif self._type in ['c2d8r', 'c3d8r']:
            return 1

    def give_id(self):
        return self._id

    def give_rank(self):
        return self._rank

    def set_rank(self, r):
        self._rank = r

    def get_center_of_mass(self):
        com = numpy.array([0., 0., 0.])
        for node in self._nodelist:
            com[0] += node._x
            com[1] += node._y
            com[2] += node._z
        com /= len(self._nodelist)
        return com
