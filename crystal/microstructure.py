'''
The microstructure module provide elementary classes to describe a 
crystallographic granular microstructure such as mostly present in 
metallic materials.
'''
import numpy as np
import vtk
from matplotlib import pyplot as plt, colors, cm
from xml.dom.minidom import Document, parse

class Orientation:
  '''
  Crystallographic orientation class.
  '''

  '''Euler angles must be specified in degrees.'''
  def __init__(self, phi1, Phi, phi2, type='euler'):
    if (type != 'euler'):
      raise TypeError('unsupported orientation type', type)
    self.type = type
    self.phi1 = phi1
    self.Phi = Phi
    self.phi2 = phi2

  '''Provide a string representation of the class.'''
  def __repr__(self):
    return "orientation = (%.3f, %.3f, %.3f)" % (self.phi1, self.Phi, self.phi2)

  '''
  Returns an XML representation of the Orientation instance.
  '''
  def to_xml(self, doc):
    orientation = doc.createElement('Orientation')
    orientation_type = doc.createElement('Type')
    orientation_type_text = doc.createTextNode('%s' % self.type)
    orientation_type.appendChild(orientation_type_text)
    orientation.appendChild(orientation_type)
    orientation_phi1 = doc.createElement('phi1')
    orientation_phi1_text = doc.createTextNode('%f' % self.phi1)
    orientation_phi1.appendChild(orientation_phi1_text)
    orientation.appendChild(orientation_phi1)
    orientation_Phi = doc.createElement('Phi')
    orientation_Phi_text = doc.createTextNode('%f' % self.Phi)
    orientation_Phi.appendChild(orientation_Phi_text)
    orientation.appendChild(orientation_Phi)
    orientation_phi2 = doc.createElement('phi2')
    orientation_phi2_text = doc.createTextNode('%f' % self.phi2)
    orientation_phi2.appendChild(orientation_phi2_text)
    orientation.appendChild(orientation_phi2)
    return orientation

  @staticmethod
  def from_xml(orientation_node):
    orientation_type = orientation_node.childNodes[0]
    orientation_phi1 = orientation_node.childNodes[1]
    orientation_Phi = orientation_node.childNodes[2]
    orientation_phi2 = orientation_node.childNodes[3]
    t = orientation_type.childNodes[0].nodeValue
    phi1 = float(orientation_phi1.childNodes[0].nodeValue)
    Phi = float(orientation_Phi.childNodes[0].nodeValue)
    phi2 = float(orientation_phi2.childNodes[0].nodeValue)
    orientation = Orientation(phi1, Phi, phi2, type=t)
    return orientation

  '''Compute the orientation matrix. '''
  def orientation_matrix(self):
    c1 = np.cos(self.phi1*np.pi/180.)
    s1 = np.sin(self.phi1*np.pi/180.)
    c = np.cos(self.Phi*np.pi/180.)
    s = np.sin(self.Phi*np.pi/180.)
    c2 = np.cos(self.phi2*np.pi/180.)
    s2 = np.sin(self.phi2*np.pi/180.)

    # rotation matrix B
    b11 = c1*c2-s1*s2*c
    b12 = s1*c2+c1*s2*c
    b13 = s2*s
    b21 = -c1*s2-s1*c2*c
    b22 = -s1*s2+c1*c2*c
    b23 = c2*s
    b31 = s1*s
    b32 = -c1*s
    b33 = c
    B = np.array([[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]])
    return B

class Grain:
  '''
  Class defining a crystallographic grain.

  A grain has its own crystallographic orientation.
  An optional id for the grain may be specified.
  The position field is the center of mass of the grain in world coordinates.
  The volume of the grain is expressed in pixel/voxel unit.
  '''

  def __init__(self, grain_id, phi1, Phi, phi2):
    print '*** INIT ***'
    self.__init__(grain_id, Orientation(phi1, Phi, phi2, type='euler'))

  def __init__(self, grain_id, grain_orientation):
    self.id = grain_id
    self.orientation = grain_orientation
    self.position = (0, 0, 0)
    self.volume = 0
    self.vtkmesh = None
    #self.records = []

  '''Provide a string representation of the class.'''
  def __repr__(self):
    s = '%s\n * id = %d\n' % (self.__class__.__name__, self.id)
    s += ' * %s\n' % (self.orientation)
    s += ' * position (%f, %f, %f)\n' % (self.position)
    s += ' * has vtk mesh ? %s\n' % (self.vtkmesh != None)
    return s
    
  def SetVtkMesh(self, mesh):
    self.vtkmesh = mesh
    
  def add_vtk_mesh(self, array):
    label = self.id # we use the grain id here... 
    # create vtk structure
    from scipy import ndimage
    from vtk.util import numpy_support
    grain_size = np.shape(array)
    local_com = ndimage.measurements.center_of_mass(array == label, array)
    vtk_data_array = numpy_support.numpy_to_vtk(np.ravel(array, order='F'), deep=1)
    grid = vtk.vtkUniformGrid()
    grid.SetExtent(0, grain_size[0], 0, grain_size[1], 0, grain_size[2])
    grid.SetOrigin(-local_com[0], -local_com[1], -local_com[2])
    grid.SetSpacing(1, 1, 1)
    grid.SetScalarType(vtk.VTK_UNSIGNED_CHAR)
    grid.GetCellData().SetScalars(vtk_data_array)
    # threshold selected grain
    print 'thresholding label', label
    thresh = vtk.vtkThreshold()
    thresh.ThresholdBetween(label-0.5, label+0.5)
    thresh.SetInput(grid)
    thresh.Update()
    self.SetVtkMesh(thresh.GetOutput())

  '''
  Returns an XML representation of the Grain instance.
  '''
  def to_xml(self, doc):
    grain = doc.createElement('Grain')
    grain_id = doc.createElement('Id')
    grain_id_text = doc.createTextNode('%s' % self.id)
    grain_id.appendChild(grain_id_text)
    grain.appendChild(grain_id)
    grain.appendChild(self.orientation.to_xml(doc))
    grain_position = doc.createElement('Position')
    grain_position_x = doc.createElement('X')
    grain_position.appendChild(grain_position_x)
    grain_position_x_text = doc.createTextNode('%f' % self.position[0])
    grain_position_x.appendChild(grain_position_x_text)
    grain_position_y = doc.createElement('Y')
    grain_position.appendChild(grain_position_y)
    grain_position_y_text = doc.createTextNode('%f' % self.position[1])
    grain_position_y.appendChild(grain_position_y_text)
    grain_position_z = doc.createElement('Z')
    grain_position.appendChild(grain_position_z)
    grain_position_z_text = doc.createTextNode('%f' % self.position[2])
    grain_position_z.appendChild(grain_position_z_text)
    grain.appendChild(grain_position)
    grain_mesh = doc.createElement('Mesh')
    grain_mesh_text = doc.createTextNode('%s' % self.vtk_file_name())
    grain_mesh.appendChild(grain_mesh_text)
    grain.appendChild(grain_mesh)
    return grain

  @staticmethod
  def from_xml(grain_node):
    grain_id = grain_node.childNodes[0]
    grain_orientation = grain_node.childNodes[1]
    orientation = Orientation.from_xml(grain_orientation)
    id = int(grain_id.childNodes[0].nodeValue)
    grain = Grain(id, orientation)
    grain_position = grain_node.childNodes[2]
    xg = float(grain_position.childNodes[0].childNodes[0].nodeValue)
    yg = float(grain_position.childNodes[1].childNodes[0].nodeValue)
    zg = float(grain_position.childNodes[2].childNodes[0].nodeValue)
    grain.position = (xg, yg, zg)
    grain_mesh = grain_node.childNodes[3]
    grain_mesh_file = grain_mesh.childNodes[0].nodeValue
    print grain_mesh_file
    grain.load_vtk_repr(grain_mesh_file)
    return grain
    
  def vtk_file_name(self):
    return 'grain_%d.vtu' % self.id
    
  def save_vtk_repr(self):
    import vtk
    print 'writting ' + self.vtk_file_name()
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(self.vtk_file_name())
    writer.SetInput(self.vtkmesh)
    writer.Write()

  def load_vtk_repr(self, file_name):
    import vtk
    print 'reading ' + file_name
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()
    self.vtkmesh = reader.GetOutput()

  def orientation_matrix(self):
    return self.orientation.orientation_matrix()

class Microstructure:
  '''
  Class used to manipulate a full microstructure.
  
  It is typically defined as a list of grains objects.
  '''
  def __init__(self, name='empty'):
    self.name = name
    self.grains = []
    self.vtkmesh = None

  @staticmethod
  def from_xml(xml_file_name):
    micro = Microstructure()
    dom = parse(xml_file_name)
    root = dom.childNodes[0]
    name = root.childNodes[0]
    micro.name = name.childNodes[0].nodeValue
    grains = root.childNodes[1]
    for node in grains.childNodes:
      print node
      micro.grains.append(Grain.from_xml(node))
    return micro
    

  '''Provide a string representation of the class.'''
  def __repr__(self):
    s = '%s\n' % self.__class__.__name__
    s += '* name: %s\n' % self.name
    for g in self.grains:
      s += '* %s' % g.__repr__
    return s
  
  def SetVtkMesh(self, mesh):
    self.vtkmesh = mesh

  '''
  Returns an XML representation of the Microstructure instance.
  '''
  def to_xml(self, doc):
    root = doc.createElement('Microstructure')
    doc.appendChild(root)
    name = doc.createElement('Name')
    root.appendChild(name)
    name_text = doc.createTextNode(self.name)
    name.appendChild(name_text)
    grains = doc.createElement('Grains')
    root.appendChild(grains)
    for grain in self.grains:
      grains.appendChild(grain.to_xml(doc))
              
  '''
  Saving the microstructure, only save the vtk representation 
  of the grain for now.
  '''
  def save(self):
    # save the microstructure instance as xml
    doc = Document()
    self.to_xml(doc)
    xml_file_name = '%s.xml' % self.name[:-4]
    print 'writting ' + xml_file_name
    f= open(xml_file_name, 'wb')
    doc.writexml(f, encoding= 'utf-8')
    f.close()
    # now save the vtk representation
    if self.vtkmesh != None:
      import vtk
      vtk_file_name = '%s.vtm' % self.name[:-4]
      print 'writting ' + vtk_file_name
      writer = vtk.vtkXMLMultiBlockDataWriter()
      writer.SetFileName(vtk_file_name)
      writer.SetInput(self.vtkmesh)
      writer.Write()    
    
class EbsdMicrostructure:
  '''
  Class used to manipulate a full microstructure read from an EBSD 
  measurement for instance.
  '''
  def __init__(self, name='empty'):
    self.name = name
    self.type = None
    self.shape = None
    self.records = None

  # this should produce a single array from which you can plot the usual 
  # EBSD stuff. Then add another method which extract a grain list given
  # a criterion (can be grain ID or misorientation)
  def read_from_ebsd(self, filename, grid='square'):
    if (grid != 'square' and grid != 'hex'):
      raise TypeError('unsupported grid type', grid)
    self.name = filename.split('/')[-1]
    self.records = np.loadtxt(filename, usecols=range(9))
    # guess the grid size
    size_x = sum(self.records[:,4] == self.records[0,4])
    size_y = len(self.records[:,4])/size_x
    print 'size is ',size_x, size_y
    self.shape = (size_y,size_x)
    
  def extract_grains(self):
    grain_id_list = np.unique(self.records[:,8]).tolist()
    grain_record_list = np.empty(len(grain_id_list))
    # todo: filter out all records to build array for each grain...
    for record in a:
      # add this line to the corresponding grain_record
      grain_record_list[grain_id_list.index(record[8])].append(record)
    #for record_list in grain_record_list:
    # create a new grain

  # plot the ebsd data using pyplot. have a look at enum to handle all possibilities
  def plot(self, type='Euler', save=False, display=True):
    if type == 'phi1':
      phi1 = self.records[:,0].reshape(self.shape)/np.pi*180.
      plt.imshow(phi1, cmap=cm.hsv, interpolation='nearest')
      plt.clim([0,360])
      plt.colorbar()
      print np.min(phi1), np.max(phi1)
    elif type == 'Phi':
      Phi = self.records[:,1].reshape(self.shape)/np.pi*180.
      plt.imshow(Phi, cmap=cm.hsv, interpolation='nearest')
      plt.clim([0,180])
    elif type == 'phi2':
      phi2 = self.records[:,2].reshape(self.shape)/np.pi*180.
      plt.imshow(phi2, cmap=cm.hsv, interpolation='nearest')
      plt.clim([0,360])
    elif type == 'Euler':
      # provide a MxNx3 array to imshow
      rgb = np.empty((self.shape[0],self.shape[1],3), dtype=float)
      rgb[:,:,0] = self.records[:,0].reshape(self.shape)/(2*np.pi)
      rgb[:,:,1] = self.records[:,1].reshape(self.shape)/(np.pi)
      rgb[:,:,2] = self.records[:,2].reshape(self.shape)/(2*np.pi)
      plt.imshow(rgb, interpolation='nearest')
    elif type == 'IQ':
      iq = self.records[:,5].reshape(self.shape)
      plt.imshow(iq, cmap=cm.gray, interpolation='nearest')
    elif type == 'GID':
      np.random.seed(13)
      rand_colors = np.random.rand(4096,3)
      rand_colors[0] = [0., 0., 0.] # enforce black background (value 0)
      rand_cmap = colors.ListedColormap(rand_colors)
      gid = self.records[:,8].reshape(self.shape)
      plt.imshow(gid, cmap=rand_cmap, interpolation='nearest')
    else:
      raise TypeError('unsupported ebsd plot type', type)
    if save:
      plt.imsave(name + '_' + type + '.png', format='png')
    if display:
      plt.show()
    
if __name__ == '__main__':
  import os
  data_dir = '/home/proudhon/python/samples/dct'
  micro = Microstructure.from_xml(os.path.join(data_dir, 'grains_302x302x100_uint8.xml'))
  print micro
  
  '''
  #g6 = Grain(6, 142.845, 31.966, 214.384)
  #print g6.__repr__
  m = EbsdMicrostructure()
  m.read_from_ebsd('/home/proudhon/students/claudio/ebsd/grainfilepointssquare.txt')
  #m.read_from_ebsd('/home/proudhon/data/ebsd/AlMgSi_Reza/80min_80Cr_17mm_export.txt')
  print m.records.shape
  m.plot(type='GID')
  '''
  print 'done'
