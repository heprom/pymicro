from pymicro.core.samples import SampleData
import tables
import numpy as np
from scipy import ndimage


class FeatureData(tables.IsDescription):
    """
       Description class specifying structured storage of feature data in the
       SampleWithFeatures Class, in HDF5 node /GrainData/GrainDataTable
    """
    # feature identity number
    id = tables.Int32Col()  # Signed 32-bit integer
    # feature volume
    volume = tables.Float32Col()  # float
    # feature center of mass coordinates
    center = tables.Float32Col(shape=(3,))  # float  (double-precision)
    # Feature Bounding box
    bounding_box = tables.Int32Col(shape=(3, 2))  # Signed 64-bit integer


class SampleWithFeatures(SampleData):
    """
    Class used to manipulate a sample containing a set of features and derived
    from the `SampleData` class.

    This class is a data container for a mechanical sample containing features.
    Features are labeled object that can be described in an image and studied
    individually like pores in a matrix or features in a polycrystalline
    microstructure. As a SampleData instance, the object in memory is always
    synchronized with a HDF5 file and a XDMF file.

    The dataset maintains a `FeatureData` instance which inherits from
    tables.IsDescription and acts as a structured array containing each feature
    attributes such as id, center of mass, volume and bounding box.

    The image data typically contains a feature map which carries integer values
    of all the feature labels (zero being the rest), a mask used to regroup the
    sample (with the features).
    """

    def __init__(self,
                 filename=None, name='sample', description='',
                 verbose=False, overwrite_hdf5=False, autodelete=False):
        if filename is None:
            # only add '_' if not present at the end of name
            filename = name + (not name.endswith('_')) * '_' + 'data'
        # call SampleData constructor
        SampleData.__init__(self, filename=filename, sample_name=name,
                            sample_description=description, verbose=verbose,
                            overwrite_hdf5=overwrite_hdf5,
                            autodelete=autodelete,
                            after_file_open_args={})
        return

    def _after_file_open(self, **kwargs):
        """Initialization code to run after opening a Sample Data file."""
        self.features = self.get_node('FeatureDataTable')
        self.default_compression_options = {'complib': 'zlib', 'complevel': 5}
        return

    def minimal_data_model(self):
        """Data model for a sample containing features.

        Specify the minimal contents of the hdf5 (Group names, paths and group
        types) in the form of a dictionary {content: location}. This extends
        `~pymicro.core.SampleData.minimal_data_model` method.

        :return: a tuple containing the two dictionaries.
        """
        minimal_content_index_dic = {'Image_data': '/CellData',
                                     'feature_map': '/CellData/feature_map',
                                     'mask': '/CellData/mask',
                                     'Mesh_data': '/MeshData',
                                     'Feature_data': '/FeatureData',
                                     'FeatureDataTable': ('/FeatureData/'
                                                          'FeatureDataTable')
                                     }
        minimal_content_type_dic = {'Image_data': '3DImage',
                                    'feature_map': 'field_array',
                                    'mask': 'field_array',
                                    'Mesh_data': 'Mesh',
                                    'Feature_data': 'Group',
                                    'FeatureDataTable': FeatureData
                                    }
        return minimal_content_index_dic, minimal_content_type_dic

    def get_number_of_features(self, from_feature_map=False):
        """Return the number of features in this sample.

        :param bool from_feature_map: controls if the returned number of
        features comes from the feature data table or from the feature map.
        :return: the number of features in the sample.
        """
        if from_feature_map:
            # do not count label 0 (reserved for background)
            return len(np.unique(self.get_feature_map())) - 1
        else:
            return self.features.nrows

    def get_feature_map(self):
        """Get the feature map as a numpy array.

        The feature map is the image constituted by the feature ids or labels.
        Label zero is reserved for the background or unattributed voxels.

        :return: the feature map as a numpy array.
        """
        feature_map = self.get_field('feature_map')
        if self._is_empty('feature_map'):
            feature_map = None
        elif feature_map.ndim == 2:
            # reshape to 3D
            new_dim = self.get_attribute('dimension', 'CellData')
            if len(new_dim) == 3:
                feature_map = feature_map.reshape(new_dim)
            else:
                feature_map = feature_map.reshape(
                    (feature_map.shape[0], feature_map.shape[1], 1))
        return feature_map

    def get_mask(self):
        """Get the mask as a numpy array.

        The mask represent the sample outline. The value 1 means we are inside
        the sample (or inside a feature), the value 0 means we are outside the
        sample.

        :return: the mask as a numpy array.
        """
        mask = self.get_field('mask')
        if self._is_empty('mask'):
            mask = None
        elif mask.ndim == 2:
            # reshape to 3D
            new_dim = self.get_attribute('dimension', 'CellData')
            if len(new_dim) == 3:
                mask = mask.reshape(new_dim)
            else:
                mask = mask.reshape(
                    (mask.shape[0], mask.shape[1], 1))
        return mask

    def get_ids_from_map(self):
        """Return the list of feature ids found in the feature map.

        By convention, only positive values are taken into account, 0 is
        reserved for the background.

        :return: a 1D numpy array containing the feature ids.
        """
        feature_map = self.get_node('feature_map')
        features_id = np.unique(feature_map)
        features_id = features_id[features_id > 0]
        return features_id

    @staticmethod
    def id_list_to_condition(id_list):
        """Convert a list of id to a condition to filter the feature table.

        The condition will be interpreted using Numexpr typically using
        a `read_where` call on the feature data table.

        :param list id_list: a non empty list of the feature ids.
        :return: the condition as a string .
        """
        if not len(id_list) > 0:
            raise ValueError('the list of feature ids must not be empty')
        condition = "\'(id == %d)" % id_list[0]
        for feature_id in id_list[1:]:
            condition += " | (id == %d)" % feature_id
        condition += "\'"
        return condition

    def get_ids(self):
        """Return the feature ids found in the FeatureDataTable.

        :return: a 1D numpy array containing the feature ids.
        """
        return self.get_tablecol('FeatureDataTable', 'id')

    def get_volumes(self, id_list=None):
        """Get the feature volumes.

        The feature data table is queried and the volumes of the features are
        returned in a single array. An optional list of feature ids can be used
        to restrict the features, by default all the feature volumes are returned.

        :param list id_list: a non empty list of the feature ids.
        :return: a numpy array containing the feature volumes.
        """
        if id_list:
            condition = SampleWithFeatures.id_list_to_condition(id_list)
            return self.features.read_where(eval(condition))['volume']
        else:
            return self.get_tablecol('FeatureDataTable', 'volume')

    def get_centers(self, id_list=None):
        """Get the feature centers.

        The feature data table is queried and the centers of the features are
        returned in a single array. An optional list of feature ids can be used
        to restrict the features, by default all the feature centers are returned.

        :param list id_list: a non empty list of the feature ids.
        :return: a numpy array containing the feature centers.
        """
        if id_list:
            condition = SampleWithFeatures.id_list_to_condition(id_list)
            return self.features.read_where(eval(condition))['center']
        else:
            return self.get_tablecol('FeatureDataTable', 'center')

    def get_bounding_boxes(self, id_list=None):
        """Get the feature bounding boxes.

        The feature data table is queried and the bounding boxes of the features
        are returned in a single array. An optional list of feature ids can be
        used to restrict the features, by default all the feature bounding boxes
        are returned.

        .. note::

          The bounding boxes are returned in ascending order of the feature ids
          (not necessary the same order than the list if it is not ordered).
          The maximum length of the ids list is 256.

        :param list id_list: a non empty (preferably ordered) list of the
        selected feature ids (with a maximum number of ids of 256).
        :return: a numpy array containing the feature bounding boxes.
        :raise: a ValueError if the length of the id list is larger than 256.
        """
        if id_list:
            if len(id_list) > 256:
                raise(ValueError("the id_list can only have 256 values"))
            condition = SampleWithFeatures.id_list_to_condition(id_list)
            return self.features.read_where(eval(condition))['bounding_box']
        else:
            return self.get_tablecol('FeatureDataTable', 'bounding_box')

    def get_voxel_size(self):
        """Get the voxel size for image data of the sample.

        If this instance of `SampleWithFeatures` has no image data, None is returned.
        """
        try:
            return self.get_attribute(attrname='spacing',
                                      nodename='/CellData')[0]
        except:
            return None

    def set_centers(self, centers):
        """Store the feature centers array in the FeatureDataTable

        :param ndarray centers: a numpy array of size (n_features, 3) containing
        all features center of mass.
        """
        self.set_tablecol('FeatureDataTable', 'center', column=centers)

    def set_bounding_boxes(self, bounding_boxes):
        """ Store feature bounding boxes array in the FeatureDataTable

        :param ndarray bounding_boxes: a numpy array containing all features
        bounding boxes.
        """
        self.set_tablecol('GrainDataTable', 'bounding_box', column=bounding_boxes)
        return

    def set_volumes(self, volumes):
        """ Store feature volumes array in the FeatureDataTable.

        :param ndarray volumes: a numpy array of size (n_features) containing
        all features volumes.
        """
        self.set_tablecol('FeatureDataTable', 'volume', column=volumes)
        return

    def set_feature_map(self, feature_map, voxel_size=None, compression=None):
        """Set the feature map for this sample.

        :param ndarray feature_map: a 2D or 3D numpy array.
        :param float voxel_size: the size of the voxels in mm unit. Used only
            if the CellData image node must be created.
        :param dict compression: a dictionary with compression settings to use.
        """
        if compression is None:
            compression = self.default_compression_options
        create_image = True
        if self.__contains__('CellData'):
            empty = self.get_attribute(attrname='empty', nodename='CellData')
            if not empty:
                create_image = False
        if create_image:
            if voxel_size is None:
                msg = 'Please specify voxel size for CellData image'
                raise ValueError(msg)
            if np.isscalar(voxel_size):
                dim = len(feature_map.shape)
                spacing_array = voxel_size * np.ones((dim,))
            else:
                if len(voxel_size) != len(feature_map.shape):
                    raise ValueError('voxel_size array must have a length '
                                     'equal to feature_map shape')
                spacing_array = voxel_size
            self.add_image_from_field(field_array=feature_map,
                                      fieldname='feature_map',
                                      imagename='CellData', location='/',
                                      spacing=spacing_array,
                                      replace=True,
                                      compression_options=compression)
        else:
            # Handle case of a 2D SampleWithFeatures: squeeze feature map to
            # ensure (Nx, Ny, 1) array will be stored as (Nx, Ny)
            if self._get_group_type('CellData') == '2DImage':
                feature_map = feature_map.squeeze()
            self.add_field(gridname='CellData', fieldname='feature_map',
                           array=feature_map, replace=True,
                           compression_options=compression)
        return

    def set_mask(self, mask, voxel_size=None, compression=None):
        """Set the mask for this sample.

        :param ndarray mask: a 2D or 3D numpy array.
        :param float voxel_size: the size of the voxels in mm unit. Used only
            if the CellData image Node must be created.
        """
        if compression is None:
            compression = self.default_compression_options
        create_image = True
        if self.__contains__('CellData'):
            empty = self.get_attribute(attrname='empty', nodename='CellData')
            if not empty:
                create_image = False
        if mask.dtype == 'bool':
            # use uint8 encoding
            mask = mask.astype(np.uint8)
        if create_image:
            if voxel_size is None:
                msg = 'Please specify voxel size for CellData image'
                raise ValueError(msg)
            if np.isscalar(voxel_size):
                dim = len(mask.shape)
                spacing_array = voxel_size * np.ones((dim, ))
            else:
                if len(voxel_size) != len(mask.shape):
                    raise ValueError('voxel_size array must have a length '
                                     'equal to feature_map shape')
                spacing_array = voxel_size
            self.add_image_from_field(mask, 'mask',
                                      imagename='CellData', location='/',
                                      spacing=spacing_array,
                                      replace=True,
                                      compression_options=compression)
        else:
            self.add_field(gridname='CellData', fieldname='mask',
                           array=mask, replace=True, indexname='mask',
                           compression_options=compression)
        return

    def remove_features_not_in_map(self):
        """Remove from FeatureDataTable features that are not in the feature map."""
        _, not_in_map, _ = self.compute_features_map_table_intersection()
        self.remove_features_from_table(not_in_map)
        return

    def remove_features_from_table(self, ids):
        """Remove from FeatureDataTable the features with given ids.

        :param list ids: list of feature ids to remove from GrainDataTable
        """
        for feature_id in ids:
            where = self.features.get_where_list('id == feature_id')[:]
            self.features.remove_row(int(where))
        return

    def compute_volume(self, feature_id):
        """Compute the volume of the feature given its id.

        The total number of voxels with the given id is computed. The value is
        converted to mm unit using the `voxel_size`. The unit will be squared
        mm for a 2D feature map or cubed mm for a 3D feature map.

        .. warning::

          This function assume the feature bounding box is correct, call
          `recompute_feature_bounding_boxes()` if this is not the case.

        :param int feature_id: the feature id to consider.
        :return: the volume of the feature.
        """
        bb = self.features.read_where('id == %d' % feature_id)['bounding_box'][0]
        feature_map = self.get_feature_map()[bb[0][0]:bb[0][1],
                                             bb[1][0]:bb[1][1],
                                             bb[2][0]:bb[2][1]]
        voxel_size = self.get_attribute('spacing', 'CellData')
        volume_vx = np.sum(feature_map == np.array(feature_id))
        return volume_vx * np.prod(voxel_size)

    def compute_center(self, feature_id):
        """Compute the center of masses of a feature given its id.

        .. warning::

          This function assume the feature bounding box is correct, call
          `recompute_feature_bounding_boxes()` if this is not the case.

        :param int feature_id: the feature id to consider.
        :return: a tuple with the center of mass in mm units
                 (or voxel if the voxel_size is not specified).
        """
        # isolate the feature within the complete feature map
        bb = self.features.read_where('id == %d' % feature_id)['bounding_box'][0]
        feature_map = self.get_feature_map()[bb[0][0]:bb[0][1],
                                             bb[1][0]:bb[1][1],
                                             bb[2][0]:bb[2][1]]
        voxel_size = self.get_attribute('spacing', 'CellData')
        if len(voxel_size) == 2:
            voxel_size = np.concatenate((voxel_size, np.array([0])), axis=0)
        offset = bb[:, 0]
        feature_data_bin = (feature_map == feature_id).astype(np.uint8)
        local_com = ndimage.measurements.center_of_mass(feature_data_bin)
        local_com += np.array([0.5, 0.5, 0.5])  # account for first voxel coordinates
        com = voxel_size * (offset + local_com
                            - 0.5 * np.array(self.get_feature_map().shape))
        return com

    def compute_bounding_box(self, feature_id, as_slice=False):
        """Compute the feature bounding box indices in the feature map.

        :param int feature_id: the id of the feature.
        :param bool as_slice: a flag to return the feature bounding box as a slice.
        :return: the bounding box coordinates.
        """
        slices = ndimage.find_objects(self.get_feature_map() == np.array(feature_id))[0]
        if as_slice:
            return slices
        x_indices = (slices[0].start, slices[0].stop)
        y_indices = (slices[1].start, slices[1].stop)
        z_indices = (slices[2].start, slices[2].stop)
        return x_indices, y_indices, z_indices

    def recompute_volumes(self, verbose=False):
        """Compute the volume of all feature in the sample.

        Each feature volume is computed using the feature map. The value is
        assigned to the volume column of the FeatureDataTable node.
        If the voxel size is specified, the feature volumes will be in mm unit,
        if not in voxel unit.

        .. note::

          A feature map need to be associated with this SampleWithFeatures instance
          for the method to run.

        :param bool verbose: flag for verbose mode.
        :return: a 1D array with all feature volumes.
        """
        if self._is_empty('feature_map'):
            print('warning: needs a feature map to recompute the volumes '
                  'of the features')
            return
        for f in self.features:
            try:
                vol = self.compute_volume(f['id'])
            except ValueError:
                print('skipping feature %d' % f['id'])
                continue
            if verbose:
                print('feature {}, computed volume is {}'.format(f['id'], vol))
            f['volume'] = vol
            f.update()
        self.features.flush()
        return self.get_volumes()

    def recompute_centers(self, verbose=False):
        """Compute and assign the center of all feature in the sample.

        Each feature center is computed using its center of mass. The value is
        assigned to the feature.center attribute. If the voxel size is specified,
        the feature centers will be in mm unit, if not in voxel unit.

        .. note::

          A feature map need to be associated with this SampleWithFeatures
          instance for the method to run.

        :param bool verbose: flag for verbose mode.
        :return: a 1D array with all feature centers.
        """
        if self._is_empty('feature_map'):
            print('warning: need a feature map to recompute the center of mass'
                  ' of the features')
            return
        for f in self.features:
            try:
                com = self.compute_center(f['id'])
            except ValueError:
                print('skipping feature %d' % f['id'])
                continue
            if verbose:
                print('feature %d center: %.3f, %.3f, %.3f'
                      % (f['id'], com[0], com[1], com[2]))
            f['center'] = com
            f.update()
        self.features.flush()
        return self.get_centers()

    def recompute_bounding_boxes(self, verbose=False):
        """Compute and assign the center of all features in the SampleWithFeatures.

        Each feature bounding box is computed in voxel unit. The value is
        assigned to the feature.bounding_box attribute.

        .. note::

          A feature map need to be associated with this SampleWithFeatures
          instance for the method to run.

        :param bool verbose: flag for verbose mode.
        """
        if self._is_empty('feature_map'):
            print('warning: need a feature map to recompute the bounding boxes'
                  ' of the features')
            return
        # find_objects will return a list of N slices, N being the max feature id
        slices = ndimage.find_objects(self.get_feature_map())
        for f in self.features:
            try:
                f_slice = slices[f['id'] - 1]
                x_indices = (f_slice[0].start, f_slice[0].stop)
                y_indices = (f_slice[1].start, f_slice[1].stop)
                z_indices = (f_slice[2].start, f_slice[2].stop)
                bbox = x_indices, y_indices, z_indices
            except (ValueError, TypeError, IndexError):
                '''
                value or type error can be risen for features in the data table 
                that are not in the feature map (None will be returned from 
                find_objects). IndexError can occur if these feature ids are 
                larger than the maximum id in the feature map.
                '''
                print('skipping feature %d' % f['id'])
                continue
            if verbose:
                print('feature %d bounding box: [%d:%d, %d:%d, %d:%d]'
                      % (f['id'], bbox[0][0], bbox[0][1], bbox[1][0],
                         bbox[1][1], bbox[2][0], bbox[2][1]))
            f['bounding_box'] = bbox
            f.update()
        self.features.flush()
        return self.get_bounding_boxes()

    def compute_geometry(self, overwrite_table=False):
        """Compute each feature geometry from the feature map.

        This method computes the feature centers, volume and bounding boxes
        from the feature map and update the feature data table. This applies
        only to features represented in the feature map. If other features are
        present, their information is unchanged unless the option
        `overwrite_table` is activated.

        :param bool overwrite_table: if this is True, the features present in
        the data table and not in the feature map are removed from it.
        """
        feature_ids = self.get_ids_from_map()
        if overwrite_table and self.features.nrows > 0:
            self.features.remove_rows(start=0)
            for feature_id in feature_ids:
                index = self.features.get_where_list('(id == i)')
                if len(index) > 0:
                    feature = self.features[index]
                else:
                    feature = np.zeros((1,), dtype=self.features.dtype)
                feature['bounding_box'] = self.compute_bounding_box(feature_id)
                feature['center'] = self.compute_center(feature_id)
                feature['volume'] = self.compute_volume(feature_id)
                if len(index) > 0:
                    self.features[index] = feature
                else:
                    self.features.append(feature)
            self.features.flush()
        else:
            self.recompute_bounding_boxes()
            self.recompute_centers()
            self.recompute_volumes()
        return
