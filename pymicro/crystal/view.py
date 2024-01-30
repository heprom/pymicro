from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import filters
from pymicro.crystal.microstructure import Microstructure

class View_slice:

    def __init__(self, m, plane='XY', unit='mm'):
        self.microstructure = m
        self.plane = plane
        self.unit = unit
        self.allowed_units = ['mm', 'pixel']
        self.allowed_planes = ['YZ', 'XZ', 'XY']
        self.allowed_annotation = ['grain_ids', 'lattices', 'slip_traces']
        self.annotate = None

    def set_unit(self, unit):
        if unit in self.allowed_units:
            self.unit = unit
        else:
            print('unit must be one of', self.allowed_units)
            print('unit is', self.unit)

    def set_plane(self, plane):
        if plane in self.allowed_planes:
            self.plane = plane
        else:
            print('view plane must be one of', self.allowed_planes)
            print('view plane is', self.plane)

    def view_map_slice(self, map_name='grain_map', slice=None,
                    color='random', show_mask=True, highlight_ids=None,
                    show_grain_ids=False, circle_ids=False, show_lattices=False,
                    slip_system=None, axis=[0., 0., 1], show_slip_traces=False,
                    hkl_planes=None, show_phases=False, show_gb=False,
                    display=True):
        """A simple utility method to show one microstructure slice.
        
        The method extract a 2D orthogonal slice from the microstructure to
        display it. The extract plane can be one of 'XY', 'YZ' or 'XZ'.
        The microstructure can be colored using different fields such as a
        random color (default), the grain ids, the Schmid factor or the grain
        orientation using IPF coloring.
        The plot can be customized in several ways. Annotations can be added
        in the grains (ids, lattice plane traces) and the list of grains where
        annotations are shown can be controlled using the `highlight_ids`
        argument. By default, if present, the mask will be shown.

        :param str map_name: name of field to plot from the CellData image
            ('grain_map', 'phase_map', 'my_data_field'...). Default is
            'grain_map'.
        :param int slice: the slice number
        :param str plane: the cut plane, must be one of 'XY', 'YZ' or 'XZ'
        :param str color: a string to chose the colormap from ('random',
            'grain_ids', 'schmid', 'ipf')
        :param bool show_mask: a flag to show the mask by transparency.
        :param list highlight_ids: a list of grain ids to restrict the
            annotations (by default all grains are annotated).
        :param bool show_grain_ids: a flag to annotate the plot with the grain
            ids.
        :param bool circle_ids: activate larger and circled grain ids
            annotations.
        :param bool show_lattices: a flag to annotate the plot with the crystal 
            lattices showing the grain orientations. 
        :param slip_system: an instance (or a list of instances) of the class
            SlipSystem to compute the Schmid factor.
        :param axis: the unit vector for the load direction to compute
            the Schmid factor or to display IPF coloring.
        :param bool show_slip_traces: activate slip traces plot in each grain.
        :param list hkl_planes: the list of planes to plot the slip traces.
        :param str unit: switch between mm and pixel units.
        :param bool show_gb: show the grain boundaries.
        :param bool show_phases: show the phases from the phase mapby hatching
            regions on the slice. Works only if the phase map exists.
        :param bool display: if True, the show method is called, otherwise,
            the figure is simply returned.
        """
        # TODO: too many arguments, needs refactoring
        m = self.microstructure
        # get the slice to plot
        slice_map, vmin, vmax = self._get_slice_map(map_name, slice,
                                                    color, slip_system, axis=axis)
        if slice_map is None:
            print('Could not get slice to visualize, returning. ')
            return
        extent = self._set_extent(slice_map)
        # prepare figure
        fig, ax = plt.subplots()
        # plot map
        self._print_slice(ax, slice_map, extent, color, vmin, vmax)
        if show_mask:
            mask_slice, vmin, vmax = self._get_slice_map("mask", slice, color)
            if mask_slice is None:
                print('Could not get mask slice, proceeding... ')
            else:
                self._overlay_slice(ax, mask_slice, extent)
        # specific grain related annotations for grain map
        if (show_grain_ids or show_lattices or show_slip_traces):
            # recompute grain map slice : call with random cmap to handle
            # schmid and ipf case
            grains_slice, vmin, vmax = self._get_slice_map("grain_map",
                                slice, "random", slip_system, axis=axis)
            # compute grain ids and grain centers
            if not highlight_ids:
                highlight_ids = np.unique(grains_slice).tolist()
                highlight_ids.remove(0)
            centers, sizes, highlight_ids = self._get_slice_grain_centers(grains_slice, highlight_ids)
            if show_grain_ids:
                # print grain id on center on each grain
                self._overlay_ids(ax, centers, highlight_ids, circle_ids)
            if show_lattices:
                # plot the crystal lattice on each grain
                self._overlay_crystal_lattices(ax, centers, highlight_ids,
                                                sizes, extent)
            if show_slip_traces and hkl_planes:
                # print slip traces for desired slip system on each grain
                # TODO: adapt for multi phase material
                self._overlay_slip_traces(ax, centers, highlight_ids,
                                            sizes, hkl_planes, extent)
        # show grain boundaries on the map
        if show_gb:
            # recompute grain map slice : call with random cmap to handle
            # schid and ipf case
            grains_slice, vmin, vmax = self._get_slice_map("grain_map",
                                slice, "random", slip_system, axis=axis)
            if grains_slice is None:
                print('Could not get grain map to plot grain boundaries'
                        ', proceeding... ')
            else:
                extent = self._set_extent(grains_slice)
                self._overlay_gb(ax, grains_slice, extent)
        # phases on the map
        if show_phases:
            phase_slice, vmin, vmax = self._get_slice_map("phase_map",
                                    slice, color, slip_system)
            if phase_slice is None:
                print('Could not get phase slice, proceeding... ')
            else:
                extent = self._set_extent(phase_slice, vertical_reverse=False)
                self._hatch_phases(ax, phase_slice, extent)
        if display:
            plt.show()
        return fig, ax

    def _get_slice_map(self, map_name, slice, color, slip_system=None,
                        axis=[0., 0., 1]):
        """Get slice of a CellData field for view slice visualization."""
        vmin=None
        vmax=None
        m = self.microstructure
        # check map presence
        if m._is_empty(map_name):
            msg = (f'Microstructure instance must have a {map_name} field to '
                    'view it with "view_slice"')
            print(msg)
            return None, None, None
        cut_axis = self.allowed_planes.index(self.plane)
        # get right map to plot --> special case for grain and phase map to
        # get active map
        if map_name == "grain_map":
            map_array = m.get_grain_map()
            vmin = 0
        elif map_name == "phase_map":
            map_array = m.get_phase_map()
        elif map_name == "mask":
            map_array = m.get_mask()
        else:
            map_array = m[map_name]
        # Check if slice value fits and otherwise computes half size slice
        if slice is None or slice > map_array.shape[cut_axis] - 1 or slice < 0:
            slice = map_array.shape[cut_axis] // 2
            print('using slice value %d' % slice)
        # cut slice from map
        map_slice = map_array.take(indices=slice, axis=cut_axis)
        if map_name == "grain_map":
            # Compute slice for schmid factor
            if (color == 'schmid'):
                map_slice = self._build_schmid_factor_map(map_slice,
                                                          slip_system, axis)
                vmin = 0
                vmax = 0.5
            # Compute image for IPF
            if (color == 'ipf'):
                map_slice = self._build_ipf_map(map_slice, axis)
        return map_slice, vmin, vmax

    def _get_slice_grain_centers(self, grains_slice, highlight_ids):
        """Get coordinates of grain centers in slice."""
        gids = np.intersect1d(np.unique(grains_slice), highlight_ids)
        centers = np.zeros((len(gids), 2), dtype='f')
        sizes = np.zeros(len(gids), dtype='f')
        for i, gid in enumerate(gids):
            sizes[i] = np.sum(grains_slice == gid)
            centers[i] = ndimage.measurements.center_of_mass(
                grains_slice == gid, grains_slice)
        if self.unit == 'mm':
            centers += np.array([-0.5 * grains_slice.shape[0],
                                 -0.5 * grains_slice.shape[1]])
            centers *= self.microstructure.get_voxel_size()
        return centers, sizes, gids

    def _build_schmid_factor_map(self, grains_slice, slip_system, axis):
        """Compute Schmid factor map for view slice."""
        schmid_image = np.zeros_like(grains_slice, dtype=float)
        # get ids of the grain whose schmid factor must be computed
        gids = np.intersect1d(np.unique(grains_slice), self.microstructure.get_grain_ids())
        for gid in gids:
            o = self.microstructure.get_grain(gid).orientation
            if type(slip_system) == list:
                # compute max Schmid factor
                sf = max(o.compute_all_schmid_factors(
                    slip_systems=slip_system, load_direction=axis))
            else:
                sf = o.schmid_factor(slip_system, axis)
            schmid_image[grains_slice == gid] = sf
        return schmid_image

    def _build_ipf_map(self, grains_slice, axis):
        """Compute IPF map for view slice."""
        ipf_image = np.zeros((*grains_slice.shape, 3), dtype=float)
        gids = np.intersect1d(np.unique(grains_slice), self.microstructure.get_grain_ids())
        for gid in gids:
            o = self.microstructure.get_grain(gid).orientation
            try:
                c = o.ipf_color(axis,
                                symmetry=self.microstructure.get_lattice().get_symmetry(),
                                saturate=True)
            except ValueError:
                print('problem moving to the fundamental zone for '
                      'rodrigues vector {}'.format(o.rod))
                c = np.array([0., 0., 0.])
            # print(c)
            ipf_image[grains_slice == gid] = c
        return ipf_image

    def _set_slice_colors(self, color):
        """Set colormap to plot grain map in slice view."""
        if color == 'random':
            cmap = Microstructure.rand_cmap(first_is_black=True)
        elif (color == 'grain_ids') or (color == 'values'):
            cmap = 'viridis'
        elif color == 'schmid':
            cmap = plt.cm.gray
        elif color == 'ipf':
            cmap = None
        else:
            print('unknown color scheme requested, please chose between '
                  '{random, grain_ids, values, schmid, ipf}, returning')
            return
        return cmap

    def _set_extent(self, slice_map, vertical_reverse=True):
        """Set length unit to plot grain map in slice view."""
        # extent is used to control size the axes in which the images are
        # plotted by "imshow" or "contourf"
        if vertical_reverse:
            # extent seems to need to be reversed in Y direction for contour plots
            # Why ?? --> used to plot hatched regions with contourf
            extent = [0, slice_map.shape[0], slice_map.shape[1], 0]
        else:
            extent = [0, slice_map.shape[0], 0, slice_map.shape[1]]
        if self.unit == 'mm':
            # work out extent in mm unit
            extent += np.array([-0.5 * slice_map.shape[0],
                                -0.5 * slice_map.shape[0],
                                -0.5 * slice_map.shape[1],
                                -0.5 * slice_map.shape[1]])
            extent *= self.microstructure.get_voxel_size()
        return extent

    def _print_slice(self, ax, slice_map, extent, color, vmin, vmax):
        """Plot slice with viridis color map."""
        if color == 'ipf':
            image = slice_map.transpose(1, 0, 2)
        else:
            image = slice_map.T
        cmap = self._set_slice_colors(color)
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='nearest', extent=extent)
        # add color bar if schmid factor plot requested
        if color == 'schmid':
            cb = plt.colorbar(im)
            cb.set_label('Schmid factor')
        # add axes legend
        ax.xaxis.set_label_position('top')
        plt.xlabel(self.plane[0] + ' [%s]' % self.unit)
        plt.ylabel(self.plane[1] + ' [%s]' % self.unit)
        return

    def _overlay_slice(self, ax, slice_map, extent):
        """Overlay a field over the view slice plot (usually mask)."""
        from pymicro.view.vol_utils import alpha_cmap
        ax.imshow(slice_map.T, cmap=alpha_cmap(opacity=0.3), extent=extent)

    def _overlay_ids(self, ax, centers, highlight_ids, circle_ids):
        """Overlay grain ids on grain map slice view."""
        if circle_ids:
            bbox_dic = dict(boxstyle="circle,pad=0.1", fc="white")
        else:
            bbox_dic = None
        for i, gid in enumerate(highlight_ids):
            ax.annotate('%d' % gid, xycoords='data',
                         xy=(centers[i, 0], centers[i, 1]),
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='k', fontsize=12,
                         bbox=bbox_dic)

    def _overlay_crystal_lattices(self, ax, centers, highlight_ids, sizes, extent):
        """Overlay crystal lattice on grain map slice view."""
        #FIXME adapt for multi phase material
        centers = self.microstructure.get_grain_centers(id_list=highlight_ids)
        sizes = self.microstructure.get_grain_volumes(id_list=highlight_ids)
        coords, edge_point_ids = self.microstructure.get_lattice().get_points(origin='mid')
        # list of point ids to draw the cell edges
        edge_point_ids = np.array([[0, 1], [1, 3], [2, 3], [0, 2], 
                                   [4, 5], [5, 7], [6, 7], [4, 6], 
                                   [0, 4], [1, 5], [2, 6], [3, 7]], dtype=np.uint8)
        #FIXME handle all 3 planes
        if self.plane != 'XY':
            print('lattices can only be plotted in the XY plane for now, skipping this...')
            return
        for k, gid in enumerate(highlight_ids):
            # now apply the crystal orientation
            g = self.microstructure.get_grain(gid).orientation.orientation_matrix()
            coords_rot = np.empty_like(coords)
            for i in range(8):
                # scale coordinates with the grain size and center on the grain
                coords_rot[i] = centers[k] + np.sqrt(sizes[k]) * np.dot(g.T, coords[i])
            # scale coords to the proper unit
            if self.unit == 'pixel':
                coords_rot = coords_rot / self.microstructure.get_voxel_size() + 0.5 * np.array(self.microstructure.get_grain_map().shape)
            for edge_id in range(12):
                ax.plot(coords_rot[edge_point_ids[edge_id, :], 0],
                        coords_rot[edge_point_ids[edge_id, :], 1], 'k-')
        # prevent axis to move due to lines spanning outside the map
        plt.axis(extent)

    def _overlay_slip_traces(self, ax, centers, highlight_ids, sizes,
                             hkl_planes, extent):
        """Overlay slip traces on grain map slice view."""
        # TODO: adapt for multi phase material
        # TODO: scale slip traces with surface/slip plane angle ?
        cut_axis = self.allowed_planes.index(self.plane)
        n_int = np.zeros(3)
        n_int[cut_axis] = 1.
        view_up = [0, -1, 0]
        if self.plane == 'XZ':
            view_up = [0, 0, -1]
            n_int[cut_axis] = -1.
        elif self.plane == 'YZ':
            view_up = [0, 0, -1]
        for i, gid in enumerate(highlight_ids):
            g = self.microstructure.get_grain(gid)
            for hkl in hkl_planes:
                trace = hkl.slip_trace(g.orientation,
                                       n_int=n_int,
                                       view_up=view_up,
                                       trace_size=0.8 * np.sqrt(sizes[i]),
                                       verbose=False)
                if self.unit == 'mm':
                    trace *= self.microstructure.get_voxel_size()
                color = 'k'
                x = centers[i][0] + np.array([-trace[0] / 2, trace[0] / 2])
                y = centers[i][1] + np.array([-trace[1] / 2, trace[1] / 2])
                ax.plot(x, y, '-', linewidth=1, color=color)
            # prevent axis to move due to traces spanning outside the map
            plt.axis(extent)

    def _overlay_gb(self, ax, grains_slice, extent):
        """Plot grain boundaries in slice view."""
        grain_boundaries = filters.roberts(grains_slice) > 0
        gb_slice = np.ma.masked_where(grain_boundaries == 0, grain_boundaries)
        ax.imshow(gb_slice.T, extent=extent, cmap='Greys_r')
        return

    def _hatch_phases(self, ax, phase_slice, extent):
        """Plot phases in slice view by hatching regions from phase map."""
        # get number of levels
        n_phases = np.unique(phase_slice).shape[0]
        # create hatches list  --> set for only 10 phases
        H_list = [None, '+','x','.','/','\\','-','|','.-','/-']
        # plot phases as hatched regions
        cs = ax.contourf(phase_slice.T, colors='black', vmin=0,
                     levels=n_phases-1, alpha=0.0, hatches=H_list,
                     extent=extent)
        # build legend for phase hatches
        artists, labels = cs.legend_elements(variable_name='phase',
                                        str_format='{:2.1f}'.format)
        # overwrite labels with name of phases
        for i in range(len(artists)):
            labels[i] = f'$\\varphi_{i+1}$'
        legend = plt.legend(artists, labels, handleheight=2,
                        framealpha=1, bbox_to_anchor=(0.5, 1.05),
                        loc="lower center", mode="None", ncol=4,
                        markerscale=1.5)
        handles = legend.legendHandles
        # reduce width off hatches to allow visualization of grain
        # ids etc...
        for i, handle in enumerate(handles):
            handle.set_edgecolor("black") # set_edgecolors
            handle.set_facecolor("white") # set_edgecolors
            handle.set_hatch(H_list[i])
            handle.set_alpha(0.8)
        plt.rcParams['hatch.linewidth'] = 0.5
        return
