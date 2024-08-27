from pymicro.crystal.microstructure import Microstructure, Orientation
import networkx as nx
import numpy as np
import vtk
from vtk.util import numpy_support
from BasicTools.Bridges import vtkBridge
try:
    from skimage.future import graph
except ImportError:
    from skimage import graph

def create_graph(m: Microstructure) -> graph.RAG:
    """Create the graph of this microstructure.

    This method processes a `Microstructure` instance using a Region Adgency
    Graph built with the crystal misorientation between neighbors as weights.
    The graph has a node per grain and a connection between neighboring
    grains of the same phase. The misorientation angle is attach to each edge.
    
    :return rag: the region adjency graph of this microstructure.
    """

    print('build the region agency graph for this microstructure')
    rag = graph.RAG(m.get_grain_map(), connectivity=1)

    # remove node and connections to the background
    if 0 in rag.nodes:
        rag.remove_node(0)

    # get the grain infos
    grain_ids = m.get_grain_ids()
    rodrigues = m.get_grain_rodrigues()
    centers = m.get_grain_centers()
    volumes = m.get_grain_volumes()
    phases = m.grains[:]['phase']
    for grain_id, d in rag.nodes(data=True):
        d['label'] = [grain_id]
        index = grain_ids.tolist().index(grain_id)
        d['rod'] = rodrigues[index]
        d['center'] = centers[index]
        d['volume'] = volumes[index]
        d['phase'] = phases[index]

    # assign grain misorientation between neighbors to each edge of the graph
    for x, y, d in rag.edges(data=True):
        if rag.nodes[x]['phase'] != rag.nodes[y]['phase']:
            # skip edge between neighboring grains of different phases
            continue
        sym = m.get_phase(phase_id=rag.nodes[x]['phase']).get_symmetry()
        o_x = Orientation.from_rodrigues(rag.nodes[x]['rod'])
        o_y = Orientation.from_rodrigues(rag.nodes[y]['rod'])
        mis = np.degrees(o_x.disorientation(o_y, crystal_structure=sym)[0])
        d['misorientation'] = mis
    
    # return our graph
    return rag

def store_graph(m: Microstructure, rag: graph.RAG):
    """Store the microstructure graph in a form that can be visualized in ParaView."""
    if rag is None:
        rag = create_graph(m)
    # create  points for each node in the graph
    points = vtk.vtkPoints()
    # vtkUnstructuredGrid instance for all the cells
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    for grain_id, d in rag.nodes(data=True):
        # scale coordinates with the grain size and center on the grain
        points.InsertNextPoint(d['center'])
    # allocate memory for the cells representing the edges
    grid.Allocate(len(rag.edges), 1)
    for e in rag.edges:
        Ids = vtk.vtkIdList()
        Ids.InsertNextId(e[0] - 1)
        Ids.InsertNextId(e[1] - 1)
        grid.InsertNextCell(vtk.VTK_LINE, Ids)
    # add arrays containing useful information to the grid
    grain_ids = m.get_grain_ids()
    grain_sizes = m.compute_grain_equivalent_diameters()
    grain_ids_array = numpy_support.numpy_to_vtk(grain_ids)
    grain_sizes_array = numpy_support.numpy_to_vtk(grain_sizes)
    grain_ids_array.SetName('grain_ids')
    grain_sizes_array.SetName('grain_sizes')
    grid.GetCellData().AddArray(grain_ids_array)
    grid.GetCellData().AddArray(grain_sizes_array)

    # now add the created mesh to the microstructure
    mesh = vtkBridge.VtkToMesh(grid)
    m.add_mesh(mesh_object=mesh, location='/MeshData', meshname='graph', indexname='mesh_graph', replace=True)

def segment_mtr(m: Microstructure, labels_seg=None, mis_thr=20., min_area=500, store=False):
    """Segment micro-textured regions (MTR).

    This method processes a `Microstructure` instance to segment the MTR
    with the specified parameters.

    :param ndarray labels_seg: a pre-segmentation of the grain map, the full
    grain map will be used if not specified.
    :param float mis_thr: threshold in misorientation used to cut the graph.
    :param int min_area: minimum area used to define a MTR.
    :param bool store: flag to store the segmented array in the microstructure.
    :return mtr_labels: array with the labels of the segmented regions.
    """
    rag_seg = m.graph()

    # cut our graph with the misorientation threshold
    rag = rag_seg.copy()
    edges_to_remove = [(x, y) for x, y, d in rag.edges(data=True)
                        if d['misorientation'] >= mis_thr]
    rag.remove_edges_from(edges_to_remove)

    comps = nx.connected_components(rag)
    map_array = np.arange(labels_seg.max() + 1, dtype=labels_seg.dtype)
    for i, nodes in enumerate(comps):
        # compute area of this component
        area = np.sum(np.isin(labels_seg, list(nodes)))
        if area < min_area:
            # ignore small MTR (simply assign them to label zero)
            i = 0
        for node in nodes:
            for label in rag.nodes[node]['label']:
                map_array[label] = i
    mtr_labels = map_array[labels_seg]
    print('%d micro-textured regions were segmented' % len(np.unique(mtr_labels)))
    if store:
        m.add_field(gridname='CellData', fieldname='mtr_segmentation',
                        array=mtr_labels, replace=True)
    return mtr_labels

