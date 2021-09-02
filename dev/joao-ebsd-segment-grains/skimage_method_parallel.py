
import numpy as np
from skimage.future.graph import RAG
import itertools
from scipy.spatial.distance import cdist
import progressbar
from pymicro.crystal.microstructure import Orientation
from pymicro.crystal.lattice import Symmetry
from skimage.future.graph import cut_threshold
from multiprocessing import Pool
import os

import contextlib

    
    
def worker(lab1, lab2, euler1, euler2):
    o1 = Orientation.from_euler(np.degrees(euler1))
    o2 = Orientation.from_euler(np.degrees(euler2))
    disor = o1.disorientation(o2, crystal_structure=Symmetry.hexagonal)  # angles (weights) are in radians
    return (lab1, lab2, disor[0])
    

def skimage_method_parallel(scan_, threshold_deg, nprocs):
    shape = scan_.iq.shape
    size = shape[0] * shape[1]
    # each pixel has its own label
    labels = np.arange(size).reshape(shape)
    rag = RAG(label_image=labels, connectivity=1) 
    for node in rag.nodes:
        rag.nodes[node]['labels'] = [node]
    pixel_indices = np.array(list(itertools.product(range(shape[0]), range(shape[1]))))
    pixel_position_distances = cdist(pixel_indices, pixel_indices, metric='cityblock')
    pixels_to_get = np.stack(np.where(pixel_position_distances == 1)).T
    pixels_to_get = pixels_to_get[pixels_to_get[:, 0] < pixels_to_get[:, 1]]  # get rid of duplicates
    args_list = []
    for pix1_idx, pix2_idx in pixels_to_get:
        pixel1_pos = tuple(pixel_indices[pix1_idx])
        pixel2_pos = tuple(pixel_indices[pix2_idx])
        euler1 = scan_.euler[pixel1_pos]
        euler2 = scan_.euler[pixel2_pos]
        lab1 = labels[pixel1_pos]
        lab2 = labels[pixel2_pos]
        args_list.append((lab1, lab2, euler1, euler2))
    
    with contextlib.closing(Pool(processes=nprocs,)) as pool:
        edges = pool.starmap_async(worker, args_list)
    pool.join()
    assert edges.ready()
    assert edges.successful()
    edges = [e for e in edges.get()]
    
    for lab1, lab2, angle in edges:
        rag.add_edge(lab1, lab2, dict(weight=angle))
        
     
    threshold_rad = np.radians(threshold_deg)
    
    return cut_threshold(labels, rag, threshold_rad)    
