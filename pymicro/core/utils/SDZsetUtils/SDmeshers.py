#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SD meshers module to allow mesher tools & SampleData instances interactions


"""

## Imports
import os
import shutil
import pathlib
import subprocess
import numpy as np
from pymicro.core.samples import SampleData


# SD Zset mesher class
class SDZsetMesher():
    """Class to use Zset meshers on SampleData objects mesh and image data.

    This class is an inteface between the SampleData data platform class and
    the mesh tools used and/or developed at Centre des Matériaux, Mines Paris.

    The use of this class methods requires a Zset compatible environnement.
    That means that the Zset software must be available in your PATH
    environnement variable at least. Some methods also rely on the MATLAB
    software, and hence also require a compatible envrionnement.

    .. rubric:: CONSTRUCTOR PARAMETERS

    :data: SampleData object
        SampleData object containing the input data for the meshing operations.
        If `None`, the class will open a SampleData instance from a datafile
        path.
    :sd_datafile: str, optional ('')
        Path of the hdf5/xdmf data file to open as the SampleData instance
        containing mesh tools input data.
    :verbose: `bool`, optional (False)
        verbosity flag
    """

    def __init__(self, data=None, sd_datafile='', verbose=False):
        """SDZsetMesher class constructor."""
        if data is None:
            self.data = SampleData(filename=sd_datafile, verbose=verbose)
        else:
            self.data = data
        self._verbose = verbose
        return

    def multi_phase_mesher(self, multiphase_image_name='', meshname='',
                           indexname='', location='', load_surface_mesh=False,
                           bin_fields_from_sets=True, replace=False,
                           **keywords):
        """Create a conformal mesh from a multiphase image.

        A Matlab multiphase mesher is called to create a conformal mesh of a
        multiphase image: a voxelized/pixelized field of integers identifying
        the different phases of a microstructure. Then, the mesh is stored in
        the calling SampleData instance at the desired location with the
        desired name and Indexname.

        The meshing procedure involves the construction of a surface mesh that
        is conformant with the phase boundaries in the image. The space
        between boundary elements is then filled with tetrahedra to construct
        a volumic mesh. This intermediate surface mesh can be store into the
        SampleData instance if required.

        The mesher path must be correctly set in the `global_variables.py`
        file, as well as the definition and path of the Matlab command. The
        multiphase mesher is a Matlab program that has been developed by
        Franck Nguyen (Centre des Matériaux).

        :param str multiphase_image_name: Path, Name, Index Name or Alias of
            the multiphase image field to mesh in the SampleData instance.
        :param str meshname: name used to create the Mesh group in dataset
        :param indexname: Index name used to reference the Mesh group
        :param str location: Path, Name, Index Name or Alias of the parent
            group where the Mesh group is to be created
        :param bool load_surface_mesh: If `True`, load the intermediate
            surface mesh in the dataset.
        :param bool bin_fields_from_sets: If `True`, stores all Node and
            Element Sets in mesh_object as binary fields (1 on Set, 0 else)
        :param bool replace: if `True`, overwrites pre-existing Mesh group
            with the same `meshname` to add the new mesh.
        """
        self.data.sync()
        # Set data and file pathes
        DATA_DIR, _ = os.path.split(self.data.h5_path)
        DATA_PATH = self.data._name_or_node_to_path(multiphase_image_name)
        OUT_DIR = os.path.join(DATA_DIR, 'Tmp/')
        # create temp directory for mesh files
        if not os.path.exists(OUT_DIR):
           os.mkdir(OUT_DIR)
        # Get meshing parameters eventually passed as keyword arguments
        mesh_params = self._get_mesher_parameters(**keywords)
        # launch mesher
        self._launch_mesher(DATA_PATH, self.data.h5_path, OUT_DIR, mesh_params)
        # Add mesh to SD instance
        out_file = os.path.join(OUT_DIR,'Tmp_mesh_vor_tetra_p.geof')
        self.data.add_mesh(file=out_file, meshname=meshname,
                           indexname=indexname, location=location,
                           replace=replace,
                           bin_fields_from_sets=bin_fields_from_sets)
        # Add surface mesh if required
        if load_surface_mesh:
            out_file = os.path.join(OUT_DIR,'Tmp_mesh_vor.geof')
            self.data.add_mesh(file=out_file, meshname=meshname+'_surface',
                               location=location, replace=replace,
                               bin_fields_from_sets=bin_fields_from_sets)
        # Remove tmp mesh files
        shutil.rmtree(OUT_DIR)
        return

    def morphological_image_cleaner(self, target_image_field='',
                                    clean_fieldname='', indexname='',
                                    replace=False, **keywords):
        """Apply a morphological cleaning treatment to a multiphase image.

        A Matlab morphological cleaner is called to smooth the morphology of
        the different phases of a multiphase image: a voxelized/pixelized
        field of integers identifying the different phases of a
        microstructure.

        This cleaning treatment is typically used to improve the quality of a
        mesh produced from the multiphase image, or improved image based
        mechanical modelisation techniques results, such as FFT-based
        computational homogenization solvers.

        The cleaner path must be correctly set in the `global_variables.py`
        file, as well as the definition and path of the Matlab command. The
        multiphase cleaner is a Matlab program that has been developed by
        Franck Nguyen (Centre des Matériaux).

        :param str target_image_field: Path, Name, Index Name or Alias of
            the multiphase image field to clean in the SampleData instance
        :param str clean_fieldname: name used to add the morphologically
            cleaned field to the image group
        :param indexname: Index name used to reference the Mesh group
        :param bool replace: If `True`, overwrite any preexisting field node
            with the name `clean_fieldname` in the image group with the
            morphologically cleaned field.
        """
        imagename = self.data._get_parent_name(target_image_field)
        self.data.sync()
        # Set data and file pathes
        DATA_DIR, _ = os.path.split(self.data.h5_path)
        DATA_PATH = self.data._name_or_node_to_path(target_image_field)
        OUT_FILE = os.path.join(DATA_DIR, 'Clean_image.mat')
        # launch mesher
        self._launch_morphocleaner(DATA_PATH, self.data.h5_path, OUT_FILE)
        # Add image to SD instance
        from scipy.io import loadmat
        mat_dic = loadmat(OUT_FILE)
        image = mat_dic['mat3D_clean']
        self.data.add_field(gridname=imagename, fieldname=clean_fieldname,
                            array=image, replace=replace, **keywords)
        # Remove tmp mesh files
        os.remove(OUT_FILE)
        return

    def _launch_morphocleaner(self, path, filename, out_file):
        from SDUtilsGlobals import MATLAB, MATLAB_OPTS
        from SDUtilsGlobals import CLEANER_TEMPLATE, CLEANER_TMP
        # Create specific mesher script
        shutil.copyfile(CLEANER_TEMPLATE, CLEANER_TMP)
        with open(CLEANER_TMP,'r') as file:
            lines = file.read()
        lines = lines.replace('DATA_PATH', path)
        lines = lines.replace('DATA_H5FILE', filename)
        lines = lines.replace('OUT_FILE', out_file)
        with open(CLEANER_TMP,'w') as file:
            file.write(lines)
        # Launch mesher
        CWD = os.getcwd()
        matlab_command = '"'+"run('" + CLEANER_TMP + "');exit;"+'"'
        subprocess.run(args=[MATLAB,MATLAB_OPTS,matlab_command])
        os.remove(CLEANER_TMP)
        os.chdir(CWD)
        return

    def _get_mesher_parameters(self, **keywords):
        MEM = 50
        HGRAD = 1.5
        HMIN = 5
        HMAX = 50
        HAUSD = 3
        ANG = 50
        if 'MEM' in keywords: MEM = keywords['MEM']
        if 'HGRAD' in keywords: HGRAD = keywords['HGRAD']
        if 'HMIN' in keywords: HMIN = keywords['HMIN']
        if 'HMAX' in keywords: HMAX = keywords['HMAX']
        if 'HAUSD' in keywords: HAUSD = keywords['HAUSD']
        if 'ANG' in keywords: ANG = keywords['ANG']
        return {'MEM':MEM, 'HGRAD':HGRAD, 'HMIN':HMIN, 'HMAX':HMAX,
                'HAUSD':HAUSD, 'ANG':ANG}

    def _launch_mesher(self, path, filename, out_dir, params):
        from SDUtilsGlobals import MATLAB, MATLAB_OPTS
        from SDUtilsGlobals import MESHER_TEMPLATE, MESHER_TMP
        print(filename)
        # Create specific mesher script
        shutil.copyfile(MESHER_TEMPLATE, MESHER_TMP)
        with open(MESHER_TMP,'r') as file:
            lines = file.read()
        lines = lines.replace('DATA_PATH', path)
        lines = lines.replace('DATA_H5FILE', filename)
        lines = lines.replace('OUT_DIR', out_dir)
        lines = lines.replace('MEM', str(params['MEM']))
        lines = lines.replace('HGRAD', str(params['HGRAD']))
        lines = lines.replace('HMIN', str(params['HMIN']))
        lines = lines.replace('HMAX', str(params['HMAX']))
        lines = lines.replace('HAUSD', str(params['HAUSD']))
        lines = lines.replace('ANG', str(params['ANG']))
        with open(MESHER_TMP,'w') as file:
            file.write(lines)
        # Launch mesher
        CWD = os.getcwd()
        matlab_command = '"'+"run('" + MESHER_TMP + "');exit;"+'"'
        subprocess.run(args=[MATLAB,MATLAB_OPTS,matlab_command])
        os.remove(MESHER_TMP)
        os.chdir(CWD)
        return

