#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SD meshers module to allow mesher tools & SampleData instances interactions


"""

## Imports
import os
import shutil
import subprocess
from pymicro.core.utils.templateUtils import ScriptTemplate
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
        the Mesher SampleData instance at the desired location with the
        desired meshname and Indexname.

        The meshing procedure involves the construction of a surface mesh that
        is conformant with the phase boundaries in the image. The space
        between boundary elements is then filled with tetrahedra to construct
        a volumic mesh. This intermediate surface mesh can be store into the
        SampleData instance if required.

        The mesher path must be correctly set in the `global_variables.py`
        file, as well as the definition and path of the Matlab command. The
        multiphase mesher is a Matlab program that has been developed by
        Franck Nguyen (Centre des Matériaux). It also relies on the Zset
        and the MeshGems-Tetra softwares.

        Mesh parameters can be passed as optional keyword arguments (see
        MeshGems - Tetra software User Manual for more details.)

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
        :param float MEM: (optional) memory (in Mb) to reserve for the mesh
            construction
        :param float HGRAD: Controls the ratio between the size of elements
            close to surfaces and elements in the bulk of the mesh.
        :param float HMIN: Minimal size for the mesh edges
        :param float HMAX: Maximal size for the mesh edges
        :param float HAUSD: Hausdorff distance between the initially produced
            surface mesh and the volume mesh created by MeshGems from the
            surface mesh.
        """
        # Defaults mesh parameters
        default_params = {'MEM':50,'HGRAD':1.5,'HMIN':5,'HMAX':50, 'HAUSD':3,
                          'ANG':50}
        # Local Imports
        from pymicro.core.utils.SDUtilsGlobals import MATLAB, MATLAB_OPTS
        from pymicro.core.utils.SDUtilsGlobals import (MESHER_TEMPLATE,
                                                       MESHER_TMP)
        # get data and output pathes
        DATA_PATH = self.data._name_or_node_to_path(multiphase_image_name)
        DATA_DIR, _ = os.path.split(self.data.h5_path)
        OUT_DIR = os.path.join(DATA_DIR, 'Tmp/')
        # create temp directory for mesh files
        if not os.path.exists(OUT_DIR): os.mkdir(OUT_DIR)
        # perpare mesher script
        mesher = ScriptTemplate(template_file=MESHER_TEMPLATE,
                                script_file = MESHER_TMP, autodelete=True,
                                script_command=MATLAB)
        # set command line options
        matlab_command = '"'+"run('" + MESHER_TMP + "');exit;"+'"'
        mesher.set_script_command_options([MATLAB_OPTS, matlab_command])
        # set mesher script parameters
        mesher_arguments = {'DATA_PATH':DATA_PATH,'OUT_DIR':OUT_DIR,
                            'DATA_H5FILE': self.data.h5_path}
        mesher_arguments.update(default_params)
        mesher.set_arguments(mesher_arguments, **keywords)
        mesher.createScript(filename=MESHER_TMP)
        # launch mesher
        CWD = os.getcwd()
        self.data.sync() # flushes H5 dataset
        mesher.launchScript(workdir=OUT_DIR, append_filename=False)
        os.chdir(CWD)
        # Add mesh to SD instance
        out_file = os.path.join(OUT_DIR,'Tmp_mesh_vor_tetra.geof')
        self.data.add_mesh(file=out_file, meshname=meshname, replace=replace,
                           indexname=indexname, location=location,
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
        # Local Imports
        from pymicro.core.utils.SDUtilsGlobals import MATLAB, MATLAB_OPTS
        from pymicro.core.utils.SDUtilsGlobals import (CLEANER_TEMPLATE,
                                                       CLEANER_TMP)
        self.data.sync()
        # get data and output pathes
        imagename = self.data._get_parent_name(target_image_field)
        DATA_DIR, _ = os.path.split(self.data.h5_path)
        DATA_PATH = self.data._name_or_node_to_path(target_image_field)
        OUT_FILE = os.path.join(DATA_DIR, 'Clean_image.mat')
        # perpare cleaner script
        cleaner = ScriptTemplate(template_file=CLEANER_TEMPLATE,
                                 script_file = CLEANER_TMP, autodelete=True,
                                 script_command=MATLAB)
        # set command line options
        matlab_command = '"'+"run('" + CLEANER_TMP + "');exit;"+'"'
        cleaner.set_script_command_options([MATLAB_OPTS, matlab_command])
        # set mesher script parameters and create script file
        cleaner_arguments = {'DATA_PATH':DATA_PATH,'OUT_FILE':OUT_FILE,
                            'DATA_H5FILE': self.data.h5_path}
        cleaner.set_arguments(cleaner_arguments)
        cleaner.createScript(filename=CLEANER_TMP)
        # launch cleaner
        CWD = os.getcwd()
        self.data.sync() # flushes H5 dataset
        cleaner.launchScript(append_filename=False)
        os.chdir(CWD)
        # Add image to SD instance
        from scipy.io import loadmat
        mat_dic = loadmat(OUT_FILE)
        image = mat_dic['mat3D_clean']
        self.data.add_field(gridname=imagename, fieldname=clean_fieldname,
                            array=image, replace=replace, **keywords)
        # Remove tmp mesh files
        os.remove(OUT_FILE)
        return
