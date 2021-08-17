Overview
========

This page provides a quick overview of the pymicro code base.

Pymicro is organised around 8 packages:

1. :py:mod:`pymicro.apps` contains a simple application to visualise and interact 
   with 2D images and 3D volumes;
2. :py:mod:`pymicro.core` gather the samples.py module at the root of the data
   management in Pymicro and some related utilities.
3. :py:mod:`pymicro.crystal` contains various modules to handle crystal lattices,
   crystallographic grains and to organise them into microstructures;
4. :py:mod:`pymicro.external` regroup external utilities used for instance to
   read TIF or CIF files;
5. :py:mod:`pymicro.fe` contains a module to manipulate files for finite-elements
   analysis;
6. :py:mod:`pymicro.file` contains many helpers methods to handle typical file
   reading and writting found at ESRF, ANKA and Soleil;
7. :py:mod:`pymicro.view` contains the two principal modules for 3D visualisation:
   vtk_utils.py for rendering still 3D images and vtk_anim.py to create 
   animations;
8. :py:mod:`pymicro.xray` regroup modules to simulate and analyse xray experiments like tomography and
   diffraction..

Installation
============

You can grab a copy of the latest version on the git repository at Centre des Materiaux, open a terminal and type:

::

  git clone http://github.com/heprom/pymicro

This will create a pymicro folder containing the source code (another pymicro folder), the documentation and the examples. Then you must add this folder to the PYTHONPATH environement variable. To do this you can add this line to your .cshrc file:

::

  setenv PYTHONPATH /path/to/source/folder/pymicro

Then you will be able to use pymicro (you may have to source the .cshrc or restart the terminal). For instance after starting a ipython shell:

::

  from pymicro.file.file_utils import HST_read

If you want to use pymicro interactively and import all modules at ipython startup you may apply the following recipe: :doc:`cookbook/pymicro_and_ipython`

Dependencies
============

1. Python 3.7+ required. Tested with 3.7, please report any problem on
the github page.
2. numpy - For array, matrix and other numerical manipulations. Used extensively
   by all modules.
3. scipy, mainly used for ndimage filters.
4. skimage for additional image analysis (http://scikit-image.org).
5. matplotlib 1.1+ for plotting (e.g. pole figures or 3D image slices).
6. VTK with Python bindings 5.8+ (http://www.vtk.org/) for visualization of
   3D data using the :py:mod:`pymicro.view` package.
7. h5py and pytables to deal with HDF5 files.
8. basictools. Starting with version 0.5, we rely on this library to support mesh data; basictools is open source and
 can be installed from conda using `conda install -c conda-forge basictools`.

External
========

1. Crystal lattices can be created using CIF files usig the :py:mod:`pymicro.crystal.lattice.from_cif` method. We use PyCifRW to read and parse CIF files.
2. reading and writing 3d Tiff files is supported via the TiffFile module.

API documentation
=================

For detailed documentation of all modules and classes, please refer to the
:doc:`API docs </modules>`.

