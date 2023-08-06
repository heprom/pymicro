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

You can grab a copy of the latest version on the git repository at Centre des 
Materiaux, open a terminal and type:

::

  git clone http://github.com/heprom/pymicro

This will create a pymicro folder containing the source code (another pymicro 
folder), the documentation and the examples. Then you must add this folder to 
the PYTHONPATH environement variable. To do this you can use the following 
command in your shell (for C shell) :

::

  setenv PYTHONPATH /path/to/source/folder/pymicro

or if you rely on a Bash shell : 

::

  export PYTHONPATH=/path/to/source/folder/pymicro

You may include the command into your shell configuration file (.cshrc for C 
shell, or .bashrc for Bash shell), so that Pymicro will be in your PYTHONPATH 
for all your future sessions. 

Then you will be able to use pymicro. For instance  after starting a ipython 
shell:

::

  from pymicro.file.file_utils import HST_read

If you want to use pymicro interactively and import all modules at ipython 
startup you may apply the following recipe: :doc:`cookbook/pymicro_and_ipython`

Dependencies
============

1. Python 3.7+ required, please report any problem on
   the github page.
2. numpy - For array, matrix and other numerical manipulations. Used extensively
   by all modules. Currently, version must be <= 1.23
3. scipy, mainly used for ndimage filters.
4. skimage for additional image analysis (http://scikit-image.org).
5. matplotlib 1.1+ for plotting (e.g. pole figures or 3D image slices).
6. VTK with Python bindings (http://www.vtk.org/) for visualization of
   3D data using the :py:mod:`pymicro.view` package. Version should be 
   > 5.8.0, 9.0.1 recommended.
7. h5py and pytables to deal with HDF5 files.
8. basictools. Starting with version 0.5, we rely on this library to support mesh data; 
   basictools is open source and can be installed from conda using:

   ::

      `conda install -c conda-forge basictools`.

9. *Optional* , To build documentation locally : sphinx <= 6.21, 
   sphinxcontrib-bibtex, nbsphinx, pandoc, ipykernel

External
========

1. Crystal lattices can be created using CIF files usig the 
   :py:mod:`pymicro.crystal.lattice.from_cif` method. We use PyCifRW to read and
   parse CIF files.
2. reading and writing 3d Tiff files is supported via the TiffFile module.


Learning
========

The :doc:`cookbook` provides short examples in the form of scripts, that
illustrate different uses of the library to perform tasks that are of use for
material science data magement tasks. 

The :doc:`User's guide <userguide>` provides a step-by-step introduction to the code, and a
detailed overview of the data model, as well as the various code functionalities.
**The User guide is currently under development. It is not yet guaranteed to be
complete.**

For detailed documentation of all modules and classes, please refer to the
:doc:`API docs </modules>`.

