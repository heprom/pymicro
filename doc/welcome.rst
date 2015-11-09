Overview
========

This page provides a quick overview of the pymicro code base.

As of version 0.1 (august 2014), pymicro is organised around 4 packages:

1. :py:mod:`pymicro.apps` contains a simple application to visualise and interact 
   with 2D images and 3D volumes;
2. :py:mod:`pymicro.crystal` contains various modules to handle crystal lattices, 
   crystallographic grains and to organise them into microstructures;
3. :py:mod:`pymicro.file` contains many helpers methods to handle typical file 
   reading and writting found at ESRF, ANKA and Soleil;
4. :py:mod:`pymicro.view` contains the two principal modules for 3D visualisation: 
   vtk_utils.py for rendering still 3D images and vtk_anim.py to create 
   animations.

Installation
============

You can grab a copy of the latest version on the git repository at Centre des Materiaux, open a terminal and type:

::

  git clone git://vcs/pymicro

This will create a pymicro folder. Then you must add the folder containing pymicro to the PYTHONPATH environement variable. To do this you can add this line to your .cshrc file:

::

  setenv PYTHONPATH /path/to/folder/containing/pymicro

Then you will be able to use pymicro (you may have to source the .cshrc or restart the terminal). For instance after starting a ipython shell:

::

  from pymicro.file.file_utils import HST_read

If you want to use pymicro interactively and import all modules at ipython startup you may apply the following recipe: :doc:`cookbook/pymicro_and_ipython`

Dependencies
============

1. Python 2.6+ required. Tested with 2.6 and 2.7, please report any problem.
2. numpy - For array, matrix and other numerical manipulations. Used extensively
   by all modules.
3. scipy 0.10+, mainly used for ndimage filters.
4. matplotlib 1.1+ for plotting (e.g. pole figures or 3D image slices).
5. VTK with Python bindings 5.8+ (http://www.vtk.org/) for visualization of
   3D data using the :py:mod:`pymicro.view` package.

External
========

1. Crystal lattices can be created using CIF files usig the :py:mod:`pymicro.crystal.lattice.from_cif` method. We use PyCifRW to read and parse CIF files.
2. reading and writing 3d Tiff files is supported via the TiffFile module.

API documentation
=================

For detailed documentation of all modules and classes, please refer to the
:doc:`API docs </modules>`.

