Overview
========

This page provides a quick overview of the pymicro code base.

As of version 0.1 (august 2014), pymicro is organised around 4 packages:

1. :mod:`pymicro.apps` contains a simple application to visualise and interact 
   with 3D volumes;
2. :mod:`pymicro.crystal` contains various modules to handle crystal lattices, 
   crystallographic grains and to organise them into microstructures;
3. :mod:`pymicro.file` contains many helpers methods to handle typical file 
   reading and writting found at ESRF, ANKA and Soleil;
4. :mod:`pymicro.view` contains the two principal modules for 3D visualisation: 
   vtk_utils.py for rendering still 3D images and vtk_anim.py to create 
   animations.
