.. pymicro documentation master file, created by sphinx-quickstart on Tue Oct 22 11:17:23 2013.

.. image:: _static/pymicro_logo_600.png
   :width: 300 px
   :alt: pymicro
   :align: center


.. include:: ../README.rst


Introduction
============

**Pymicro** is a free python package designed to load, manage, study and vizualize 
three dimensional material data, with a particular focus on crystalline
microstructures. This library has been developed primarily for data from imaging
and field measurement methods in materials science experiments, relying on 
techniques such as optical or electronical microscopy, or synchrotron X-ray 
diffraction. The goal of **Pymicro** is to make it easy (or easier) to search, 
browse and process your 3D datasets especially when it comes to automated 
processing needed for in situ data analysis.

**Pymicro** combines the generic power of the main scientific python 
libraries (namely *numpy*, *scipy* and *matplotlib*), with a hierarchical, 
generic and versatile data platform, that is built on top of the `Pytables 
<https://www.pytables.org/>`_ package and the HDF5 file format. **Pymicro** 
is also integrates 3D visualisation capabilities through the use of VTK library,
and its ability to generate XDMF files, to enable native visualization of the 
datasets with the free software Paraview. 

If you wish to contribute to the code, report bugs, or suggest new features,
please contact us via `GitHub <https://github.com/heprom/pymicro>`_. 

Contents:
=========

.. toctree::
   :maxdepth: 2
    
   welcome
   auto_examples/index.rst
   cookbook
   User's Guide<userguide>
   API Docs<pymicro>
   changelog
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

