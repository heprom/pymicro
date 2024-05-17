User's Guide
============

The *pymicro* package aim is to help scientists to handle complex multi-modal
datasets stemming from material science experiments and simulations, with a
particular focus on polycrystalline materials and their mechanical properties.

In practical applications, these datasets may contain various **data items**:
numerical values, text, data arrays... These arrays can have multiple forms:
uniform arrays, structured tables, or arrays representing 3D data, that must be
linked to a geometrical topology. For instance, studying a polycrystalline
material sample may require to store images obtained via scanning electron
microscopy, strain measurements from in-situ mechanical tests, the material
mechanical and chemical properties, and all sort of testing metadata. Nowadays,
datasets increasingly tend to become large i.e* contain large arrays), and
complex (*i.e.* contain many **data items**).

**Pymicro's first goal is to provide a data management platform to efficiently
gather, organize, search and store these data items into organized datasets**.
Each dataset should correspond to a studied physical object and gather all data
relative to it. These tools rely are implemented within the ``pymicro.core``
module.

**Pymicro's second goal is to provide specialized tools to analyze data for
polycrystalline material science applications:** cristallography, X-ray
diffraction simulation, microstructural statistics... They are mainly
implemented in the ``pymicro.crystal`` and ``pymicro.xray`` packages. I/O
functionnalities to load data from imaging devices output files
(synchrotron X-ray diffraction, electronic microscopy...) are implemented into
the ``pymicro.file`` module.

Pre-requisites
----------------

To use Pymicro, it is highly recommended to learn the basics of:

    * Python (syntax, data types, imports, classes, interactive Python...)
    * `HDF5 <https://portal.hdfgroup.org/display/HDF5/Introduction+to+HDF5>`_ library
      and data model
    * `XDMF <https://www.xdmf.org/index.php/XDMF_Model_and_Format>`_ data model
    * `Numpy <https://numpy.org/doc/stable/index.html>`_
    * `Paraview <https://docs.paraview.org/en/latest/>`_ (visualization software)
    * `Pytables <https://www.pytables.org/>`_

How to use this Guide
---------------------

This Guide contains information sections and tutorials that gradually introduce
*Pymicro's* concepts and features. The tutorial sections in this documentation
are written as Jupyter **notebooks**, that can be found in the
``pymicro_package_path/examples/UserGuide`` directory. They can be browsed and
executed by all applications that can open jupyter-notebooks (jupyter-lab,
spyder....), to serve as an interactive Pymicro tutorial. If you do not have
*Jupyter* installed in your python environement, you may consult the dedicated
webpage of the `Jupyter <https://jupyter.org/install>_` project.

The Guide also include Refence sheets that provide a short summary of the
the code features and associated Python commands.

*Even without jupyter-notebooks, you may try the code presented in the this
user guide within any Python interactive console (like ipython). We strongly
encourage you to do so while reading through it !*

**Please note that the User Guide is currently an undergoing work. As such, it
is not yet completed.** Do not hesitate to check the
:doc:`cookbook`, or the :doc:`API docs </modules>`,
where you might find usefull examples and information.

Contents
--------

.. toctree::
   :maxdepth: 1

   Data Management with Pymicro<../examples/UserGuide/Data_Management.rst>
   ../examples/UserGuide/Data_Management_tutorial.rst
   Data Management Reference Sheet<../examples/UserGuide/Reference_Sheet_Datasets.ipynb>
   Material science datasets <../examples/UserGuide/Polycrystalline_Datasets.rst>