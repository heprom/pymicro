Polycrystalline Materials Science with Pymicro 
===============================================

The main scientific focus of the *Pymicro* library is the management, processing
and visualization of three dimensional data for the study of polycrystalline
material microstructures. The review of the concepts and tools for data 
management with Pymicro are presented in the data management 
:ref:`page <data_management_label>` and :ref:`tutorials <sampledata_tutorial>`.
This user's guide section is dedicated to tutorials for Pymicro's target 
application: polycrystalline microstructures. 

*Pymicro* is designed to study polycrystalline material samples. *Pymicro* allows
to use the following data as descriptors for the microstructure of these samples:

* the 2D or 3D geometry of the sample
* identification of phases that compose the sample and their crystallographic
  and physical properties
* the mapping of these phases in the sample
* identification and properties of each grains composing each crystalline phase
  in the sample 
* the mapping of these grains in the sample

The package provides essential tools based on crystallography, mechanics or 
microstructural analysis, to collect, organize, process and visualize these various
types of information. The tutorial listed below cover most of these features.

.. toctree:: 
   :maxdepth: 2

   Polycrystalline Datasets <./Microstructure_class.ipynb>
   Material Phases <./Phases.ipynb>
   Microstructure Maps <./Cell_Data.ipynb>
   Grain Data <./Grain_Data.ipynb>
   