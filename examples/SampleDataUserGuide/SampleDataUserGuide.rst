Pymicro Data Platform: The Sample Data Class User Guide
========================================================

The *pymicro* package is based on a data format designed
to create, organize and manage efficiently complex multi-modal datasets, 
used in material science, with a particular focus on the mechanics of
microstructures. These data sets are said to be multi-modal because they
bring together data from various measurement and numerical simulation 
techniques, originally produced in very different formats.

The `pymicro.core` package implements a data platform 
class allowing to meet the numerous challenges raised by the complexity 
and size of these datasets. *Pymicro*'s main class `Microstructure`
is now based on this data format. The data platform is  
based on a central class, `SampleData`, that implements the backend and
user interface with the datasets. 

This User Guide objective is to serve as a documentation of the classe, 
and as a guide to learn step by step the SampleData file format, data 
model, and the use of the class to manipulate complex and multimodal 
datasets. It is composed of a serie of Jupyter notebooks, that have 
been integrated into the *Pymicro* package documentation. 

The first documentation page/notebook is an introduction, and does not
contain Python code. It is dedicated to the presentation of the
challenges that motivated the implementation of this data platform, and
of its main features. 

The other pages/notebooks are tutorials that will introduce you to the
various functionnalites of the platform, the different data item types
that can be stored into dataset, and their data model. 
They are introduced from the simplest to the most complex. Hence, it is 
adviced to read through them in the indicated order. **You will find at
the end of each page/notebook a short summary of the functionalities and
code lines presented. All these summary have been gathered in a complete
quick reference sheet, in the last page/notebook of this guide.**  

If you are reading through this on the on-line documentation, please
note that the associated notebooks can be found in the *pymicro* package 
directories, under the path 
`pymicro_package_path/examples/SampleDataUserGuide/`. You can open 
these files with the *jupyter-notebook* program so that you can run the 
code as you read this documentation, and get familiar with the use of 
SampleData. If you do not have *jupyter* installed in your python 
environement, you may consult the dedicated webpage of the 
*Jupyter* project to help you get it: https://jupyter.org/install. 

**Even without jupyter-notebooks, you may reproduce the code content of this guide 
within any Python interactive console (like ipython).
We strongly encourage you to do so while reading through this user 
guide !**

.. toctree::
   :maxdepth: 2

   SampleData_Introduction.ipynb
   1_Getting_Information_from_SampleData_datasets.ipynb
   2_SampleData_basic_data_items.ipynb
   3_SampleData_Image_groups.ipynb
   4_SampleData_Mesh_groups.ipynb
   5_SampleData_data_compression.ipynb
   6_SampleData_inheritance.ipynb
   7_SampleData_Interfaces.ipynb
   8_SampleData_Quick_Reference_Sheet.ipynb
