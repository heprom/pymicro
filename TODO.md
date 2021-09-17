# TODO list for the Pymicro package development

This file describes important features and ideas to be developed or bugs to be
fixed for code improvements. Each entry may contain a few words/lines on how to
implement feature of fix bug, to guide future work.

Small TODOs that require only a few lines of code should be written directly
in the source code, with a `# TODO:` flag. This file should not list minor
development projects.

# Features to develop

## SampleData class

- [] Automatic building of XDMF file from HDF5 dataset content (structure and
     metadata). How to: iterate through dataset like in `print_dataset_content`
     method, and use methods like `_add_mesh_to_xdmf`, `_add_image_to_xdmf`,
     `_add_field_to_xdmf` to fill XDMF tree.

- [] Dataset assembly functionality: add all the content of one SampleData
     dataset to another one. In practice, one group of the target dataset will
     receive all content of the source dataset root group. XDMF file, dataset
     Index and aliases, pathes in attributes etc... must be checked and
     updated.
     Potential application: dataset composition.

- [] Read only mode of the SampleData Interface

- [] Visualization solution for Integration Point fields to see
     all field values.

- [] Mesh file writer wrapper for all formats supported by BasicTools

- [] Graphical interface

# Bug to solve

## SampleData class

- [] Grid Temporal Collection are not automatically written in the XDMF file
     in ascending order of Time value, leading to a bug in Paraview.




