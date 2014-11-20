Examples
========

This page provides a few showcase examples of pymicro.

3D visualisation
----------------

* `display a cubic crystal in 3d <../examples/cubic_crystal_3d.py>`

  .. figure:: ../examples/cubic_crystal_3d.png
      :width: 800 px
      :height: 800 px
      :alt: cubic_crystal_3d
      :align: center

      A 3D view of a cubic lattice with the (111) slip plane family.

  .. literalinclude:: ../examples/cubic_crystal_3d.py
      :linenos:
      :language: python

* `display an hexagonal crystal in 3d <../examples/hexagonal_crystal_3d.py>`

  .. figure:: ../examples/hexagonal_crystal_3d.png
      :width: 800 px
      :height: 800 px
      :alt: hexagonal_crystal_3d
      :align: center

      A 3D view of an hexagonal lattice with two slip planes (-3,6,5) and (0,0,1).

  .. literalinclude:: ../examples/hexagonal_crystal_3d.py
      :linenos:
      :language: python

* `display an isosurface in 3d <../examples/mousse_3d.py>`

  .. figure:: ../examples/mousse_250x250x250_uint8_3d.png
      :width: 600 px
      :height: 600 px
      :alt: mousse_250x250x250_uint8_3d
      :align: center

      A 3D view of a tomographic stack from a polymer foam represented by an isosurface at level 80.

  .. literalinclude:: ../examples/mousse_3d.py
      :linenos:
      :language: python

* `show a cracked single crystal with one or two slip systems in 3d <../examples/cracked_single_crystal_with_slip_systems.py>`

  .. literalinclude:: ../examples/cracked_single_crystal_with_slip_systems.py
      :linenos:
      :language: python

2D plotting
-----------

* `plot a pole figure associated with a Microstructure object <../examples/pole_6grains.py>`
* `plot crystallographic slip traces <../examples/slip_traces.py>`

Animation
---------

* `show a grain in 3d and rotate it around a vertical axis <../examples/animation/grain_hkl_anim_3d.py>`

  .. figure:: ../examples/animation/grain1_anim_3d.gif
      :width: 600 px
      :height: 700 px
      :alt: grain1_anim_3d.gif
      :align: center

  .. literalinclude:: ../examples/animation/grain_hkl_anim_3d.py
      :linenos:
      :language: python
