Using pymicro interactively
---------------------------

pymicro can be used interactively within the ipython framework without any change to your system. Just type some code in !

Example:

.. code-block:: python

  >>> from pymicro.crystal.lattice import Lattice
  >>> al = Lattice.from_symbol('Al')
  >>> from pymicro.crystal.lattice import HklPlane
  >>> p111 = HklPlane(1, 1, 1, al)
  >>> p111.bragg_angle(20)*180/pi
  7.618

As seen it can quick become boring to import all the required modules. To use pymicro seamlessly within ipython, it is recommended to tweak the ipython config in the following way:

you can add the import_modules.py file to the ipython startup directory. This way all the pymicro modules will be imported into the namespace at startup. You can copy it there but the best way is to use a hardlink. This way if pymicro is updated and this file changes, you will have nothing special to do:

.. code-block:: bash

  $ cd .ipython/profile_default/startup
  $ ln -s /path/to/pymicro/import_modules.py

That's it, now you may just type:

.. code-block:: python

  >>> al = Lattice.from_symbol('Al')
  >>> p111 = HklPlane(1, 1, 1, al)
  >>> p111.bragg_angle(20)*180/pi
  7.618

pymicro features several static methods to directly compute or plot with one liners, for instance to plot a pole figure for a single orientation without creating a Grain instance and so on you may just type:

.. code-block:: python

  >>> PoleFigure.plot(Orientation.from_rodrigues([0.0885, 0.3889, 0.3268]))

To compute all Schmid factor for this orientation and the {111}[110] slip systems, you can use:

.. code-block:: python

  >>> o = Orientation.from_rodrigues([0.0885, 0.3889, 0.3268])
  >>> o.compute_all_schmid_factors(SlipSystem.get_slip_systems(plane_type='111'))

