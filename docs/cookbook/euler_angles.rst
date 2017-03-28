Understanding Euler angles and the orientation matrix
-----------------------------------------------------

In crystallography, the orientation of a lattice can be described with respect to the laboratory frame by a rotation.
In material science, this description follows the passive convention (as used by pymicro) which means the rotation is
defines such as it brings the laboratory frame in coincidence with the crystal frame.

Rotations in Euclidian space have 3 independent components as shown by Euler. They can be described in a number of ways
such as:

 * Euler angles
 * Orientation matrix
 * Rodrigues vector
 * Quaternion

Euler angles are a common way of defining a rotation by combining 3 successive rotations around different axes. Here we
use the convention of Bunge which is to rotate first around Z then around the new X and finally around the new Z. This
example will show how the 3 successive rotations are carried out and that they indeed bring the laboratory frame (XYZ)
in coincidence with the crystal frame.

**Get the complete Python source code:** :download:`euler_angles_anim.py <../examples/animation/euler_angles_anim.py>`

As an example, take the following triplet of Euler angles (in degrees): :math:`(\phi_1, \Phi, \phi_2) = (142.8, 32.0, 214.4)`

.. literalinclude:: ../examples/animation/euler_angles_anim.py
   :lines: 8-10

This instance of `Orientation` can be used to display a crystal lattice in a 3D scene

.. literalinclude:: ../examples/animation/euler_angles_anim.py
   :lines: 7,32-35

.. figure:: ../_static/euler_angles_anim_001.png
   :align: center
   :width: 30%

   A 3D view of a cubic lattice with a given orientation.

Now by applying successively the 3 rotation, we show that the laboratory frame is made coincident with the crystal
frame. The first rotation of angle :math:`\phi_1` is around Z, the second rotation of angle :math:`\Phi`is around the
new X and the third rotation angle :math:`\phi_2` is around the new Z.

.. image:: ../_static/euler_angles_anim_071.png
   :width: 30%
.. image:: ../_static/euler_angles_anim_121.png
   :width: 30%
.. image:: ../_static/euler_angles_anim_171.png
   :width: 30%
