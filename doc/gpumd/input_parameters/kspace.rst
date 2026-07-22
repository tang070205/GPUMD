.. _kw_kspace:
.. index::
   single: kspace (keyword in run.in)

:attr:`kspace`
==============

This keyword is used to set the computation method for the reciprocal space contribution to the electristatic energy.

Syntax
------

This keyword is used as follows::

  kspace <method> [slab <volfac>]

where :attr:`<method>` can be either `ewald` or `pppm`.
The default is `pppm`, which implies that the particle-particle particle-mesh (PPPM) method is used.

The optional `slab` keyword activates the Yeh-Berkowitz dipole correction [Yeh1999]_ for slab geometries that are periodic in the :math:`x` and :math:`y` directions and have a vacuum layer along :math:`z`.
It removes the artificial electrostatic interaction between periodically repeated slabs along the out-of-plane direction, adding the correction energy

.. math::

   E_{\mathrm{corr}} = \frac{2\pi M_z^2}{V'},

where :math:`M_z = \sum_i q_i z_i` is the total dipole moment along :math:`z` and :math:`V'` is :attr:`<volfac>` times the simulation box volume.
:attr:`<volfac>` must be no smaller than 1; the default usage is :attr:`<volfac>` = 1, meaning that the correction is evaluated with the volume of the actual simulation box, which should contain a sufficient vacuum layer along :math:`z`.
The correction requires an orthogonal box with the slab normal along :math:`z`, and the system is still treated as three-dimensionally periodic by the k-space solver.

Example
-------

To use the Ewald method use::

   kspace ewald

To use the PPPM method with the slab dipole correction use::

   kspace pppm slab 1.0

References
----------

.. [Yeh1999]
   I.-C. Yeh and M. L. Berkowitz,
   *Ewald summation for systems with slab geometry*,
   Journal of Chemical Physics **111**, 3155 (1999).
