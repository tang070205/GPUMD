.. _kw_kspace:
.. index::
   single: kspace (keyword in run.in)

:attr:`kspace`
==============

This keyword is used to set the computation method for the reciprocal space contribution to the electristatic energy.

Syntax
------

This keyword is used as follows::

  kspace <method>

where :attr:`<method>` can be either `ewald` or `pppm`.
The default is `pppm`, which implies that the particle-particle particle-mesh (PPPM) method is used.

For slab geometries that are periodic in the :math:`x` and :math:`y` directions and have a vacuum layer along :math:`z`, the Yeh-Berkowitz dipole correction can be activated using the :ref:`correct_slab <kw_correct_slab>` keyword.

Example
-------

To use the Ewald method use::

   kspace ewald
