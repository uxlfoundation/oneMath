.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack:

LAPACK Routines
+++++++++++++++

oneMath provides a DPC++ interface to select routines from the Linear Algebra PACKage (LAPACK), as well as several LAPACK-like extension routines.
LAPACK routines require column major layout of matrices.

.. include:: lapack-linear-equation-routines.inc.rst
.. include:: lapack-singular-value-eigenvalue-routines.inc.rst
.. include:: lapack-like-extensions.inc.rst


.. container::

   .. container:: Note


      .. rubric:: Note
         :class: NoteTipHead


      Different arrays used as parameters to oneMath LAPACK routines must
      not overlap.


   .. container:: Note


      .. rubric:: Warning
         :name: warning
         :class: NoteTipHead


      LAPACK routines assume that input matrices do not contain IEEE 754
      special values such as INF or NaN values. Using these special
      values may cause LAPACK to return unexpected results or become
      unstable.

**Parent topic:** :ref:`onemath_dense_linear_algebra`
