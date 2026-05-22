.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_unmtr_scratchpad_size:

unmtr_scratchpad_size
=====================

Computes size of scratchpad memory required for :ref:`onemath_lapack_unmtr` function.

.. container:: section

  .. rubric:: Description
         
``unmtr_scratchpad_size`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``std::complex<float>`` 
        * -  ``std::complex<double>`` 

Computes the number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemath_lapack_unmtr` function should be able to hold.
Calls to this routine must specify the template parameter explicitly.

unmtr_scratchpad_size
---------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      template <typename T>
      std::int64_t unmtr_scratchpad_size(cl::sycl::queue &queue, oneapi::math::side side, oneapi::math::uplo upper_lower, oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t lda, std::int64_t ldc) 
    }

.. container:: section

  .. rubric:: Input Parameters

In the descriptions below, ``r`` denotes the order of :math:`Q`:

.. container:: tablenoborder

     .. list-table::
        :header-rows: 0

        * -  :math:`r = m`
          -  if ``side = side::left``
        * -  :math:`r = n`
          -  if ``side = side::right``

queue
   Device queue where calculations by :ref:`onemath_lapack_unmtr` function will be performed.

side
   Must be either ``side::left`` or ``side::right``.

   If ``side=side::left``, :math:`Q` or :math:`Q^{H}` is
   applied to :math:`C` from the left.

   If ``side=side::right``, :math:`Q` or :math:`Q^{H}` is
   applied to :math:`C` from the right.

upper_lower
   Must be either ``uplo::upper`` or ``uplo::lower``. Uses the
   same ``upper_lower`` as supplied to
   :ref:`onemath_lapack_hetrd`.

trans
   Must be either ``transpose::nontrans`` or
   ``transpose::conjtrans``.

   If ``trans=transpose::nontrans``, the routine multiplies :math:`C`
   by :math:`Q`.

   If ``trans=transpose::conjtrans``, the routine multiplies :math:`C`
   by :math:`Q^{H}`.

m
   The number of rows in the matrix :math:`C` (:math:`m \ge 0`).

n
   The number of columns the matrix :math:`C` (:math:`n \ge 0`).

k
   The number of elementary reflectors whose product defines the
   matrix :math:`Q` (:math:`0 \le k \le n`).

lda
   The leading dimension of :math:`a` :math:`(\text{lda} \ge \max(1,r))`.

ldc
   The leading dimension of :math:`c` :math:`(\text{ldc} \ge \max(1,m))`.

.. container:: section

  .. rubric:: Throws
         
This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`

:ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

:ref:`oneapi::math::lapack::invalid_argument<onemath_lapack_exception_invalid_argument>`

   Exception is thrown in case of incorrect supplied argument value.
   Position of wrong argument can be determined by `info()` method of exception object.

.. container:: section

  .. rubric:: Return Value
         
The number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemath_lapack_unmtr` function should be able to hold.

**Parent topic:** :ref:`onemath_lapack-singular-value-eigenvalue-routines`

