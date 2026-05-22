.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_gesvd_scratchpad_size:

gesvd_scratchpad_size
=====================

Computes size of scratchpad memory required for :ref:`onemath_lapack_gesvd` function.

.. container:: section

  .. rubric:: Description

``gesvd_scratchpad_size`` supports the following precisions.

      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 

Computes the number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemath_lapack_gesvd` function should be able to hold.
Calls to this routine must specify the template parameter explicitly.

gesvd_scratchpad_size
---------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      template <typename T>
      std::int64_t gesvd_scratchpad_size(cl::sycl::queue &queue, oneapi::math::job jobu, oneapi::math::job jobvt, std::int64_t m, std::int64_t n, std::int64_t lda, std::int64_t ldu, std::int64_t ldvt) 
    }

.. container:: section

   .. rubric:: Input Parameters

queue
   Device queue where calculations by :ref:`onemath_lapack_gesvd` function will be performed.

jobu
   Must be ``job::allvec``, ``job::somevec``,
   ``job::overwritevec``, or ``job::novec``. Specifies options for
   computing all or part of the matrix :math:`U`.

   If ``jobu = job::allvec``, all :math:`m` columns of :math:`U` are
   returned in the buffer ``u``;

   if ``jobu = job::somevec``, the first :math:`\min(m, n)` columns of
   :math:`U` (the left singular vectors) are returned in the buffer ``v``;

   if ``jobu = job::overwritevec``, the first :math:`\min(m, n)`
   columns of :math:`U` (the left singular vectors) are overwritten on
   the buffer ``a``;

   if ``jobu = job::novec``, no columns of :math:`U` (no left singular
   vectors) are computed.

jobvt
   Must be ``job::allvec``, ``job::somevec``,
   ``job::overwritevec``, or ``job::novec``. Specifies options for
   computing all or part of the matrix :math:`V^T/V^H`.

   If ``jobvt = job::allvec``, all :math:`n` columns of :math:`V^T/V^H` are
   returned in the buffer ``vt``;

   if ``jobvt = job::somevec``, the first :math:`\min(m, n)` columns of
   :math:`V^T/V^H` (the left singular vectors) are returned in the
   buffer ``vt``;

   if ``jobvt = job::overwritevec``, the first :math:`\min(m, n)`
   columns of :math:`V^T/V^H` (the left singular vectors) are
   overwritten on the buffer ``a``;

   if ``jobvt = job::novec``, no columns of :math:`V^T/V^H` (no left
   singular vectors) are computed.

m
   The number of rows in the matrix :math:`A` (:math:`0 \le m`).

n
   The number of columns in the matrix :math:`A` (:math:`0 \le n`).

lda
   The leading dimension of ``a``.

ldu
   The leading dimension of ``u``.

ldvt
   The leading dimension of ``vt``.

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

The number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemath_lapack_gesvd` function should be able to hold.

**Parent topic:** :ref:`onemath_lapack-singular-value-eigenvalue-routines`


