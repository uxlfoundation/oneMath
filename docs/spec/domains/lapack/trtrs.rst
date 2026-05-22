.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_trtrs:

trtrs
=====

Solves a system of linear equations with a triangular coefficient
matrix, with multiple right-hand sides.

.. container:: section

  .. rubric:: Description

``trtrs`` supports the following precisions.

      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 

The routine solves for :math:`X` the following systems of linear
equations with a triangular matrix :math:`A`, with multiple right-hand
sides stored in :math:`B`:

    .. list-table::
       :header-rows: 1
 
       * -     :math:`AX = B`
         -
         -     if ``transa`` =\ ``transpose::nontrans``,
       * -     \ :math:`A^TX = B`\
         -
         -     if ``transa`` =\ ``transpose::trans``,
       * -     :math:`A^HX = B`
         -
         -     if ``transa`` =\ ``transpose::conjtrans`` (for complex    matrices only).

trtrs (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      void trtrs(cl::sycl::queue &queue, oneapi::math::uplo upper_lower, oneapi::math::transpose transa, oneapi::math::diag unit_diag, std::int64_t n, std::int64_t nrhs, cl::sycl::buffer<T,1> &a, std::int64_t lda, cl::sycl::buffer<T,1> &b, std::int64_t ldb, cl::sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

upper_lower
   Indicates whether :math:`A` is upper or lower    triangular:

      If upper_lower = ``uplo::upper``, then   :math:`A` is upper triangular.

      If upper_lower =   ``uplo::lower``, then :math:`A` is lower triangular.

transa
   If transa = ``transpose::nontrans``, then    :math:`AX = B` is solved for :math:`X`.

   If   transa = ``transpose::trans``, then :math:`A^{T}X = B` is solved for :math:`X`.

   If transa =   ``transpose::conjtrans``, then :math:`A^{H}X = B` is   solved for :math:`X`.

unit_diag
   If unit_diag = ``diag::nonunit``, then :math:`A` is not a    unit triangular matrix.

   If unit_diag = ``diag::unit``,   then :math:`A` is unit triangular: diagonal elements of :math:`A` are assumed   to be 1 and not referenced in the array ``a``.

n
   The order of :math:`A`; the number of rows in :math:`B`;    :math:`n \ge 0`.

nrhs
   The number of right-hand sides; :math:`\text{nrhs} \ge 0`.

a
   Buffer containing the matrix :math:`A`.      The    second dimension of ``a`` must be at least :math:`\max(1,n)`.

lda
   The leading dimension of ``a``;    :math:`\text{lda} \ge \max(1, n)`.

b
   Buffer containing the matrix :math:`B` whose columns are the    right-hand sides for the systems of equations.      The   second dimension of ``b`` at least :math:`\max(1,\text{nrhs})`.

ldb
   The leading dimension of ``b``; :math:`\text{ldb} \ge \max(1, n)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_trtrs_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters
      
b
   Overwritten by the solution matrix :math:`X`.

scratchpad
   Buffer holding scratchpad memory to be used by routine for storing intermediate results.

.. container:: section

  .. rubric:: Throws
         
This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`

:ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`

:ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`

:ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

:ref:`oneapi::math::lapack::invalid_argument<onemath_lapack_exception_invalid_argument>`

:ref:`oneapi::math::lapack::computation_error<onemath_lapack_exception_computation_error>`

   Exception is thrown in case of problems during calculations. The ``info`` code of the problem can be obtained by `info()` method of exception object:

   If :math:`\text{info}=-i`, the :math:`i`-th parameter had an illegal value.

   If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should not be less than value return by `detail()` method of exception object.

trtrs (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::math::lapack {
      cl::sycl::event trtrs(cl::sycl::queue &queue, oneapi::math::uplo upper_lower, oneapi::math::transpose transa, oneapi::math::diag unit_diag, std::int64_t n, std::int64_t nrhs, const T *a, std::int64_t lda, T *b, std::int64_t ldb, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

upper_lower
   Indicates whether :math:`A` is upper or lower    triangular:

      If upper_lower = ``uplo::upper``, then   :math:`A` is upper triangular.

      If upper_lower =   ``uplo::lower``, then :math:`A` is lower triangular.

transa
   If transa = ``transpose::nontrans``, then    :math:`AX = B` is solved for :math:`X`.

   If   transa = ``transpose::trans``, then :math:`A^{T}X = B` is solved for :math:`X`.

   If transa =   ``transpose::conjtrans``, then :math:`A^{H}X = B` is   solved for :math:`X`.

unit_diag
   If unit_diag = ``diag::nonunit``, then :math:`A` is not a    unit triangular matrix.

   If unit_diag = ``diag::unit``,   then :math:`A` is unit triangular: diagonal elements of :math:`A` are assumed   to be 1 and not referenced in the array ``a``.

n
   The order of :math:`A`; the number of rows in :math:`B`;    :math:`n \ge 0`.

nrhs
   The number of right-hand sides; :math:`\text{nrhs} \ge 0`.

a
   Array containing the matrix :math:`A`.      The    second dimension of ``a`` must be at least :math:`\max(1,n)`.

lda
   The leading dimension of ``a``;    :math:`\text{lda} \ge \max(1, n)`.

b
   Array containing the matrix :math:`B` whose columns are the    right-hand sides for the systems of equations.      The   second dimension of ``b`` at least :math:`\max(1,\text{nrhs})`.

ldb
   The leading dimension of ``b``; :math:`\text{ldb} \ge \max(1, n)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_trtrs_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters
      
b
   Overwritten by the solution matrix :math:`X`.

scratchpad
   Pointer to scratchpad memory to be used by routine for storing intermediate results.

.. container:: section

  .. rubric:: Throws
         
This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`

:ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`

:ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`

:ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

:ref:`oneapi::math::lapack::invalid_argument<onemath_lapack_exception_invalid_argument>`

:ref:`oneapi::math::lapack::computation_error<onemath_lapack_exception_computation_error>`

   Exception is thrown in case of problems during calculations. The ``info`` code of the problem can be obtained by `info()` method of exception object:

   If :math:`\text{info}=-i`, the :math:`i`-th parameter had an illegal value.

   If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should not be less than value return by `detail()` method of exception object.

.. container:: section

  .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemath_lapack-linear-equation-routines`

