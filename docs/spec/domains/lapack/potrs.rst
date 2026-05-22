.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_potrs:

potrs
=====

Solves a system of linear equations with a Cholesky-factored
symmetric (Hermitian) positive-definite coefficient matrix.

.. container:: section

  .. rubric:: Description
      
``potrs`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``float`` 
        * -  ``double`` 
        * -  ``std::complex<float>`` 
        * -  ``std::complex<double>`` 

The routine solves for :math:`X` the system of linear equations
:math:`AX = B` with a symmetric positive-definite or, for complex data,
Hermitian positive-definite matrix :math:`A`, given the Cholesky
factorization of :math:`A`:

.. list-table:: 
   :header-rows: 1

   * -  :math:`A = U^TU` for real data, :math:`A = U^HU` for complex data
     -  if ``upper_lower=oneapi::math::uplo::upper``
   * -  :math:`A = LL^T` for real data, :math:`A = LL^H` for complex data
     -  if ``upper_lower=oneapi::math::uplo::lower``

where :math:`L` is a lower triangular matrix and :math:`U` is upper
triangular. The system is solved with multiple right-hand sides
stored in the columns of the matrix :math:`B`.

Before calling this routine, you must call :ref:`onemath_lapack_potrf` to compute
the Cholesky factorization of :math:`A`.

potrs (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      void potrs(cl::sycl::queue &queue, oneapi::math::uplo upper_lower, std::int64_t n, std::int64_t nrhs, cl::sycl::buffer<T,1> &a, std::int64_t lda, cl::sycl::buffer<T,1> &b, std::int64_t ldb, cl::sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

upper_lower
   Indicates how the input matrix has been factored:

   If ``upper_lower = oneapi::math::uplo::upper``, the upper triangle   :math:`U` of :math:`A` is stored, where :math:`A` = :math:`U^{T}`U`   for real data, :math:`A` = :math:`U^{H}U` for complex data.

   If ``upper_lower = oneapi::math::uplo::lower``, the lower triangle   :math:`L` of :math:`A` is stored, where :math:`A` = :math:`LL^{T}`   for real data, :math:`A` = :math:`LL^{H}` for complex   data.

n
   The order of matrix :math:`A` (:math:`0 \le n`).\

nrhs
   The number of right-hand sides (:math:`0 \le \text{nrhs}`).

a
   Buffer containing the factorization of the matrix A, as    returned by   :ref:`onemath_lapack_potrf`.   The second dimension of ``a`` must be at least :math:`\max(1, n)`.

lda
   The leading dimension of ``a``.

b
   The array ``b`` contains the matrix :math:`B` whose columns    are the right-hand sides for the systems of equations. The second   dimension of ``b`` must be at least :math:`\max(1,\text{nrhs})`.

ldb
   The leading dimension of ``b``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_potrs_scratchpad_size` function.

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

   If :math:`\text{info}=i`, the :math:`i`-th diagonal element of the Cholesky factor is zero, and the solve could not be completed.

   If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should not be less than value return by `detail()` method of exception object.

potrs (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      cl::sycl::event potrs(cl::sycl::queue &queue, oneapi::math::uplo upper_lower, std::int64_t n, std::int64_t nrhs, const T *a, std::int64_t lda, T *b, std::int64_t ldb, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

upper_lower
   Indicates how the input matrix has been factored:

   If ``upper_lower = oneapi::math::uplo::upper``, the upper triangle   :math:`U` of :math:`A` is stored, where :math:`A` = :math:`U^{T}U`   for real data, :math:`A` = :math:`U^{H}U` for complex data.

   If ``upper_lower = oneapi::math::uplo::lower``, the lower triangle   :math:`L` of :math:`A` is stored, where :math:`A` = :math:`LL^{T}`   for real data, :math:`A` = :math:`LL^{H}` for complex   data.

n
   The order of matrix :math:`A` (:math:`0 \le n`).\

nrhs
   The number of right-hand sides (:math:`0 \le \text{nrhs}`).

a
   Pointer to array containing the factorization of the matrix :math:`A`, as    returned by   :ref:`onemath_lapack_potrf`.   The second dimension of ``a`` must be at least :math:`\max(1, n)`.

lda
   The leading dimension of ``a``.

b
   The array ``b`` contains the matrix :math:`B` whose columns    are the right-hand sides for the systems of equations. The second   dimension of ``b`` must be at least :math:`\max(1,\text{nrhs})`.

ldb
   The leading dimension of ``b``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_potrs_scratchpad_size` function.

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

   If :math:`\text{info}=i`, the :math:`i`-th diagonal element of the Cholesky factor is zero, and the solve could not be completed.

   If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should not be less than value return by `detail()` method of exception object.

.. container:: section

  .. rubric:: Return Values
         
Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemath_lapack-linear-equation-routines`


