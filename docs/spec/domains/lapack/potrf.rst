.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_potrf:

potrf
=====

Computes the Cholesky factorization of a symmetric (Hermitian)
positive-definite matrix.

.. container:: section

  .. rubric:: Description
      
``potrf`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``float`` 
        * -  ``double`` 
        * -  ``std::complex<float>`` 
        * -  ``std::complex<double>`` 

The routine forms the Cholesky factorization of a symmetric
positive-definite or, for complex data, Hermitian positive-definite
matrix :math:`A`:

    .. list-table:: 
       :header-rows: 1
 
       * -  :math:`A` = :math:`U^{T}U` for real data, :math:`A = U^{H}U` for complex data
         -  if upper_lower=\ ``oneapi::math::uplo::upper`` 
       * -  :math:`A` = :math:`LL^{T}` for real data, :math:`A = LL^{H}` for complex data
         -  if upper_lower=\ ``oneapi::math::uplo::lower`` 

where :math:`L` is a lower triangular matrix and :math:`U` is upper
triangular.

potrf (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      void potrf(cl::sycl::queue &queue, oneapi::math::uplo upper_lower, std::int64_t n, cl::sycl::buffer<T,1> &a, std::int64_t lda, cl::sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

upper_lower
   Indicates whether the upper or lower triangular part of :math:`A` is
   stored and how :math:`A` is factored:

   If upper_lower=\ ``oneapi::math::uplo::upper``, the array ``a`` stores the
   upper triangular part of the matrix :math:`A`, and the strictly lower
   triangular part of the matrix is not referenced.

   If upper_lower=\ ``oneapi::math::uplo::lower``, the array ``a`` stores the
   lower triangular part of the matrix :math:`A`, and the strictly upper
   triangular part of the matrix is not referenced.

n
   Specifies the order of the matrix :math:`A` (:math:`0 \le n`).

a
   Buffer holding input matrix :math:`A`. The buffer ``a`` contains either
   the upper or the lower triangular part of the matrix :math:`A` (see
   upper_lower). The second dimension of ``a`` must be at least
   :math:`\max(1, n)`.

lda
   The leading dimension of ``a``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_potrf_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

a
   The buffer ``a`` is overwritten by the Cholesky factor :math:`U` or :math:`L`,
   as specified by ``upper_lower``.

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

   If :math:`\text{info}=i`, and `detail()` returns 0, then the leading minor of order :math:`i` (and therefore the
   matrix :math:`A` itself) is not positive-definite, and the
   factorization could not be completed. This may indicate an error
   in forming the matrix :math:`A`.

   If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should not be less than value return by `detail()` method of exception object.

potrf (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::math::lapack {
      cl::sycl::event potrf(cl::sycl::queue &queue, oneapi::math::uplo upper_lower, std::int64_t n, T *a, std::int64_t lda, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

upper_lower
   Indicates whether the upper or lower triangular part of :math:`A` is
   stored and how :math:`A` is factored:

   If upper_lower=\ ``oneapi::math::uplo::upper``, the array ``a`` stores the
   upper triangular part of the matrix :math:`A`, and the strictly lower
   triangular part of the matrix is not referenced.

   If upper_lower=\ ``oneapi::math::uplo::lower``, the array ``a`` stores the
   lower triangular part of the matrix :math:`A`, and the strictly upper
   triangular part of the matrix is not referenced.

n
   Specifies the order of the matrix :math:`A` (:math:`0 \le n`).

a
   Pointer to input matrix :math:`A`. The array ``a`` contains either
   the upper or the lower triangular part of the matrix :math:`A` (see
   upper_lower). The second dimension of ``a`` must be at least
   :math:`\max(1, n)`.

lda
   The leading dimension of ``a``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_potrf_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

a
   The memory pointer to by pointer ``a`` is overwritten by the Cholesky factor :math:`U` or :math:`L`,
   as specified by ``upper_lower``.

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

   If :math:`\text{info}=i`, and `detail()` returns 0, then the leading minor of order :math:`i` (and therefore the
   matrix :math:`A` itself) is not positive-definite, and the
   factorization could not be completed. This may indicate an error
   in forming the matrix :math:`A`.

   If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should not be less than value return by `detail()` method of exception object.

.. container:: section

  .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemath_lapack-linear-equation-routines`


