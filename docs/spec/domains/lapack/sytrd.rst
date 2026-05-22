.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_sytrd:

sytrd
=====

Reduces a real symmetric matrix to tridiagonal form.

.. container:: section

  .. rubric:: Description
      
``sytrd`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``float`` 
        * -  ``double`` 

The routine reduces a real symmetric matrix :math:`A` to symmetric
tridiagonal form :math:`T` by an orthogonal similarity transformation:
:math:`A = QTQ^T`. The orthogonal matrix :math:`Q` is not formed explicitly
but is represented as a product of :math:`n-1` elementary reflectors.
Routines are provided for working with :math:`Q` in this representation .

sytrd (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::math::lapack {
      void sytrd(cl::sycl::queue &queue, oneapi::math::uplo upper_lower, std::int64_t n, cl::sycl::buffer<T,1> &a, std::int64_t lda, cl::sycl::buffer<T,1> &d, cl::sycl::buffer<T,1> &e, cl::sycl::buffer<T,1> &tau, cl::sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

upper_lower
   Must be ``uplo::upper`` or ``uplo::lower``.

   If ``upper_lower = uplo::upper``, ``a`` stores the upper
   triangular part of :math:`A`.

   If ``upper_lower = uplo::lower``, ``a`` stores the lower
   triangular part of :math:`A`.

n
   The order of the matrices :math:`A` :math:`(0 \le n)`.

a
   The buffer ``a``, size ``(lda,*)``. Contains the upper or lower
   triangle of the symmetric matrix :math:`A`, as specified by
   ``upper_lower``.

   The second dimension of ``a`` must be at least :math:`\max(1,n)`.

lda
   The leading dimension of ``a``; at least :math:`\max(1,n)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_sytrd_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

a
   On exit,

   if ``upper_lower = uplo::upper``, the diagonal and first
   superdiagonal of :math:`A` are overwritten by the corresponding
   elements of the tridiagonal matrix :math:`T`, and the elements above
   the first superdiagonal, with the buffer ``tau``, represent the
   orthogonal matrix :math:`Q` as a product of elementary reflectors;

   if ``upper_lower = uplo::lower``, the diagonal and first
   subdiagonal of :math:`A` are overwritten by the corresponding elements
   of the tridiagonal matrix :math:`T`, and the elements below the first
   subdiagonal, with the buffer ``tau``, represent the orthogonal matrix
   :math:`Q` as a product of elementary reflectors.

d
   Buffer containing the diagonal elements of the matrix :math:`T`. The
   dimension of ``d`` must be at least :math:`\max(1, n)`.

e
   Buffer containing the off diagonal elements of the matrix :math:`T`.
   The dimension of ``e`` must be at least :math:`\max(1, n-1)`.

tau
   Buffer, size at least :math:`\max(1, n)`. Stores :math:`(n-1)` scalars that
   define elementary reflectors in decomposition of the unitary
   matrix :math:`Q` in a product of :math:`n-1` elementary reflectors.
   :math:`\tau(n)` is used as workspace.

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

sytrd (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      cl::sycl::event sytrd(cl::sycl::queue &queue, oneapi::math::uplo upper_lower, std::int64_t n, T *a, std::int64_t lda, T *d, T *e, T *tau, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

upper_lower
   Must be ``uplo::upper`` or ``uplo::lower``.

   If ``upper_lower = uplo::upper``, ``a`` stores the upper
   triangular part of :math:`A`.

   If ``upper_lower = uplo::lower``, ``a`` stores the lower
   triangular part of :math:`A`.

n
   The order of the matrices :math:`A` :math:`(0 \le n)`.

a
   The pointer to matrix :math:`A`, size ``(lda,*)``. Contains the upper or lower
   triangle of the symmetric matrix :math:`A`, as specified by
   ``upper_lower``.
   The second dimension of ``a`` must be at least :math:`\max(1,n)`.

lda
   The leading dimension of ``a``; at least :math:`\max(1,n)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_sytrd_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters
   
a
   On exit,

   if ``upper_lower = uplo::upper``, the diagonal and first
   superdiagonal of :math:`A` are overwritten by the corresponding
   elements of the tridiagonal matrix :math:`T`, and the elements above
   the first superdiagonal, with the array ``tau``, represent the
   orthogonal matrix :math:`Q` as a product of elementary reflectors;

   if ``upper_lower = uplo::lower``, the diagonal and first
   subdiagonal of :math:`A` are overwritten by the corresponding elements
   of the tridiagonal matrix :math:`T`, and the elements below the first
   subdiagonal, with the array ``tau``, represent the orthogonal matrix
   :math:`Q` as a product of elementary reflectors.

d
   Pointer to diagonal elements of the matrix :math:`T`. The
   dimension of ``d`` must be at least :math:`\max(1, n)`.

e
   Pointer to off diagonal elements of the matrix :math:`T`.
   The dimension of ``e`` must be at least :math:`\max(1, n-1)`.

tau
   Pointer to array of size at least :math:`\max(1, n)`. Stores :math:`(n-1)` scalars that
   define elementary reflectors in decomposition of the unitary
   matrix :math:`Q` in a product of :math:`n-1` elementary reflectors.
   :math:`\tau(n)` is used as workspace.

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

**Parent topic:** :ref:`onemath_lapack-singular-value-eigenvalue-routines`


