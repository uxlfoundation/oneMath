.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_ungtr:

ungtr
=====

Generates the complex unitary matrix :math:`Q` determined by
:ref:`onemath_lapack_hetrd`.

.. container:: section

  .. rubric:: Description
      
``ungtr`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``std::complex<float>`` 
        * -  ``std::complex<double>`` 

The routine explicitly generates the :math:`n \times n` unitary matrix
:math:`Q` formed by :ref:`onemath_lapack_hetrd` when
reducing a complex Hermitian matrix :math:`A` to tridiagonal form:
:math:`A = QTQ^H`. Use this routine after a call to
:ref:`onemath_lapack_hetrd`.

ungtr (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      void ungtr(cl::sycl::queue &queue, oneapi::math::uplo upper_lower, std::int64_t n, cl::sycl::buffer<T,1> &a, std::int64_t lda, cl::sycl::buffer<T,1> &tau, cl::sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

upper_lower
   Must be ``uplo::upper`` or ``uplo::lower``. Uses the same
   ``upper_lower`` as supplied to
   :ref:`onemath_lapack_hetrd`.

n
   The order of the matrix :math:`Q` :math:`(0 \le n)`.

a
   The buffer ``a`` as returned by
   :ref:`onemath_lapack_hetrd`. The
   second dimension of ``a`` must be at least :math:`\max(1, n)`.

lda
   The leading dimension of ``a`` :math:`(n \le \text{lda})`.

tau
   The buffer ``tau`` as returned by
   :ref:`onemath_lapack_hetrd`. The
   dimension of ``tau`` must be at least :math:`\max(1, n-1)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_ungtr_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

a
   Overwritten by the unitary matrix :math:`Q`.

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

ungtr (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::math::lapack {
      cl::sycl::event ungtr(cl::sycl::queue &queue, oneapi::math::uplo upper_lower, std::int64_t n, T *a, std::int64_t lda, const T *tau, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

upper_lower
   Must be ``uplo::upper`` or ``uplo::lower``. Uses the same
   ``upper_lower`` as supplied to
   :ref:`onemath_lapack_hetrd`.

n
   The order of the matrix :math:`Q` :math:`(0 \le n)`.

a
   The pointer to ``a`` as returned by
   :ref:`onemath_lapack_hetrd`. The
   second dimension of ``a`` must be at least :math:`\max(1, n)`.

lda
   The leading dimension of ``a`` :math:`(n \le \text{lda})`.

tau
   The pointer to ``tau`` as returned by
   :ref:`onemath_lapack_hetrd`. The
   dimension of ``tau`` must be at least :math:`\max(1, n-1)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_ungtr_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

a
   Overwritten by the unitary matrix :math:`Q`.

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


