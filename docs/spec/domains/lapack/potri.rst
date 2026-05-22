.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_potri:

potri
=====

Computes the inverse of a symmetric (Hermitian) positive-definite
matrix using the Cholesky factorization.

.. container:: section

  .. rubric:: Description

``potri`` supports the following precisions.

      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 

The routine computes the inverse :math:`A^{-1}` of a symmetric positive
definite or, for complex flavors, Hermitian positive-definite matrix
:math:`A`. Before calling this routine, call :ref:`onemath_lapack_potrf`
to factorize :math:`A`.

potri (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      void potri(cl::sycl::queue &queue, oneapi::math::uplo upper_lower, std::int64_t n, cl::sycl::buffer<T,1> &a, std::int64_t lda, cl::sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

upper_lower
   Indicates how the input matrix :math:`A` has been    factored:

   If ``upper_lower = oneapi::math::uplo::upper``, the upper   triangle of :math:`A` is stored.

   If   ``upper_lower = oneapi::math::uplo::lower``, the lower triangle of :math:`A` is   stored.

n
   Specifies the order of the matrix    :math:`A` (:math:`0 \le n`).

a
   Contains the factorization of the matrix :math:`A`, as    returned by   :ref:`onemath_lapack_potrf`.   The second dimension of ``a`` must be at least :math:`\max(1, n)`.

lda
   The leading dimension of ``a``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_potri_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters
      
a
   Overwritten by the upper or lower triangle of the inverse    of :math:`A`. Specified by ``upper_lower``.

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

   If :math:`\text{info}=i`, the :math:`i`-th diagonal element of the Cholesky factor
   (and therefore the factor itself) is zero, and the inversion could not be completed.

   If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should not be less than value return by `detail()` method of exception object.

potri (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::math::lapack {
      cl::sycl::event potri(cl::sycl::queue &queue, oneapi::math::uplo upper_lower, std::int64_t n, T *a, std::int64_t lda, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

upper_lower
   Indicates how the input matrix :math:`A` has been    factored:

   If ``upper_lower = oneapi::math::uplo::upper``, the upper   triangle of :math:`A` is stored.

   If   ``upper_lower = oneapi::math::uplo::lower``, the lower triangle of :math:`A` is   stored.

n
   Specifies the order of the matrix    :math:`A` (:math:`0 \le n`).

a
   Contains the factorization of the matrix :math:`A`, as    returned by   :ref:`onemath_lapack_potrf`.   The second dimension of ``a`` must be at least :math:`\max(1, n)`.

lda
   The leading dimension of ``a``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_potri_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters
      
a
   Overwritten by the upper or lower triangle of the inverse    of :math:`A`. Specified by ``upper_lower``.

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

   If :math:`\text{info}=i`, the :math:`i`-th diagonal element of the Cholesky factor
   (and therefore the factor itself) is zero, and the inversion could not be completed.

   If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should not be less than value return by `detail()` method of exception object.

.. container:: section

  .. rubric:: Return Values
         
Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemath_lapack-linear-equation-routines`


