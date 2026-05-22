.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_getrs:

getrs
=====

Solves a system of linear equations with an LU-factored square
coefficient matrix, with multiple right-hand sides.

.. container:: section

  .. rubric:: Description
      
``getrs`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1
  
      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

The routine solves for :math:`X` the following systems of linear
equations:

    .. list-table:: 
       :header-rows: 1
    
       * -     \ :math:`AX = B`\     
         -     if ``trans``\ =\ ``oneapi::math::transpose::nontrans``\     
       * -     \ :math:`A^TX = B`\     
         -     if ``trans``\ =\ ``oneapi::math::transpose::trans``\     
       * -     \ :math:`A^HX = B`\     
         -     if ``trans``\ =\ ``oneapi::math::transpose::conjtrans``\     

Before calling this routine, you must call
:ref:`onemath_lapack_getrf`
to compute the LU factorization of :math:`A`.

getrs (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax
      
.. code-block:: cpp

    namespace oneapi::math::lapack {
      void getrs(cl::sycl::queue &queue, oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs, cl::sycl::buffer<T,1> &a, std::int64_t lda, cl::sycl::buffer<std::int64_t,1> &ipiv, cl::sycl::buffer<T,1> &b, std::int64_t ldb, cl::sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

trans
   Indicates the form of the equations:

   If ``trans=oneapi::math::transpose::nontrans``, then :math:`AX = B` is solved
   for :math:`X`.

   If ``trans=oneapi::math::transpose::trans``, then :math:`A^TX = B` is solved
   for :math:`X`.

   If ``trans=oneapi::math::transpose::conjtrans``, then :math:`A^HX = B` is
   solved for :math:`X`.

n
   The order of the matrix :math:`A` and the number of rows in matrix
   :math:`B(0 \le n)`.

nrhs
   The number of right-hand sides (:math:`0 \le \text{nrhs}`).

a
   Buffer containing the factorization of the matrix :math:`A`, as
   returned by :ref:`onemath_lapack_getrf`. The second dimension of ``a`` must be at least
   :math:`\max(1, n)`.

lda
   The leading dimension of ``a``.

ipiv
   Array, size at least :math:`\max(1, n)`. The ``ipiv`` array, as returned by
   :ref:`onemath_lapack_getrf`.

b
   The array ``b`` contains the matrix :math:`B` whose columns are the
   right-hand sides for the systems of equations. The second
   dimension of ``b`` must be at least :math:`\max(1,\text{nrhs})`.

ldb
   The leading dimension of ``b``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_getrs_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters
      
b
   The buffer ``b`` is overwritten by the solution matrix :math:`X`.

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

   If ``info=-i``, the :math:`i`-th parameter had an illegal value.

   If ``info=i``, the :math:`i`-th diagonal element of :math:`U` is zero,
   and the solve could not be completed.

   If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should not be less than value return by `detail()` method of exception object.

getrs (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      cl::sycl::event getrs(cl::sycl::queue &queue, oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs, const T *a, std::int64_t lda, const std::int64_t *ipiv, T *b, std::int64_t ldb, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

trans
   Indicates the form of the equations:

   If ``trans=oneapi::math::transpose::nontrans``, then :math:`AX = B` is solved
   for :math:`X`.

   If ``trans=oneapi::math::transpose::trans``, then :math:`A^TX = B` is solved
   for :math:`X`.

   If ``trans=oneapi::math::transpose::conjtrans``, then :math:`A^HX = B` is
   solved for :math:`X`.

n
   The order of the matrix :math:`A` and the number of rows in matrix
   :math:`B(0 \le n)`.

nrhs
   The number of right-hand sides (:math:`0 \le \text{nrhs}`).

a
   Pointer to array containing the factorization of the matrix :math:`A`, as
   returned by :ref:`onemath_lapack_getrf`. The second dimension of ``a`` must be at least
   :math:`\max(1, n)`.

lda
   The leading dimension of ``a``.

ipiv
   Array, size at least :math:`\max(1, n)`. The ``ipiv`` array, as returned by
   :ref:`onemath_lapack_getrf`.

b
   The array ``b`` contains the matrix :math:`B` whose columns are the
   right-hand sides for the systems of equations. The second
   dimension of ``b`` must be at least :math:`\max(1,\text{nrhs})`.

ldb
   The leading dimension of ``b``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_getrs_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters
      
b
   The array ``b`` is overwritten by the solution matrix :math:`X`.

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

   If ``info=-i``, the :math:`i`-th parameter had an illegal value.

   If ``info=i``, the :math:`i`-th diagonal element of :math:`U` is zero,
   and the solve could not be completed.

   If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should not be less than value return by `detail()` method of exception object.

.. container:: section

  .. rubric:: Return Values
     
Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemath_lapack-linear-equation-routines`
