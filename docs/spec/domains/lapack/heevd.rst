.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_heevd:

heevd
=====

Computes all eigenvalues and, optionally, all eigenvectors of a
complex Hermitian matrix using divide and conquer algorithm.

.. container:: section

  .. rubric:: Description

``heevd`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``std::complex<float>`` 
        * -  ``std::complex<double>`` 

The routine computes all the eigenvalues, and optionally all the
eigenvectors, of a complex Hermitian matrix :math:`A`. In other words, it
can compute the spectral factorization of :math:`A` as: :math:`A = Z\Lambda Z^H`.

Here :math:`\Lambda` is a real diagonal matrix whose diagonal elements are the
eigenvalues :math:`\lambda_i`, and :math:`Z` is the (complex) unitary matrix
whose columns are the eigenvectors :math:`z_{i}`. Thus,

:math:`Az_i = \lambda_i z_i` for :math:`i = 1, 2, ..., n`.

If the eigenvectors are requested, then this routine uses a divide
and conquer algorithm to compute eigenvalues and eigenvectors.
However, if only eigenvalues are required, then it uses the
Pal-Walker-Kahan variant of the QL or QR algorithm.

heevd (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::math::lapack {
      void heevd(cl::sycl::queue &queue, oneapi::math::job jobz, oneapi::math::uplo upper_lower, std::int64_t n, butter<T,1> &a, std::int64_t lda, cl::sycl::buffer<realT,1> &w, cl::sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

jobz
   Must be ``job::novec`` or ``job::vec``.

   If ``jobz = job::novec``, then only eigenvalues are computed.

   If ``jobz = job::vec``, then eigenvalues and eigenvectors are
   computed.

upper_lower
   Must be ``uplo::upper`` or ``uplo::lower``.

   If ``upper_lower = job::upper``, a stores the upper triangular
   part of :math:`A`.

   If ``upper_lower = job::lower``, a stores the lower triangular
   part of :math:`A`.

n
   The order of the matrix :math:`A` (:math:`0 \le n`).

a
   The buffer ``a``, size (``lda,*``). The buffer ``a`` contains the matrix
   :math:`A`. The second dimension of ``a`` must be at least :math:`\max(1, n)`.

lda
   The leading dimension of ``a``. Must be at least :math:`\max(1,n)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_heevd_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters
      
a
   If ``jobz = job::vec``, then on exit this buffer is overwritten by
   the unitary matrix :math:`Z` which contains the eigenvectors of :math:`A`.

w
   Buffer, size at least n. Contains the eigenvalues
   of the matrix :math:`A` in ascending order.

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

   If ``info=i``, and ``jobz = oneapi::math::job::novec``, then the algorithm
   failed to converge; :math:`i` indicates the number of off-diagonal
   elements of an intermediate tridiagonal form which did not
   converge to zero.

   If ``info=i``, and ``jobz = oneapi::math::job::vec``, then the algorithm failed
   to compute an eigenvalue while working on the submatrix lying in
   rows and columns :math:`\text{info}/(n+1)` through :math:`\text{mod}(\text{info},n+1)`.

   If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should not be less than value return by `detail()` method of exception object.

heevd (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      cl::sycl::event heevd(cl::sycl::queue &queue, oneapi::math::job jobz, oneapi::math::uplo upper_lower, std::int64_t n, butter<T,1> &a, std::int64_t lda, RealT *w, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

jobz
   Must be ``job::novec`` or ``job::vec``.

   If ``jobz = job::novec``, then only eigenvalues are computed.

   If ``jobz = job::vec``, then eigenvalues and eigenvectors are
   computed.

upper_lower
   Must be ``uplo::upper`` or ``uplo::lower``.

   If ``upper_lower = job::upper``, a stores the upper triangular
   part of :math:`A`.

   If ``upper_lower = job::lower``, a stores the lower triangular
   part of :math:`A`.

n
   The order of the matrix :math:`A` (:math:`0 \le n`).

a
   Pointer to array containing :math:`A`, size (``lda,*``).The second dimension of ``a`` must be at least :math:`\max(1, n)`.

lda
   The leading dimension of ``a``. Must be at least :math:`\max(1,n)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_heevd_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters
      
a
   If ``jobz = job::vec``, then on exit this array is overwritten by
   the unitary matrix :math:`Z` which contains the eigenvectors of :math:`A`.

w
   Pointer to array of size at least :math:`n`. Contains the eigenvalues
   of the matrix :math:`A` in ascending order.

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

   If ``info=i``, and ``jobz = oneapi::math::job::novec``, then the algorithm
   failed to converge; :math:`i` indicates the number of off-diagonal
   elements of an intermediate tridiagonal form which did not
   converge to zero.

   If ``info=i``, and ``jobz = oneapi::math::job::vec``, then the algorithm failed
   to compute an eigenvalue while working on the submatrix lying in
   rows and columns :math:`\text{info}/(n+1)` through :math:`\text{mod}(\text{info},n+1)`.

   If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should not be less than value return by `detail()` method of exception object.

.. container:: section

  .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemath_lapack-singular-value-eigenvalue-routines`

