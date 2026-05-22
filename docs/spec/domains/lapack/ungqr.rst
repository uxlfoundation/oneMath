.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_ungqr:

ungqr
=====

Generates the complex unitary matrix :math:`Q` of the QR factorization formed
by :ref:`onemath_lapack_geqrf`.

.. container:: section

  .. rubric:: Description
      
``ungqr`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``std::complex<float>`` 
        * -  ``std::complex<double>`` 

The routine generates the whole or part of :math:`m \times m` unitary
matrix :math:`Q` of the QR factorization formed by the routines
:ref:`onemath_lapack_geqrf`.

Usually :math:`Q` is determined from the QR factorization of an :math:`m \times p` matrix :math:`A` with :math:`m \ge p`. To compute the whole matrix
:math:`Q`, use:

::

    oneapi::math::lapack::ungqr(queue, m, m, p, a, lda, tau, scratchpad, scratchpad_size)

To compute the leading :math:`p` columns of :math:`Q` (which form an
orthonormal basis in the space spanned by the columns of :math:`A`):

::

    oneapi::math::lapack::ungqr(queue, m, p, p, a, lda, tau, scratchpad, scratchpad_size)

To compute the matrix :math:`Q^{k}` of the QR factorization of
the leading :math:`k` columns of the matrix :math:`A`:

::

    oneapi::math::lapack::ungqr(queue, m, m, k, a, lda, tau, scratchpad, scratchpad_size)

To compute the leading :math:`k` columns of :math:`Q^{k}` (which form
an orthonormal basis in the space spanned by the leading :math:`k`
columns of the matrix :math:`A`):

::

    oneapi::math::lapack::ungqr(queue, m, k, k, a, lda, tau, scratchpad, scratchpad_size)

ungqr (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      void ungqr(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, cl::sycl::buffer<T,1> &a, std::int64_t lda, cl::sycl::buffer<T,1> &tau, cl::sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

m
   The number of rows in the matrix :math:`A` (:math:`0 \le m`).

n
   The number of columns in the matrix :math:`A` (:math:`0 \le n`).

k
   The number of elementary reflectors whose product defines the
   matrix :math:`Q` (:math:`0 \le k \le n`).

a
   The buffer ``a`` as returned by
   :ref:`onemath_lapack_geqrf`.

lda
   The leading dimension of ``a`` (:math:`\text{lda} \le m`).

tau
   The buffer ``tau`` as returned by
   :ref:`onemath_lapack_geqrf`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_ungqr_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

a
   Overwritten by :math:`n` leading columns of the :math:`m \times m`
   orthogonal matrix :math:`Q`.

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

ungqr (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      cl::sycl::event ungqr(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, T *a, std::int64_t lda, const T *tau, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

m
   The number of rows in the matrix :math:`A` (:math:`0 \le m`).

n
   The number of columns in the matrix :math:`A` (:math:`0 \le n`).

k
   The number of elementary reflectors whose product defines the
   matrix :math:`Q` (:math:`0 \le k \le n`).

a
   The pointer to ``a`` as returned by
   :ref:`onemath_lapack_geqrf`.

lda
   The leading dimension of ``a`` (:math:`\text{lda} \le m`).

tau
   The pointer to ``tau`` as returned by
   :ref:`onemath_lapack_geqrf`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_ungqr_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters
      
a
   Overwritten by :math:`n` leading columns of the :math:`m \times m`
   orthogonal matrix :math:`Q`.

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

