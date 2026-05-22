.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_geqrf:

geqrf
=====

Computes the QR factorization of a general :math:`m \times n` matrix.

.. rubric:: Description

``geqrf`` supports the following precisions:

.. list-table:: 
   :header-rows: 1

   * -  T 
   * -  ``float`` 
   * -  ``double`` 
   * -  ``std::complex<float>`` 
   * -  ``std::complex<double>`` 

The routine forms the QR factorization of a general
:math:`m \times n` matrix :math:`A`. No pivoting is performed.

The routine does not form the matrix :math:`Q` explicitly. Instead, :math:`Q`
is represented as a product of :math:`\min(m, n)` elementary
reflectors. Routines are provided to work with :math:`Q` in this
representation.

geqrf (Buffer Version)
----------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      void geqrf(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, cl::sycl::buffer<T,1> &a, std::int64_t lda, cl::sycl::buffer<T,1> &tau, cl::sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

    .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

m
   The number of rows in the matrix :math:`A` (:math:`0 \le m`).

n
   The number of columns in :math:`A` (:math:`0 \le n`).

a
   Buffer holding input matrix :math:`A`. Must have size at least
   :math:`\text{lda} \cdot n`.

lda
   The leading dimension of :math:`A`; at least :math:`\max(1, m)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_geqrf_scratchpad_size` function.

.. container:: section

    .. rubric:: Output Parameters

a
   Output buffer, overwritten by the factorization data as follows:

   The elements on and above the diagonal of the array contain the
   :math:`\min(m,n) \times n` upper trapezoidal matrix :math:`R` (:math:`R` is upper
   triangular if :math:`m \ge n`); the elements below the diagonal, with the
   array tau, represent the orthogonal matrix :math:`Q` as a product of
   :math:`\min(m,n)` elementary reflectors.

tau
   Output buffer, size at least :math:`\max(1, \min(m, n))`. Contains scalars
   that define elementary reflectors for the matrix :math:`Q` in its
   decomposition in a product of elementary reflectors.

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

   If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should not be less than value return by `detail()` method of exception object.

geqrf (USM Version)
----------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      cl::sycl::event geqrf(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, T *a, std::int64_t lda, T *tau, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    }

.. container:: section

    .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

m
   The number of rows in the matrix :math:`A` (:math:`0 \le m`).

n
   The number of columns in :math:`A` (:math:`0 \le n`).

a
   Pointer to memory holding input matrix :math:`A`. Must have size at least
   :math:`\text{lda} \cdot n`.

lda
   The leading dimension of :math:`A`; at least :math:`\max(1, m)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_geqrf_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.


.. container:: section

    .. rubric:: Output Parameters

a
   Overwritten by the factorization data as follows:

   The elements on and above the diagonal of the array contain the
   :math:`\min(m,n) \times n` upper trapezoidal matrix :math:`R` (:math:`R` is upper
   triangular if :math:`m \ge n`); the elements below the diagonal, with the
   array tau, represent the orthogonal matrix :math:`Q` as a product of
   :math:`\min(m,n)` elementary reflectors.

tau
   Array, size at least :math:`\max(1, \min(m, n))`. Contains scalars
   that define elementary reflectors for the matrix :math:`Q` in its
   decomposition in a product of elementary reflectors.

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

   If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should not be less than value return by `detail()` method of exception object.

.. container:: section

    .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemath_lapack-linear-equation-routines`


