.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_ungbr_scratchpad_size:

ungbr_scratchpad_size
=====================

Computes size of scratchpad memory required for :ref:`onemath_lapack_ungbr` function.

.. container:: section

  .. rubric:: Description
         
``ungbr_scratchpad_size`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``std::complex<float>`` 
        * -  ``std::complex<double>`` 

Computes the number of elements of type :math:`T` the scratchpad memory to be passed to :ref:`onemath_lapack_ungbr` function should be able to hold.
Calls to this routine must specify the template parameter explicitly.

ungbr_scratchpad_size
---------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::math::lapack {
      template <typename T>
      std::int64_t ungbr_scratchpad_size(cl::sycl::queue &queue, oneapi::math::generate gen, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda, std::int64_t &scratchpad_size) 
    }

.. container:: section

  .. rubric:: Input Parameters
         
queue
   Device queue where calculations by :ref:`onemath_lapack_ungbr` function will be performed.

gen
   Must be ``generate::q`` or ``generate::p``.

   If ``gen = generate::q``, the routine generates the matrix
   :math:`Q`.

   If ``gen = generate::p``, the routine generates the matrix
   :math:`P^{T}`.

m
   The number of rows in the matrix :math:`Q` or :math:`P^{T}` to be
   returned :math:`(0 \le m)`.

   If ``gen = generate::q``, :math:`m \ge n \ge \min(m, k)`.

   If ``gen = generate::p``, :math:`n \ge m \ge \min(n, k)`.

n
   The number of columns in the matrix :math:`Q` or :math:`P^{T}` to
   be returned :math:`(0 \le n)`. See m for constraints.

k
   If ``gen = generate::q``, the number of columns in the original
   :math:`m \times k` matrix reduced by
   :ref:`onemath_lapack_gebrd`.

   If ``gen = generate::p``, the number of rows in the original
   :math:`k \times n` matrix reduced by
   :ref:`onemath_lapack_gebrd`.

lda
   The leading dimension of ``a``.

.. container:: section

  .. rubric:: Throws

This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`

:ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

:ref:`oneapi::math::lapack::invalid_argument<onemath_lapack_exception_invalid_argument>`

   Exception is thrown in case of incorrect supplied argument value.
   Position of wrong argument can be determined by `info()` method of exception object.

.. container:: section

  .. rubric:: Return Value
         
The number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemath_lapack_ungbr` function should be able to hold.

**Parent topic:** :ref:`onemath_lapack-singular-value-eigenvalue-routines`


