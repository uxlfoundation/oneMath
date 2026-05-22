.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_getrs_scratchpad_size:

getrs_scratchpad_size
=====================

Computes size of scratchpad memory required for :ref:`onemath_lapack_getrs` function.

.. container:: section

  .. rubric:: Description

``getrs_scratchpad_size`` supports the following precisions.

    .. list-table:: 
       :header-rows: 1

       * -  T 
       * -  ``float`` 
       * -  ``double`` 
       * -  ``std::complex<float>`` 
       * -  ``std::complex<double>`` 
      
Computes the number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemath_lapack_getrs` function should be able to hold.
Calls to this routine must specify the template parameter
explicitly.

getrs_scratchpad_size
---------------------

.. container:: section

  .. rubric:: Syntax
      
.. code-block:: cpp

    namespace oneapi::math::lapack {
      template <typename T>
      std::int64_t getrs_scratchpad_size(cl::sycl::queue &queue, oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs, std::int64_t lda, std::int64_t ldb) 
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   Device queue where calculations by :ref:`onemath_lapack_getrs` function will be performed.

trans
   Indicates the form of the equations:

   If ``trans=oneapi::math::transpose::nontrans``, then :math:`AX = B` is solved
   for :math:`X`.

   If ``trans=oneapi::math::transpose::trans``, then :math:`A^TX = B` is solved
   for :math:`X`.

   If ``trans=oneapi::math::transpose::conjtrans``, then :math:`A^HX = B` is
   solved for :math:`X`.

n
   The order of the matrix :math:`A` :math:`(0 \le n)` and the number of rows in matrix
   :math:`B(0 \le n)`.

nrhs
   The number of right-hand sides (:math:`0 \le \text{nrhs}`).

lda
   The leading dimension of ``a``.

ldb
   The leading dimension of ``b``.

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

The number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemath_lapack_getrs` function should be able to hold.

**Parent topic:** :ref:`onemath_lapack-linear-equation-routines` 


