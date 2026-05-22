.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_potrs_scratchpad_size:

potrs_scratchpad_size
=====================

Computes size of scratchpad memory required for :ref:`onemath_lapack_potrs` function.

.. container:: section

  .. rubric:: Description
         
``potrs_scratchpad_size`` supports the following precisions.

    .. list-table:: 
       :header-rows: 1
    
       * -  T 
       * -  ``float`` 
       * -  ``double`` 
       * -  ``std::complex<float>`` 
       * -  ``std::complex<double>`` 

Computes the number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemath_lapack_potrs` function should be able to hold.
Calls to this routine must specify the template parameter explicitly.

potrs_scratchpad_size
---------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      template <typename T>
      std::int64_t potrs_scratchpad_size(cl::sycl::queue &queue, oneapi::math::uplo upper_lower, std::int64_t n, std::int64_t nrhs, std::int64_t lda, std::int64_t ldb) 
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   Device queue where calculations by :ref:`onemath_lapack_potrs` function will be performed.

upper_lower
   Indicates how the input matrix has been factored:

   If ``upper_lower = oneapi::math::uplo::upper``, the upper triangle   :math:`U` of :math:`A` is stored, where :math:`A = U^{T}U`   for real data, :math:`A = U^{H}U` for complex data.

   If ``upper_lower = oneapi::math::uplo::lower``, the lower triangle   :math:`L` of :math:`A` is stored, where :math:`A = LL^{T}`   for real data, :math:`A = LL^{H}` for complex   data.

n
   The order of matrix :math:`A` (:math:`0 \le n`).

nrhs
   The number of right-hand sides (:math:`0 \le nrhs`).

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
         
The number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemath_lapack_potrs` function should be able to hold.

**Parent topic:** :ref:`onemath_lapack-linear-equation-routines`

