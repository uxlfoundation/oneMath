.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_sytrd_scratchpad_size:

sytrd_scratchpad_size
=====================

Computes size of scratchpad memory required for :ref:`onemath_lapack_sytrd` function.

.. container:: section

  .. rubric:: Description
         
``sytrd_scratchpad_size`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``float`` 
        * -  ``double``

Computes the number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemath_lapack_sytrd` function should be able to hold.
Calls to this routine must specify the template parameter explicitly.

sytrd_scratchpad_size
---------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::math::lapack {
      template <typename T>
      std::int64_t sytrd_scratchpad_size(cl::sycl::queue &queue, oneapi::math::uplo upper_lower, std::int64_t n, std::int64_t lda) 
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   Device queue where calculations by :ref:`onemath_lapack_sytrd` function will be performed.

upper_lower
   Must be ``uplo::upper`` or ``uplo::lower``.

   If ``upper_lower = uplo::upper``, a stores the upper triangular
   part of :math:`A`.

   If ``upper_lower = uplo::lower``, a stores the lower triangular
   part of :math:`A`.

n
   The order of the matrices :math:`A` :math:`(0 \le n)`.

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
         
The number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemath_lapack_sytrd` function should be able to hold.

**Parent topic:** :ref:`onemath_lapack-singular-value-eigenvalue-routines`


