.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_sygvd_scratchpad_size:

sygvd_scratchpad_size
=====================

Computes size of scratchpad memory required for :ref:`onemath_lapack_sygvd` function.

.. container:: section

  .. rubric:: Description
         
`sygvd_scratchpad_size`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``float`` 
        * -  ``double`` 

Computes the number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemath_lapack_sygvd` function should be able to hold.
Calls to this routine must specify the template parameter explicitly.

sygvd_scratchpad_size
---------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::math::lapack {
      template <typename T>
      std::int64_t sygvd_scratchpad_size(cl::sycl::queue &queue, std::int64_t itype, oneapi::math::job jobz, oneapi::math::uplo upper_lower, std::int64_t n, std::int64_t lda, std::int64_t ldb) 
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   Device queue where calculations by :ref:`onemath_lapack_sygvd` function will be performed.

itype
   Must be 1 or 2 or 3. Specifies the problem type to be solved:

   if :math:`\text{itype} = 1`, the problem type is :math:`Ax = \lambda Bx`;

   if :math:`\text{itype} = 2`, the problem type is :math:`ABx = \lambda x`;

   if :math:`\text{itype} = 3`, the problem type is :math:`BAx = \lambda x`.

jobz
   Must be ``job::novec`` or ``job::vec``.

   If ``jobz = job::novec``, then only eigenvalues are computed.

   If ``jobz = job::vec``, then eigenvalues and eigenvectors are
   computed.

upper_lower
   Must be ``uplo::upper`` or ``uplo::lower``.

   If ``upper_lower = job::upper``, ``a`` and ``b`` store the upper
   triangular part of :math:`A` and :math:`B`.

   If ``upper_lower = job::lower``, ``a`` and ``b`` stores the lower
   triangular part of :math:`A` and :math:`B`.

n
   The order of the matrices :math:`A` and :math:`B` :math:`(0 \le n)`.

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
         
The number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemath_lapack_sygvd` function should be able to hold.

**Parent topic:** :ref:`onemath_lapack-singular-value-eigenvalue-routines`


