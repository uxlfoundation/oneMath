.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_potrs_batch_scratchpad_size:

potrs_batch_scratchpad_size
===========================

Computes size of scratchpad memory required for the :ref:`onemath_lapack_potrs_batch` function.

.. container:: section

  .. rubric:: Description

``potrs_batch_scratchpad_size`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

**Group API**

Computes the number of elements of type ``T`` the scratchpad memory should able to hold to be passed to the Group API of the :ref:`onemath_lapack_potrs_batch` function.

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      template <typename T>
      std::int64_t potrs_batch_scratchpad_size(cl::sycl::queue &queue, math::uplo *uplo, std::int64_t *n, std::int64_t *nrhs, std::int64_t *lda, std::int64_t *ldb, std::int64_t group_count, std::int64_t *group_sizes)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

uplo
 | Array of ``group_count`` :math:`\text{uplo}_g` parameters.
 | Each of :math:`\text{uplo}_g` indicates whether the upper or lower triangular parts of the input matrices are provided:
 | If :math:`\text{uplo}_g` is ``math::uplo::upper``, input matrices from array ``a`` belonging to group :math:`g` store the upper triangular parts,
 | If :math:`\text{uplo}_g` is ``math::uplo::lower``, input matrices from array ``a`` belonging to group :math:`g` store the lower triangular parts.

n
 | Array of ``group_count`` :math:`n_g` parameters.
 | Each :math:`n_g` specifies the order of the input matrices belonging to group :math:`g`.

nrhs
 | Array of ``group_count`` :math:`\text{nrhs}_g` parameters.
 | Each :math:`rhs_g` specifies the number of right-hand sides supplied for group :math:`g`.

lda
 | Array of ``group_count`` :math:`\text{lda}_g` parameters.
 | Each :math:`\text{lda}_g` specifies the leading dimensions of the matrices belonging to group :math:`g`.

ldb
 | Array of ``group_count`` :math:`\text{ldb}_g` parameters.
 | Each :math:`\text{ldb}_g` specifies the leading dimensions of the matrices belonging to group :math:`g`.

group_count
  Number of groups of parameters. Must be at least 0.

group_sizes Array of group_count integers. Array element with index :math:`g` specifies the number of problems to solve for each of the groups of parameters :math:`g`. So the total number of problems to solve, ``batch_size``, is a sum of all parameter group sizes.

.. container:: section
   
  .. rubric:: Return Values

Number of elements of type ``T`` the scratchpad memory should able to hold to be passed to the Group API of the :ref:`onemath_lapack_potrs_batch` function.

.. container:: section

  .. rubric:: Throws

This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`

:ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

:ref:`oneapi::math::lapack::invalid_argument<onemath_lapack_exception_invalid_argument>`

   Exception is thrown in case of incorrect supplied argument value.
   Position of wrong argument can be determined by `info()` method of exception object.

**Strided API**

Computes the number of elements of type ``T`` the scratchpad memory should able to hold to be passed to the Strided API of the :ref:`onemath_lapack_potrs_batch` function.

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      template <typename T>
      std::int64_t potrs_batch_scratchpad_size(cl::sycl::queue &queue, math::uplo uplo, std::int64_t n, std::int64_t nrhs, std::int64_t lda, std::int64_t stride_a, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size)
    };

.. container:: section

  .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.
uplo
 | Indicates how the input matrices have been factored:
 | If ``uplo = math::uplo::upper``, the upper triangle :math:`U_i` of :math:`A_i` is stored, where :math:`A_i = U_i^TU_i` for real data, :math:`A_i = U_i^HU_i` for complex data.
 | If ``uplo = math::uplo::lower``, the upper triangle :math:`L_i` of :math:`A_i` is stored, where :math:`A_i = L_iL_i^T` for real data, :math:`A_i = L_iL_i^H` for complex data.

n
  Order of matrices :math:`A_i` (:math:`0 \le n`).

nrhs  
  Number of right-hand sides (:math:`0 \le \text{nrhs}`).

lda
  Leading dimension of :math:`A_i`.

stride_a
  Stride between the beginnings of matrices inside the batch array ``a``.

ldb
  Leading dimensions of :math:`B_i`.

stride_b
  Stride between the beginnings of matrices :math:`B_i` inside the batch array ``b``.

batch_size
  Number of problems in a batch.

.. container:: section
   
  .. rubric:: Return Values

Number of elements of type ``T`` the scratchpad memory should able to hold to be passed to the Strided API of the :ref:`onemath_lapack_potrs_batch` function.

.. container:: section

  .. rubric:: Throws

This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`

:ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

:ref:`oneapi::math::lapack::invalid_argument<onemath_lapack_exception_invalid_argument>`

   Exception is thrown in case of incorrect supplied argument value.
   Position of wrong argument can be determined by `info()` method of exception object.

**Parent topic:** :ref:`onemath_lapack-like-extensions-routines`

