.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_getrs_batch_scratchpad_size:

getrs_batch_scratchpad_size
===========================

Computes size of scratchpad memory required for the :ref:`onemath_lapack_getrs_batch` function.

.. container:: section

  .. rubric:: Description

``getrs_batch_scratchpad_size`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

**Group API**

Computes the number of elements of type ``T`` the scratchpad memory should able to hold to be passed to the Group API of the :ref:`onemath_lapack_getrs_batch` function.

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      template <typename T>
      std::int64_t getrs_batch_scratchpad_size(cl::sycl::queue &queue, math::transpose *trans, std::int64_t *n, std::int64_t *nrhs, std::int64_t *lda, std::int64_t *ldb, std::int64_t group_count, std::int64_t *group_sizes)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

trans
 | Array of ``group_count`` parameters :math:`\text{trans}_g` indicating the form of the equations for the group :math:`g`:
 | If ``trans = math::transpose::nontrans``, then :math:`A_iX_i = B_i` is solved for :math:`X_i`.
 | If ``trans = math::transpose::trans``, then :math:`A_i^TX_i = B_i` is solved for :math:`X_i`.
 | If ``trans = math::transpose::conjtrans``, then :math:`A_iHX_i = B_i` is solved for :math:`X_i`.

n
  Array of ``group_count`` parameters :math:`n_g` specifying the order of the matrices :math:`A_i` and the number of rows in matrices :math:`B_i` (:math:`0 \le n_g`) belonging to group :math:`g`.

nrhs
  Array of ``group_count`` parameters nrhsg specifying the number of right-hand sides (:math:`0 \le \text{nrhs}_g`) for group :math:`g`.

lda
  Array of ``group_count`` parameters :math:`\text{lda}_g` specifying the leading dimensions of :math:`A_i` from group :math:`g`.

ldb
  Array of ``group_count`` parameters :math:`\text{ldb}_g` specifying the leading dimensions of :math:`B_i` in the group :math:`g`.

group_count
  Number of groups of parameters. Must be at least 0.

group_sizes
  Array of ``group_count`` integers. Array element with index :math:`g` specifies the number of problems to solve for each of the groups of parameters :math:`g`. So the total number of problems to solve, ``batch_size``, is a sum of all parameter group sizes.

.. container:: section
   
   .. rubric:: Return Values

Number of elements of type ``T`` the scratchpad memory should able to hold to be passed to the Group API of the :ref:`onemath_lapack_getrs_batch` function.

.. container:: section

  .. rubric:: Throws

This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`

:ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

:ref:`oneapi::math::lapack::invalid_argument<onemath_lapack_exception_invalid_argument>`

   Exception is thrown in case of incorrect supplied argument value.
   Position of wrong argument can be determined by `info()` method of exception object.

**Strided API**

Computes the number of elements of type ``T`` the scratchpad memory should able to hold to be passed to the Strided API of the :ref:`onemath_lapack_getrs_batch` function.

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      template <typename T>
      std::int64_t getrs_batch_scratchpad_size(cl::sycl::queue &queue, math::transpose trans, std::int64_t n, std::int64_t nrhs, std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size)
    };

.. container:: section

   .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

trans
 | Indicates the form of the equations:
 | ``If trans = math::transpose::nontrans``, then :math:`A_iX_i = B_i` is solved for :math:`X_i`.
 | If ``trans = math::transpose::trans``, then :math:`A_i^TX_i = B_i` is solved for :math:`X_i`.
 | If ``trans = math::transpose::conjtrans``, then :math:`A_i^HX_i = B_i` is solved for :math:`X_i`.

n
  Order of the matrices :math:`A_i` and the number of rows in matrices :math:`B_i` (:math:`0 \le n`).

nrhs
  Number of right-hand sides (:math:`0 \le \text{nrhs}`).

lda
  Leading dimension of :math:`A_i`.

stride_a
  Stride between the beginnings of matrices :math:`A_i` inside the batch array ``a``.

stride_ipiv
  Stride between the beginnings of arrays :math:`ipiv_i` inside the array ``ipiv``.

ldb
  Leading dimension of :math:`B_i`.

stride_b
  Stride between the beginnings of matrices :math:`B_i` inside the batch array ``b``.

batch_size
  Number of problems in a batch.

.. container:: section
   
   .. rubric:: Return Values

Number of elements of type ``T`` the scratchpad memory should able to hold to be passed to the Strided API of the :ref:`onemath_lapack_getrs_batch` function.

.. container:: section

  .. rubric:: Throws

This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`

:ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

:ref:`oneapi::math::lapack::invalid_argument<onemath_lapack_exception_invalid_argument>`

   Exception is thrown in case of incorrect supplied argument value.
   Position of wrong argument can be determined by `info()` method of exception object.

**Parent topic:** :ref:`onemath_lapack-like-extensions-routines`

