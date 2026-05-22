.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_ungqr_batch_scratchpad_size:

ungqr_batch_scratchpad_size
===========================

Computes size of scratchpad memory required for the :ref:`onemath_lapack_ungqr_batch` function.

.. container:: section

  .. rubric:: Description

``ungqr_batch_scratchpad_size`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

**Group API**

Computes the number of elements of type ``T`` the scratchpad memory should able to hold to be passed to the Group API of the :ref:`onemath_lapack_ungqr_batch` function.

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      template <typename T>
      std::int64_t ungqr_batch_scratchpad_size(cl::sycl::queue &queue, std::int64_t *m, std::int64_t *n, std::int64_t *k, std::int64_t *lda, std::int64_t group_count, std::int64_t *group_sizes)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

m
  Array of ``group_count`` :math:`m_g` parameters.

n
  Array of ``group_count`` :math:`n_g` parameters.

k
 | Array of ``group_count`` :math:`k_g` parameters.
 | Number of elementary reflectors whose product defines the matrices :math:`Q_i` (:math:`0 \le k_g \le n_g`).
 
lda
  Array of leading dimensions of :math:`A_i`.

group_count
  Number of groups of parameters. Must be at least 0.

group_sizes
  Array of ``group_count`` integers. Array element with index :math:`g` specifies the number of problems to solve for each of the groups of parameters :math:`g`. So the total number of problems to solve, ``batch_size``, is a sum of all parameter group sizes.

.. container:: section
   
  .. rubric:: Return Values

Number of elements of type ``T`` the scratchpad memory should able to hold to be passed to the Group API of the :ref:`onemath_lapack_ungqr_batch` function.

.. container:: section

  .. rubric:: Throws

This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`

:ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

:ref:`oneapi::math::lapack::invalid_argument<onemath_lapack_exception_invalid_argument>`

   Exception is thrown in case of incorrect supplied argument value.
   Position of wrong argument can be determined by `info()` method of exception object.

**Strided API**

Computes the number of elements of type ``T`` the scratchpad memory should able to hold to be passed to the Strided API of the :ref:`onemath_lapack_ungqr_batch` function.

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      template <typename T>
      std::int64_t ungqr_batch_scratchpad_size(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda, std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size)
    };

.. container:: section

  .. rubric:: Input Parameters


queue
  Device queue where calculations will be performed.

m
  Number of rows in the matrices :math:`A_i` (:math:`0 \le m`).

n
  Number of columns in the matrices :math:`A_i` (:math:`0 \le n`).

k
  Number of elementary reflectors whose product defines the matrices :math:`Q_i` (:math:`0 \le k \le n`).

lda
  Leading dimensions of :math:`A_i` (:math:`\text{lda} \le m`).

stride_a
  Stride between the beginnings of matrices :math:`A_i` inside the batch array ``a``.

stride_tau
  Stride between the beginnings of arrays :math:`\tau_i` inside the array ``tau``.

batch_size
  Number of problems in a batch.

.. container:: section
   
  .. rubric:: Return Values

Number of elements of type ``T`` the scratchpad memory should able to hold to be passed to the Strided API of the :ref:`onemath_lapack_ungqr_batch` function.

.. container:: section

  .. rubric:: Throws

This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`

:ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

:ref:`oneapi::math::lapack::invalid_argument<onemath_lapack_exception_invalid_argument>`

   Exception is thrown in case of incorrect supplied argument value.
   Position of wrong argument can be determined by `info()` method of exception object.

**Parent topic:** :ref:`onemath_lapack-like-extensions-routines`

