.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_getri_batch:

getri_batch
===========

Computes the inverses of a batch of LU-factored matrices determined by :ref:`onemath_lapack_getrf_batch`.

.. container:: section

  .. rubric:: Description

``getri_batch`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_lapack_getri_batch_buffer:

getri_batch (Buffer Version)
----------------------------

.. container:: section

  .. rubric:: Description

The buffer version of ``getri_batch`` supports only the strided API. 

**Strided API**

The routine computes the inverses :math:`A_i^{-1}` of general matrices :math:`A_i`. Before calling this routine, call the Strided API of the :ref:`onemath_lapack_getrf_batch_buffer` function to factorize :math:`A_i`.

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      void getri_batch(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<T> &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, cl::sycl::buffer<T> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

n
  Order of the matrices :math:`A_i` (:math:`0 \le n`).

a
  Result of the Strided API of the :ref:`onemath_lapack_getrf_batch_buffer` function.

lda
  Leading dimension of :math:`A_i` (:math:`n\le \text{lda}`).

stride_a
  Stride between the beginnings of matrices :math:`A_i` inside the batch array ``a``.

ipiv
  Arrays returned by the Strided API of the :ref:`onemath_lapack_getrf_batch_buffer` function.

stride_ipiv
  Stride between the beginnings of arrays :math:`\text{ipiv}_i` inside the array ``ipiv``.

batch_size
  Number of problems in a batch.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.

scratchpad_size
  Size of scratchpad memory as a number of floating point elements of type ``T``. Size should not be less than the value returned by the Strided API of the :ref:`onemath_lapack_getri_batch_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

a
  Inverse :math:`n \times n` matrices :math:`A_i^{-1}`.

.. container:: section

  .. rubric Throws

This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::math::lapack::batch_error<onemath_lapack_exception_batch_error>`

:ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`

:ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

:ref:`oneapi::math::lapack::invalid_argument<onemath_lapack_exception_invalid_argument>`
 

The ``info`` code of the problem can be obtained by `info()` method of exception object:

    If ``info = -n``, the :math:`n`-th parameter had an illegal value.

    If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should be not less then value returned by `detail()` method of exception object.

    If ``info`` is not zero and `detail()` returns zero, then there were some errors for some of the problems in the supplied batch and ``info`` code contains the number of failed calculations in a batch.

getri_batch (USM Version)
-------------------------

.. container:: section

  .. rubric:: Description

The USM version of ``getri_batch`` supports the group API and strided API. 

**Group API**

The routine computes the inverses :math:`A_i^{-1}` of general matrices :math:`A_i`, :math:`i \in \{1...batch\_size\}`. Before calling this routine, call the Group API of the :ref:`onemath_lapack_getrf_batch_usm` function to factorize :math:`A_i`.
Total number of problems to solve, ``batch_size``, is a sum of sizes of all of the groups of parameters as provided by ``group_sizes`` array.

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      cl::sycl::event getri_batch(cl::sycl::queue &queue, std::int64_t *n, T **a, std::int64_t *lda, const std::int64_t * const *ipiv, std::int64_t group_count, std::int64_t *group_sizes, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

n
  Array of ``group_count`` :math:`n_g` parameters specifying the order of the matrices :math:`A_i` (:math:`0 \le n_g`) belonging to group :math:`g`.

a
  Result of the Group API of the :ref:`onemath_lapack_getrf_batch_usm` function.

lda
  Array of ``group_count`` :math:`\text{lda}_g` parameters specifying the leading dimensions of the matrices :math:`A_i` (:math:`n_g \le \text{lda}_g`) belonging to group :math:`g`.

ipiv
  Arrays returned by the Group API of the :ref:`onemath_lapack_getrf_batch_usm` function.

group_count
  Number of groups of parameters. Must be at least 0.

group_sizes
  Array of ``group_count`` integers. Array element with index :math:`g` specifies the number of problems to solve for each of the groups of parameters :math:`g`. So the total number of problems to solve, ``batch_size``, is a sum of all parameter group sizes.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.

scratchpad_size
  Size of scratchpad memory as a number of floating point elements of  type ``T``. Size should not be less than the value returned by the Group API of the :ref:`onemath_lapack_getri_batch_scratchpad_size` function.

events
  List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

a
  Inverse :math:`n_g \times n_g` matrices :math:`A_i^{-1}`.

.. container:: section
   
  .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

.. container:: section

  .. rubric:: Throws

This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::math::lapack::batch_error<onemath_lapack_exception_batch_error>`

:ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`

:ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

:ref:`oneapi::math::lapack::invalid_argument<onemath_lapack_exception_invalid_argument>`
 
   The ``info`` code of the problem can be obtained by `info()` method of exception object:

    If ``info = -n``, the :math:`n`-th parameter had an illegal value.

    If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should be not less then value returned by `detail()` method of exception object.

    If ``info`` is not zero and `detail()` returns zero, then there were some errors for some of the problems in the supplied batch and ``info`` code contains the number of failed calculations in a batch.

**Strided API**

The routine computes the inverses :math:`A_i^{-1}` of general matrices :math:`A_i`. Before calling this routine, call the Strided API of the :ref:`onemath_lapack_getrf_batch_usm` function to factorize :math:`A_i`.

.. container:: section
   
  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      cl::sycl::event getri_batch(cl::sycl::queue &queue, std::int64_t n, T *a, std::int64_t lda, std::int64_t stride_a, const std::int64_t *ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    };

.. container:: section

  .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

n
  Order of the matrices :math:`A_i` (:math:`0 \le n`).

a
  Result of the Strided API of the :ref:`onemath_lapack_getrf_batch_usm` function.

lda
  Leading dimension of :math:`A_i` (:math:`n \le \text{lda}`).

stride_a
  Stride between the beginnings of matrices :math:`A_i` inside the batch array ``a``.

ipiv
  Arrays returned by the Strided API of the :ref:`onemath_lapack_getrf_batch_usm` function.

stride_ipiv
  Stride between the beginnings of arrays :math:`\text{ipiv}_i` inside the array ``ipiv``.

batch_size
  Number of problems in a batch.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.

scratchpad_size 
  Size of scratchpad memory as a number of floating point elements of type ``T``. Size should not be less than the value returned by the Strided API of the :ref:`onemath_lapack_getri_batch_scratchpad_size` function.

events
  List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

a
  Inverse :math:`n \times n` matrices :math:`A_i^{-1}`.

.. container:: section
   
  .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

.. container:: section

  .. rubric:: Throws

This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::math::lapack::batch_error<onemath_lapack_exception_batch_error>`

:ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`

:ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

:ref:`oneapi::math::lapack::invalid_argument<onemath_lapack_exception_invalid_argument>`

   The ``info`` code of the problem can be obtained by `info()` method of exception object:

    If ``info = -n``, the :math:`n`-th parameter had an illegal value.

    If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should be not less then value returned by `detail()` method of exception object.

    If ``info`` is not zero and `detail()` returns zero, then there were some errors for some of the problems in the supplied batch and ``info`` code contains the number of failed calculations in a batch.

**Parent topic:** :ref:`onemath_lapack-like-extensions-routines`

