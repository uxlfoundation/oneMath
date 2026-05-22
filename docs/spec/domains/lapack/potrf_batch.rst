.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_potrf_batch:

potrf_batch
===========

Computes the LU factorizations of a batch of general matrices.

.. container:: section

  .. rubric:: Description

``potrf_batch`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_lapack_potrf_batch_buffer:

potrf_batch (Buffer Version)
----------------------------

.. container:: section

  .. rubric:: Description

The buffer version of ``potrf_batch`` supports only the strided API. 
   
**Strided API**

 | The routine forms the Cholesky factorizations of a symmetric positive-definite or, for complex data, Hermitian positive-definite matrices :math:`A_i`, :math:`i \in \{1...batch\_size\}`:
 | :math:`A_i = U_i^TU_i` for real data, :math:`A_i = U_i^HU_i` for complex data if ``uplo = math::uplo::upper``,
 | :math:`A_i = L_iL_i^T` for real data, :math:`A_i = L_iL_i^H` for complex data if ``uplo = math::uplo::lower``,
 | where :math:`L_i` is a lower triangular matrix and :math:`U_i` is upper triangular.

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      void potrf_batch(cl::sycl::queue &queue, math::uplo uplo, std::int64_t n, cl::sycl::buffer<T> &a, std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size, cl::sycl::buffer<T> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

uplo
   | Indicates whether the upper or lower triangular part of :math:`A_i` is stored and how :math:`A_i` is factored:
   | If ``uplo = math::uplo::upper``, the array ``a`` stores the upper triangular parts of the matrices :math:`A_i`,
   | If ``uplo = math::uplo::lower``, the array ``a`` stores the lower triangular parts of the matrices :math:`A_i`.

n
  Order of the matrices :math:`A_i`, (:math:`0 \le n`).

a
  Array containing batch of input matrices :math:`A_i`, each of :math:`A_i` being of size :math:`\text{lda} \cdot n` and holding either upper or lower triangular parts of the matrices :math:`A_i` (see ``uplo``).

lda
  Leading dimension of :math:`A_i`.

stride_a
  Stride between the beginnings of matrices :math:`A_i` inside the batch.

batch_size
  Number of problems in a batch.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.

scratchpad_size
  Size of scratchpad memory as a number of floating point elements of type ``T``. Size should not be less then the value returned by the Strided API of the :ref:`onemath_lapack_potrf_batch_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

a
	Cholesky factors :math:`U_i` or :math:`L_i`, as specified by ``uplo``.

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
   
   If ``info`` is zero, then the leading minors of some of matrices (and therefore some matrices :math:`A_i` themselves) are not positive-definite, and the factorizations could not be completed for these matrices from the batch. The indices of such matrices in the batch can be obtained with `ids()` method of the exception object. The orders of corresponding not positive-definite leading minors of these matrices can be obtained by `exceptions()` method of exception object.

.. _onemath_lapack_potrf_batch_usm:

potrf_batch (USM Version)
-------------------------

.. container:: section

  .. rubric:: Description

The USM version of ``potrf_batch`` supports the group API and strided API. 

**Group API**

 | The routine forms the Cholesky factorizations of symmetric positive-definite or, for complex data, Hermitian positive-definite matrices :math:`A_i`, :math:`i \in \{1...batch\_size\}`:
 | :math:`A_i = U_i^TU_i` for real data (:math:`A_i = U_i^HU_i` for complex), if :math:`\text{uplo}_g` is ``math::uplo::upper``,
 | :math:`A_i = L_iL_i^T` for real data (:math:`A_i = L_iL_i^H` for complex), if :math:`\text{uplo}_g` is ``math::uplo::lower``,
 | where :math:`L_i` is a lower triangular matrix and :math:`U_i` is upper triangular, :math:`g` is an index of group of parameters corresponding to :math:`A_i`, and total number of problems to solve, ``batch_size``, is a sum of sizes of all of the groups of parameters as provided by ``group_sizes`` array

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      cl::sycl::event potrf_batch(cl::sycl::queue &queue, math::uplo *uplo, std::int64_t *n, T **a, std::int64_t *lda, std::int64_t group_count, std::int64_t *group_sizes, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

uplo
  | Array of ``group_count`` :math:`\text{uplo}_g` parameters. Each :math:`\text{uplo}_g` indicates whether the upper or lower triangular parts of the input matrices are provided:
  | If :math:`\text{uplo}_g` is ``math::uplo::upper``, input matrices from array ``a`` belonging to group :math:`g` store the upper triangular parts,
  | If :math:`\text{uplo}_g` is ``math::uplo::lower``, input matrices from array ``a`` belonging to group :math:`g` store the lower triangular parts.

n
  Array of ``group_count`` :math:`n_g` parameters. Each :math:`n_g` specifies the order of the input matrices from array a belonging to group :math:`g`.

a
  Array of ``batch_size`` pointers to input matrices :math:`A_i`, each being of size :math:`\text{lda}_g \cdot n_g` (:math:`g` is an index of group to which :math:`A_i` belongs to) and holding either upper or lower triangular part as specified by :math:`\text{uplo}_g`.

lda
  Array of ``group_count`` :math:`\text{lda}_g` parameters. Each :math:`\text{lda}_g` specifies the leading dimensions of the matrices from a belonging to group :math:`g`.

group_count
  Number of groups of parameters. Must be at least 0.

group_sizes
  Array of group_count integers. Array element with index :math:`g` specifies the number of problems to solve for each of the groups of parameters :math:`g`. So the total number of problems to solve, ``batch_size``, is a sum of all parameter group sizes.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.

scratchpad_size
  Size of scratchpad memory as a number of floating point elements of type ``T``. Size should not be less then the value returned by the Group API of the :ref:`onemath_lapack_potrf_batch_scratchpad_size` function.

events
  List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

a
	Cholesky factors :math:`U_i` or :math:`L_i`, as specified by :math:`\text{uplo}_g` from corresponding group of parameters.

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
   
   If ``info`` is zero, then the leading minors of some of the input matrices (and therefore some matrices themselves) are not positive-definite, and the factorizations could not be completed for these matrices from the batch. The indices of such matrices in the batch can be obtained with `ids()` method of the exception object. The orders of corresponding not positive-definite leading minors of these matrices can be obtained by `exceptions()` method of the exception object.

**Strided API**

 | The routine forms the Cholesky factorizations of a symmetric positive-definite or, for complex data, Hermitian positive-definite matrices :math:`A_i`, :math:`i \in \{1...batch\_size\}`:
 | :math:`A_i = U_i^TU_i` for real data, :math:`A_i = U_i^HU_i` for complex data if ``uplo = math::uplo::upper``,
 | :math:`A_i = L_iL_i^T` for real data, :math:`A_i = L_iL_i^H` for complex data if ``uplo = math::uplo::lower``,
 | where :math:`L_i` is a lower triangular matrix and :math:`U_i` is upper triangular.

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      cl::sycl::event potrf_batch(cl::sycl::queue &queue, math::uplo uplo, std::int64_t n, T *a, std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    };

.. container:: section

  .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

uplo
   | Indicates whether the upper or lower triangular part of :math:`A_i` is stored and how :math:`A_i` is factored:
   | If ``uplo = math::uplo::upper``, the array ``a`` stores the upper triangular parts of the matrices :math:`A_i`,
   | If ``uplo = math::uplo::lower``, the array ``a`` stores the lower triangular parts of the matrices :math:`A_i`.

n
  Order of the matrices :math:`A_i`, (:math:`0 \le n`).

a
  Array containing batch of input matrices :math:`A_i`, each of :math:`A_i` being of size :math:`\text{lda} \cdot n` and holding either upper or lower triangular parts of the matrices :math:`A_i` (see ``uplo``).

lda
  Leading dimension of :math:`A_i`.

stride_a
  Stride between the beginnings of matrices :math:`A_i` inside the batch.

batch_size
  Number of problems in a batch.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.

scratchpad_size
  Size of scratchpad memory as a number of floating point elements of type ``T``. Size should not be less then the value returned by the Strided API of the :ref:`onemath_lapack_potrf_batch_scratchpad_size` function.

events
  List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

a
	Cholesky factors :math:`U_i` or :math:`L_i`, as specified by ``uplo``.

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
        
   If ``info`` is zero, then the leading minors of some of matrices (and therefore some matrices :math:`A_i` themselves) are not positive-definite, and the factorizations could not be completed for these matrices from the batch. The indices of such matrices in the batch can be obtained with `ids()` method of the exception object. The orders of corresponding not positive-definite leading minors of these matrices can be obtained by `exceptions()` method of exception object.

**Parent topic:** :ref:`onemath_lapack-like-extensions-routines`

