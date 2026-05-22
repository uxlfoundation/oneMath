.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_getrs_batch:

getrs_batch
===========

Solves a system of linear equations with a batch of LU-factored square coefficient matrices, with multiple right-hand sides.

.. container:: section

  .. rubric:: Description

``getrs_batch`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_lapack_getrs_batch_buffer:

getrs_batch (Buffer Version)
----------------------------

.. container:: section

  .. rubric:: Description

The buffer version of ``getrs_batch`` supports only the strided API. 
   
**Strided API**

 | The routine solves for the following systems of linear equations :math:`X_i`: 
 | :math:`A_iX_i = B_i`, if ``trans=math::transpose::nontrans``
 | :math:`A_i^TX_i = B_i`, if ``trans=math::transpose::trans``
 | :math:`A_i^HX_i = B_i`, if ``trans=math::transpose::conjtrans``
 | Before calling this routine, the Strided API of the :ref:`onemath_lapack_getrf_batch_buffer` function should be called to compute the LU factorizations of :math:`A_i`.

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      void getrs_batch(cl::sycl::queue &queue, math::transpose trans, std::int64_t n, std::int64_t nrhs, cl::sycl::buffer<T> &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv, cl::sycl::buffer<T> &b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size, cl::sycl::buffer<T> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

trans
 | Form of the equations:
 | If ``trans = math::transpose::nontrans``, then :math:`A_iX_i = B_i` is solved for :math:`Xi`.
 | If ``trans = math::transpose::trans``, then :math:`A_i^TX_i = B_i` is solved for :math:`X_i`.
 | If ``trans = math::transpose::conjtrans``, then :math:`A_i^HX_i = B_i` is solved for :math:`X_i`.

n
  Order of the matrices :math:`A_i` and the number of rows in matrices :math:`B_i` (:math:`0 \le n`).

nrhs
  Number of right-hand sides (:math:`0 \le \text{nrhs}`).

a
  Array containing the factorizations of the matrices :math:`A_i`, as returned the Strided API of the :ref:`onemath_lapack_getrf_batch_buffer` function.

lda
  Leading dimension of :math:`A_i`.

stride_a
  Stride between the beginnings of matrices :math:`A_i` inside the batch array ``a``.

ipiv
  ``ipiv`` array, as returned by the Strided API of the :ref:`onemath_lapack_getrf_batch_buffer` function.

stride_ipiv
  Stride between the beginnings of arrays :math:`\text{ipiv}_i` inside the array ``ipiv``.

b 
  Array containing the matrices :math:`B_i` whose columns are the right-hand sides for the systems of equations.

ldb
  Leading dimension of :math:`B_i`.

stride_b
  Stride between the beginnings of matrices :math:`B_i` inside the batch array ``b``.
  
batch_size
  Specifies the number of problems in a batch.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.

scratchpad_size
  Size of scratchpad memory as a number of floating point elements of type ``T``. Size should not be less then the value returned by the Strided API of the :ref:`onemath_lapack_getrs_batch_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

b  
  Solution matrices :math:`X_i`.

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

    If ``info`` is zero, then diagonal element of some of :math:`U_i` is zero, and the solve could not be completed. The indices of such matrices in the batch can be obtained with `ids()` method of the exception object. The indices of first zero diagonal elements in these :math:`U_i` matrices can be obtained by `exceptions()` method of exception object.

.. _onemath_lapack_getrs_batch_usm:

getrs_batch (USM Version)
-------------------------

.. container:: section

  .. rubric:: Description

The USM version of ``getrs_batch`` supports the group API and strided API. 

**Group API**

 | The routine solves the following systems of linear equations for :math:`X_i` (:math:`i \in \{1...batch\_size\}`):
 | :math:`A_iX_i = B_i`, if ``trans=math::transpose::nontrans``
 | :math:`A_i^TX_i = B_i`, if ``trans=math::transpose::trans``
 | :math:`A_i^HX_i = B_i`, if ``trans=math::transpose::conjtrans``
 | Before calling this routine, call the Group API of the :ref:`onemath_lapack_getrf_batch_usm` function to compute the LU factorizations of :math:`A_i`.
 | Total number of problems to solve, ``batch_size``, is a sum of sizes of all of the groups of parameters as provided by ``group_sizes`` array.

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      cl::sycl::event getrs_batch(cl::sycl::queue &queue, math::transpose *trans, std::int64_t *n, std::int64_t *nrhs, const T * const *a, std::int64_t *lda, const std::int64_t * const *ipiv, T **b, std::int64_t *ldb, std::int64_t group_count, std::int64_t *group_sizes, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

trans
 | Array of ``group_count`` parameters :math:`trans_g` indicating the form of the equations for the group :math:`g`:
 | If ``trans = math::transpose::nontrans``, then :math:`A_iX_i = B_i` is solved for :math:`X_i`.
 | If ``trans = math::transpose::trans``, then :math:`A_i^TX_i = B_i` is solved for :math:`X_i`.
 | If ``trans = math::transpose::conjtrans``, then :math:`A_i^HX_i = B_i` is solved for :math:`X_i`.

n
  Array of ``group_count`` parameters :math:`n_g` specifying the order of the matrices :math:`A_i` and the number of rows in matrices :math:`B_i` (:math:`0 \le n_g`) belonging to group :math:`g`.

nrhs
  Array of ``group_count`` parameters :math:`\text{nrhs}_g` specifying the number of right-hand sides (:math:`0 \le \text{nrhs}_g`) for group :math:`g`.

a
  Array of ``batch_size`` pointers to factorizations of the matrices :math:`A_i`, as returned by the Group API of the:ref:`onemath_lapack_getrf_batch_usm` function.

lda
  Array of ``group_count`` parameters :math:`\text{lda}_g` specifying the leading dimensions of :math:`A_i` from group :math:`g`.

ipiv
  ``ipiv`` array, as returned by the Group API of the :ref:`onemath_lapack_getrf_batch_usm` function.

b 
  The array containing ``batch_size`` pointers to the matrices :math:`B_i` whose columns are the right-hand sides for the systems of equations.

ldb
  Array of ``group_count`` parameters :math:`\text{ldb}_g` specifying the leading dimensions of :math:`B_i` in the group :math:`g`.

group_count
  Specifies the number of groups of parameters. Must be at least 0.
    
group_sizes
  Array of ``group_count`` integers. Array element with index :math:`g` specifies the number of problems to solve for each of the groups of parameters :math:`g`. So the total number of problems to solve, ``batch_size``, is a sum of all parameter group sizes.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.
    
scratchpad_size
  Size of scratchpad memory as a number of floating point elements of type ``T``. Size should not be less then the value returned by the Group API of the :ref:`onemath_lapack_getrs_batch_scratchpad_size` function.
  
events
  List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

b  
  Solution matrices :math:`X_i`.

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

   Exception is thrown in case of problems during calculations. The info code of the problem can be obtained by info() method of exception object:

    If ``info = -n``, the :math:`n`-th parameter had an illegal value.

    If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should be not less then value returned by `detail()` method of exception object.

    If ``info`` is not zero and `detail()` returns zero, then there were some errors for some of the problems in the supplied batch and ``info`` code contains the number of failed calculations in a batch.

    If ``info`` is zero, then diagonal element of some of :math:`U_i` is zero, and the solve could not be completed. The indices of such matrices in the batch can be obtained with `ids()` method of the exception object. The indices of first zero diagonal elements in these :math:`U_i` matrices can be obtained by `exceptions()` method of exception object.

**Strided API**

 | The routine solves the following systems of linear equations for :math:`X_i`:
 | :math:`A_iX_i = B_i`, if ``trans=math::transpose::nontrans``
 | :math:`A_i^TX_i = B_i`, if ``trans=math::transpose::trans``
 | :math:`A_i^HX_i = B_i`, if ``trans=math::transpose::conjtrans``
 | Before calling this routine, the Strided API of the :ref:`onemath_lapack_getrf_batch` function should be called to compute the LU factorizations of :math:`A_i`.

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      cl::sycl::event getrs_batch(cl::sycl::queue &queue, math::transpose trans, std::int64_t n, std::int64_t nrhs, const T *a, std::int64_t lda, std::int64_t stride_a, const std::int64_t *ipiv, std::int64_t stride_ipiv, T *b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    };

.. container:: section

  .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

trans
 | Form of the equations:
 | If ``trans = math::transpose::nontrans``, then :math:`A_iX_i = B_i` is solved for :math:`X_i`.
 | If ``trans = math::transpose::trans``, then :math:`A_i^TX_i = B_i` is solved for :math:`X_i`.
 | If ``trans = math::transpose::conjtrans``, then :math:`A_i^HX_i = B_i` is solved for :math:`X_i`.

n
  Order of the matrices :math:`A_i` and the number of rows in matrices :math:`B_i` (:math:`0 \le n`).

nrhs
  Number of right-hand sides (:math:`0 \le \text{nrhs}`).

a
  Array containing the factorizations of the matrices :math:`A_i`, as returned by the Strided API of the:ref:`onemath_lapack_getrf_batch_usm` function.

lda
  Leading dimension of :math:`A_i`.

stride_a  
  Stride between the beginnings of matrices :math:`A_i` inside the batch array ``a``.

ipiv
  ``ipiv`` array, as returned by getrf_batch (USM) function.

stride_ipiv
  Stride between the beginnings of arrays :math:`\text{ipiv}_i` inside the array ``ipiv``.

b
  Array containing the matrices :math:`B_i` whose columns are the right-hand sides for the systems of equations.

ldb
  Leading dimensions of :math:`B_i`.

stride_b  
  Stride between the beginnings of matrices :math:`B_i` inside the batch array ``b``.
  
batch_size
  Number of problems in a batch.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.
    
scratchpad_size 
  Size of scratchpad memory as a number of floating point elements of type ``T``. Size should not be less then the value returned by the Strided API of the :ref:`onemath_lapack_getrs_batch_scratchpad_size` function.

events
  List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

b  
  Solution matrices :math:`X_i`.

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
   
   If ``info`` is zero, then diagonal element of some of :math:`U_i` is zero, and the solve could not be completed. The indices of such matrices in the batch can be obtained with `ids()` method of the exception object. The indices of first zero diagonal elements in these :math:`U_i` matrices can be obtained by `exceptions()` method of exception object.


**Parent topic:** :ref:`onemath_lapack-like-extensions-routines`

