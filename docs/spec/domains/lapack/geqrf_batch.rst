.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_geqrf_batch:

geqrf_batch
===========

Computes the QR factorizations of a batch of general matrices.

.. container:: section

  .. rubric:: Description

``geqrf_batch`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_lapack_geqrf_batch_buffer:

geqrf_batch (Buffer Version)
----------------------------

.. container:: section

  .. rubric:: Description

The buffer version of ``geqrf_batch`` supports only the strided API. 
 
**Strided API**

.. container:: section

   .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      void geqrf_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, cl::sycl::buffer<T> &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<T> &tau, std::int64_t stride_tau, std::int64_t batch_size, cl::sycl::buffer<T> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

   .. rubric:: Input Parameters

queue  
   Device queue where calculations will be performed.
 
m
   Number of rows in matrices :math:`A_i` (:math:`0 \le m`).

n  
   Number of columns in matrices :math:`A_i` (:math:`0 \le n`).

a
   Array holding input matrices :math:`A_i`. 

lda
   Leading dimension of matrices :math:`A_i`.

stride_a
   Stride between the beginnings of matrices :math:`A_i` inside the batch array ``a``.

stride_tau
   Stride between the beginnings of arrays :math:`\tau_i` inside the array ``tau``.

batch_size
   Number of problems in a batch.

scratchpad
   Scratchpad memory to be used by routine for storing intermediate results.
         
scratchpad_size
   Size of scratchpad memory as the number of floating point elements of type ``T``. Size should not be less than the value returned by the Strided API of the :ref:`onemath_lapack_geqrf_batch_scratchpad_size` function.

.. container:: section

   .. rubric:: Output Parameters
 
a
  Factorization data as follows: The elements on and above the diagonal of :math:`A_i` contain the :math:`\min(m,n) \times n` upper trapezoidal matrices :math:`R_i` (:math:`R_i` is upper triangular if :math:`m \ge n`); the elements below the diagonal, with the array :math:`\tau_i`, contain the orthogonal matrix :math:`Q_i` as a product of :math:`\min(m,n)` elementary reflectors.

tau 
    Array to store batch of :math:`\tau_i`, each of size :math:`\min(m,n)`, containing scalars that define elementary reflectors for the matrices :math:`Q_i` in its decomposition in a product of elementary reflectors.

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

.. _onemath_lapack_geqrf_batch_usm:

geqrf_batch (USM Version)
-------------------------

.. container:: section

  .. rubric:: Description

The USM version of ``geqrf_batch`` supports the group API and strided API. 

**Group API**

The routine forms the :math:`Q_iR_i` factorizations of a general :math:`m \times n` matrices :math:`A_i`, :math:`i \in \{1...batch\_size\}`, where ``batch_size`` is the sum of all parameter group sizes as provided with ``group_sizes`` array.
No pivoting is performed during factorization.
The routine does not form the matrices :math:`Q_i` explicitly. Instead, :math:`Q_i` is represented as a product of :math:`\min(m,n)` elementary reflectors. Routines are provided to work with :math:`Q_i` in this representation.
The total number of problems to solve, ``batch_size``, is a sum of sizes of all of the groups of parameters as provided by ``group_sizes`` array.

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      cl::sycl::event geqrf_batch(cl::sycl::queue &queue, std::int64_t *m, std::int64_t *n, T **a, std::int64_t *lda, T **tau, std::int64_t group_count, std::int64_t *group_sizes, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    }

.. container:: section

   .. rubric:: Input Parameters

queue 
  Device queue where calculations will be performed.

m
  Array of ``group_count`` :math:`m_g` parameters. Each :math:`m_g` specifies the number of rows in matrices :math:`A_i` from array ``a``, belonging to group :math:`g`.

n 
  Array of ``group_count`` :math:`n_g` parameters.
  Each :math:`n_g` specifies the number of columns in matrices :math:`A_i` from array ``a``, belonging to group :math:`g`.

a  
  Array of ``batch_size`` pointers to input matrices :math:`A_i`, each of size :math:`\text{lda}_g\cdot n_g` (:math:`g` is an index of group to which :math:`A_i` belongs)

lda
  Array of ``group_count`` :math:`\text{lda}_g`` parameters, each representing the leading dimensions of input matrices :math:`A_i` from array ``a``, belonging to group :math:`g`.

group_count
  Specifies the number of groups of parameters. Must be at least 0.

group_sizes 
  Array of ``group_count`` integers. Array element with index :math:`g` specifies the number of problems to solve for each of the groups of parameters :math:`g`. So the total number of problems to solve, ``batch_size``, is a sum of all parameter group sizes.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.

scratchpad_size
  Size of scratchpad memory as the number of floating point elements of type ``T``. Size should not be less than the value returned by the Group API of the :ref:`onemath_lapack_geqrf_batch_scratchpad_size` function.

events
  List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

   .. rubric:: Output Parameters

a
  Factorization data as follows: The elements on and above the diagonal of :math:`A_i` contain the :math:`\min(m_g,n_g) \times n_g` upper trapezoidal matrices :math:`R_i` (:math:`R_i` is upper triangular if :math:`m_g \ge n_g`); the elements below the diagonal, with the array :math:`\tau_i`, contain the orthogonal matrix :math:`Q_i` as a product of :math:`\min(m_g,n_g)` elementary reflectors. Here :math:`g` is the index of the parameters group corresponding to the :math:`i`-th decomposition.

tau
  Array of pointers to store arrays :math:`\tau_i`, each of size :math:`\min(m_g,n_g)`, containing scalars that define elementary reflectors for the matrices :math:`Q_i` in its decomposition in a product of elementary reflectors. Here :math:`g` is the index of the parameters group corresponding to the :math:`i`-th decomposition.

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

The routine forms the :math:`Q_iR_i` factorizations of general :math:`m \times n` matrices :math:`A_i`. No pivoting is performed.
The routine does not form the matrices :math:`Q_i` explicitly. Instead, :math:`Q_i` is represented as a product of :math:`\min(m,n)` elementary reflectors. Routines are provided to work with :math:`Q_i` in this representation.

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      sycl::event geqrf_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, T *a, std::int64_t lda, std::int64_t stride_a, T *tau, std::int64_t stride_tau, std::int64_t batch_size, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    }

.. container:: section

   .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

m 
  Number of rows in matrices :math:`A_i` (:math:`0 \le m`).

n
  Number of columns in matrices :math:`A_i` (:math:`0 \le n`).

a
  Array holding input matrices :math:`A_i`.

lda
  Leading dimensions of :math:`A_i`.

stride_a
  Stride between the beginnings of matrices :math:`A_i` inside the batch array ``a``.

stride_tau
  Stride between the beginnings of arrays :math:`\tau_i` inside the array ``tau``.

batch_size
  Number of problems in a batch.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.

scratchpad_size
  Size of scratchpad memory as the number of floating point elements of type ``T``. Size should not be less than the value returned by the Strided API of the :ref:`onemath_lapack_geqrf_batch_scratchpad_size` function.

events
  List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

   .. rubric:: Output Parameters

a
  Factorization data as follows: The elements on and above the diagonal of :math:`A_i` contain the :math:`\min(m,n) \times n` upper trapezoidal matrices :math:`R_i` (:math:`R_i` is upper triangular if :math:`m \ge n`); the elements below the diagonal, with the array :math:`\tau_i`, contain the orthogonal matrix :math:`Q_i` as a product of :math:`\min(m,n)` elementary reflectors.

tau
  Array to store batch of :math:`\tau_i`, each of size :math:`\min(m,n)`, containing scalars that define elementary reflectors for the matrices :math:`Q_i` in its decomposition in a product of elementary reflectors.

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
