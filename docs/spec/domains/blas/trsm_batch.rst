.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_trsm_batch:

trsm_batch
==========

Computes a group of ``trsm`` operations.

.. _onemath_blas_trsm_batch_description:

.. rubric:: Description

The ``trsm_batch`` routines are batched versions of :ref:`onemath_blas_trsm`, performing
multiple ``trsm`` operations in a single call. Each ``trsm`` 
solves an equation of the form op(A) \* X = alpha \* B or X \* op(A) = alpha \* B. 
   
``trsm_batch`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_blas_trsm_batch_buffer:

trsm_batch (Buffer Version)
---------------------------

.. rubric:: Description

The buffer version of ``trsm_batch`` supports only the strided API. 
   
The strided API operation is defined as:
::

   for i = 0 … batch_size – 1
       A and B are matrices at offset i * stridea and i * strideb in a and b.
       if (left_right == oneapi::math::side::left) then
           compute X such that op(A) * X = alpha * B
       else
           compute X such that X * op(A) = alpha * B
       end if
       B := X
   end for

where:

op(``A``) is one of op(``A``) = ``A``, or op(A) = ``A``\ :sup:`T`,
or op(``A``) = ``A``\ :sup:`H`,

``alpha`` is a scalar,

``A`` is a triangular matrix,

``B`` and ``X`` are ``m`` x ``n`` general matrices,

``A`` is either ``m`` x ``m`` or ``n`` x ``n``,depending on whether
it multiplies ``X`` on the left or right. On return, the matrix ``B``
is overwritten by the solution matrix ``X``.

The ``a`` and ``b`` buffers contain all the input matrices. The stride 
between matrices is given by the stride parameter. The total number
of matrices in ``a`` and ``b`` buffers are given by the ``batch_size`` parameter.

**Strided API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void trsm_batch(sycl::queue &queue,
                       oneapi::math::side left_right,
                       oneapi::math::uplo upper_lower,
                       oneapi::math::transpose trans,
                       oneapi::math::diag unit_diag,
                       std::int64_t m,
                       std::int64_t n,
                       T alpha,
                       sycl::buffer<T,1> &a,
                       std::int64_t lda,
                       std::int64_t stridea,
                       sycl::buffer<T,1> &b,
                       std::int64_t ldb,
                       std::int64_t strideb,
                       std::int64_t batch_size)
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void trsm_batch(sycl::queue &queue,
                       oneapi::math::side left_right,
                       oneapi::math::uplo upper_lower,
                       oneapi::math::transpose trans,
                       oneapi::math::diag unit_diag,
                       std::int64_t m,
                       std::int64_t n,
                       T alpha,
                       sycl::buffer<T,1> &a,
                       std::int64_t lda,
                       std::int64_t stridea,
                       sycl::buffer<T,1> &b,
                       std::int64_t ldb,
                       std::int64_t strideb,
                       std::int64_t batch_size)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   left_right
      Specifies whether the matrices ``A`` multiply ``X`` on the left
      (``side::left``) or on the right (``side::right``). See :ref:`onemath_datatypes` for more details.

   upper_lower
      Specifies whether the matrices ``A`` are upper or lower
      triangular. See :ref:`onemath_datatypes` for more details.

   trans
      Specifies op(``A``), the transposition operation applied to the
      matrices ``A``. See :ref:`onemath_datatypes` for more details.

   unit_diag
      Specifies whether the matrices ``A`` are assumed to be unit
      triangular (all diagonal elements are 1). See :ref:`onemath_datatypes` for more details.

   m
      Number of rows of the ``B`` matrices. Must be at least zero.

   n
      Number of columns of the ``B`` matrices. Must be at least zero.

   alpha
      Scaling factor for the solutions.

   a
      Buffer holding the input matrices ``A`` with size ``stridea`` * ``batch_size``.

   lda
      Leading dimension of the matrices ``A``. Must be at least ``m`` if
      ``left_right`` = ``side::left``, and at least ``n`` if ``left_right`` =
      ``side::right``. Must be positive.

   stridea
      Stride between different ``A`` matrices.

   b
      Buffer holding the input matrices ``B`` with size ``strideb`` * ``batch_size``.

   ldb
      Leading dimension of the matrices ``B``. It must be positive and at least
      ``m`` if column major layout is used to store matrices or at
      least ``n`` if row major layout is used to store matrices.

   strideb
      Stride between different ``B`` matrices.

   batch_size
      Specifies the number of triangular linear systems to solve.

.. container:: section

   .. rubric:: Output Parameters

   b
      Output buffer, overwritten by ``batch_size`` solution matrices
      ``X``.

.. container:: section

   .. rubric:: Notes

   If ``alpha`` = 0, matrix ``B`` is set to zero and the matrices ``A``
   and ``B`` do not need to be initialized before calling ``trsm_batch``.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`

trsm_batch (USM Version)
---------------------------

.. rubric:: Description

The USM version of ``trsm_batch`` supports the group API and strided API. 

The group API operation is defined as:
::

   idx = 0
   for i = 0 … group_count – 1
       for j = 0 … group_size – 1
           A and B are matrices in a[idx] and b[idx]
           if (left_right == oneapi::math::side::left) then
               compute X such that op(A) * X = alpha[i] * B
           else
               compute X such that X * op(A) = alpha[i] * B
           end if
           B := X
           idx = idx + 1
       end for
   end for     


The strided API operation is defined as:
::

   for i = 0 … batch_size – 1
       A and B are matrices at offset i * stridea and i * strideb in a and b.
       if (left_right == oneapi::math::side::left) then
           compute X such that op(A) * X = alpha * B
       else
           compute X such that X * op(A) = alpha * B
       end if
       B := X
   end for

   where:

op(``A``) is one of op(``A``) = ``A``, or op(A) = ``A``\ :sup:`T`,
or op(``A``) = ``A``\ :sup:`H`,

``alpha`` is a scalar,

``A`` is a triangular matrix,

``B`` and ``X`` are ``m`` x ``n`` general matrices,

``A`` is either ``m`` x ``m`` or ``n`` x ``n``,depending on whether
it multiplies ``X`` on the left or right. On return, the matrix ``B``
is overwritten by the solution matrix ``X``.

For group API, ``a`` and ``b`` arrays contain the pointers for all the input matrices. 
The total number of matrices in ``a`` and ``b`` are given by: 
 
.. math::
      
      total\_batch\_count = \sum_{i=0}^{group\_count-1}group\_size[i]

For strided API, ``a`` and ``b`` arrays contain all the input matrices. The total number of matrices 
in ``a`` and ``b`` are given by the ``batch_size`` parameter.  

**Group API**

.. rubric:: Syntax
      
.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event trsm_batch(sycl::queue &queue,
                              const oneapi::math::side *left_right,
                              const oneapi::math::uplo *upper_lower,
                              const oneapi::math::transpose *trans,
                              const oneapi::math::diag *unit_diag,
                              const std::int64_t *m,
                              const std::int64_t *n,
                              const T *alpha,
                              const T **a,
                              const std::int64_t *lda,
                              T **b,
                              const std::int64_t *ldb,
                              std::int64_t group_count,
                              const std::int64_t *group_size,
                              const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event trsm_batch(sycl::queue &queue,
                              const oneapi::math::side *left_right,
                              const oneapi::math::uplo *upper_lower,
                              const oneapi::math::transpose *trans,
                              const oneapi::math::diag *unit_diag,
                              const std::int64_t *m,
                              const std::int64_t *n,
                              const T *alpha,
                              const T **a,
                              const std::int64_t *lda,
                              T **b,
                              const std::int64_t *ldb,
                              std::int64_t group_count,
                              const std::int64_t *group_size,
                              const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   left_right
      Array of ``group_count`` ``oneapi::math::side`` values. ``left_right[i]`` specifies whether ``A`` multiplies
      ``X`` on the left (``side::left``) or on the right
      (``side::right``) for every ``trsm`` operation in group ``i``. See :ref:`onemath_datatypes` for more details.

   upper_lower
      Array of ``group_count`` ``oneapi::math::uplo`` values. ``upper_lower[i]`` specifies whether ``A`` is upper or lower
      triangular for every matrix in group ``i``. See :ref:`onemath_datatypes` for more details.

   trans
      Array of ``group_count`` ``oneapi::math::transpose`` values. ``trans[i]`` specifies the form of op(``A``) used
      for every ``trsm`` operation in group ``i``. See :ref:`onemath_datatypes` for more details.

   unit_diag
      Array of ``group_count`` ``oneapi::math::diag`` values. ``unit_diag[i]`` specifies whether ``A`` is assumed to
      be unit triangular (all diagonal elements are 1) for every matrix in group ``i``. See :ref:`onemath_datatypes` for more details.

   m
      Array of ``group_count`` integers. ``m[i]`` specifies the
      number of rows of ``B`` for every matrix in group ``i``. All entries must be at least zero.

   n
      Array of ``group_count`` integers. ``n[i]`` specifies the
      number of columns of ``B`` for every matrix in group ``i``. All entries must be at least zero.

   alpha
      Array of ``group_count`` scalar elements. ``alpha[i]`` specifies the scaling factor in group ``i``.

   a
      Array of pointers to input matrices ``A`` with size ``total_batch_count``. See :ref:`matrix-storage` for more details.

   lda
      Array of ``group_count`` integers. ``lda[i]`` specifies the leading dimension of ``A`` for every matrix in group ``i``. 
      All entries must be at least ``m``
      if ``left_right`` is ``side::left``, and at least 
      ``n`` if ``left_right`` is ``side::right``. All entries must be positive.

   b
      Array of pointers to input matrices ``B`` with size ``total_batch_count``. See :ref:`matrix-storage` for more details.

   ldb
      Array of ``group_count`` integers. ``ldb[i]`` specifies the
      leading dimension of ``B`` for every matrix in group ``i``.  All
      entries must be positive and at least ``m`` and positive if
      column major layout is used to store matrices or at least ``n``
      if row major layout is used to store matrices.

   group_count
      Specifies the number of groups. Must be at least 0.

   group_size
      Array of ``group_count`` integers. ``group_size[i]`` specifies the
      number of ``trsm`` operations in group ``i``. All entries must be at least 0.

   dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   b
      Output buffer, overwritten by the ``total_batch_count`` solution
      matrices ``X``.

.. container:: section

   .. rubric:: Notes

   If ``alpha`` = 0, matrix ``B`` is set to zero and the matrices ``A``
   and ``B`` do not need to be initialized before calling ``trsm_batch``.

.. container:: section
   
   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.

**Strided API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event trsm_batch(sycl::queue &queue,
                              oneapi::math::side left_right,
                              oneapi::math::uplo upper_lower,
                              oneapi::math::transpose trans,
                              oneapi::math::diag unit_diag,
                              std::int64_t m,
                              std::int64_t n,
                              value_or_pointer<T> alpha,
                              const T *a,
                              std::int64_t lda,
                              std::int64_t stridea,
                              T *b,
                              std::int64_t ldb,
                              std::int64_t strideb,
                              std::int64_t batch_size,
                              const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event trsm_batch(sycl::queue &queue,
                              oneapi::math::side left_right,
                              oneapi::math::uplo upper_lower,
                              oneapi::math::transpose trans,
                              oneapi::math::diag unit_diag,
                              std::int64_t m,
                              std::int64_t n,
                              value_or_pointer<T> alpha,
                              const T *a,
                              std::int64_t lda,
                              std::int64_t stridea,
                              T *b,
                              std::int64_t ldb,
                              std::int64_t strideb,
                              std::int64_t batch_size,
                              const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   left_right
      Specifies whether the matrices ``A`` multiply ``X`` on the left
      (``side::left``) or on the right (``side::right``). See :ref:`onemath_datatypes` for more details.

   upper_lower
      Specifies whether the matrices ``A`` are upper or lower
      triangular. See :ref:`onemath_datatypes` for more details.

   trans
      Specifies op(``A``), the transposition operation applied to the
      matrices ``A``. See :ref:`onemath_datatypes` for more details.

   unit_diag
      Specifies whether the matrices ``A`` are assumed to be unit
      triangular (all diagonal elements are 1). See :ref:`onemath_datatypes` for more details.

   m
      Number of rows of the ``B`` matrices. Must be at least zero.

   n
      Number of columns of the ``B`` matrices. Must be at least zero.

   alpha
      Scaling factor for the solutions. See :ref:`value_or_pointer` for more details.

   a
      Pointer to input matrices ``A`` with size ``stridea`` * ``batch_size``.

   lda
      Leading dimension of the matrices ``A``. Must be at least ``m`` if
      ``left_right`` = ``side::left``, and at least ``n`` if ``left_right`` =
      ``side::right``. Must be positive.

   stridea
      Stride between different ``A`` matrices.

   b
      Pointer to input matrices ``B`` with size ``strideb`` * ``batch_size``.

   ldb
      Leading dimension of the matrices ``B``. It must be positive and at least
      ``m`` if column major layout is used to store matrices or at
      least ``n`` if row major layout is used to store matrices.

   strideb
      Stride between different ``B`` matrices. 

   batch_size
      Specifies the number of triangular linear systems to solve.

.. container:: section

   .. rubric:: Output Parameters

   b
      Output matrices, overwritten by ``batch_size`` solution matrices
      ``X``.

.. container:: section

   .. rubric:: Notes

   If ``alpha`` = 0, matrix ``B`` is set to zero and the matrices ``A``
   and ``B`` do not need to be initialized before calling ``trsm_batch``.

.. container:: section
   
   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

   **Parent topic:** :ref:`blas-like-extensions`
