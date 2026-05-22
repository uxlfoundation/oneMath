.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_syrk_batch:

syrk_batch
==========

Computes a group of ``syrk`` operations.

.. _onemath_blas_syrk_batch_description:

.. rubric:: Description

The ``syrk_batch`` routines are batched versions of :ref:`onemath_blas_syrk`, performing
multiple ``syrk`` operations in a single call. Each ``syrk`` 
operation perform a rank-k update with general matrices.
   
``syrk_batch`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_blas_syrk_batch_buffer:

syrk_batch (Buffer Version)
---------------------------

.. rubric:: Description

The buffer version of ``syrk_batch`` supports only the strided API. 

The strided API operation is defined as:
::

   for i = 0 … batch_size – 1
       A and C are matrices at offset i * stridea, i * stridec in a and c.
       C := alpha * op(A) * op(A)^T + beta * C
   end for

where:

op(X) is one of op(X) = X, or op(X) = X\ :sup:`T`, or op(X) = X\ :sup:`H`,

``alpha`` and ``beta`` are scalars,

``A`` and ``C`` are matrices,

op(``A``) is ``n`` x ``k`` and ``C`` is ``n`` x ``n``.

The ``a`` and ``c`` buffers contain all the input matrices. The stride 
between matrices is given by the stride parameter. The total number
of matrices in ``a`` and ``c`` buffers is given by the ``batch_size`` parameter.

**Strided API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void syrk_batch(sycl::queue &queue,
                       oneapi::math::uplo upper_lower,
                       oneapi::math::transpose trans,
                       std::int64_t n,
                       std::int64_t k,
                       T alpha,
                       sycl::buffer<T,1> &a,
                       std::int64_t lda,
                       std::int64_t stridea,
                       T beta,
                       sycl::buffer<T,1> &c,
                       std::int64_t ldc,
                       std::int64_t stridec,
                       std::int64_t batch_size)
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void syrk_batch(sycl::queue &queue,
                       oneapi::math::uplo upper_lower,
                       oneapi::math::transpose trans,
                       std::int64_t n,
                       std::int64_t k,
                       T alpha,
                       sycl::buffer<T,1> &a,
                       std::int64_t lda,
                       std::int64_t stridea,
                       T beta,
                       sycl::buffer<T,1> &c,
                       std::int64_t ldc,
                       std::int64_t stridec,
                       std::int64_t batch_size)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether data in ``C`` is stored in its upper or lower triangle.
      For more details, see :ref:`onemath_datatypes`.

   trans
      Specifies op(``A``) the transposition operation applied to the
      matrix ``A``. Conjugation is never performed, even if trans =
      transpose::conjtrans. See :ref:`onemath_datatypes` for more
      details.

   n
      Number of rows and columns of ``C``.
      Must be at least zero.

   k
      Number of columns of op(``A``).
      Must be at least zero.

   alpha
      Scaling factor for the rank-k update.

   a
      Buffer holding the input matrices ``A`` with size ``stridea`` * ``batch_size``.

   lda
      The leading dimension of the matrices ``A``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``A`` not transposed
           - ``A`` transposed
         * - Column major
           - ``lda`` must be at least ``n``.
           - ``lda`` must be at least ``k``.
         * - Row major
           - ``lda`` must be at least ``k``.
           - ``lda`` must be at least ``n``.

   stridea
      Stride between different ``A`` matrices.

   beta
      Scaling factor for the matrices ``C``.

   c
      Buffer holding input/output matrices ``C`` with size ``stridec`` * ``batch_size``.

   ldc
      The leading dimension of the matrices ``C``. It must be positive
      and at least ``n``.

   stridec
      Stride between different ``C`` matrices. Must be at least
      ``ldc`` * ``n``.

   batch_size
      Specifies the number of rank-k update operations to perform.

.. container:: section

   .. rubric:: Output Parameters

   c
      Output buffer, overwritten by ``batch_size`` rank-k update
      operations of the form ``alpha`` * op(``A``)*op(``A``)^T + ``beta`` * ``C``.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

.. _onemath_blas_syrk_batch_usm:

syrk_batch (USM Version)
---------------------------

.. rubric:: Description

The USM version of ``syrk_batch`` supports the group API and strided API. 

The group API operation is defined as:
::

   idx = 0
   for i = 0 … group_count – 1
       for j = 0 … group_size – 1
           A, B, and C are matrices in a[idx] and c[idx]
           C := alpha[i] * op(A) * op(A)^T + beta[i] * C
           idx = idx + 1
       end for
   end for

The strided API operation is defined as
::

   for i = 0 … batch_size – 1
       A, B and C are matrices at offset i * stridea, i * stridec in a and c.
       C := alpha * op(A) * op(A)^T + beta * C
   end for

where:

op(X) is one of op(X) = X, or op(X) = X\ :sup:`T`, or op(X) = X\ :sup:`H`,

``alpha`` and ``beta`` are scalars,

``A`` and ``C`` are matrices,

op(``A``) is ``n`` x ``k`` and ``C`` is ``n`` x ``n``.

 
For group API, ``a`` and ``c`` arrays contain the pointers for all the input matrices. 
The total number of matrices in ``a`` and ``c`` are given by: 

.. math::

      total\_batch\_count = \sum_{i=0}^{group\_count-1}group\_size[i]    
 
For strided API, ``a`` and ``c`` arrays contain all the input matrices. The total number of matrices 
in ``a`` and ``c`` are given by the ``batch_size`` parameter.  
   
**Group API**

.. rubric:: Syntax
   
.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event syrk_batch(sycl::queue &queue,
                              const uplo *upper_lower,
                              const transpose *trans,
                              const std::int64_t *n,
                              const std::int64_t *k,
                              const T *alpha,
                              const T **a,
                              const std::int64_t *lda,
                              const T *beta,
                              T **c,
                              const std::int64_t *ldc,
                              std::int64_t group_count,
                              const std::int64_t *group_size,
                              const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event syrk_batch(sycl::queue &queue,
                              const uplo *upper_lower,
                              const transpose *trans,
                              const std::int64_t *n,
                              const std::int64_t *k,
                              const T *alpha,
                              const T **a,
                              const std::int64_t *lda,
                              const T *beta,
                              T **c,
                              const std::int64_t *ldc,
                              std::int64_t group_count,
                              const std::int64_t *group_size,
                              const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Array of ``group_count`` ``oneapi::math::upper_lower``
      values. ``upper_lower[i]`` specifies whether data in C for every
      matrix in group ``i`` is in upper or lower triangle.

   trans
      Array of ``group_count`` ``oneapi::math::transpose`` values. ``trans[i]`` specifies the form of op(``A``) used in
      the rank-k update in group ``i``. See :ref:`onemath_datatypes` for more details.

   n
      Array of ``group_count`` integers. ``n[i]`` specifies the
      number of rows and columns of ``C`` for every matrix in group ``i``. All entries must be at least zero.

   k
      Array of ``group_count`` integers. ``k[i]`` specifies the
      number of columns of op(``A``) for every matrix in group ``i``. All entries must be at
      least zero.

   alpha
      Array of ``group_count`` scalar elements. ``alpha[i]`` specifies the scaling factor for every rank-k update in group ``i``.

   a
      Array of pointers to input matrices ``A`` with size ``total_batch_count``. 
      
      See :ref:`matrix-storage` for more details.

   lda
      Array of ``group_count`` integers. ``lda[i]`` specifies the
      leading dimension of ``A`` for every matrix in group ``i``. All
      entries must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``A`` not transposed
           - ``A`` transposed
         * - Column major
           - ``lda[i]`` must be at least ``n[i]``.
           - ``lda[i]`` must be at least ``k[i]``.
         * - Row major
           - ``lda[i]`` must be at least ``k[i]``.
           - ``lda[i]`` must be at least ``n[i]``.
             
   beta
      Array of ``group_count`` scalar elements. ``beta[i]`` specifies the scaling factor for matrix ``C`` 
      for every matrix in group ``i``.

   c
      Array of pointers to input/output matrices ``C`` with size ``total_batch_count``. 
      
      See :ref:`matrix-storage` for more details.

   ldc
      Array of ``group_count`` integers. ``ldc[i]`` specifies the
      leading dimension of ``C`` for every matrix in group ``i``.  All
      entries must be positive and ``ldc[i]`` must be at least ``n[i]``.

   group_count
      Specifies the number of groups. Must be at least 0.

   group_size
      Array of ``group_count`` integers. ``group_size[i]`` specifies the
      number of rank-k update products in group ``i``. All entries must be at least 0.

   dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   c
      Overwritten by the ``n[i]``-by-``n[i]`` matrix calculated by 
      (``alpha[i]`` * op(``A``)*op(``A``)^T + ``beta[i]`` * ``C``) for group ``i``.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.

**Strided API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event syrk_batch(sycl::queue &queue,
                              uplo upper_lower,
                              transpose trans,
                              std::int64_t n,
                              std::int64_t k,
                              value_or_pointer<T> alpha,
                              const T *a,
                              std::int64_t lda,
                              std::int64_t stride_a,
                              value_or_pointer<T> beta,
                              T *c,
                              std::int64_t ldc,
                              std::int64_t stride_c,
                              std::int64_t batch_size,
                              const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event syrk_batch(sycl::queue &queue,
                              uplo upper_lower,
                              transpose trans,
                              std::int64_t n,
                              std::int64_t k,
                              value_or_pointer<T> alpha,
                              const T *a,
                              std::int64_t lda,
                              std::int64_t stride_a,
                              value_or_pointer<T> beta,
                              T *c,
                              std::int64_t ldc,
                              std::int64_t stride_c,
                              std::int64_t batch_size,
                              const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether data in ``C`` is stored in its upper or lower triangle.
      For more details, see :ref:`onemath_datatypes`.

   trans
      Specifies op(``A``) the transposition operation applied to the
      matrices ``A``. Conjugation is never performed, even if trans =
      transpose::conjtrans. See :ref:`onemath_datatypes` for more
      details.

   n
      Number of rows and columns of ``C``.
      Must be at least zero.

   k
      Number of columns of op(``A``).
      Must be at least zero.

   alpha
      Scaling factor for the rank-k updates. See :ref:`value_or_pointer` for more details.

   a
      Pointer to input matrices ``A`` with size ``stridea`` * ``batch_size``.

   lda
      The leading dimension of the matrices ``A``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``A`` not transposed
           - ``A`` transposed
         * - Column major
           - ``lda`` must be at least ``n``.
           - ``lda`` must be at least ``k``.
         * - Row major
           - ``lda`` must be at least ``k``.
           - ``lda`` must be at least ``n``.

   stridea
      Stride between different ``A`` matrices.

   beta
      Scaling factor for the matrices ``C``. See :ref:`value_or_pointer` for more details.

   c
      Pointer to input/output matrices ``C`` with size ``stridec`` * ``batch_size``.

   ldc
      The leading dimension of the matrices ``C``. It must be positive
      and at least ``n``.

   stridec
      Stride between different ``C`` matrices.

   batch_size
      Specifies the number of rank-k update operations to perform.

   dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   c
      Output matrices, overwritten by ``batch_size`` rank-k update
      operations of the form ``alpha`` * op(``A``)*op(``A``)^T + ``beta`` * ``C``.

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
