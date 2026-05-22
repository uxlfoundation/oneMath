.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_gemm_batch:

gemm_batch
==========

Computes a group of ``gemm`` operations.

.. _onemath_blas_gemm_batch_description:

.. rubric:: Description

The ``gemm_batch`` routines are batched versions of :ref:`onemath_blas_gemm`, performing
multiple ``gemm`` operations in a single call. Each ``gemm`` 
operation perform a matrix-matrix product with general matrices.
   
``gemm_batch`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * - | Ta
          | (A matrix) 
        - | Tb
          | (B matrix) 
        - | Tc 
          | (C matrix) 
        - | Ts
          | (alpha/beta) 
      * -  ``std::int8_t`` 
        -  ``std::int8_t`` 
        -  ``std::int32_t`` 
        -  ``float`` 
      * -  ``std::int8_t`` 
        -  ``std::int8_t`` 
        -  ``float`` 
        -  ``float`` 
      * -  ``half`` 
        -  ``half`` 
        -  ``float`` 
        -  ``float`` 
      * -  ``half`` 
        -  ``half`` 
        -  ``half`` 
        -  ``half`` 
      * -  ``bfloat16`` 
        -  ``bfloat16`` 
        -  ``float`` 
        -  ``float`` 
      * -  ``bfloat16`` 
        -  ``bfloat16`` 
        -  ``bfloat16`` 
        -  ``float`` 
      * -  ``float`` 
        -  ``float`` 
        -  ``float`` 
        -  ``float`` 
      * -  ``double`` 
        -  ``double`` 
        -  ``double`` 
        -  ``double`` 
      * -  ``std::complex<float>`` 
        -  ``std::complex<float>`` 
        -  ``std::complex<float>`` 
        -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 
        -  ``std::complex<double>`` 
        -  ``std::complex<double>`` 
        -  ``std::complex<double>`` 

.. _onemath_blas_gemm_batch_buffer:

gemm_batch (Buffer Version)
---------------------------

.. rubric:: Description

The buffer version of ``gemm_batch`` supports only the strided API. 

The strided API operation is defined as:
::

   for i = 0 … batch_size – 1
       A, B and C are matrices at offset i * stridea, i * strideb, i * stridec in a, b and c.
       C := alpha * op(A) * op(B) + beta * C
   end for

where:

op(X) is one of op(X) = X, or op(X) = X\ :sup:`T`, or op(X) = X\ :sup:`H`,

``alpha`` and ``beta`` are scalars,

``A``, ``B``, and ``C`` are matrices,

op(``A``) is ``m`` x ``k``, op(``B``) is 
``k`` x ``n``, and ``C`` is ``m`` x ``n``.

The ``a``, ``b`` and ``c`` buffers contain all the input matrices. The stride 
between matrices is given by the stride parameter. The total number
of matrices in ``a``, ``b`` and ``c`` buffers is given by the ``batch_size`` parameter.

**Strided API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void gemm_batch(sycl::queue &queue,
                       oneapi::math::transpose transa,
                       oneapi::math::transpose transb,
                       std::int64_t m,
                       std::int64_t n,
                       std::int64_t k,
                       Ts alpha,
                       sycl::buffer<Ta,1> &a,
                       std::int64_t lda,
                       std::int64_t stridea,
                       sycl::buffer<Tb,1> &b,
                       std::int64_t ldb,
                       std::int64_t strideb,
                       Ts beta,
                       sycl::buffer<Tc,1> &c,
                       std::int64_t ldc,
                       std::int64_t stridec,
                       std::int64_t batch_size)
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void gemm_batch(sycl::queue &queue,
                       oneapi::math::transpose transa,
                       oneapi::math::transpose transb,
                       std::int64_t m,
                       std::int64_t n,
                       std::int64_t k,
                       Ts alpha,
                       sycl::buffer<Ta,1> &a,
                       std::int64_t lda,
                       std::int64_t stridea,
                       sycl::buffer<Tb,1> &b,
                       std::int64_t ldb,
                       std::int64_t strideb,
                       Ts beta,
                       sycl::buffer<Tc,1> &c,
                       std::int64_t ldc,
                       std::int64_t stridec,
                       std::int64_t batch_size)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   transa
      Specifies op(``A``) the transposition operation applied to the
      matrices ``A``. See :ref:`onemath_datatypes` for more details.

   transb
      Specifies op(``B``) the transposition operation applied to the
      matrices ``B``. See :ref:`onemath_datatypes` for more details.

   m
      Number of rows of op(``A``) and ``C``. Must be at least zero.


   n
      Number of columns of op(``B``) and ``C``. Must be at least zero.


   k
      Number of columns of op(``A``) and rows of op(``B``). Must be at
      least zero.

   alpha
      Scaling factor for the matrix-matrix products.

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
           - ``lda`` must be at least ``m``.
           - ``lda`` must be at least ``k``.
         * - Row major
           - ``lda`` must be at least ``k``.
           - ``lda`` must be at least ``m``.

   stridea
      Stride between different ``A`` matrices.

   b
      Buffer holding the input matrices ``B`` with size ``strideb`` * ``batch_size``.

   ldb
      The leading dimension of the matrices``B``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``B`` not transposed
           - ``B`` transposed
         * - Column major
           - ``ldb`` must be at least ``k``.
           - ``ldb`` must be at least ``n``.
         * - Row major
           - ``ldb`` must be at least ``n``.
           - ``ldb`` must be at least ``k``.

   strideb
      Stride between different ``B`` matrices.

   beta
      Scaling factor for the matrices ``C``.

   c
      Buffer holding input/output matrices ``C`` with size ``stridec`` * ``batch_size``.

   ldc
      The leading dimension of the matrices ``C``. It must be positive and at least
      ``m`` if column major layout is used to store matrices or at
      least ``n`` if row major layout is used to store matrices.

   stridec
      Stride between different ``C`` matrices. Must be at least
      ``ldc`` * ``n``.

   batch_size
      Specifies the number of matrix multiply operations to perform.

.. container:: section

   .. rubric:: Output Parameters

   c
      Output buffer, overwritten by ``batch_size`` matrix multiply
      operations of the form ``alpha`` * op(``A``)*op(``B``) + ``beta`` * ``C``.

.. container:: section

   .. rubric:: Notes

   If ``beta`` = 0, matrix ``C`` does not need to be initialized before
   calling ``gemm_batch``.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

.. _onemath_blas_gemm_batch_usm:

gemm_batch (USM Version)
---------------------------

.. rubric:: Description

The USM version of ``gemm_batch`` supports the group API and the strided API.
The group API supports pointer and span inputs.

The group API operation is defined as:
::

   idx = 0
   for i = 0 … group_count – 1
       for j = 0 … group_size – 1
           A, B, and C are matrices in a[idx], b[idx] and c[idx]
           C := alpha[i] * op(A) * op(B) + beta[i] * C
           idx = idx + 1
       end for
   end for

The advantage of using span instead of pointer is that the sizes of
the array can vary and the size of the span can be queried at
runtime. For each GEMM parameter, except the output matrices, the span
can be of size 1, the number of groups or the total batch size. For
the output matrices, to ensure all computation are independent, the size
of the span must be the total batch size.

Depending on the size of the spans, each parameter for the GEMM computation is used as follows:

  - If the span has size 1, the parameter is reused for all GEMM
    computation.

  - If the span has size group_count, the parameter is reused for all
    GEMM within a group, but each group will have a different value
    for this parameter.  This is like the gemm_batch group API with pointers.

  - If the span has size equal to the total batch size, each GEMM
    computation will use a different value for this parameter.

The strided API operation is defined as
::

   for i = 0 … batch_size – 1
       A, B and C are matrices at offset i * stridea, i * strideb, i * stridec in a, b and c.
       C := alpha * op(A) * op(B) + beta * C
   end for

where:

op(X) is one of op(X) = X, or op(X) = X\ :sup:`T`, or op(X) = X\ :sup:`H`,

``alpha`` and ``beta`` are scalars,

``A``, ``B``, and ``C`` are matrices,

op(``A``) is ``m`` x ``k``, op(``B``) is ``k`` x ``n``, and ``C`` is ``m`` x ``n``.

 
For group API, ``a``, ``b`` and ``c`` arrays contain the pointers for all the input matrices. 
The total number of matrices in ``a``, ``b`` and ``c`` are given by: 

.. math::

      total\_batch\_count = \sum_{i=0}^{group\_count-1}group\_size[i]    
 
For strided API, ``a``, ``b``, ``c`` arrays contain all the input matrices. The total number of matrices 
in ``a``, ``b`` and ``c`` are given by the ``batch_size`` parameter.  
   
**Group API**

.. rubric:: Syntax
   
.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event gemm_batch(sycl::queue &queue,
                              const oneapi::math::transpose *transa,
                              const oneapi::math::transpose *transb,
                              const std::int64_t *m,
                              const std::int64_t *n,
                              const std::int64_t *k,
                              const Ts *alpha,
                              const Ta **a,
                              const std::int64_t *lda,
                              const Tb **b,
                              const std::int64_t *ldb,
                              const Ts *beta,
                              Tc **c,
                              const std::int64_t *ldc,
                              std::int64_t group_count,
                              const std::int64_t *group_size,
                              const std::vector<sycl::event> &dependencies = {})

       sycl::event gemm_batch(sycl::queue &queue,
                              const sycl::span<oneapi::math::transpose> &transa,
                              const sycl::span<oneapi::math::transpose> &transb,
                              const sycl::span<std::int64_t> &m,
                              const sycl::span<std::int64_t> &n,
                              const sycl::span<std::int64_t> &k,
                              const sycl::span<Ts> &alpha,
                              const sycl::span<const Ta*> &a,
                              const sycl::span<std::int64_t> &lda,
                              const sycl::span<const Tb*> &b,
                              const sycl::span<std::int64_t> &ldb,
                              const sycl::span<Ts> &beta,
                              sycl::span<Tc*> &c,
                              const sycl::span<std::int64_t> &ldc,
                              size_t group_count,
                              const sycl::span<size_t> &group_sizes,
                              const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event gemm_batch(sycl::queue &queue,
                              const oneapi::math::transpose *transa,
                              const oneapi::math::transpose *transb,
                              const std::int64_t *m,
                              const std::int64_t *n,
                              const std::int64_t *k,
                              const Ts *alpha,
                              const Ta **a,
                              const std::int64_t *lda,
                              const Tb **b,
                              const std::int64_t *ldb,
                              const Ts *beta,
                              Tc **c,
                              const std::int64_t *ldc,
                              std::int64_t group_count,
                              const std::int64_t *group_size,
                              const std::vector<sycl::event> &dependencies = {})

       sycl::event gemm_batch(sycl::queue &queue,
                              const sycl::span<oneapi::math::transpose> &transa,
                              const sycl::span<oneapi::math::transpose> &transb,
                              const sycl::span<std::int64_t> &m,
                              const sycl::span<std::int64_t> &n,
                              const sycl::span<std::int64_t> &k,
                              const sycl::span<Ts> &alpha,
                              const sycl::span<const Ta*> &a,
                              const sycl::span<std::int64_t> &lda,
                              const sycl::span<const Tb*> &b,
                              const sycl::span<std::int64_t> &ldb,
                              const sycl::span<Ts> &beta,
                              sycl::span<Tc*> &c,
                              const sycl::span<std::int64_t> &ldc,
                              size_t group_count,
                              const sycl::span<size_t> &group_sizes,
                              const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   transa
      Array or span of ``group_count`` ``oneapi::math::transpose`` values. ``transa[i]`` specifies the form of op(``A``) used in
      the matrix multiplication in group ``i``. See :ref:`onemath_datatypes` for more details.

   transb
      Array or span of ``group_count`` ``oneapi::math::transpose`` values. ``transb[i]`` specifies the form of op(``B``) used in
      the matrix multiplication in group ``i``. See :ref:`onemath_datatypes` for more details.

   m
      Array or span of ``group_count`` integers. ``m[i]`` specifies the
      number of rows of op(``A``) and ``C`` for every matrix in group ``i``. All entries must be at least zero.

   n
      Array or span of ``group_count`` integers. ``n[i]`` specifies the
      number of columns of op(``B``) and ``C`` for every matrix in group ``i``. All entries must be at least zero.

   k
      Array or span of ``group_count`` integers. ``k[i]`` specifies the
      number of columns of op(``A``) and rows of op(``B``) for every matrix in group ``i``. All entries must be at
      least zero.

   alpha
      Array or span of ``group_count`` scalar elements. ``alpha[i]`` specifies the scaling factor for every matrix-matrix
      product in group ``i``.

   a
      Array of pointers or span of input matrices ``A`` with size ``total_batch_count``. 
      
      See :ref:`matrix-storage` for more details.

   lda
      Array or span of ``group_count`` integers. ``lda[i]`` specifies the
      leading dimension of ``A`` for every matrix in group ``i``. All
      entries must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``A`` not transposed
           - ``A`` transposed
         * - Column major
           - ``lda[i]`` must be at least ``m[i]``.
           - ``lda[i]`` must be at least ``k[i]``.
         * - Row major
           - ``lda[i]`` must be at least ``k[i]``.
           - ``lda[i]`` must be at least ``m[i]``.
             
   b
      Array of pointers or span of input matrices ``B`` with size ``total_batch_count``. 
      
      See :ref:`matrix-storage` for more details.

   ldb
      Array or span of ``group_count`` integers. ``ldb[i]`` specifies the
      leading dimension of ``B`` for every matrix in group ``i``. All
      entries must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``B`` not transposed
           - ``B`` transposed
         * - Column major
           - ``ldb[i]`` must be at least ``k[i]``.
           - ``ldb[i]`` must be at least ``n[i]``.
         * - Row major
           - ``ldb[i]`` must be at least ``n[i]``.
           - ``ldb[i]`` must be at least ``k[i]``.
             
   beta
      Array or span of ``group_count`` scalar elements. ``beta[i]`` specifies the scaling factor for matrix ``C`` 
      for every matrix in group ``i``.

   c
      Array of pointers or span of input/output matrices ``C`` with size ``total_batch_count``. 
      
      See :ref:`matrix-storage` for more details.

   ldc
      Array or span of ``group_count`` integers. ``ldc[i]`` specifies the
      leading dimension of ``C`` for every matrix in group ``i``.  All
      entries must be positive and ``ldc[i]`` must be at least
      ``m[i]`` if column major layout is used to store matrices or at
      least ``n[i]`` if row major layout is used to store matrices.

   group_count
      Specifies the number of groups. Must be at least 0.

   group_size
      Array or span of ``group_count`` integers. ``group_size[i]`` specifies the
      number of matrix multiply products in group ``i``. All entries must be at least 0.

   dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   c
      Overwritten by the ``m[i]``-by-``n[i]`` matrix calculated by 
      (``alpha[i]`` * op(``A``)*op(``B``) + ``beta[i]`` * ``C``) for group ``i``.

.. container:: section

   .. rubric:: Notes

   If ``beta`` = 0, matrix ``C`` does not need to be initialized
   before calling ``gemm_batch``.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.

.. container:: section

   .. rubric:: Output Parameters

   c
      Overwritten by the ``m[i]``-by-``n[i]`` matrix calculated by
      (``alpha[i]`` * op(``A``)*op(``B``) + ``beta[i]`` * ``C``) for group ``i``.

.. container:: section

   .. rubric:: Notes

   If ``beta`` = 0, matrix ``C`` does not need to be initialized
   before calling ``gemm_batch``.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.

**Strided API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event gemm_batch(sycl::queue &queue,
                              oneapi::math::transpose transa,
                              oneapi::math::transpose transb,
                              std::int64_t m,
                              std::int64_t n,
                              std::int64_t k,
                              value_or_pointer<Ts> alpha,
                              const Ta *a,
                              std::int64_t lda,
                              std::int64_t stridea,
                              const Tb *b,
                              std::int64_t ldb,
                              std::int64_t strideb,
                              value_or_pointer<Ts> beta,
                              Tc *c,
                              std::int64_t ldc,
                              std::int64_t stridec,
                              std::int64_t batch_size,
                              const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event gemm_batch(sycl::queue &queue,
                              oneapi::math::transpose transa,
                              oneapi::math::transpose transb,
                              std::int64_t m,
                              std::int64_t n,
                              std::int64_t k,
                              value_or_pointer<Ts> alpha,
                              const Ta *a,
                              std::int64_t lda,
                              std::int64_t stridea,
                              const Tb *b,
                              std::int64_t ldb,
                              std::int64_t strideb,
                              value_or_pointer<Ts> beta,
                              Tc *c,
                              std::int64_t ldc,
                              std::int64_t stridec,
                              std::int64_t batch_size,
                              const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   transa
      Specifies op(``A``) the transposition operation applied to the
      matrices ``A``. See :ref:`onemath_datatypes` for more details.

   transb
      Specifies op(``B``) the transposition operation applied to the
      matrices ``B``. See :ref:`onemath_datatypes` for more details.

   m
      Number of rows of op(``A``) and ``C``. Must be at least zero.

   n
      Number of columns of op(``B``) and ``C``. Must be at least zero.

   k
      Number of columns of op(``A``) and rows of op(``B``). Must be at
      least zero.

   alpha
      Scaling factor for the matrix-matrix products. See :ref:`value_or_pointer` for more details.

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
           - ``lda`` must be at least ``m``.
           - ``lda`` must be at least ``k``.
         * - Row major
           - ``lda`` must be at least ``k``.
           - ``lda`` must be at least ``m``.

   stridea
      Stride between different ``A`` matrices.

   b
      Pointer to input matrices ``B`` with size ``strideb`` * ``batch_size``.

   ldb
      The leading dimension of the matrices``B``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``B`` not transposed
           - ``B`` transposed
         * - Column major
           - ``ldb`` must be at least ``k``.
           - ``ldb`` must be at least ``n``.
         * - Row major
           - ``ldb`` must be at least ``n``.
           - ``ldb`` must be at least ``k``.

   strideb
      Stride between different ``B`` matrices.

   beta
      Scaling factor for the matrices ``C``. See :ref:`value_or_pointer` for more details.

   c
      Pointer to input/output matrices ``C`` with size ``stridec`` * ``batch_size``.

   ldc
      The leading dimension of the matrices ``C``. It must be positive and at least
      ``m`` if column major layout is used to store matrices or at
      least ``n`` if row major layout is used to store matrices.

   stridec
      Stride between different ``C`` matrices.

   batch_size
      Specifies the number of matrix multiply operations to perform.

   dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   c
      Output matrices, overwritten by ``batch_size`` matrix multiply
      operations of the form ``alpha`` * op(``A``)*op(``B``) + ``beta`` * ``C``.

.. container:: section

   .. rubric:: Notes

   If ``beta`` = 0, matrix ``C`` does not need to be initialized before
   calling ``gemm_batch``.

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
