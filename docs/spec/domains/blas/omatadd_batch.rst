.. SPDX-FileCopyrightText: 2022 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_omatadd_batch:

omatadd_batch
=============

Computes a group of out-of-place scaled matrix additions using general
dense matrices.

.. _onemath_blas_omatadd_batch_description:

.. rubric:: Description

The ``omatadd_batch`` routines perform a series of out-of-place scaled matrix
additions. They are batched versions of :ref:`onemath_blas_omatadd`,
but the ``omatadd_batch`` routines perform their operations with
groups of matrices. Each group contains matrices with the same parameters.

There is a *strided API*, in which the matrices in a batch are a set
distance away from each other in memory and in which all matrices
share the same parameters (for example matrix size), and a more
flexible *group API* where each group of matrices has the same
parameters but the user may provide multiple groups that have
different parameters. The group API argument structure is better
suited to USM pointers than to ``sycl::buffer`` arguments, so we
only specify it for USM inputs. The strided API works with both USM
and buffer memory.

   .. list-table::
      :header-rows: 1

      * -
        - strided API
        - group API
      * - Buffer memory
        - supported
        - not supported
      * - USM pointers
        - supported
        - supported

``omatadd_batch`` supports the following precisions:

   .. list-table::
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_blas_omatadd_batch_buffer:

omatadd_batch (Buffer Version)
------------------------------

.. rubric:: Description

The buffer version of ``omatcopy_batch`` supports only the strided API.

The operation of ``omatadd_batch`` is defined as:
::

   for i = 0 … batch_size – 1
       A is a matrix at offset i * stridea in a
       B is a matrix at offset i * strideb in b
       C is a matrix at offset i * stridec in c
       C := alpha * op(A) + beta * op(B)
   end for

where:

op(X) is one of op(X) = X, or op(X) = X\ :sup:`T`, or op(X) = X\ :sup:`H`,

``alpha`` and ``beta`` are scalars,

``A`` and ``B`` are input matrices while ``C`` is an output matrix,

``C`` is ``m`` x ``n``,

``A`` is ``m`` x ``n`` if the ``op(A)`` is not transposed or ``n`` by ``m`` if it is,

and ``B`` is ``m`` x ``n`` if the ``op(B)`` is not transposed or ``n`` by ``m`` if it is.

The ``a`` and ``b`` buffers contain all the input matrices while the
``c`` buffer contains all the output matrices. The locations of the
individual matrices within the buffer are given by the ``stride_a``,
``stride_b``, and ``stride_c`` parameters, while the total number of
matrices in each buffer is given by the ``batch_size`` parameter.

In general, the ``a``, ``b``, and ``c`` buffers should not overlap in
memory, with the exception of the following in-place operations:

   - ``a`` and ``c`` may point to the same memory if ``op(A)`` is non-transpose and all the ``A`` matrices have the same parameters as all the respective ``C`` matrices;

   - ``b`` and ``c`` may point to the same memory if ``op(B)`` is non-transpose and all the ``B`` matrices have the same parameters as all the respective ``C`` matrices.


**Strided API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void omatadd_batch(sycl::queue &queue,
                          oneapi::math::transpose transa,
                          oneapi::math::transpose transb,
                          std::int64_t m,
                          std::int64_t n,
                          T alpha,
                          sycl::buffer<T, 1> &a,
                          std::int64_t lda,
                          std::int64_t stride_a,
                          T beta,
                          sycl::buffer<T, 1> &b,
                          std::int64_t ldb,
                          std::int64_t stride_b,
                          sycl::buffer<T, 1> &c,
                          std::int64_t ldc,
                          std::int64_t stride_c,
                          std::int64_t batch_size);
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void omatadd_batch(sycl::queue &queue,
                          oneapi::math::transpose transa,
                          oneapi::math::transpose transb,
                          std::int64_t m,
                          std::int64_t n,
                          T alpha,
                          sycl::buffer<T, 1> &a,
                          std::int64_t lda,
                          std::int64_t stride_a,
                          T beta,
                          sycl::buffer<T, 1> &b,
                          std::int64_t ldb,
                          std::int64_t stride_b,
                          sycl::buffer<T, 1> &c,
                          std::int64_t ldc,
                          std::int64_t stride_c,
                          std::int64_t batch_size);
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   transa
      Specifies op(``A``), the transposition operation applied to the
      matrices ``A``. See :ref:`onemath_datatypes` for more details.

   transb
      Specifies op(``B``), the transposition operation applied to the
      matrices ``B``. See :ref:`onemath_datatypes` for more details.

   m
      Number of rows for the result matrix ``C``. Must be at least zero.

   n
      Number of columns for the result matrix ``C``. Must be at least zero.

   alpha
      Scaling factor for the matrices ``A``.

   a
      Buffer holding the input matrices ``A``. Must have size at least ``stride_a`` * ``batch_size``.

   lda
      The leading dimension of the matrices ``A``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``A`` not transposed
           - ``A`` transposed
         * - Column major
           - ``lda`` must be at least ``m``.
           - ``lda`` must be at least ``n``.
         * - Row major
           - ``lda`` must be at least ``n``.
           - ``lda`` must be at least ``m``.

   stride_a
      Stride between the different ``A`` matrices within the buffer.

      .. list-table::
         :header-rows: 1

         * -
           - ``A`` not transposed
           - ``A`` transposed
         * - Column major
           - ``stride_a`` must be at least ``lda*n``.
           - ``stride_a`` must be at least ``lda*m``.
         * - Row major
           - ``stride_a`` must be at least ``lda*m``.
           - ``stride_a`` must be at least ``lda*n``.

   beta
      Scaling factor for the matrices ``B``.

   b
      Buffer holding the input matrices ``B``. Must have size at least
      ``stride_b`` * ``batch_size``.

   ldb
      The leading dimension of the ``B`` matrices. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``B`` not transposed
           - ``B`` transposed
         * - Column major
           - ``ldb`` must be at least ``m``.
           - ``ldb`` must be at least ``n``.
         * - Row major
           - ``ldb`` must be at least ``n``.
           - ``ldb`` must be at least ``m``.

   stride_b
      Stride between different ``B`` matrices.

      .. list-table::
         :header-rows: 1

         * -
           - ``B`` not transposed
           - ``B`` transposed
         * - Column major
           - ``stride_b`` must be at least ``ldb`` x ``n``.
           - ``stride_b`` must be at least ``ldb`` x ``m``.
         * - Row major
           - ``stride_b`` must be at least ``ldb`` x ``m``.
           - ``stride_b`` must be at least ``ldb`` x ``n``.

   c
      Buffer holding the output matrices ``C``. Must have size at least
      ``stride_c`` * ``batch_size``.

   ldc
      Leading dimension of the ``C`` matrices. If matrices are stored using
      column major layout, ``ldc`` must be at least ``m``. If matrices are
      stored using row major layout, ``ldc`` must be at least ``n``. Must be
      positive.

   stride_c
      Stride between the different ``C`` matrices. If matrices are stored using
      column major layout, ``stride_c`` must be at least ``ldc*n``. If matrices
      are stored using row major layout, ``stride_c`` must be at least
      ``ldc*m``.

   batch_size
      Specifies the number of matrix transposition or copy operations to perform.

.. container:: section

   .. rubric:: Output Parameters

   c
      Output buffer, overwritten by ``batch_size`` matrix addition operations
      of the form ``alpha*op(A) + beta*op(B)``. Must have size at least
      ``stride_c*batch_size``.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated
   condition is detected. An implementation may throw additional
   implementation-specific exception(s) in case of error conditions
   not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

.. _onemath_blas_omatadd_batch_usm:
   
omatadd_batch (USM Version)
---------------------------

.. rubric:: Description

The USM version of ``omatadd_batch`` supports the group API and the strided API.

The operation for the group API is defined as:
::

   idx = 0
   for i = 0 … group_count – 1
       m, n, alpha, beta, lda, ldb, ldc and group_size at position i in their respective arrays
       for j = 0 … group_size – 1
           A, B and C are matrices at position idx in their respective arrays
           C := alpha * op(A) + beta * op(B)
           idx := idx + 1
       end for
   end for

The operation for the strided API is defined as:
::

   for i = 0 … batch_size – 1
       A is a matrix at offset i * stridea in a
       B is a matrix at offset i * strideb in b
       C is a matrix at offset i * stridec in c
       C := alpha * op(A) + beta * op(B)
   end for

where:

op(X) is one of op(X) = X, or op(X) = X\ :sup:`T`, or op(X) = X\ :sup:`H`,

``alpha`` and ``beta`` are scalars,

``A`` and ``B`` are input matrices while ``C`` is an output matrix,

``C`` is ``m`` x ``n``,

``A`` is ``m`` x ``n`` if the ``op(A)`` is not transposed or ``n`` by ``m`` if it is,

and ``B`` is ``m`` x ``n`` if the ``op(B)`` is not transposed or ``n`` by ``m`` if it is.

For the group API, the matrices are given by arrays of pointers. ``A``, ``B``, and ``C``
represent matrices stored at addresses pointed to by ``a_array``, ``b_array``,
and ``c_array`` respectively. The number of entries in ``a_array``, ``b_array``,
and ``c_array`` is given by:

.. math::

      total\_batch\_count = \sum_{i=0}^{group\_count-1}group\_size[i]    

For the strided API, the ``a`` and ``b`` arrays contain all the input matrices
while the ``c`` array contains all the output matrices. The locations of the
individual matrices within the array are given by the ``stride_a``,
``stride_b``, and ``stride_c`` parameters, while the total number of
matrices in each array is given by the ``batch_size`` parameter.

In general, the batches of matrices indicated by ``a``, ``b``, and ``c``
should not overlap in memory, with the exception of the the following
in-place operations:

   - ``a`` and ``c`` may point to the same memory if ``op(A)`` is non-transpose and all the ``A`` matrices have identical parameters as all the respective ``C`` matrices;

   - ``b`` and ``c`` may point to the same memory if ``op(B)`` is non-transpose and all the the ``B`` matrices have identical parameters as all the respective ``C`` matrices.


**Group API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event omatadd_batch(sycl::queue &queue,
                                 const oneapi::math::transpose *transa_array,
                                 const oneapi::math::transpose *transb_array,
                                 const std::int64_t *m_array,
                                 const std::int64_t *n_array,
                                 const T *alpha_array,
                                 const T **a_array,
                                 const std::int64_t *lda_array,
                                 const T *beta_array,
                                 const T **b_array,
                                 const std::int64_t *ldb_array,
                                 const T **c_array,
                                 const std::int64_t *ldc_array,
                                 std::int64_t group_count,
                                 const std::int64_t *groupsize,
                                 const std::vector<sycl::event> &dependencies = {});
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event omatadd_batch(sycl::queue &queue,
                                 const oneapi::math::transpose *transa_array,
                                 const oneapi::math::transpose *transb_array,
                                 const std::int64_t *m_array,
                                 const std::int64_t *n_array,
                                 const T *alpha_array,
                                 const T **a_array,
                                 const std::int64_t *lda_array,
                                 const T *beta_array,
                                 const T **b_array,
                                 const std::int64_t *ldb_array,
                                 const T **c_array,
                                 const std::int64_t *ldc_array,
                                 std::int64_t group_count,
                                 const std::int64_t *groupsize,
                                 const std::vector<sycl::event> &dependencies = {});
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   transa_array
      Array of size ``group_count``. Each element ``i`` in the array specifies
      ``op(A)`` the transposition operation applied to the matrices A.

   transb_array
      Array of size ``group_count``. Each element ``i`` in the array specifies
      ``op(B)`` the transposition operation applied to the matrices B.

   m_array
      Array of size ``group_count`` of number of rows of ``C``. Each
      must be at least 0.

   n_array
      Array of size ``group_count`` of number of columns of ``C``. Each
      must be at least 0.

   alpha_array
      Array of size ``group_count`` containing scaling factors for the matrices
      ``A``.

   a_array
      Array of size ``total_batch_count``, holding pointers to arrays used to
      store ``A`` matrices. The array allocated for each ``A`` matrix of the group
      ``i`` must be of size at least:

      .. list-table::
         :header-rows: 1

         * -
           - ``transa[i]`` = ``transpose::nontrans``
           - ``transa[i]`` = ``transpose::trans`` or ``transa[i]`` = ``transpose::conjtrans``
         * - Column major
           - ``lda_array[i]`` * ``n_array[i]``
           - ``lda_array[i]`` * ``m_array[i]``
         * - Row major
           - ``lda_array[i]`` * ``m_array[i]``
           - ``lda_array[i]`` * ``n_array[i]``

   lda_array
      Array of size ``group_count`` of leading dimension of the A matrices.
      All must be positive and satisfy:

      .. list-table::
         :header-rows: 1

         * -
           - ``transa[i]`` = ``transpose::nontrans``
           - ``transa[i]`` = ``transpose::trans`` or ``transa`` = ``transpose::conjtrans``
         * - Column major
           - ``lda_array[i]`` must be at least ``m_array[i]``.
           - ``lda_array[i]`` must be at least ``n_array[i]``.
         * - Row major
           - ``lda_array[i]`` must be at least ``n_array[i]``.
           - ``lda_array[i]`` must be at least ``m_array[i]``.

   beta_array
      Array of size ``group_count`` containing scaling factors for the matrices
      ``B``.

   b_array
      Array of size ``total_batch_count`` of pointers used to store the B matrices.
      The array allocated for each B matrix of the group ``i`` must be of size at least:
 
      .. list-table::
         :header-rows: 1
     
         * -
           - ``transb[i]`` = ``transpose::nontrans``
           - ``transb[i]`` = ``transpose::trans`` or ``transb[i]`` = ``transpose::conjtrans``
         * - Column major
           - ``ldb_array[i]`` * ``n_array[i]``
           - ``ldb_array[i]`` * ``m_array[i]``
         * - Row major
           - ``ldb_array[i]`` * ``m_array[i]``
           - ``ldb_array[i]`` * ``n_array[i]``

   ldb_array
      Array of size ``group_count``. The leading dimension of ``B`` matrices.
      All must be positive and satisfy:

      .. list-table::
         :header-rows: 1

         * -
           - ``transb[i]`` = ``transpose::nontrans``
           - ``transb[i]`` = ``transpose::trans`` or ``transb[i]`` = ``transpose::conjtrans``
         * - Column major
           - ``ldb_array[i]`` must be at least ``m_array[i]``.
           - ``ldb_array[i]`` must be at least ``n_array[i]``.
         * - Row major
           - ``ldb_array[i]`` must be at least ``n_array[i]``.
           - ``ldb_array[i]`` must be at least ``m_array[i]``.

   c_array
      Array of size ``total_batch_count`` of pointers used to store the ``C`` output
      matrices. The array allocated for each C matrix of the group ``i`` must be of size
      at least:
 
      .. list-table::

         * - Column major
           - ``ldc_array[i]`` * ``n_array[i]``
         * - Row major
           - ``ldc_array[i]`` * ``m_array[i]``

   ldc_array
      Array of size ``group_count``. The leading dimension of the ``C`` matrices. If
      matrices are stored using column major layout, ``ldc_array[i]`` must be at least
      ``m_array[i]``. If matrices are stored using row major layout, ``ldc_array[i]``
      must be at least ``n_array[i]``. All entries must be positive.

   group_count
      Number of groups. Must be at least 0.

   group_size
      Array of size ``group_count``. The element ``group_size[i]`` is the
      number of matrices in the group ``i``. Each element in ``group_size``
      must be at least 0.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   c_array
      Output array of pointers to C matrices, overwritten by
      ``total_batch_count`` matrix addition operations of the form
      ``alpha*op(A) + beta*op(B)``.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.


**Strided API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event omatadd_batch(sycl::queue &queue,
                                 oneapi::math::transpose transa,
                                 oneapi::math::transpose transb,
                                 std::int64_t m,
                                 std::int64_t n,
                                 value_or_pointer<T> alpha,
                                 const T *a,
                                 std::int64_t lda,
                                 std::int64_t stride_a,
                                 value_or_pointer<T> beta,
                                 T *b,
                                 std::int64_t ldb,
                                 std::int64_t stride_b,
                                 T *c,
                                 std::int64_t ldc,
                                 std::int64_t stride_c,
                                 std::int64_t batch_size,
                                 const std::vector<sycl::event> &dependencies = {});
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event omatadd_batch(sycl::queue &queue,
                                 oneapi::math::transpose transa,
                                 oneapi::math::transpose transb,
                                 std::int64_t m,
                                 std::int64_t n,
                                 value_or_pointer<T> alpha,
                                 const T *a,
                                 std::int64_t lda,
                                 std::int64_t stride_a,
                                 value_or_pointer<T> beta,
                                 T *b,
                                 std::int64_t ldb,
                                 std::int64_t stride_b,
                                 T *c,
                                 std::int64_t ldc,
                                 std::int64_t stride_c,
                                 std::int64_t batch_size,
                                 const std::vector<sycl::event> &dependencies = {});
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   transa
      Specifies op(``A``), the transposition operation applied to the
      matrices ``A``. See :ref:`onemath_datatypes` for more details.

   transb
      Specifies op(``B``), the transposition operation applied to the
      matrices ``B``. See :ref:`onemath_datatypes` for more details.

   m
      Number of rows for the result matrix ``C``. Must be at least zero.

   n
      Number of columns for the result matrix ``C``. Must be at least zero.

   alpha
      Scaling factor for the matrices ``A``. See :ref:`value_or_pointer` for more details.

   a
      Array holding the input matrices ``A``. Must have size at least ``stride_a`` * ``batch_size``.

   lda
      The leading dimension of the matrices ``A``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``A`` not transposed
           - ``A`` transposed
         * - Column major
           - ``lda`` must be at least ``m``.
           - ``lda`` must be at least ``n``.
         * - Row major
           - ``lda`` must be at least ``n``.
           - ``lda`` must be at least ``m``.

   stride_a
      Stride between the different ``A`` matrices within the array.

      .. list-table::
         :header-rows: 1

         * -
           - ``A`` not transposed
           - ``A`` transposed
         * - Column major
           - ``stride_a`` must be at least ``lda*n``.
           - ``stride_a`` must be at least ``lda*m``.
         * - Row major
           - ``stride_a`` must be at least ``lda*m``.
           - ``stride_a`` must be at least ``lda*n``.

   beta
      Scaling factor for the matrices ``B``. See :ref:`value_or_pointer` for more details.

   b
      Array holding the input matrices ``B``. Must have size at least
      ``stride_b`` * ``batch_size``.

   ldb
      The leading dimension of the ``B`` matrices. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``B`` not transposed
           - ``B`` transposed
         * - Column major
           - ``ldb`` must be at least ``m``.
           - ``ldb`` must be at least ``n``.
         * - Row major
           - ``ldb`` must be at least ``n``.
           - ``ldb`` must be at least ``m``.

   stride_b
      Stride between different ``B`` matrices.

      .. list-table::
         :header-rows: 1

         * -
           - ``B`` not transposed
           - ``B`` transposed
         * - Column major
           - ``stride_b`` must be at least ``ldb`` x ``n``.
           - ``stride_b`` must be at least ``ldb`` x ``m``.
         * - Row major
           - ``stride_b`` must be at least ``ldb`` x ``m``.
           - ``stride_b`` must be at least ``ldb`` x ``n``.

   c
      Array holding the output matrices ``C``. Must have size at least
      ``stride_c`` * ``batch_size``.

   ldc
      Leading dimension of the ``C`` matrices. If matrices are stored using
      column major layout, ``ldc`` must be at least ``m``. If matrices are
      stored using row major layout, ``ldc`` must be at least ``n``. Must be
      positive.

   stride_c
      Stride between the different ``C`` matrices. If matrices are stored using
      column major layout, ``stride_c`` must be at least ``ldc*n``. If matrices
      are stored using row major layout, ``stride_c`` must be at least
      ``ldc*m``.

   batch_size
      Specifies the number of matrix transposition or copy operations to perform.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   c
      Output array, overwritten by ``batch_size`` matrix addition operations
      of the form ``alpha*op(A) + beta*op(B)``. Must have size at least
      ``stride_c*batch_size``.

.. container:: section
      
   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated
   condition is detected. An implementation may throw additional
   implementation-specific exception(s) in case of error conditions
   not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`


   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

   **Parent topic:** :ref:`blas-like-extensions`

