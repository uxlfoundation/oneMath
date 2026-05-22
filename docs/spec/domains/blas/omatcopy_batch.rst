.. SPDX-FileCopyrightText: 2022 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_omatcopy_batch:

omatcopy_batch
==============

Computes a group of out-of-place scaled matrix transpose or copy operations
using general dense matrices.

.. _onemath_blas_omatcopy_batch_description:

.. rubric:: Description

The ``omatcopy_batch`` routines perform a series of out-of-place scaled matrix
copies or transpositions. They are batched versions of :ref:`onemath_blas_omatcopy`,
but the ``omatcopy_batch`` routines perform their operations with
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

``omatcopy_batch`` supports the following precisions:

   .. list-table::
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_blas_omatcopy_batch_buffer:

omatcopy_batch (Buffer Version)
-------------------------------

.. rubric:: Description

The buffer version of ``omatadd_batch`` supports only the strided API.

The operation for the strided API is defined as:
::

   for i = 0 … batch_size – 1
       A and B are matrices at offset i * stridea in a and i * strideb in b
       B := alpha * op(A)
   end for

where:

op(X) is one of op(X) = X, or op(X) = X\ :sup:`T`, or op(X) = X\ :sup:`H`,

``alpha`` is a scalar,

``A`` and ``B`` are input and output matrices,

``A`` is ``m`` x ``n``,

and ``B`` is ``m`` x ``n`` if the matrix is not transposed or ``n`` by ``m`` if it is.

The ``a`` buffer contains all the input matrices while the ``b``
buffer contains all the output matrices. The locations of the
individual matrices within the buffer are given by the ``stride_a``
and ``stride_b`` parameters, while the total number of matrices in
each buffer is given by the ``batch_size`` parameter.

**Strided API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void omatcopy_batch(sycl::queue &queue,
                           oneapi::math::transpose trans,
                           std::int64_t m,
                           std::int64_t n,
                           T alpha,
                           sycl::buffer<T, 1> &a,
                           std::int64_t lda,
                           std::int64_t stride_a,
                           sycl::buffer<T, 1> &b,
                           std::int64_t ldb,
                           std::int64_t stride_b,
                           std::int64_t batch_size);
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void omatcopy_batch(sycl::queue &queue,
                           oneapi::math::transpose trans,
                           std::int64_t m,
                           std::int64_t n,
                           T alpha,
                           sycl::buffer<T, 1> &a,
                           std::int64_t lda,
                           std::int64_t stride_a,
                           sycl::buffer<T, 1> &b,
                           std::int64_t ldb,
                           std::int64_t stride_b,
                           std::int64_t batch_size);
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   trans
      Specifies op(``A``), the transposition operation applied to the
      matrices ``A``. See :ref:`onemath_datatypes` for more details.

   m
      Number of rows for each matrix ``A``. Must be at least zero.

   n
      Number of columns for each matrix ``A``. Must be at least zero.

   alpha
      Scaling factor for the matrix transposition or copy operations.

   a
      Buffer holding the input matrices ``A`` with size ``stride_a`` * ``batch_size``.

   lda
      The leading dimension of the matrices ``A``. It must be
      positive, and must be at least ``m`` if column major layout is
      used, and at least ``n`` if row-major layout is used.

   stride_a
      Stride between the different ``A`` matrices. If matrices are stored using
      column major layout, ``stride_a`` must be at least ``lda*n``. If matrices
      are stored using row major layout, ``stride_a`` must be at least
      ``lda*m``.

   b
      Buffer holding the output matrices ``B`` with size ``stride_b`` * ``batch_size``.

   ldb
      The leading dimension of the matrices ``B``. It must be positive.

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

   batch_size
      Specifies the number of matrix transposition or copy operations to perform.

.. container:: section

   .. rubric:: Output Parameters

   b
      Output buffer, overwritten by ``batch_size`` matrix copy or transposition
      operations of the form ``alpha`` * op(``A``).

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
      

.. _onemath_blas_omatcopy_batch_usm:
   
omatcopy_batch (USM Version)
----------------------------

.. rubric:: Description

The USM version of ``omatcopy_batch`` supports the group API and the strided API.
            
The operation for the group API is defined as:
::

   idx = 0
   for i = 0 … group_count – 1
       m, n, alpha, lda, ldb and group_size at position i in their respective arrays
       for j = 0 … group_size – 1
           A and B are matrices at position idx in their respective arrays
           B := alpha * op(A)
           idx := idx + 1
       end for
   end for

The operation for the strided API is defined as:
::

   for i = 0 … batch_size – 1
       A and B are matrices at offset i * stridea in a and i * strideb in b
       B := alpha * op(A)
   end for

where:

op(X) is one of op(X) = X, or op(X) = X\ :sup:`T`, or op(X) = X\ :sup:`H`,

``alpha`` is a scalar,

``A`` and ``B`` are input and output matrices,

``A`` is ``m`` x ``n``,

and ``B`` is ``m`` x ``n`` if the matrix is not transposed or ``n`` by ``m`` if it is.

For the group API, the matrices are given by arrays of pointers. A and B
represent matrices stored at addresses pointed to by ``a_array`` and ``b_array``
respectively. The number of entries in ``a_array`` and ``b_array`` is given by:

.. math::

      total\_batch\_count = \sum_{i=0}^{group\_count-1}group\_size[i]    

For the strided API, the single input array contains all the input
matrices, and the single output array contains all the output
matrices. The locations of the individual matrices within the array
are given by stride lengths, while the number of matrices is given by
the ``batch_size`` parameter.


**Group API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event omatcopy_batch(sycl::queue &queue,
                                  const oneapi::math::transpose *trans_array,
                                  const std::int64_t *m_array,
                                  const std::int64_t *n_array,
                                  const T *alpha_array,
                                  const T **a_array,
                                  const std::int64_t *lda_array,
                                  T **b_array,
                                  const std::int64_t *ldb_array,
                                  std::int64_t group_count,
                                  const std::int64_t *groupsize,
                                  const std::vector<sycl::event> &dependencies = {});
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event omatcopy_batch(sycl::queue &queue,
                                  const oneapi::math::transpose *trans_array,
                                  const std::int64_t *m_array,
                                  const std::int64_t *n_array,
                                  const T *alpha_array,
                                  const T **a_array,
                                  const std::int64_t *lda_array,
                                  T **b_array,
                                  const std::int64_t *ldb_array,
                                  std::int64_t group_count,
                                  const std::int64_t *groupsize,
                                  const std::vector<sycl::event> &dependencies = {});
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   trans_array
      Array of size ``group_count``. Each element ``i`` in the array specifies
      ``op(A)`` the transposition operation applied to the matrices A.

   m_array
      Array of size ``group_count`` of number of rows of A. Each
      must be at least 0.

   n_array
      Array of size ``group_count`` of number of columns of A. Each
      must be at least 0.

   alpha_array
      Array of size ``group_count`` containing scaling factors for the matrix
      transpositions or copies.

   a_array
      Array of size ``total_batch_count``, holding pointers to arrays used to
      store A matrices.

   lda_array
      Array of size ``group_count`` of leading dimension of the A matrices.
      If matrices are stored using column major layout, ``lda_array[i]`` must be
      at least ``m_array[i]``. If matrices are stored using row major layout,
      ``lda_array[i]`` must be at least ``n_array[i]``. Each must be positive.

   b_array
      Array of size ``total_batch_count`` of pointers used to store B matrices.
      The array allocated for each B matrix of the group ``i`` must be of size
      at least:
 
      .. list-table::
         :header-rows: 1

         * -
           - ``B`` not transposed
           - ``B`` transposed
         * - Column major
           - ``ldb_array[i]`` x ``n_array[i]``
           - ``ldb_array[i]`` x ``m_array[i]``
         * - Row major
           - ``ldb_array[i]`` x ``m_array[i]``
           - ``ldb_array[i]`` x ``n_array[i]``

   ldb_array
      Array of size ``group_count``. The leading dimension of the output
      matrix B. Each entry ``ldb_array[i]`` must be positive and at least:

      .. list-table::
         :header-rows: 1

         * -
           - ``B`` not transposed
           - ``B`` transposed
         * - Column major
           - ``ldb[i]`` must be at least ``m_array[i]``.
           - ``ldb[i]`` must be at least ``n_array[i]``.
         * - Row major
           - ``ldb[i]`` must be at least ``n_array[i]``.
           - ``ldb[i]`` must be at least ``m_array[i]``.

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

   b_array
      Output array of pointers to B matrices, overwritten by
      ``total_batch_count`` matrix transpose or copy operations of the form
      ``alpha*op(A)``.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.


**Strided API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       event omatcopy_batch(queue &queue,
           transpose trans,
           std::int64_t m,
           std::int64_t n,
           value_or_pointer<T> alpha,
           const T *a,
           std::int64_t lda,
           std::int64_t stride_a,
           T *b,
           std::int64_t ldb,
           std::int64_t stride_b,
           std::int64_t batch_size,
           const std::vector<sycl::event> &dependencies = {});
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       event omatcopy_batch(queue &queue,
           transpose trans,
           std::int64_t m,
           std::int64_t n,
           value_or_pointer<T> alpha,
           const T *a,
           std::int64_t lda,
           std::int64_t stride_a,
           T *b,
           std::int64_t ldb,
           std::int64_t stride_b,
           std::int64_t batch_size,
           const vector_class<event> &dependencies = {});
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine will be executed.

   trans
      Specifies ``op(A)``, the transposition operation applied to the
      matrices A.

   m
      Number of rows for each matrix A. Must be at least 0.

   n
      Number of columns for each matrix B. Must be at least 0.

   alpha
      Scaling factor for the matrix transpose or copy operation. See :ref:`value_or_pointer` for more details.

   a
      Array holding the matrices A. Must have size at least
      ``stride_a*batch_size``.

   lda
      Leading dimension of the A matrices. If matrices are stored using
      column major layout, ``lda`` must be at least ``m``. If matrices are
      stored using row major layout, ``lda`` must be at least ``n``. Must be
      positive.

   stride_a
      Stride between the different A matrices. If matrices are stored using
      column major layout, ``stride_a`` must be at least ``lda*n``. If matrices
      are stored using row major layout, ``stride_a`` must be at least
      ``lda*m``.

   b
      Array holding the matrices B. Must have size at least
      ``stride_b*batch_size``.

   ldb
      Leading dimension of the B matrices. Must be positive.

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

   batch_size
      Specifies the number of matrices to transpose or copy.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   b
      Output array, overwritten by ``batch_size`` matrix transposition
      or copy operations of the form ``alpha*op(A)``.

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

