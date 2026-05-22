.. SPDX-FileCopyrightText: 2022 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_imatcopy_batch:

imatcopy_batch
==============

Computes a group of in-place scaled matrix transpose or copy operations
using general dense matrices.

.. _onemath_blas_imatcopy_batch_description:

.. rubric:: Description

The ``imatcopy_batch`` routines perform a series of in-place scaled matrix
copies or transpositions. They are batched versions of :ref:`onemath_blas_imatcopy`,
but the ``imatcopy_batch`` routines perform their operations with
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

``imatcopy_batch`` supports the following precisions:

   .. list-table::
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_blas_imatcopy_batch_buffer:

imatcopy_batch (Buffer Version)
-------------------------------

.. rubric:: Description

The buffer version of ``imatcopy_batch`` supports only the strided API.

The operation for the strided API is defined as:
::

   for i = 0 … batch_size – 1
       C is a matrix at offset i * stride in matrix_array_in_out
       C := alpha * op(C)
   end for

where:

op(X) is one of op(X) = X, or op(X) = X\ :sup:`T`, or op(X) = X\ :sup:`H`,

``alpha`` is a scalar,

``C`` is a matrix to be transformed in place,

and ``C`` is ``m`` x ``n``.

The ``matrix_array_in_out`` buffer contains all the input matrices. The stride
between matrices is given by the ``stride`` parameter. The total
number of matrices in ``matrix_array_in_out`` is given by the ``batch_size``
parameter.

**Strided API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void imatcopy_batch(sycl::queue &queue,
                           oneapi::math::transpose trans,
                           std::int64_t m,
                           std::int64_t n,
                           T alpha,
                           sycl::buffer<T, 1> &matrix_array_in_out,
                           std::int64_t ld_in,
                           std::int64_t ld_out,
                           std::int64_t stride,
                           std::int64_t batch_size);
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void imatcopy_batch(sycl::queue &queue,
                           oneapi::math::transpose trans,
                           std::int64_t m,
                           std::int64_t n,
                           T alpha,
                           sycl::buffer<T, 1> &matrix_array_in_out,
                           std::int64_t ld_in,
                           std::int64_t ld_out,
                           std::int64_t stride,
                           std::int64_t batch_size);
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   trans
      Specifies op(``C``), the transposition operation applied to the
      matrices ``C``. See :ref:`onemath_datatypes` for more details.

   m
      Number of rows of each matrix ``C`` on input. Must be at least zero.


   n
      Number of columns of each matrix ``C`` on input. Must be at least zero.

   alpha
      Scaling factor for the matrix transpositions or copies.

   matrix_array_in_out
      Buffer holding the input matrices ``C`` with size ``stride`` * ``batch_size``.

   ld_in
      The leading dimension of the matrices ``C`` on input. It must be
      positive, and must be at least ``m`` if column major layout is
      used, and at least ``n`` if row-major layout is used.

   ld_out
      The leading dimension of the matrices ``C`` on output. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``C`` not transposed
           - ``C`` transposed
         * - Column major
           - ``ld_out`` must be at least ``m``.
           - ``ld_out`` must be at least ``n``.
         * - Row major
           - ``ld_out`` must be at least ``n``.
           - ``ld_out`` must be at least ``m``.

   stride
      Stride between different ``C`` matrices.

      .. list-table::
         :header-rows: 1

         * -
           - ``C`` not transposed
           - ``C`` transposed
         * - Column major
           - ``stride`` must be at least ``max(ld_in*m, ld_out*m)``.
           - ``stride`` must be at least ``max(ld_in*m, ld_out*n)``.
         * - Row major
           - ``stride`` must be at least ``max(ld_in*n, ld_out*n)``.
           - ``stride`` must be at least ``max(ld_in*n, ld_out*m)``.

   batch_size
      Specifies the number of matrix transposition or copy operations to perform.

.. container:: section

   .. rubric:: Output Parameters

   matrix_array_in_out
      Output buffer, overwritten by ``batch_size`` matrix copy or transposition
      operations of the form ``alpha`` * op(``C``).

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
      

.. _onemath_blas_imatcopy_batch_usm:
   
imatcopy_batch (USM Version)
----------------------------

.. rubric:: Description

The USM version of ``imatcopy_batch`` supports the group API and the strided API.
            
The operation for the group API is defined as:
::

   idx = 0
   for i = 0 … group_count – 1
       m,n, alpha, ld_in, ld_out and group_size at position i in their respective arrays
       for j = 0 … group_size – 1
           C is a matrix at position idx in matrix_array_in_out
           C := alpha * op(C)
           idx := idx + 1
       end for
   end for

The operation for the strided API is defined as:
::

   for i = 0 … batch_size – 1
       C is a matrix at offset i * stride in matrix_array_in_out
       C := alpha * op(C)
   end for
   
where:

op(X) is one of op(X) = X, or op(X) = X\ :sup:`T`, or op(X) = X\ :sup:`H`,

``alpha`` is a scalar,

``C`` is a matrix to be transformed in place,

and ``C`` is ``m`` x ``n``.

For the group API, the matrices are given by arrays of pointers. ``C``
represents a matrix stored at the address pointed to by ``matrix_array_in_out``.
The number of entries in ``matrix_array_in_out`` is given by:

.. math::

      total\_batch\_count = \sum_{i=0}^{group\_count-1}group\_size[i]    

For the strided API, the single array C contains all the matrices
to be transformed in place. The locations of the individual matrices within
the buffer or array are given by stride lengths, while the number of
matrices is given by the ``batch_size`` parameter.


**Group API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       event imatcopy_batch(sycl::queue &queue,
                            const oneapi::math::transpose *trans_array,
                            const std::int64_t *m_array,
                            const std::int64_t *n_array,
                            const T *alpha_array,
                            T **matrix_array_in_out,
                            const std::int64_t *ld_in_array,
                            const std::int64_t *ld_out_array,
                            std::int64_t group_count,
                            const std::int64_t *groupsize,
                            const std::vector<sycl::event> &dependencies = {});
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       event imatcopy_batch(sycl::queue &queue,
                            const oneapi::math::transpose *trans_array,
                            const std::int64_t *m_array,
                            const std::int64_t *n_array,
                            const T *alpha_array,
                            T **matrix_array_in_out,
                            const std::int64_t *ld_in_array,
                            const std::int64_t *ld_out_array,
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
      ``op(C)`` the transposition operation applied to the matrices C.

   m_array
      Array of size ``group_count`` of number of rows of C on input. Each
      must be at least 0.

   n_array
      Array of size ``group_count`` of number of columns of C on input. Each
      must be at least 0.

   alpha_array
      Array of size ``group_count`` containing scaling factors for the matrix
      transpositions or copies.

   matrix_array_in_out
      Array of size ``total_batch_count``, holding pointers to arrays used to
      store C matrices.

   ld_in_array
      Array of size ``group_count``. The leading dimension of the matrix input
      ``C``. If matrices are stored using column major layout, ``ld_in_array[i]``
      must be at least ``m_array[i]``. If matrices are stored using row major
      layout, ``ld_in_array[i]`` must be at least ``n_array[i]``.
      Must be positive.

   ld_out_array
      Array of size ``group_count``. The leading dimension of the output matrix
      ``C``. Each entry ``ld_out_array[i]`` must be positive and at least:

      - ``m_array[i]`` if column major layout is used and ``C`` is not transposed

      - ``m_array[i]`` if row major layout is used and ``C`` is transposed

      - ``n_array[i]`` otherwise

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

   matrix_array_in_out
      Output array of pointers to ``C`` matrices, overwritten by
      ``total_batch_count`` matrix transpose or copy operations of the form
      ``alpha*op(C)``.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.


**Strided API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event imatcopy_batch(sycl::queue &queue,
                                  oneapi::math::transpose trans,
                                  std::int64_t m,
                                  std::int64_t n,
                                  value_or_pointer<T> alpha,
                                  const T *matrix_array_in_out,
                                  std::int64_t ld_in,
                                  std::int64_t ld_out,
                                  std::int64_t stride,
                                  std::int64_t batch_size,
                                  const std::vector<sycl::event> &dependencies = {});
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event imatcopy_batch(sycl::queue &queue,
                                  oneapi::math::transpose trans,
                                  std::int64_t m,
                                  std::int64_t n,
                                  value_or_pointer<T> alpha,
                                  const T *matrix_array_in_out,
                                  std::int64_t ld_in,
                                  std::int64_t ld_out,
                                  std::int64_t stride,
                                  std::int64_t batch_size,
                                  const std::vector<sycl::event> &dependencies = {});

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   trans
      Specifies ``op(C)``, the transposition operation applied to the
      matrices C.

   m
      Number of rows for each matrix ``C`` on input. Must be at least 0.

   n
      Number of columns for each matrix ``C`` on input. Must be at least 0.

   alpha
      Scaling factor for the matrix transpose or copy operation. See :ref:`value_or_pointer` for more details.

   matrix_array_in_out
      Array holding the matrices ``C``. Must have size at least
      ``stride*batch_size``.

   ld_in
      Leading dimension of the ``C`` matrices on input. If matrices are stored
      using column major layout, ``ld_in`` must be at least ``m``. If matrices
      are stored using row major layout, ``ld_in`` must be at least ``n``. 
      Must be positive.

   ld_out
      Leading dimension of the ``C`` matrices on output. If matrices are stored
      using column major layout, ``ld_out`` must be at least ``m`` if ``C`` is not
      transposed or ``n`` if ``C`` is transposed. If matrices are stored using
      row major layout, ``ld_out`` must be at least ``n`` if ``C`` is not transposed
      or at least ``m`` if ``C`` is transposed. Must be positive.

   stride
      Stride between different ``C`` matrices within ``matrix_array_in_out``.

      .. list-table::
         :header-rows: 1

         * -
           - ``C`` not transposed
           - ``C`` transposed
         * - Column major
           - ``stride`` must be at least ``max(ld_in*m, ld_out*m)``.
           - ``stride`` must be at least ``max(ld_in*m, ld_out*n)``.
         * - Row major
           - ``stride`` must be at least ``max(ld_in*n, ld_out*n)``.
           - ``stride`` must be at least ``max(ld_in*n, ld_out*m)``.

   batch_size
      Specifies the number of matrices to transpose or copy.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   matrix_array_in_out
      Output array, overwritten by ``batch_size`` matrix transposition or copy
      operations of the form ``alpha*op(C)``.

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

