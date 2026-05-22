.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_gemv_batch:

gemv_batch
==========

Computes a group of ``gemv`` operations.

.. _onemath_blas_gemv_batch_description:

.. rubric:: Description

The ``gemv_batch`` routines are batched versions of
:ref:`onemath_blas_gemv`, performing multiple ``gemv`` operations in a
single call. Each ``gemv`` operations perform a scalar-matrix-vector
product and add the result to a scalar-vector product.
   
``gemv_batch`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_blas_gemv_batch_buffer:

gemv_batch (Buffer Version)
---------------------------

.. rubric:: Description

The buffer version of ``gemv_batch`` supports only the strided API. 

The strided API operation is defined as:
::

   for i = 0 … batch_size – 1
       A is a matrix at offset i * stridea in a.
       X and Y are matrices at offset i * stridex, i * stridey, in x and y.
       Y := alpha * op(A) * X + beta * Y
   end for

where:

op(A) is one of op(A) = A, or op(A) = A\ :sup:`T`, or op(A) = A\ :sup:`H`,

``alpha`` and ``beta`` are scalars,

``A`` is a matrix and ``X`` and ``Y`` are vectors,

The ``x`` and ``y`` buffers contain all the input matrices. The stride
between vectors is given by the stride parameter. The total number of
vectors in ``x`` and ``y`` buffers is given by the ``batch_size``
parameter.

**Strided API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void gemv_batch(sycl::queue &queue,
                       oneapi::math::transpose trans,
                       std::int64_t m,
                       std::int64_t n,
                       T alpha,
                       sycl::buffer<T,1> &a,
                       std::int64_t lda,
                       std::int64_t stridea,
                       sycl::buffer<T,1> &x,
                       std::int64_t incx,
                       std::int64_t stridex,
                       T beta,
                       sycl::buffer<T,1> &y,
                       std::int64_t incy,
                       std::int64_t stridey,
                       std::int64_t batch_size)
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void gemv_batch(sycl::queue &queue,
                       oneapi::math::transpose trans,
                       std::int64_t m,
                       std::int64_t n,
                       T alpha,
                       sycl::buffer<T,1> &a,
                       std::int64_t lda,
                       std::int64_t stridea,
                       sycl::buffer<T,1> &x,
                       std::int64_t incx,
                       std::int64_t stridex,
                       T beta,
                       sycl::buffer<T,1> &y,
                       std::int64_t incy,
                       std::int64_t stridey,
                       std::int64_t batch_size)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   trans
      Specifies op(``A``) the transposition operation applied to the
      matrices ``A``. See :ref:`onemath_datatypes` for more details.

   m
      Number of rows of ``A``. Must be at least zero.

   n
      Number of columns of ``A``. Must be at least zero.

   alpha
      Scaling factor for the matrix-vector products.

   a
      Buffer holding the input matrices ``A`` with size ``stridea`` * ``batch_size``.

   lda
      The leading dimension of the matrices ``A``. It must be positive
      and at least ``m`` if column major layout is used or at least
      ``n`` if row major layout is used.

   stridea
      Stride between different ``A`` matrices. Must be at least zero.

   x
      Buffer holding the input vectors ``X`` with size ``stridex`` * ``batch_size``.

   incx
      The stride of the vector ``X``.  Must not be zero.

   stridex
      Stride between different consecutive ``X`` vectors, must be at least 0.

   beta
      Scaling factor for the vector ``Y``.

   y
      Buffer holding input/output vectors ``Y`` with size ``stridey`` * ``batch_size``.

   incy
      Stride between two consecutive elements of the ``Y`` vectors. Must not be zero.

   stridey
      Stride between two consecutive ``Y`` vectors. Must be at least (1 + (``m`` - 1)*abs(``incy``)) if layout is column major or (1 + (``n`` - 1)*abs(``incy``)) if row major layout is used.

   batch_size
      Specifies the number of matrix-vector operations to perform.

.. container:: section

   .. rubric:: Output Parameters

   y
      Output overwritten by ``batch_size`` matrix-vector product
      operations of the form ``alpha`` * op(``A``) * ``X`` + ``beta`` * ``Y``.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

.. _onemath_blas_gemv_batch_usm:

gemv_batch (USM Version)
---------------------------

.. rubric:: Description

The USM version of ``gemv_batch`` supports the group API and strided API. 

The group API operation is defined as:
::

   idx = 0
   for i = 0 … group_count – 1
       for j = 0 … group_size – 1
           A is an m x n matrix in a[idx]
           X and Y are vectors in x[idx] and y[idx]
           Y := alpha[i] * op(A) * X + beta[i] * Y
           idx = idx + 1
       end for
   end for

The strided API operation is defined as
::

   for i = 0 … batch_size – 1
       A is a matrix at offset i * stridea in a.
       X and Y are vectors at offset i * stridex, i * stridey in x and y.
       Y := alpha * op(A) * X + beta * Y
   end for

where:

op(A) is one of op(A) = A, or op(A) = A\ :sup:`T`, or op(A) = A\ :sup:`H`,

``alpha`` and ``beta`` are scalars,

``A`` is a matrix and ``X`` and ``Y`` are vectors,

For group API, ``x`` and ``y`` arrays contain the pointers for all the input vectors. 
``A`` array contains the pointers to all input matrices.
The total number of vectors in ``x`` and ``y`` and matrices in ``A`` are given by: 

.. math::

      total\_batch\_count = \sum_{i=0}^{group\_count-1}group\_size[i]    
 
For strided API, ``x`` and ``y`` arrays contain all the input
vectors. ``A`` array contains the pointers to all input matrices.  The
total number of vectors in ``x`` and ``y`` and matrices in ``A`` are given by the
``batch_size`` parameter.
   
**Group API**

.. rubric:: Syntax
   
.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event gemv_batch(sycl::queue &queue,
                              const oneapi::math::transpose *trans,
                              const std::int64_t *m,
                              const std::int64_t *n,
                              const T *alpha,
                              const T **a,
                              const std::int64_t *lda,
                              const T **x,
                              const std::int64_t *incx,
                              const T *beta,
                              T **y,
                              const std::int64_t *incy,
                              std::int64_t group_count,
                              const std::int64_t *group_size,
                              const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event gemv_batch(sycl::queue &queue,
                              const oneapi::math::transpose *trans,
                              const std::int64_t *m,
                              const std::int64_t *n,
                              const T *alpha,
                              const T **a,
                              const std::int64_t *lda,
                              const T **x,
                              const std::int64_t *incx,
                              const T *beta,
                              T **y,
                              const std::int64_t *incy,
                              std::int64_t group_count,
                              const std::int64_t *group_size,
                              const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   trans
      Array of ``group_count`` ``oneapi::math::transpose`` values. ``trans[i]`` specifies the form of op(``A``) used in
      the matrix-vector product in group ``i``. See :ref:`onemath_datatypes` for more details.

   m
      Array of ``group_count`` integers. ``m[i]`` specifies the
      number of rows of ``A`` for every matrix in group ``i``. All entries must be at least zero.

   n
      Array of ``group_count`` integers. ``n[i]`` specifies the
      number of columns of ``A`` for every matrix in group ``i``. All entries must be at least zero.

   alpha
      Array of ``group_count`` scalar elements. ``alpha[i]`` specifies
      the scaling factor for every matrix-vector product in group
      ``i``.

   a
      Array of pointers to input matrices ``A`` with size ``total_batch_count``. 
      
      See :ref:`matrix-storage` for more details.

   lda
      Array of ``group_count`` integers. ``lda[i]`` specifies the
      leading dimension of ``A`` for every matrix in group ``i``. All
      entries must be positive and at least ``m`` if column major
      layout is used or at least ``n`` if row major layout is used.
             
   x
      Array of pointers to input vectors ``X`` with size ``total_batch_count``. 
      
      See :ref:`matrix-storage` for more details.

   incx
      Array of ``group_count`` integers. ``incx[i]`` specifies the
      stride of ``X`` for every vector in group ``i``. Must not
      be zero.
             
   beta
      Array of ``group_count`` scalar elements. ``beta[i]`` specifies
      the scaling factor for vector ``Y`` for every vector in group
      ``i``.

   y
      Array of pointers to input/output vectors ``Y`` with size ``total_batch_count``. 
      
      See :ref:`matrix-storage` for more details.

   incy
      Array of ``group_count`` integers. ``incy[i]`` specifies the
      leading dimension of ``Y`` for every vector in group ``i``. Must
      not be zero.

   group_count
      Specifies the number of groups. Must be at least 0.

   group_size
      Array of ``group_count`` integers. ``group_size[i]`` specifies the
      number of matrix-vector products in group ``i``. All entries must be at least 0.

   dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   y
      Overwritten by vector calculated by 
      (``alpha[i]`` * op(``A``) * ``X`` + ``beta[i]`` * ``Y``) for group ``i``.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.

**Strided API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event gemv_batch(sycl::queue &queue,
                              oneapi::math::transpose trans,
                              std::int64_t m,
                              std::int64_t n,
                              value_or_pointer<T> alpha,
                              const T *a,
                              std::int64_t lda,
                              std::int64_t stridea,
                              const T *x,
                              std::int64_t incx,
                              std::int64_t stridex,
                              value_or_pointer<T> beta,
                              T *y,
                              std::int64_t incy,
                              std::int64_t stridey,
                              std::int64_t batch_size,
                              const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event gemv_batch(sycl::queue &queue,
                              oneapi::math::transpose trans,
                              std::int64_t m,
                              std::int64_t n,
                              value_or_pointer<T> alpha,
                              const T *a,
                              std::int64_t lda,
                              std::int64_t stridea,
                              const T *x,
                              std::int64_t incx,
                              std::int64_t stridex,
                              value_or_pointer<T> beta,
                              T *y,
                              std::int64_t incy,
                              std::int64_t stridey,
                              std::int64_t batch_size,
                              const std::vector<sycl::event> &dependencies = {})
   }


.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   trans
      Specifies op(``A``) the transposition operation applied to the
      matrices ``A``. See :ref:`onemath_datatypes` for more details.

   m
      Number of rows of ``A``. Must be at least zero.

   n
      Number of columns of ``A``. Must be at least zero.

   alpha
      Scaling factor for the matrix-vector products. See :ref:`value_or_pointer` for more details.

   a
      Pointer to the input matrices ``A`` with size ``stridea`` * ``batch_size``.

   lda
      The leading dimension of the matrices ``A``. It must be positive
      and at least ``m`` if column major layout is used or at least
      ``n`` if row major layout is used.

   stridea
      Stride between different ``A`` matrices. Must be at least zero.

   x
      Pointer to the input vectors ``X`` with size ``stridex`` * ``batch_size``.

   incx
      Stride of the vector ``X``. Must not be zero.

   stridex
      Stride between different consecutive ``X`` vectors, must be at least 0.

   beta
      Scaling factor for the vector ``Y``. See :ref:`value_or_pointer` for more details.

   y
      Pointer to the input/output vectors ``Y`` with size ``stridey`` * ``batch_size``.

   incy
      Stride between two consecutive elements of the ``y`` vectors. Must not be zero.

   stridey
      Stride between two consecutive ``Y`` vectors. Must be at least (1 + (``m`` - 1)*abs(``incy``)) if layout is column major or (1 + (``n`` - 1)*abs(``incy``)) if row major layout is used.

   batch_size
      Specifies the number of matrix-vector operations to perform.

.. container:: section

   .. rubric:: Output Parameters

   y
      Output overwritten by ``batch_size`` matrix-vector product
      operations of the form ``alpha`` * op(``A``) * ``X`` + ``beta`` * ``Y``.

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
