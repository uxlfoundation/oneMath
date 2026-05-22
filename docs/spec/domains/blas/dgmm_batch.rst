.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_dgmm_batch:

dgmm_batch
==========

Computes a group of ``dgmm`` operations.

.. _onemath_blas_dgmm_batch_description:

.. rubric:: Description

The ``dgmm_batch`` routines perform
multiple diagonal matrix-matrix product operations in a single call.
   
``dgmm_batch`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_blas_dgmm_batch_buffer:

dgmm_batch (Buffer Version)
---------------------------

.. rubric:: Description

The buffer version of ``dgmm_batch`` supports only the strided API. 

The strided API operation is defined as:
::

   for i = 0 … batch_size – 1
       A and C are matrices at offset i * stridea in a, i * stridec in c.
       X is a vector at offset i * stridex in x
       C := diag(X) * A or  C = A * diag(X)
   end for

where:

``A`` is a matrix,

``X`` is a diagonal matrix stored as a vector

The ``a`` and ``x`` buffers contain all the input matrices. The stride 
between matrices is given by the stride parameter. The total number
of matrices in ``a`` and ``x`` buffers is given by the ``batch_size`` parameter.

**Strided API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void dgmm_batch(sycl::queue &queue,
                       oneapi::math::side left_right,
                       std::int64_t m,
                       std::int64_t n,
                       sycl::buffer<T,1> &a,
                       std::int64_t lda,
                       std::int64_t stridea,
                       sycl::buffer<T,1> &x,
                       std::int64_t incx,
                       std::int64_t stridex,
                       sycl::buffer<T,1> &c,
                       std::int64_t ldc,
                       std::int64_t stridec,
                       std::int64_t batch_size)
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void dgmm_batch(sycl::queue &queue,
                       oneapi::math::side left_right,
                       std::int64_t m,
                       std::int64_t n,
                       sycl::buffer<T,1> &a,
                       std::int64_t lda,
                       std::int64_t stridea,
                       sycl::buffer<T,1> &x,
                       std::int64_t incx,
                       std::int64_t stridex,
                       sycl::buffer<T,1> &c,
                       std::int64_t ldc,
                       std::int64_t stridec,
                       std::int64_t batch_size)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   left_right
      Specifies the position of the diagonal matrix in the product.
      See :ref:`onemath_datatypes` for more details.

   m
      Number of rows of matrices ``A`` and ``C``. Must be at least zero.

   n
      Number of columns of matrices ``A`` and ``C``. Must be at least zero.

   a

      Buffer holding the input matrices ``A`` with size ``stridea`` *
      ``batch_size``.  Must be of at least ``lda`` * ``j`` +
      ``stridea`` * (``batch_size`` - 1) where j is n if column major
      layout is used or m if major layout is used.

   lda
      The leading dimension of the matrices ``A``. It must be positive
      and at least ``m`` if column major layout is used or at least
      ``n`` if row major layout is used.

   stridea
      Stride between different ``A`` matrices.

   x
      Buffer holding the input matrices ``X`` with size ``stridex`` *
      ``batch_size``.  Must be of size at least 
      (1 + (``len`` - 1)*abs(``incx``)) + ``stridex`` * (``batch_size`` - 1) 
      where ``len`` is ``n`` if the diagonal matrix is on the right 
      of the product or ``m`` otherwise.

   incx
      Stride between two consecutive elements of the ``x`` vectors.

   stridex
      Stride between different ``X`` vectors, must be at least 0.

   c
      Buffer holding input/output matrices ``C`` with size ``stridec`` * ``batch_size``.

   ldc
      The leading dimension of the matrices ``C``. It must be positive and at least
      ``m`` if column major layout is used to store matrices or at
      least ``n`` if column major layout is used to store matrices.

   stridec
      Stride between different ``C`` matrices. Must be at least
      ``ldc`` * ``n`` if column major layout is used or ``ldc`` * ``m`` if row
      major layout is used.

   batch_size
      Specifies the number of diagonal matrix-matrix product operations to perform.

.. container:: section

   .. rubric:: Output Parameters

   c
      Output overwritten by ``batch_size`` diagonal matrix-matrix product
      operations.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

.. _onemath_blas_dgmm_batch_usm:

dgmm_batch (USM Version)
---------------------------

.. rubric:: Description

The USM version of ``dgmm_batch`` supports the group API and strided API. 

The group API operation is defined as:
::

   idx = 0
   for i = 0 … group_count – 1
       for j = 0 … group_size – 1
           a and c are matrices of size mxn at position idx in a_array and c_array
           x is a vector of size m or n depending on left_right, at position idx in x_array
           if (left_right == oneapi::math::side::left)
               c := diag(x) * a
           else
               c := a * diag(x)
           idx := idx + 1
       end for
   end for

The strided API operation is defined as
::

   for i = 0 … batch_size – 1
       A and C are matrices at offset i * stridea in a, i * stridec in c.
       X is a vector at offset i * stridex in x
       C := diag(X) * A or  C = A * diag(X)
   end for

where:

``A`` is a matrix,

``X`` is a diagonal matrix stored as a vector

The ``a`` and ``x`` buffers contain all the input matrices. The stride 
between matrices is given by the stride parameter. The total number
of matrices in ``a`` and ``x`` buffers is given by the ``batch_size`` parameter.
 
For group API, ``a`` and ``x`` arrays contain the pointers for all the input matrices. 
The total number of matrices in ``a`` and ``x`` are given by: 

.. math::

      total\_batch\_count = \sum_{i=0}^{group\_count-1}group\_size[i]    
 
For strided API, ``a`` and ``x`` arrays contain all the input matrices. The total number of matrices 
in ``a`` and ``x`` are given by the ``batch_size`` parameter.  
   
**Group API**

.. rubric:: Syntax
   
.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event dgmm_batch(sycl::queue &queue,
                              const oneapi::math::side *left_right,
                              const std::int64_t *m,
                              const std::int64_t *n,
                              const T **a,
                              const std::int64_t *lda,
                              const T **x,
                              const std::int64_t *incx,
                              T **c,
                              const std::int64_t *ldc,
                              std::int64_t group_count,
                              const std::int64_t *group_size,
                              const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event dgmm_batch(sycl::queue &queue,
                              const oneapi::math::side *left_right,
                              const std::int64_t *m,
                              const std::int64_t *n,
                              const T **a,
                              const std::int64_t *lda,
                              const T **x,
                              const std::int64_t *incx,
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

   left_right
      Specifies the position of the diagonal matrix in the product.
      See :ref:`onemath_datatypes` for more details.

   m
      Array of ``group_count`` integers. ``m[i]`` specifies the
      number of rows of ``A`` for every matrix in group ``i``. All entries must be at least zero.

   n
      Array of ``group_count`` integers. ``n[i]`` specifies the
      number of columns of ``A`` for every matrix in group ``i``. All entries must be at least zero.

   a
      Array of pointers to input matrices ``A`` with size
      ``total_batch_count``.  Must be of size at least ``lda[i]`` * ``n[i]`` if
      column major layout is used or at least ``lda[i]`` * ``m[i]`` if row major
      layout is used.
      See :ref:`matrix-storage` for more details.

   lda
      Array of ``group_count`` integers. ``lda[i]`` specifies the
      leading dimension of ``A`` for every matrix in group ``i``. All
      entries must be positive and at least ``m[i]`` if column major
      layout is used or at least ``n[i]`` if row major layout is used.

   x
      Array of pointers to input vectors ``X`` with size
      ``total_batch_count``.  Must be of size at least (1 + ``len[i]`` –
      1)*abs(``incx[i]``)) where ``len[i]`` is ``n[i]`` if the diagonal matrix is on the
      right of the product or ``m[i]`` otherwise.
      See :ref:`matrix-storage` for more details.

   incx
      Array of ``group_count`` integers. ``incx[i]`` specifies the
      stride of ``x`` for every vector in group ``i``. All entries
      must be positive.
   c
      Array of pointers to input/output matrices ``C`` with size ``total_batch_count``. 
      Must be of size at least
      ``ldc[i]`` * ``n[i]``
      if column major layout is used or at least
      ``ldc[i]`` * ``m[i]``
      if row major layout is used.
      See :ref:`matrix-storage` for more details.

   ldc
      Array of ``group_count`` integers. ``ldc[i]`` specifies the
      leading dimension of ``C`` for every matrix in group ``i``.  All
      entries must be positive and ``ldc[i]`` must be at least
      ``m[i]`` if column major layout is used to store matrices or at
      least ``n[i]`` if row major layout is used to store matrices.

   group_count
      Specifies the number of groups. Must be at least 0.

   group_size
      Array of ``group_count`` integers. ``group_size[i]`` specifies the
      number of diagonal matrix-matrix product operations in group ``i``.
      All entries must be at least 0.

   dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   c
      Output overwritten by ``batch_size`` diagonal matrix-matrix product
      operations.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.

**Strided API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event dgmm_batch(sycl::queue &queue,
                              oneapi::math::side left_right,
                              std::int64_t m,
                              std::int64_t n,
                              const T *a,
                              std::int64_t lda,
                              std::int64_t stridea,
                              const T *b,
                              std::int64_t incx,
                              std::int64_t stridex,
                              T *c,
                              std::int64_t ldc,
                              std::int64_t stridec,
                              std::int64_t batch_size,
                              const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event dgmm_batch(sycl::queue &queue,
                              oneapi::math::side left_right,
                              std::int64_t m,
                              std::int64_t n,
                              const T *a,
                              std::int64_t lda,
                              std::int64_t stridea,
                              const T *b,
                              std::int64_t incx,
                              std::int64_t stridex,
                              T *c,
                              std::int64_t ldc,
                              std::int64_t stridec,
                              std::int64_t batch_size,
                              const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   left_right
      Specifies the position of the diagonal matrix in the product.
      See :ref:`onemath_datatypes` for more details.

   m
      Number of rows of ``A``. Must be at least zero.

   n
      Number of columns of ``A``. Must be at least zero.

   a
      Pointer to input matrices ``A`` with size ``stridea`` *
      ``batch_size``.  Must be of size at least
      ``lda`` * ``k`` + ``stridea`` * (``batch_size`` - 1) 
      where ``k`` is ``n`` if column major layout is used 
      or ``m`` if row major layout is used.

   lda
      The leading dimension of the matrices ``A``. It must be positive
      and at least ``m``.  Must be positive and at least ``m`` if column
      major layout is used or at least ``n`` if row major layout is used.

   stridea
      Stride between different ``A`` matrices.

   x
      Pointer to input matrices ``X`` with size ``stridex`` * ``batch_size``.
      Must be of size at least
      (1 + (``len`` - 1)*abs(``incx``)) + ``stridex`` * (``batch_size`` - 1)
      where ``len`` is ``n`` if the diagonal matrix is on the right
      of the product or ``m`` otherwise.

   incx
      Stride between two consecutive elements of the ``x`` vector.

   stridex
      Stride between different ``X`` vectors, must be at least 0.

   c
      Pointer to input/output matrices ``C`` with size ``stridec`` * ``batch_size``.

   ldc
      The leading dimension of the matrices ``C``. It must be positive and at least
      ``ldc`` * ``m`` if column major layout is used to store matrices or at
      least ``ldc`` * ``n`` if column major layout is used to store matrices.

   stridec
      Stride between different ``C`` matrices. Must be at least
      ``ldc`` * ``n`` if column major layout is used or 
      ``ldc`` * ``m`` if row major layout is used.

   batch_size
      Specifies the number of diagonal matrix-matrix product operations to perform.

.. container:: section

   .. rubric:: Output Parameters

   c
      Output overwritten by ``batch_size`` diagonal matrix-matrix product
      operations.

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
