.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_axpy_batch:

axpy_batch
==========

Computes a group of ``axpy`` operations.

.. _onemath_blas_axpy_batch_description:

.. rubric:: Description

The ``axpy_batch`` routines are batched versions of :ref:`onemath_blas_axpy`, performing
multiple ``axpy`` operations in a single call. Each ``axpy`` 
operation adds a scalar-vector product to a vector.
   
``axpy_batch`` supports the following precisions for data.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_blas_axpy_batch_buffer:

axpy_batch (Buffer Version)
---------------------------

.. rubric:: Description

The buffer version of ``axpy_batch`` supports only the strided API. 

The strided API operation is defined as:
::
  
   for i = 0 … batch_size – 1
      X and Y are vectors at offset i * stridex, i * stridey in x and y
      Y := alpha * X + Y
   end for

where:

``alpha`` is scalar,

``X`` and ``Y`` are vectors.
   
**Strided API**

.. rubric:: Syntax
 
.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void axpy_batch(sycl::queue &queue,
                       std::int64_t n,
                       T alpha,
                       sycl::buffer<T,
                       1> &x,
                       std::int64_t incx,
                       std::int64_t stridex,
                       sycl::buffer<T,
                       1> &y,
                       std::int64_t incy,
                       std::int64_t stridey,
                       std::int64_t batch_size)
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void axpy_batch(sycl::queue &queue,
                       std::int64_t n,
                       T alpha,
                       sycl::buffer<T,
                       1> &x,
                       std::int64_t incx,
                       std::int64_t stridex,
                       sycl::buffer<T,
                       1> &y,
                       std::int64_t incy,
                       std::int64_t stridey,
                       std::int64_t batch_size)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   n
      Number of elements in ``X`` and ``Y``.

   alpha
       Specifies the scalar ``alpha``.

   x
      Buffer holding input vectors ``X`` with size ``stridex`` * ``batch_size``.

   incx 
      Stride of vector ``X``. Must not be zero.

   stridex 
      Stride between different ``X`` vectors. Must be at least zero.

   y
      Buffer holding input/output vectors ``Y`` with size ``stridey`` * ``batch_size``.

   incy 
      Stride of vector ``Y``. Must not be zero.
   
   stridey 
      Stride between different ``Y`` vectors. Must be at least (1 + (``n``-1)*abs(``incy``)).

   batch_size 
      Specifies the number of ``axpy`` operations to perform.

.. container:: section

   .. rubric:: Output Parameters

   y
      Output buffer, overwritten by ``batch_size`` ``axpy`` operations of the form 
      ``alpha`` * ``X`` + ``Y``.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

.. _onemath_blas_axpy_batch_usm:

axpy_batch (USM Version)
------------------------

.. rubric:: Description

The USM version of ``axpy_batch`` supports the group API and strided API. 

The group API operation is defined as
::
   
   idx = 0
   for i = 0 … group_count – 1
       for j = 0 … group_size – 1
           X and Y are vectors in x[idx] and y[idx]
           Y := alpha[i] * X + Y
           idx := idx + 1
       end for
   end for

The strided API operation is defined as
::
   
   for i = 0 … batch_size – 1
      X and Y are vectors at offset i * stridex, i * stridey in x and y
      Y := alpha * X + Y
   end for

where:

``alpha`` is scalar,

``X`` and ``Y`` are vectors.

For group API, ``x`` and ``y`` arrays contain the pointers for all the input vectors. 
The total number of vectors in ``x`` and ``y`` are given by:

.. math::

      total\_batch\_count = \sum_{i=0}^{group\_count-1}group\_size[i]    

For strided API, ``x`` and ``y`` arrays contain all the input vectors. 
The total number of vectors in ``x`` and ``y`` are given by the ``batch_size`` parameter.

**Group API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event axpy_batch(sycl::queue &queue,
                              const std::int64_t *n,
                              const T *alpha,
                              const T **x,
                              const std::int64_t *incx,
                              T **y,
                              const std::int64_t *incy,
                              std::int64_t group_count,
                              const std::int64_t *group_size,
                              const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event axpy_batch(sycl::queue &queue,
                              const std::int64_t *n,
                              const T *alpha,
                              const T **x,
                              const std::int64_t *incx,
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

   n
      Array of ``group_count`` integers. ``n[i]`` specifies the number of elements in vectors ``X`` and ``Y`` for every vector in group ``i``.

   alpha
       Array of ``group_count`` scalar elements. ``alpha[i]`` specifies the scaling factor for vector ``X`` in group ``i``.

   x
      Array of pointers to input vectors ``X`` with size ``total_batch_count``.
      The size of array allocated for the ``X`` vector of the group ``i`` must be at least (1 + (``n[i]`` – 1)*abs(``incx[i]``)). 
      See :ref:`matrix-storage` for more details.

   incx
      Array of ``group_count`` integers. ``incx[i]`` specifies the stride of vector ``X`` in group ``i``. Must not be zero.
 
   y
      Array of pointers to input/output vectors ``Y`` with size ``total_batch_count``.
      The size of array allocated for the ``Y`` vector of the group ``i`` must be at least (1 + (``n[i]`` – 1)*abs(``incy[i]``)). 
      See :ref:`matrix-storage` for more details.

   incy
      Array of ``group_count`` integers. ``incy[i]`` specifies the stride of vector ``Y`` in group ``i``. Must not be zero.

   group_count
      Number of groups. Must be at least 0.

   group_size
      Array of ``group_count`` integers. ``group_size[i]`` specifies the number of ``axpy`` operations in group ``i``. 
      Each element in ``group_size`` must be at least 0.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   y
      Array of pointers holding the ``Y`` vectors, overwritten by ``total_batch_count`` ``axpy`` operations of the form 
      ``alpha`` * ``X`` + ``Y``.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.

**Strided API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event axpy_batch(sycl::queue &queue,
                              std::int64_t n,
                              value_or_pointer<T> alpha,
                              const T *x,
                              std::int64_t incx,
                              std::int64_t stridex,
                              T *y,
                              std::int64_t incy,
                              std::int64_t stridey,
                              std::int64_t batch_size,
                              const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event axpy_batch(sycl::queue &queue,
                              std::int64_t n,
                              value_or_pointer<T> alpha,
                              const T *x,
                              std::int64_t incx,
                              std::int64_t stridex,
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

   n
      Number of elements in ``X`` and ``Y``.

   alpha
       Specifies the scalar ``alpha``. See :ref:`value_or_pointer` for more details.

   x
      Pointer to input vectors ``X`` with size ``stridex`` * ``batch_size``.

   incx 
      Stride of vector ``X``. Must not be zero.
   
   stridex 
      Stride between different ``X`` vectors. Must be at least zero.

   y
      Pointer to input/output vectors ``Y`` with size ``stridey`` * ``batch_size``.

   incy 
      Stride of vector ``Y``. Must not be zero.
   
   stridey 
      Stride between different ``Y`` vectors. Must be at least (1 + (``n``-1)*abs(``incy``)).

   batch_size 
      Specifies the number of ``axpy`` operations to perform.
  
   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   y
      Output vectors, overwritten by ``batch_size`` ``axpy`` operations of the form 
      ``alpha`` * ``X`` + ``Y``.

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
      

   **Parent topic:**:ref:`blas-like-extensions`
