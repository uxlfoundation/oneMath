.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_hbmv:

hbmv
====

Computes a matrix-vector product using a Hermitian band matrix.

.. _onemath_blas_hbmv_description:

.. rubric:: Description

The ``hbmv`` routines compute a scalar-matrix-vector product and add the
result to a scalar-vector product, with a Hermitian band matrix. The
operation is defined as

.. math::

      y \leftarrow alpha*A*x + beta*y

where:

``alpha`` and ``beta`` are scalars,

``A`` is an ``n``-by-``n`` Hermitian band matrix, with ``k``
super-diagonals,

``x`` and ``y`` are vectors of length ``n``.

``hbmv`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_blas_hbmv_buffer:

hbmv (Buffer Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void hbmv(sycl::queue &queue,
                 oneapi::math::uplo upper_lower,
                 std::int64_t n,
                 std::int64_t k,
                 T alpha,
                 sycl::buffer<T,1> &a,
                 std::int64_t lda,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx,
                 T beta,
                 sycl::buffer<T,1> &y,
                 std::int64_t incy)
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void hbmv(sycl::queue &queue,
                 oneapi::math::uplo upper_lower,
                 std::int64_t n,
                 std::int64_t k,
                 T alpha,
                 sycl::buffer<T,1> &a,
                 std::int64_t lda,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx,
                 T beta,
                 sycl::buffer<T,1> &y,
                 std::int64_t incy)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether ``A`` is upper or lower triangular. See :ref:`onemath_datatypes` for more details.

   n
      Number of rows and columns of ``A``. Must be at least zero.

   k
      Number of super-diagonals of the matrix ``A``. Must be at least
      zero.

   alpha
      Scaling factor for the matrix-vector product.

   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``n``. See :ref:`matrix-storage` for
      more details.

   lda
      Leading dimension of matrix ``A``. Must be at least (``k`` + 1),
      and positive.

   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``m`` - 1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``. Must not be zero.

   beta
      Scaling factor for vector ``y``.

   y
      Buffer holding input/output vector ``y``. The buffer must be of
      size at least (1 + (``n`` - 1)*abs(``incy``)). See :ref:`matrix-storage`
      for more details.

   incy
      Stride of vector ``y``. Must not be zero.

.. container:: section

   .. rubric:: Output Parameters

   y
      Buffer holding the updated vector ``y``.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

.. _onemath_blas_hbmv_usm:

hbmv (USM Version)
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event hbmv(sycl::queue &queue,
                        oneapi::math::uplo upper_lower,
                        std::int64_t n,
                        std::int64_t k,
                        value_or_pointer<T> alpha,
                        const T *a,
                        std::int64_t lda,
                        const T *x,
                        std::int64_t incx,
                        value_or_pointer<T> beta,
                        T *y,
                        std::int64_t incy,
                        const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event hbmv(sycl::queue &queue,
                        oneapi::math::uplo upper_lower,
                        std::int64_t n,
                        std::int64_t k,
                        value_or_pointer<T> alpha,
                        const T *a,
                        std::int64_t lda,
                        const T *x,
                        std::int64_t incx,
                        value_or_pointer<T> beta,
                        T *y,
                        std::int64_t incy,
                        const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether ``A`` is upper or lower triangular. See :ref:`onemath_datatypes` for more details.

   n
      Number of rows and columns of ``A``. Must be at least zero.

   k
      Number of super-diagonals of the matrix ``A``. Must be at least
      zero.

   alpha
      Scaling factor for the matrix-vector product. See :ref:`value_or_pointer` for more details.

   a
      Pointer to the input matrix ``A``. The array holding input
      matrix ``A`` must have size at least ``lda``\ \*\ ``n``. See
      :ref:`matrix-storage` for
      more details.

   lda
      Leading dimension of matrix ``A``. Must be at least (``k`` +
      1), and positive.

   x
      Pointer to input vector ``x``. The array holding input vector
      ``x`` must be of size at least (1 + (``m`` - 1)*abs(``incx``)).
      See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``. Must not be zero.

   beta
      Scaling factor for vector ``y``. See :ref:`value_or_pointer` for more details.

   y
      Pointer to input/output vector ``y``. The array holding
      input/output vector ``y`` must be of size at least (1 + (``n``
      - 1)*abs(``incy``)). See :ref:`matrix-storage` for
      more details.

   incy
      Stride of vector ``y``. Must not be zero.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   y
      Pointer to the updated vector ``y``.

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
      

   **Parent topic:** :ref:`blas-level-2-routines`
