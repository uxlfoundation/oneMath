.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_gemv:

gemv
====

Computes a matrix-vector product using a general matrix.

.. _onemath_blas_gemv_description:

.. rubric:: Description

The ``gemv`` routines compute a scalar-matrix-vector product and add the
result to a scalar-vector product, with a general matrix. The
operation is defined as:

.. math::
      
      y \leftarrow alpha*op(A)*x + beta*y

where:

op(``A``) is one of op(``A``) = ``A``, or op(``A``) =
``A``\ :sup:`T`, or op(``A``) = ``A``\ :sup:`H`,

``alpha`` and ``beta`` are scalars,

``A`` is an ``m``-by-``n`` matrix, and ``x``, ``y`` are vectors.

``gemv`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_blas_gemv_buffer:

gemv (Buffer Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void gemv(sycl::queue &queue,
                 oneapi::math::transpose trans,
                 std::int64_t m,
                 std::int64_t n,
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
       void gemv(sycl::queue &queue,
                 oneapi::math::transpose trans,
                 std::int64_t m,
                 std::int64_t n,
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

   trans
      Specifies ``op(A)``, the transposition operation applied to ``A``.

   m
      Specifies the number of rows of the matrix ``A``. The value of
      ``m`` must be at least zero.

   n
      Specifies the number of columns of the matrix ``A``. The value of
      ``n`` must be at least zero.

   alpha
      Scaling factor for the matrix-vector product.

   a
      The buffer holding the input matrix ``A``. Must have a size of at
      least ``lda``\ \*\ ``n`` if column major layout is used or at
      least ``lda``\ \*\ ``m`` if row major layout is used. See
      :ref:`matrix-storage` for more details.

   lda
      Leading dimension of matrix ``A``. Must be positive and at least
      ``m`` if column major layout is used or at least ``n`` if row
      major layout is used.

   x
      Buffer holding input vector ``x``. The length ``len`` of vector
      ``x`` is ``n`` if ``A`` is not transposed, and ``m`` if ``A`` is
      transposed. The buffer must be of size at least (1 + (``len`` -
      1)*abs(``incx``)). See :ref:`matrix-storage` for more details.

   incx
      The stride of vector ``x``. Must not be zero.

   beta
      The scaling factor for vector ``y``.

   y
      Buffer holding input/output vector ``y``. The length ``len`` of
      vector ``y`` is ``m``, if ``A`` is not transposed, and ``n`` if
      ``A`` is transposed. The buffer must be of size at least (1 +
      (``len`` - 1)*abs(``incy``)) where ``len`` is this length. See
      :ref:`matrix-storage` for more details.

   incy
      The stride of vector ``y``. Must not be zero.

.. container:: section

   .. rubric:: Output Parameters

   y
      The buffer holding updated vector ``y``.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

.. _onemath_blas_gemv_usm:

gemv (USM Version)
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event gemv(sycl::queue &queue,
                        oneapi::math::transpose trans,
                        std::int64_t m,
                        std::int64_t n,
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
       sycl::event gemv(sycl::queue &queue,
                        oneapi::math::transpose trans,
                        std::int64_t m,
                        std::int64_t n,
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

   trans
      Specifies ``op(A)``, the transposition operation applied to
      ``A``. See
      :ref:`onemath_datatypes` for
      more details.

   m
      Specifies the number of rows of the matrix ``A``. The value of
      ``m`` must be at least zero.

   n
      Specifies the number of columns of the matrix ``A``. The value
      of ``n`` must be at least zero.

   alpha
      Scaling factor for the matrix-vector product. See :ref:`value_or_pointer` for more details.

   a
      Pointer to the input matrix ``A``. Must have a size of at
      least ``lda``\ \*\ ``n`` if column major layout is used or at
      least ``lda``\ \*\ ``m`` if row major layout is used. See
      :ref:`matrix-storage` for more details.

   lda
      Leading dimension of matrix ``A``. Must be positive and at least
      ``m`` if column major layout is used or at least ``n`` if row
      major layout is used.

   x
      Pointer to the input vector ``x``. The length ``len`` of vector
      ``x`` is ``n`` if ``A`` is not transposed, and ``m`` if ``A``
      is transposed. The array holding vector ``x`` must be of size
      at least (1 + (``len`` - 1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      The stride of vector ``x``. Must not be zero.

   beta
      The scaling factor for vector ``y``. See :ref:`value_or_pointer` for more details.

   y
      Pointer to input/output vector ``y``. The length ``len`` of
      vector ``y`` is ``m``, if ``A`` is not transposed, and ``n`` if
      ``A`` is transposed. The array holding input/output vector
      ``y`` must be of size at least (1 + (``len`` -
      1)*abs(``incy``)) where ``len`` is this length. See :ref:`matrix-storage` for
      more details.

   incy
      The stride of vector ``y``. Must not be zero.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   y
      The pointer to updated vector ``y``.

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
