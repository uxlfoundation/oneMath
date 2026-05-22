.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_her:

her
===

Computes a rank-1 update of a Hermitian matrix.

.. _onemath_blas_her_description:

.. rubric:: Description

The ``her`` routines compute a scalar-vector-vector product and add the
result to a Hermitian matrix. The operation is defined as:

.. math::
      
      A \leftarrow alpha*x*x^H + A

where:

``alpha`` is scalar,

``A`` is an ``n``-by-``n`` Hermitian matrix,

``x`` is a vector of length ``n``.

``her`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
        -  Treal 
      * -  ``std::complex<float>`` 
        -  ``float`` 
      * -  ``std::complex<double>`` 
        -  ``double`` 

.. _onemath_blas_her_buffer:

her (Buffer Version)
--------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void her(sycl::queue &queue,
                oneapi::math::uplo upper_lower,
                std::int64_t n,
                Treal alpha,
                sycl::buffer<T,1> &x,
                std::int64_t incx,
                sycl::buffer<T,1> &a,
                std::int64_t lda)
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void her(sycl::queue &queue,
                oneapi::math::uplo upper_lower,
                std::int64_t n,
                Treal alpha,
                sycl::buffer<T,1> &x,
                std::int64_t incx,
                sycl::buffer<T,1> &a,
                std::int64_t lda)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether ``A`` is upper or lower triangular. See :ref:`onemath_datatypes` for more details.

   n
      Number of rows and columns of ``A``. Must be at least zero.

   alpha
      Scaling factor for the matrix-vector product.

   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``. Must not be zero.

   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``n``. See :ref:`matrix-storage` for
      more details.

   lda
      Leading dimension of matrix ``A``. Must be at least ``n``, and
      positive.

.. container:: section

   .. rubric:: Output Parameters

   a
      Buffer holding the updated upper triangular part of the Hermitian
      matrix ``A`` if ``upper_lower``\  \= ``upper`` or the updated
      lower triangular part of the Hermitian matrix ``A`` if
      ``upper_lower``\ \ =\ ``lower``.

      If ``alpha`` is zero, ``A`` matrix is unchanged, otherwise imaginary parts of the diagonal elements are set to zero.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

.. _onemath_blas_her_usm:

her (USM Version)
-----------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event her(sycl::queue &queue,
                       oneapi::math::uplo upper_lower,
                       std::int64_t n,
                       value_or_pointer<Treal> alpha,
                       const T *x,
                       std::int64_t incx,
                       T *a,
                       std::int64_t lda,
                       const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event her(sycl::queue &queue,
                       oneapi::math::uplo upper_lower,
                       std::int64_t n,
                       value_or_pointer<Treal> alpha,
                       const T *x,
                       std::int64_t incx,
                       T *a,
                       std::int64_t lda,
                       const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether *A* is upper or lower triangular. See :ref:`onemath_datatypes` for more details.

   n
      Number of rows and columns of ``A``. Must be at least zero.

   alpha
      Scaling factor for the matrix-vector product. See :ref:`value_or_pointer` for more details.

   x
      Pointer to input vector ``x``. The array holding input vector
      ``x`` must be of size at least (1 + (``n`` - 1)*abs(``incx``)).
      See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``. Must not be zero.

   a
      Pointer to input matrix ``A``. The array holding input matrix
      ``A`` must have size at least ``lda``\ \*\ ``n``. See :ref:`matrix-storage` for
      more details.

   lda
      Leading dimension of matrix ``A``. Must be at least ``n``, and
      positive.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   a
      Pointer to the updated upper triangular part of the Hermitian
      matrix ``A`` if ``upper_lower``\ \=\ ``upper`` or the updated
      lower triangular part of the Hermitian matrix ``A`` if
      ``upper_lower``\ \=\ ``lower``.

      If ``alpha`` is zero, ``A`` matrix is unchanged, otherwise imaginary parts of the diagonal elements are set to zero.

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
