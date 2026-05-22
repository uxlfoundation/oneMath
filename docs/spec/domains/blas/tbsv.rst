.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_tbsv:

tbsv
====

Solves a system of linear equations whose coefficients are in a
triangular band matrix.

.. _onemath_blas_tbsv_description:

.. rubric:: Description

The ``tbsv`` routines solve a system of linear equations whose
coefficients are in a triangular band matrix. The operation is
defined as:

.. math::

      op(A)*x = b

where:

op(``A``) is one of op(``A``) = ``A``, or op(``A``) =
``A``\ :sup:`T`, or op(``A``) = ``A``\ :sup:`H`,

``A`` is an ``n``-by-``n`` unit or non-unit, upper or lower
triangular band matrix, with (``k`` + 1) diagonals,

``b`` and ``x`` are vectors of length ``n``.

``tbsv`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_blas_tbsv_buffer:

tbsv (Buffer Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void tbsv(sycl::queue &queue,
                 oneapi::math::uplo upper_lower,
                 oneapi::math::transpose trans,
                 oneapi::math::diag unit_nonunit,
                 std::int64_t n,
                 std::int64_t k,
                 sycl::buffer<T,1> &a,
                 std::int64_t lda,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx)
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void tbsv(sycl::queue &queue,
                 oneapi::math::uplo upper_lower,
                 oneapi::math::transpose trans,
                 oneapi::math::diag unit_nonunit,
                 std::int64_t n,
                 std::int64_t k,
                 sycl::buffer<T,1> &a,
                 std::int64_t lda,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether ``A`` is upper or lower triangular. See :ref:`onemath_datatypes` for more details.

   trans
      Specifies op(``A``), the transposition operation applied to ``A``. See :ref:`onemath_datatypes` for more details.

   unit_nonunit
      Specifies whether the matrix ``A`` is unit triangular or not. See :ref:`onemath_datatypes` for more details.

   n
      Number of rows and columns of ``A``. Must be at least zero.

   k
      Number of sub/super-diagonals of the matrix ``A``. Must be at
      least zero.

   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``n``. See :ref:`matrix-storage` for
      more details.

   lda
      Leading dimension of matrix ``A``. Must be at least (``k`` + 1),
      and positive.

   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``. Must not be zero.

.. container:: section

   .. rubric:: Output Parameters

   x
      Buffer holding the solution vector ``x``.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

.. _onemath_blas_tbsv_usm:

tbsv (USM Version)
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event tbsv(sycl::queue &queue,
                        oneapi::math::uplo upper_lower,
                        oneapi::math::transpose trans,
                        oneapi::math::diag unit_nonunit,
                        std::int64_t n,
                        std::int64_t k,
                        const T *a,
                        std::int64_t lda,
                        T *x,
                        std::int64_t incx,
                        const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event tbsv(sycl::queue &queue,
                        oneapi::math::uplo upper_lower,
                        oneapi::math::transpose trans,
                        oneapi::math::diag unit_nonunit,
                        std::int64_t n,
                        std::int64_t k,
                        const T *a,
                        std::int64_t lda,
                        T *x,
                        std::int64_t incx,
                        const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether ``A`` is upper or lower triangular. See :ref:`onemath_datatypes` for more details.

   trans
      Specifies op(``A``), the transposition operation applied to
      ``A``. See :ref:`onemath_datatypes` for more details.

   unit_nonunit
      Specifies whether the matrix ``A`` is unit triangular or not. See :ref:`onemath_datatypes` for more details.

   n
      Number of rows and columns of ``A``. Must be at least zero.

   k
      Number of sub/super-diagonals of the matrix ``A``. Must be at
      least zero.

   a
      Pointer to input matrix ``A``. The array holding input matrix
      ``A`` must have size at least ``lda``\ \*\ ``n``. See :ref:`matrix-storage` for
      more details.

   lda
      Leading dimension of matrix ``A``. Must be at least (``k`` +
      1), and positive.

   x
      Pointer to input vector ``x``. The array holding input vector
      ``x`` must be of size at least (1 + (``n`` - 1)*abs(``incx``)).
      See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``. Must not be zero.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   x
      Pointer to the solution vector ``x``.

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
