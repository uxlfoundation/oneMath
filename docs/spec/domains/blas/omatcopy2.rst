.. SPDX-FileCopyrightText: 2022 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_omatcopy2:

omatcopy2
=========

Computes two-strided scaling and out-of-place transposition or copying of general
dense matrices.

.. _onemath_blas_omatcopy2_description:

.. rubric:: Description

The ``omatcopy2`` routine performs a two-strided scaling and out-of-place
transposition or copy of matrices. For complex matrices the transpose operation can
be a conjugate transpose.

Normally, matrices in the BLAS or LAPACK are specified by a single
stride index. For instance, in the column-major order, ``A(2,1)`` is
stored in memory one element away from ``A(1,1)``, but ``A(1,2)`` is a leading
dimension away. The leading dimension in this case is at least the
number of rows of the source matrix. If a matrix has two strides, then
both ``A(2,1)`` and ``A(1,2)`` may be an arbitrary distance from ``A(1,1)``.

The operation is defined as:

.. math::

      B \leftarrow \alpha * op(A)

where:

op(``X``) is one of op(``X``) = ``X``, or op(``X``) = ``X``\ :sup:`T`, or op(``X``) = ``X``\ :sup:`H`,

``alpha`` is a scalar,

``A`` and  ``B`` are matrices,

``A`` is ``m`` x ``n`` matrix,

``B`` is ``m`` x ``n`` matrix if ``op`` is non-transpose and an ``n`` x ``m`` matrix otherwise.,

``omatcopy2`` supports the following precisions:

   .. list-table::
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_blas_omatcopy2_buffer:

omatcopy2 (Buffer Version)
--------------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void omatcopy2(sycl::queue &queue,
                      oneapi::math::transpose trans,
                      std::int64_t m,
                      std::int64_t n,
                      T alpha,
                      sycl::buffer<T, 1> &a,
                      std::int64_t lda,
                      std::int64_t stridea,
                      sycl::buffer<T, 1> &b,
                      std::int64_t ldb,
                      std::int64_t strideb);
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void omatcopy2(sycl::queue &queue,
                      oneapi::math::transpose trans,
                      std::int64_t m,
                      std::int64_t n,
                      T alpha,
                      sycl::buffer<T, 1> &a,
                      std::int64_t lda,
                      std::int64_t stridea,
                      sycl::buffer<T, 1> &b,
                      std::int64_t ldb,
                      std::int64_t strideb);
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   trans
      Specifies op(``A``), the transposition operation applied to the
      matrix ``A``. See :ref:`onemath_datatypes` for more details.

   m
      Number of rows for the matrix ``A``. Must be at least zero.

   n
      Number of columns for the matrix ``A``. Must be at least zero.

   alpha
      Scaling factor for the matrix transposition or copy.

   a
      Buffer holding the input matrix ``A``. Must have size at least
      ``lda`` * ``n`` for column major ordering and at least ``lda`` * ``m`` for
      row major ordering.

   lda
      Leading dimension of the matrix ``A``. If matrices are stored using
      column major layout, ``lda`` is the number of elements in the array between
      adjacent columns of the matrix, and must be at least ``stridea`` * (``m``-1) + 1.
      If using row major layout, ``lda`` is the number of elements between adjacent
      rows of the matrix and must be at least ``stridea`` * (``n``-1) + 1.

   stridea
      The second stride of the matrix ``A``. For column major layout, ``stridea`` is
      the number of elements in the array between adjacent rows of the matrix. For row
      major layout ``stridea`` is the number of elements between adjacent columns of the
      matrix. In both cases ``stridea`` must be at least 1.

   b
      Buffer holding the output matrix ``B``.

      .. list-table::
         :header-rows: 1

         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``trans`` = ``transpose::conjtrans``
         * - Column major
           - ``B`` is ``m`` x ``n`` matrix. Size of buffer ``b`` must be at least ``ldb`` * ``n``
           - ``B`` is ``n`` x ``m`` matrix. Size of buffer ``b`` must be at least ``ldb`` * ``m``
         * - Row major
           - ``B`` is ``m`` x ``n`` matrix. Size of buffer ``b`` must be at least ``ldb`` * ``m``
           - ``B`` is ``n`` x ``m`` matrix. Size of buffer ``b`` must be at least ``ldb`` * ``n``

   ldb
      The leading dimension of the matrix ``B``. Must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``trans`` = ``transpose::conjtrans``
         * - Column major
           - ``ldb`` must be at least ``strideb`` * (``m``-1) + 1.
           - ``ldb`` must be at least ``strideb`` * (``n``-1) + 1.
         * - Row major
           - ``ldb`` must be at least ``strideb`` * (``n``-1) + 1.
           - ``ldb`` must be at least ``strideb`` * (``m``-1) + 1.

   strideb
      The second stride of the matrix ``B``. For column major layout, ``strideb`` is
      the number of elements in the array between adjacent rows of the matrix. For row
      major layout, ``strideb`` is the number of elements between adjacent columns of the
      matrix. In both cases ``strideb`` must be at least 1.

.. container:: section

   .. rubric:: Output Parameters

   b
      Output buffer, overwritten by ``alpha`` * op(``A``).

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
      

.. _onemath_blas_omatcopy2_usm:
   
omatcopy2 (USM Version)
-----------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event omatcopy2(sycl::queue &queue,
                             oneapi::math::transpose trans,
                             std::int64_t m,
                             std::int64_t n,
                             value_or_pointer<T> alpha,
                             const T *a,
                             std::int64_t lda,
                             std::int64_t stridea,
                             T *b,
                             std::int64_t ldb,
                             std::int64_t strideb,
                             const std::vector<sycl::event> &dependencies = {});
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event omatcopy2(sycl::queue &queue,
                             oneapi::math::transpose trans,
                             std::int64_t m,
                             std::int64_t n,
                             value_or_pointer<T> alpha,
                             const T *a,
                             std::int64_t lda,
                             std::int64_t stridea,
                             T *b,
                             std::int64_t ldb,
                             std::int64_t strideb,
                             const std::vector<sycl::event> &dependencies = {});
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine will be executed.

   trans
      Specifies op(``A``), the transposition operation applied to matrix ``A``.
      See :ref:`onemath_datatypes` for more details.

   m
      Number of rows for the matrix ``A``. Must be at least zero.

   n
      Number of columns for the matrix ``A``. Must be at least zero.

   alpha
      Scaling factor for the matrix transposition or copy. See :ref:`value_or_pointer` for more details.

   a
      Pointer to input matrix ``A``. Must have size at least
      ``lda`` * ``n`` for column-major and at least ``lda`` * ``m`` for row-major.

   lda
      Leading dimension of the matrix ``A``. If matrices are stored using
      column major layout, ``lda`` is the number of elements in the array between
      adjacent columns of the matrix, and must be at least ``stridea`` * (``m``-1) + 1.
      If using row major layout, ``lda`` is the number of elements between adjacent
      rows of the matrix and must be at least ``stridea`` * (``n``-1) + 1.

   stridea
      The second stride of the matrix ``A``. For column major layout, ``stridea`` is
      the number of elements in the array between adjacent rows of the matrix. For row
      major layout ``stridea`` is the number of elements between adjacent columns of the
      matrix. In both cases ``stridea`` must be at least 1.

   b
      Pointer to output matrix ``B``.

      .. list-table::
         :header-rows: 1
     
         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``trans`` = ``transpose::conjtrans``
         * - Column major
           - ``B`` is ``m`` x ``n`` matrix. Size of array ``b`` must be at least ``ldb`` * ``n``
           - ``B`` is ``n`` x ``m`` matrix. Size of array ``b`` must be at least ``ldb`` * ``m``
         * - Row major
           - ``B`` is ``m`` x ``n`` matrix. Size of array ``b`` must be at least ``ldb`` * ``m``
           - ``B`` is ``n`` x ``m`` matrix. Size of array ``b`` must be at least ``ldb`` * ``n``

   ldb
      The leading dimension of the matrix ``B``. Must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``trans`` = ``transpose::conjtrans``
         * - Column major
           - ``ldb`` must be at least ``strideb`` * (``m``-1) + 1.
           - ``ldb`` must be at least ``strideb`` * (``n``-1) + 1.
         * - Row major
           - ``ldb`` must be at least ``strideb`` * (``n``-1) + 1.
           - ``ldb`` must be at least ``strideb`` * (``m``-1) + 1.

   strideb
      The second stride of the matrix ``B``. For column major layout, ``strideb`` is
      the number of elements in the array between adjacent rows of the matrix. For row
      major layout, ``strideb`` is the number of elements between adjacent columns of the
      matrix. In both cases ``strideb`` must be at least 1.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   b
      Pointer to output matrix ``B`` overwritten by ``alpha`` * op(``A``).

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

