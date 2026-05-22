.. SPDX-FileCopyrightText: 2022 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_omatcopy:

omatcopy
========

Computes an out-of-place scaled matrix transpose or copy operation
using a general dense matrix.

.. _onemath_blas_omatcopy_description:

.. rubric:: Description

The ``omatcopy`` routine performs an out-of-place scaled
matrix copy or transposition.

The operation is defined as:

.. math::

      B \leftarrow \alpha * op(A)

where:

op(``X``) is one of op(``X``) = ``X``, or op(``X``) = ``X``\ :sup:`T`, or op(``X``) = ``X``\ :sup:`H`,

``alpha`` is a scalar,

``A`` and  ``B`` are matrices,

``A`` is ``m`` x ``n`` matrix,

``B`` is ``m`` x ``n`` matrix if ``op`` is non-transpose and an ``n`` x ``m`` matrix otherwise.,

``omatcopy`` supports the following precisions:

   .. list-table::
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_blas_omatcopy_buffer:

omatcopy (Buffer Version)
-------------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void omatcopy(sycl::queue &queue,
                     oneapi::math::transpose trans,
                     std::int64_t m,
                     std::int64_t n,
                     T alpha,
                     sycl::buffer<T, 1> &a,
                     std::int64_t lda,
                     sycl::buffer<T, 1> &b,
                     std::int64_t ldb);
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void omatcopy(sycl::queue &queue,
                     oneapi::math::transpose trans,
                     std::int64_t m,
                     std::int64_t n,
                     T alpha,
                     sycl::buffer<T, 1> &a,
                     std::int64_t lda,
                     sycl::buffer<T, 1> &b,
                     std::int64_t ldb);
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
      ``lda`` * ``n`` for column-major and at least ``lda`` * ``m`` for row-major.

   lda
      Leading dimension of the matrix ``A``. If matrices are stored using
      column major layout, ``lda`` must be at least ``m``. If matrices are
      stored using row major layout, ``lda`` must be at least ``n``. Must be
      positive.

   b
      Buffer holding the output matrix ``B``.

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
           - ``ldb`` must be at least ``m``.
           - ``ldb`` must be at least ``n``.
         * - Row major
           - ``ldb`` must be at least ``n``.
           - ``ldb`` must be at least ``m``.

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
      

.. _onemath_blas_omatcopy_usm:
   
omatcopy (USM Version)
----------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event omatcopy(sycl::queue &queue,
                            oneapi::math::transpose trans,
                            std::int64_t m,
                            std::int64_t n,
                            value_or_pointer<T> alpha,
                            const T *a,
                            std::int64_t lda,
                            T *b,
                            std::int64_t ldb,
                            const std::vector<sycl::event> &dependencies = {});
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event omatcopy(sycl::queue &queue,
                            oneapi::math::transpose trans,
                            std::int64_t m,
                            std::int64_t n,
                            value_or_pointer<T> alpha,
                            const T *a,
                            std::int64_t lda,
                            T *b,
                            std::int64_t ldb,
                            const std::vector<sycl::event> &dependencies = {});
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine will be executed.

   trans
      Specifies op(``A``), the transposition operation applied to the
      matrix ``A``. See :ref:`onemath_datatypes` for more details.

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
      column major layout, ``lda`` must be at least ``m``. If matrices are
      stored using row major layout, ``lda`` must be at least ``n``. Must be
      positive.

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
      Leading dimension of the matrix ``B``. Must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``trans`` = ``transpose::conjtrans``
         * - Column major
           - Must be at least ``m``
           - Must be at least ``n``
         * - Row major
           - Must be at least ``n``
           - Must be at least ``m``

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

