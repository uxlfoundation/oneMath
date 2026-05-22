.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_gemmt:

gemmt
=====

.. warning:: gemmt has been deprecated. Use level 3 :ref:`onemath_blas_gemmtr` instead.

Computes a matrix-matrix product with general matrices, but updates
only the upper or lower triangular part of the result matrix.

.. _onemath_blas_gemmt_description:

.. rubric:: Description

The gemmt routines compute a scalar-matrix-matrix product and add
the result to the upper or lower part of a scalar-matrix product,
with general matrices. The operation is defined as:

.. math::

      C \leftarrow alpha*op(A)*op(B) + beta*C 

where:

op(``X``) is one of op(``X``) = ``X``, or op(``X``) = ``X``\ :sup:`T`, or
op(``X``) = ``X``\ :sup:`H`,

``alpha`` and ``beta`` are scalars

``A``, ``B``, and ``C`` are matrices

op(``A``) is ``n`` x ``k``, op(``B``) is ``k`` x ``n``, and
``C`` is ``n`` x ``n``.

``gemmt`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_blas_gemmt_buffer:

gemmt (Buffer Version)
----------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void gemmt(sycl::queue &queue,
                  oneapi::math::uplo upper_lower,
                  oneapi::math::transpose transa,
                  oneapi::math::transpose transb,
                  std::int64_t n,
                  std::int64_t k,
                  T alpha,
                  sycl::buffer<T,1> &a,
                  std::int64_t lda,
                  sycl::buffer<T,1> &b,
                  std::int64_t ldb,
                  T beta,
                  sycl::buffer<T,1> &c,
                  std::int64_t ldc)
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void gemmt(sycl::queue &queue,
                  oneapi::math::uplo upper_lower,
                  oneapi::math::transpose transa,
                  oneapi::math::transpose transb,
                  std::int64_t n,
                  std::int64_t k,
                  T alpha,
                  sycl::buffer<T,1> &a,
                  std::int64_t lda,
                  sycl::buffer<T,1> &b,
                  std::int64_t ldb,
                  T beta,
                  sycl::buffer<T,1> &c,
                  std::int64_t ldc)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether ``C``\ ’s data is stored in its upper or
      lower triangle. See :ref:`onemath_datatypes` for more details.
   
   transa
      Specifies op(``A``), the transposition operation applied to
      ``A``. See :ref:`onemath_datatypes` for more details.

   transb
      Specifies op(``B``), the transposition operation applied to
      ``B``. See :ref:`onemath_datatypes` for more details.

   n
      Number of rows of op(``A``), columns of op(``B``), and
      columns and rows of\ ``C``. Must be at least zero.

   k
      Number of columns of op(``A``) and rows of op(``B``). Must be
      at least zero.

   alpha
      Scaling factor for the matrix-matrix product.

   a
      Buffer holding the input matrix ``A``.

      .. list-table::
         :header-rows: 1

         * -
           - ``A`` not transposed
           - ``A`` transposed
         * - Column major
           - ``A`` is an ``n``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``k``.
           - ``A`` is an ``k``-by-``n`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``n``
         * - Row major
           - ``A`` is an ``n``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``n``.
           - ``A`` is an ``k``-by-``n`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``k``.
      
      See :ref:`matrix-storage` for more details.

   lda
      The leading dimension of ``A``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``A`` not transposed
           - ``A`` transposed
         * - Column major
           - ``lda`` must be at least ``n``.
           - ``lda`` must be at least ``k``.
         * - Row major
           - ``lda`` must be at least ``k``.
           - ``lda`` must be at least ``n``.

   b
      Buffer holding the input matrix ``B``.
      
      .. list-table::
         :header-rows: 1

         * -
           - ``B`` not transposed
           - ``B`` transposed
         * - Column major
           - ``B`` is an ``k``-by-``n`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``n``.
           - ``B`` is an ``n``-by-``k`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``k``
         * - Row major
           - ``B`` is an ``k``-by-``n`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``k``.
           - ``B`` is an ``n``-by-``k`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``n``.
   
      See :ref:`matrix-storage` for more details.

   ldb
      The leading dimension of ``B``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``B`` not transposed
           - ``B`` transposed
         * - Column major
           - ``ldb`` must be at least ``k``.
           - ``ldb`` must be at least ``n``.
         * - Row major
           - ``ldb`` must be at least ``n``.
           - ``ldb`` must be at least ``k``.

   beta
      Scaling factor for matrix ``C``.

   c
      Buffer holding the input/output matrix ``C``. Must have size at
      least ``ldc`` \* ``n``. See :ref:`matrix-storage` for
      more details.

   ldc
      Leading dimension of ``C``. Must be positive and at least
      ``m``.

.. container:: section

   .. rubric:: Output Parameters

   c
      Output buffer, overwritten by the upper or lower triangular
      part of ``alpha`` * op(``A``)*op(``B``) + ``beta`` * ``C``.

.. container:: section

   .. rubric:: Notes

   If ``beta`` = 0, matrix ``C`` does not need to be initialized
   before calling gemmt.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

.. _onemath_blas_gemmt_usm:

gemmt (USM Version)
-------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event gemmt(sycl::queue &queue,
                         oneapi::math::uplo upper_lower,
                         oneapi::math::transpose transa,
                         oneapi::math::transpose transb,
                         std::int64_t n,
                         std::int64_t k,
                         value_or_pointer<T> alpha,
                         const T *a,
                         std::int64_t lda,
                         const T *b,
                         std::int64_t ldb,
                         value_or_pointer<T> beta,
                         T *c,
                         std::int64_t ldc,
                         const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event gemmt(sycl::queue &queue,
                         oneapi::math::uplo upper_lower,
                         oneapi::math::transpose transa,
                         oneapi::math::transpose transb,
                         std::int64_t n,
                         std::int64_t k,
                         value_or_pointer<T> alpha,
                         const T *a,
                         std::int64_t lda,
                         const T *b,
                         std::int64_t ldb,
                         value_or_pointer<T> beta,
                         T *c,
                         std::int64_t ldc,
                         const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether ``C``\ ’s data is stored in its upper or
      lower triangle. See
      :ref:`onemath_datatypes` for
      more details.

   transa
      Specifies op(``A``), the transposition operation applied to
      ``A``. See
      :ref:`onemath_datatypes` for
      more details.

   transb
      Specifies op(``B``), the transposition operation applied to
      ``B``. See
      :ref:`onemath_datatypes` for
      more details.

   n
      Number of columns of op(``A``), columns of op(``B``), and
      columns of\ ``C``. Must be at least zero.

   k
      Number of columns of op(``A``) and rows of op(``B``). Must be
      at least zero.

   alpha
      Scaling factor for the matrix-matrix product. See :ref:`value_or_pointer` for more details.

   a
      Pointer to input matrix ``A``.

      .. list-table::
         :header-rows: 1

         * -
           - ``A`` not transposed
           - ``A`` transposed
         * - Column major
           - ``A`` is an ``n``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``k``.
           - ``A`` is an ``k``-by-``n`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``n``
         * - Row major
           - ``A`` is an ``n``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``n``.
           - ``A`` is an ``k``-by-``n`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``k``

      See :ref:`matrix-storage` for more details.

   lda
      The leading dimension of ``A``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``A`` not transposed
           - ``A`` transposed
         * - Column major
           - ``lda`` must be at least ``n``.
           - ``lda`` must be at least ``k``.
         * - Row major
           - ``lda`` must be at least ``k``.
           - ``lda`` must be at least ``n``.

   b
      Pointer to input matrix ``B``.

      .. list-table::
         :header-rows: 1

         * -
           - ``B`` not transposed
           - ``B`` transposed
         * - Column major
           - ``B`` is an ``k``-by-``n`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``n``.
           - ``B`` is an ``n``-by-``k`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``k``
         * - Row major
           - ``B`` is an ``k``-by-``n`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``k``.
           - ``B`` is an ``n``-by-``k`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``n``

      See :ref:`matrix-storage` for more details.

   ldb
      The leading dimension of ``B``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``B`` not transposed
           - ``B`` transposed
         * - Column major
           - ``ldb`` must be at least ``k``.
           - ``ldb`` must be at least ``n``.
         * - Row major
           - ``ldb`` must be at least ``n``.
           - ``ldb`` must be at least ``k``.
      
   beta
      Scaling factor for matrix ``C``. See :ref:`value_or_pointer` for more details.

   c
      Pointer to input/output matrix ``C``. Must have size at least
      ``ldc`` \* ``n``. See :ref:`matrix-storage` for
      more details.

   ldc
      Leading dimension of ``C``. Must be positive and at least
      ``m``.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   c
      Pointer to the output matrix, overwritten by the upper or lower
      triangular part of ``alpha`` * op(``A``)*op(``B``) + ``beta`` * ``C``.

.. container:: section

   .. rubric:: Notes

   If ``beta`` = 0, matrix ``C`` does not need to be initialized
   before calling gemmt.

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
