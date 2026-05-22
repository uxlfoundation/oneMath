.. SPDX-FileCopyrightText: 2022 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_omatadd:

omatadd
=======

Computes a sum of two general dense matrices, with optional transposes.

.. _onemath_blas_omatadd_description:

.. rubric:: Description

The ``omatadd`` routine performs an out-of-place scaled
matrix addition with optional transposes in the arguments.
The operation is defined as:

.. math::

      C \leftarrow \alpha * op(A) + \beta * op(B)

where:

op(``X``) is one of op(``X``) = ``X``, or op(``X``) = ``X``\ :sup:`T`, or op(``X``) = ``X``\ :sup:`H`

``alpha`` and ``beta`` are scalars,

``A`` and ``B`` are input matrices while ``C`` is an output matrix,

``C`` is ``m`` x ``n``,

``A`` is ``m`` x ``n`` if the ``op(A)`` is not transposed or ``n`` by ``m`` if it is,

and ``B`` is ``m`` x ``n`` if the ``op(B)`` is not transposed or ``n`` by ``m`` if it is.

In general, ``A``, ``B``, and ``C`` should not overlap in memory, with the exception of
the following in-place operations:

   - ``A`` and ``C`` may point to the same memory if ``op(A)`` is non-transpose and ``lda`` = ``ldc``;

   - ``B`` and ``C`` may point to the same memory if ``op(B)`` is non-transpose and ``ldb`` = ``ldc``.


``omatadd`` supports the following precisions:

   .. list-table::
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_blas_omatadd_buffer:

omatadd (Buffer Version)
------------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void omatadd(sycl::queue &queue,
                    oneapi::math::transpose transa,
                    oneapi::math::transpose transb,
                    std::int64_t m,
                    std::int64_t n,
                    T alpha,
                    sycl::buffer<T, 1> &a,
                    std::int64_t lda,
                    T beta,
                    sycl::buffer<T, 1> &b,
                    std::int64_t ldb,
                    sycl::buffer<T, 1> &c,
                    std::int64_t ldc)
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void omatadd(sycl::queue &queue,
                    oneapi::math::transpose transa,
                    oneapi::math::transpose transb,
                    std::int64_t m,
                    std::int64_t n,
                    T alpha,
                    sycl::buffer<T, 1> &a,
                    std::int64_t lda,
                    T beta,
                    sycl::buffer<T, 1> &b,
                    std::int64_t ldb,
                    sycl::buffer<T, 1> &c,
                    std::int64_t ldc)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   transa
      Specifies op(``A``), the transposition operation applied to the
      matrix ``A``. See :ref:`onemath_datatypes` for more details.

   transb
      Specifies op(``B``), the transposition operation applied to the
      matrix ``B``. See :ref:`onemath_datatypes` for more details.

   m
      Number of rows for the result matrix ``C``. Must be at least zero.

   n
      Number of columns for the result matrix ``C``. Must be at least zero.

   alpha
      Scaling factor for the matrix ``A``.

   a
      Buffer holding the input matrix ``A``.

      .. list-table::
         :header-rows: 1

         * -
           - ``transa`` = ``transpose::nontrans``
           - ``transa`` = ``transpose::trans`` or ``transa`` = ``transpose::conjtrans``
         * - Column major
           - ``A`` is ``m`` x ``n`` matrix. Size of array ``a`` must be at least ``lda`` * ``n``
           - ``A`` is ``n`` x ``m`` matrix. Size of array ``a`` must be at least ``lda`` * ``m``
         * - Row major
           - ``A`` is ``m`` x ``n`` matrix. Size of array ``a`` must be at least ``lda`` * ``m``
           - ``A`` is ``n`` x ``m`` matrix. Size of array ``a`` must be at least ``lda`` * ``n``

   lda
      The leading dimension of the matrix ``A``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``transa`` = ``transpose::nontrans``
           - ``transa`` = ``transpose::trans`` or ``transa`` = ``transpose::conjtrans``
         * - Column major
           - ``lda`` must be at least ``m``.
           - ``lda`` must be at least ``n``.
         * - Row major
           - ``lda`` must be at least ``n``.
           - ``lda`` must be at least ``m``.

   beta
      Scaling factor for the matrix ``B``.

   b
      Buffer holding the input matrix ``B``. Must have size at least:

      .. list-table::
         :header-rows: 1
     
         * -
           - ``transb`` = ``transpose::nontrans``
           - ``transb`` = ``transpose::trans`` or ``transb`` = ``transpose::conjtrans``
         * - Column major
           - ``B`` is ``m`` x ``n`` matrix. Size of array ``b`` must be at least ``ldb`` * ``n``
           - ``B`` is ``n`` x ``m`` matrix. Size of array ``b`` must be at least ``ldb`` * ``m``
         * - Row major
           - ``B`` is ``m`` x ``n`` matrix. Size of array ``b`` must be at least ``ldb`` * ``m``
           - ``B`` is ``n`` x ``m`` matrix. Size of array ``b`` must be at least ``ldb`` * ``n``

   ldb
      The leading dimension of the ``B`` matrix. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``transb`` = ``transpose::nontrans``
           - ``transb`` = ``transpose::trans`` or ``transb`` = ``transpose::conjtrans``
         * - Column major
           - ``ldb`` must be at least ``m``.
           - ``ldb`` must be at least ``n``.
         * - Row major
           - ``ldb`` must be at least ``n``.
           - ``ldb`` must be at least ``m``.

   c
      Buffer holding the output matrix ``C``.

      .. list-table::

         * - Column major
           - ``C`` is ``m`` x ``n`` matrix. Size of array ``c`` must be at least ``ldc`` * ``n``
         * - Row major
           - ``C`` is ``m`` x ``n`` matrix. Size of array ``c`` must be at least ``ldc`` * ``m``

   ldc
      Leading dimension of the C matrices. If matrices are stored using
      column major layout, ``ldc`` must be at least ``m``. If matrices are
      stored using row major layout, ``ldc`` must be at least ``n``. Must be
      positive.

.. container:: section

   .. rubric:: Output Parameters

   c
      Output buffer overwritten by ``alpha`` * op(``A``) + ``beta`` * op(``B``).

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
      

.. _onemath_blas_omatadd_usm:
   
omatadd (USM Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event omatadd(sycl::queue &queue,
                           oneapi::math::transpose transa,
                           oneapi::math::transpose transb,
                           std::int64_t m,
                           std::int64_t n,
                           value_or_pointer<T> alpha,
                           const T *a,
                           std::int64_t lda,
                           value_or_pointer<T> beta,
                           const T *b,
                           std::int64_t ldb,
                           T *c,
                           std::int64_t ldc,
                           const std::vector<sycl::event> &dependencies = {});
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event omatadd(sycl::queue &queue,
                           oneapi::math::transpose transa,
                           oneapi::math::transpose transb,
                           std::int64_t m,
                           std::int64_t n,
                           value_or_pointer<T> alpha,
                           const T *a,
                           std::int64_t lda,
                           value_or_pointer<T> beta,
                           const T *b,
                           std::int64_t ldb,
                           T *c,
                           std::int64_t ldc,
                           const std::vector<sycl::event> &dependencies = {});
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   transa
      Specifies op(``A``), the transposition operation applied to the
      matrix ``A``. See :ref:`onemath_datatypes` for more details.

   transb
      Specifies op(``B``), the transposition operation applied to the
      matrix ``B``. See :ref:`onemath_datatypes` for more details.

   m
      Number of rows for the result matrix ``C``. Must be at least zero.

   n
      Number of columns for the result matrix ``C``. Must be at least zero.

   alpha
      Scaling factor for the matrix ``A``. See :ref:`value_or_pointer` for more details.

   a
      Array holding the input matrix ``A``.

      .. list-table::
         :header-rows: 1

         * -
           - ``transa`` = ``transpose::nontrans``
           - ``transa`` = ``transpose::trans`` or ``transa`` = ``transpose::conjtrans``
         * - Column major
           - ``A`` is ``m`` x ``n`` matrix. Size of array ``a`` must be at least ``lda`` * ``n``
           - ``A`` is ``n`` x ``m`` matrix. Size of array ``a`` must be at least ``lda`` * ``m``
         * - Row major
           - ``A`` is ``m`` x ``n`` matrix. Size of array ``a`` must be at least ``lda`` * ``m``
           - ``A`` is ``n`` x ``m`` matrix. Size of array ``a`` must be at least ``lda`` * ``n``

   lda
      The leading dimension of the matrix ``A``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``transa`` = ``transpose::nontrans``
           - ``transa`` = ``transpose::trans`` or ``transa`` = ``transpose::conjtrans``
         * - Column major
           - ``lda`` must be at least ``m``.
           - ``lda`` must be at least ``n``.
         * - Row major
           - ``lda`` must be at least ``n``.
           - ``lda`` must be at least ``m``.

   beta
      Scaling factor for the matrices ``B``. See :ref:`value_or_pointer` for more details.

   b
      Array holding the input matrices ``B``.

      .. list-table::
         :header-rows: 1
     
         * -
           - ``transb`` = ``transpose::nontrans``
           - ``transb`` = ``transpose::trans`` or ``transb`` = ``transpose::conjtrans``
         * - Column major
           - ``B`` is ``m`` x ``n`` matrix. Size of array ``b`` must be at least ``ldb`` * ``n``
           - ``B`` is ``n`` x ``m`` matrix. Size of array ``b`` must be at least ``ldb`` * ``m``
         * - Row major
           - ``B`` is ``m`` x ``n`` matrix. Size of array ``b`` must be at least ``ldb`` * ``m``
           - ``B`` is ``n`` x ``m`` matrix. Size of array ``b`` must be at least ``ldb`` * ``n``

   ldb
      The leading dimension of the ``B`` matrix. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``transb`` = ``transpose::nontrans``
           - ``transb`` = ``transpose::trans`` or ``transb`` = ``transpose::conjtrans``
         * - Column major
           - ``ldb`` must be at least ``m``.
           - ``ldb`` must be at least ``n``.
         * - Row major
           - ``ldb`` must be at least ``n``.
           - ``ldb`` must be at least ``m``.

   c
      Array holding the output matrix ``C``.

      .. list-table::

         * - Column major
           - ``C`` is ``m`` x ``n`` matrix. Size of array ``c`` must be at least ``ldc`` * ``n``
         * - Row major
           - ``C`` is ``m`` x ``n`` matrix. Size of array ``c`` must be at least ``ldc`` * ``m``

   ldc
      Leading dimension of the ``C`` matrix. If matrices are stored using
      column major layout, ``ldc`` must be at least ``m``. If matrices are
      stored using row major layout, ``ldc`` must be at least ``n``. Must be
      positive.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   c
      Output array, overwritten by ``alpha`` * op(``A``) + ``beta`` * op(``B``).

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

