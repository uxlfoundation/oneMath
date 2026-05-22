.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_gemm:

gemm
====

Computes a matrix-matrix product with general matrices.

.. _onemath_blas_gemm_description:

.. rubric:: Description

The ``gemm`` routines compute a scalar-matrix-matrix product and add the
result to a scalar-matrix product, with general matrices. The
operation is defined as:

.. math::

      C \leftarrow alpha*op(A)*op(B) + beta*C

where:

op(``X``) is one of op(``X``) = ``X``, or op(``X``) = ``X``\ :sup:`T`, or
op(``X``) = ``X``\ :sup:`H`,

``alpha`` and ``beta`` are scalars,

``A``, ``B`` and ``C`` are matrices,

``op(A)`` is an ``m``-by-``k`` matrix,

``op(B)`` is a ``k``-by-``n`` matrix,

``C`` is an ``m``-by-``n`` matrix.

``gemm`` supports the following precisions.

   .. list-table:: 
     :header-rows: 1

     * - | Ta
         | (A matrix) 
       - | Tb
         | (B matrix) 
       - | Tc 
         | (C matrix) 
       - | Ts
         | (alpha/beta) 
     * -  ``std::int8_t`` 
       -  ``std::int8_t`` 
       -  ``std::int32_t`` 
       -  ``float`` 
     * -  ``std::int8_t`` 
       -  ``std::int8_t`` 
       -  ``float`` 
       -  ``float`` 
     * -  ``half`` 
       -  ``half`` 
       -  ``float`` 
       -  ``float`` 
     * -  ``half`` 
       -  ``half`` 
       -  ``half`` 
       -  ``half`` 
     * -  ``bfloat16`` 
       -  ``bfloat16`` 
       -  ``float`` 
       -  ``float`` 
     * -  ``bfloat16`` 
       -  ``bfloat16`` 
       -  ``bfloat16`` 
       -  ``float`` 
     * -  ``float`` 
       -  ``float`` 
       -  ``float`` 
       -  ``float`` 
     * -  ``double`` 
       -  ``double`` 
       -  ``double`` 
       -  ``double`` 
     * -  ``std::complex<float>`` 
       -  ``std::complex<float>`` 
       -  ``std::complex<float>`` 
       -  ``std::complex<float>`` 
     * -  ``std::complex<double>`` 
       -  ``std::complex<double>`` 
       -  ``std::complex<double>`` 
       -  ``std::complex<double>`` 

.. _onemath_blas_gemm_buffer:

gemm (Buffer Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void gemm(sycl::queue &queue,
                 oneapi::math::transpose transa,
                 oneapi::math::transpose transb,
                 std::int64_t m,
                 std::int64_t n,
                 std::int64_t k,
                 Ts alpha,
                 sycl::buffer<Ta,1> &a,
                 std::int64_t lda,
                 sycl::buffer<Tb,1> &b,
                 std::int64_t ldb,
                 Ts beta,
                 sycl::buffer<Tc,1> &c,
                 std::int64_t ldc)
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void gemm(sycl::queue &queue,
                 oneapi::math::transpose transa,
                 oneapi::math::transpose transb,
                 std::int64_t m,
                 std::int64_t n,
                 std::int64_t k,
                 Ts alpha,
                 sycl::buffer<Ta,1> &a,
                 std::int64_t lda,
                 sycl::buffer<Tb,1> &b,
                 std::int64_t ldb,
                 Ts beta,
                 sycl::buffer<Tc,1> &c,
                 std::int64_t ldc)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   transa
      Specifies the form of op(``A``), the transposition operation
      applied to ``A``.

   transb
      Specifies the form of op(``B``), the transposition operation
      applied to ``B``.

   m
      Specifies the number of rows of the matrix op(``A``) and of the
      matrix ``C``. The value of m must be at least zero.

   n
      Specifies the number of columns of the matrix op(``B``) and the
      number of columns of the matrix ``C``. The value of n must be at
      least zero.

   k
      Specifies the number of columns of the matrix op(``A``) and the
      number of rows of the matrix op(``B``). The value of k must be at
      least zero.

   alpha
      Scaling factor for the matrix-matrix product.

   a
      The buffer holding the input matrix ``A``.

      .. list-table::
         :header-rows: 1

         * -
           - ``A`` not transposed
           - ``A`` transposed
         * - Column major
           - ``A`` is an ``m``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``k``.
           - ``A`` is an ``k``-by-``m`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``m``
         * - Row major
           - ``A`` is an ``m``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``m``.
           - ``A`` is an ``k``-by-``m`` matrix so the array ``a``
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
           - ``lda`` must be at least ``m``.
           - ``lda`` must be at least ``k``.
         * - Row major
           - ``lda`` must be at least ``k``.
           - ``lda`` must be at least ``m``.
             
   b
      The buffer holding the input matrix ``B``.

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
      Scaling factor for matrix ``C``.

   c
      The buffer holding the input/output matrix ``C``. It must have a
      size of at least ``ldc``\ \*\ ``n`` if column major layout is
      used to store matrices or at least ``ldc``\ \*\ ``m`` if row
      major layout is used to store matrices . See :ref:`matrix-storage` for more details.

   ldc
      The leading dimension of ``C``. It must be positive and at least
      ``m`` if column major layout is used to store matrices or at
      least ``n`` if row major layout is used to store matrices.

.. container:: section

   .. rubric:: Output Parameters

   c
      The buffer, which is overwritten by
      ``alpha``\ \*\ op(``A``)*op(``B``) + ``beta``\ \*\ ``C``.

.. container:: section

   .. rubric:: Notes

   If ``beta`` = 0, matrix ``C`` does not need to be initialized before
   calling ``gemm``.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

.. _onemath_blas_gemm_usm:

gemm (USM Version)
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event gemm(sycl::queue &queue,
                        oneapi::math::transpose transa,
                        oneapi::math::transpose transb,
                        std::int64_t m,
                        std::int64_t n,
                        std::int64_t k,
                        value_or_pointer<Ts> alpha,
                        const Ta *a,
                        std::int64_t lda,
                        const Tb *b,
                        std::int64_t ldb,
                        value_or_poitner<Ts> beta,
                        Tc *c,
                        std::int64_t ldc,
                        const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event gemm(sycl::queue &queue,
                        oneapi::math::transpose transa,
                        oneapi::math::transpose transb,
                        std::int64_t m,
                        std::int64_t n,
                        std::int64_t k,
                        value_or_pointer<Ts> alpha,
                        const Ta *a,
                        std::int64_t lda,
                        const Tb *b,
                        std::int64_t ldb,
                        value_or_pointer<Ts> beta,
                        Tc *c,
                        std::int64_t ldc,
                        const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   transa
      Specifies the form of op(``A``), the transposition operation
      applied to ``A``.


   transb
      Specifies the form of op(``B``), the transposition operation
      applied to ``B``.


   m
      Specifies the number of rows of the matrix op(``A``) and of the
      matrix ``C``. The value of m must be at least zero.


   n
      Specifies the number of columns of the matrix op(``B``) and the
      number of columns of the matrix ``C``. The value of n must be
      at least zero.


   k
      Specifies the number of columns of the matrix op(``A``) and the
      number of rows of the matrix op(``B``). The value of k must be
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
           - ``A`` is an ``m``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``k``.
           - ``A`` is an ``k``-by-``m`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``m``
         * - Row major
           - ``A`` is an ``m``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``m``.
           - ``A`` is an ``k``-by-``m`` matrix so the array ``a``
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
           - ``lda`` must be at least ``m``.
           - ``lda`` must be at least ``k``.
         * - Row major
           - ``lda`` must be at least ``k``.
           - ``lda`` must be at least ``m``.
             
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
      The pointer to input/output matrix ``C``. It must have a
      size of at least ``ldc``\ \*\ ``n`` if column major layout is
      used to store matrices or at least ``ldc``\ \*\ ``m`` if row
      major layout is used to store matrices . See :ref:`matrix-storage` for more details.

   ldc
      The leading dimension of ``C``. It must be positive and at least
      ``m`` if column major layout is used to store matrices or at
      least ``n`` if row major layout is used to store matrices.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   c
      Pointer to the output matrix, overwritten by
      ``alpha``\ \*\ op(``A``)*op(``B``) + ``beta``\ \*\ ``C``.
 
.. container:: section

   .. rubric:: Notes

   If ``beta`` = 0, matrix ``C`` does not need to be initialized
   before calling ``gemm``.

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
      

   **Parent topic:** :ref:`blas-level-3-routines`
