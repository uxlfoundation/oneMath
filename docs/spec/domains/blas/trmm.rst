.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_trmm:

trmm
====

Computes a matrix-matrix product where one input matrix is triangular
and one input matrix is general.

.. _onemath_blas_trmm_description:

.. rubric:: Description

The ``trmm`` routines compute a scalar-matrix-matrix product where one of
the matrices in the multiplication is triangular. The argument
``left_right`` determines if the triangular matrix, ``A``, is on the
left of the multiplication (``left_right`` = ``side::left``) or on
the right (``left_right`` = ``side::right``). Depending on
``left_right``.

There are two operations available, an in-place operation and an out-of-place operation. The in-place operation is defined as:

.. math::

      B \leftarrow alpha*op(A)*B

or

.. math::

      B \leftarrow alpha*B*op(A)

The out-of-place operation is defined as:

.. math::

      C \leftarrow alpha*op(A)*B + beta*C

or

.. math::

      C \leftarrow alpha*B*op(A) + beta*C

where:

op(``A``) is one of op(``A``) = *A*, or op(``A``) = ``A``\ :sup:`T`,
or op(``A``) = ``A``\ :sup:`H`,

``alpha`` and ``beta`` are scalars,

``A`` is a triangular matrix, and ``B`` and ``C`` are general matrices.

Here ``B`` and ``C`` are  ``m`` x ``n`` and ``A`` is either ``m`` x ``m`` or
``n`` x ``n``, depending on ``left_right``.

``trmm`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_blas_trmm_buffer:

trmm (Buffer Version)
---------------------

**In-place API**
----------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void trmm(sycl::queue &queue,
                 oneapi::math::side left_right,
                 oneapi::math::uplo upper_lower,
                 oneapi::math::transpose transa,
                 oneapi::math::diag unit_diag,
                 std::int64_t m,
                 std::int64_t n,
                 T alpha,
                 sycl::buffer<T,1> &a,
                 std::int64_t lda,
                 sycl::buffer<T,1> &b,
                 std::int64_t ldb)
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void trmm(sycl::queue &queue,
                 oneapi::math::side left_right,
                 oneapi::math::uplo upper_lower,
                 oneapi::math::transpose transa,
                 oneapi::math::diag unit_diag,
                 std::int64_t m,
                 std::int64_t n,
                 T alpha,
                 sycl::buffer<T,1> &a,
                 std::int64_t lda,
                 sycl::buffer<T,1> &b,
                 std::int64_t ldb)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   left_right
      Specifies whether ``A`` is on the left side of the multiplication
      (``side::left``) or on the right side (``side::right``). See :ref:`onemath_datatypes` for more details.

   upper_lower
      Specifies whether the matrix ``A`` is upper or lower triangular. See :ref:`onemath_datatypes` for more details.

   trans
      Specifies op(``A``), the transposition operation applied to ``A``. See :ref:`onemath_datatypes` for more details.

   unit_diag
      Specifies whether ``A`` is assumed to be unit triangular (all
      diagonal elements are 1). See :ref:`onemath_datatypes` for more details.

   m
      Specifies the number of rows of ``B``. The value of ``m`` must be
      at least zero.

   n
      Specifies the number of columns of ``B``. The value of ``n`` must
      be at least zero.

   alpha
      Scaling factor for the matrix-matrix product.

   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``m`` if ``left_right`` = ``side::left``, or
      ``lda``\ \*\ ``n`` if ``left_right`` = ``side::right``. See
      :ref:`matrix-storage` for
      more details.

   lda
      Leading dimension of ``A``. Must be at least ``m`` if
      ``left_right`` = ``side::left``, and at least ``n`` if
      ``left_right`` = ``side::right``. Must be positive.

   b
      Buffer holding input/output matrix ``B``. Must have size at
      least ``ldb``\ \*\ ``n`` if column major layout is used to store
      matrices or at least ``ldb``\ \*\ ``m`` if row major layout is
      used to store matrices. See :ref:`matrix-storage` for more details.

   ldb
      Leading dimension of ``B``. It must be positive and at least
      ``m`` if column major layout is used to store matrices or at
      least ``n`` if row major layout is used to store matrices.

.. container:: section

   .. rubric:: Output Parameters

   b
      Output buffer, overwritten by ``alpha``\ \*op(``A``)\*\ ``B`` or
      ``alpha``\ \*\ ``B``\ \*op(``A``).

.. container:: section

   .. rubric:: Notes

   If ``alpha`` = 0, matrix ``B`` is set to zero, and ``A`` and ``B`` do
   not need to be initialized at entry.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`

**Out-of-place API**
--------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void trmm(sycl::queue &queue,
                 oneapi::math::side left_right,
                 oneapi::math::uplo upper_lower,
                 oneapi::math::transpose trans,
                 oneapi::math::diag unit_diag,
                 std::int64_t m,
                 std::int64_t n,
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
       void trmm(sycl::queue &queue,
                 oneapi::math::side left_right,
                 oneapi::math::uplo upper_lower,
                 oneapi::math::transpose trans,
                 oneapi::math::diag unit_diag,
                 std::int64_t m,
                 std::int64_t n,
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

   left_right
      Specifies whether ``A`` is on the left side of the multiplication
      (``side::left``) or on the right side (``side::right``). See :ref:`onemath_datatypes` for more details.

   upper_lower
      Specifies whether the matrix ``A`` is upper or lower triangular. See :ref:`onemath_datatypes` for more details.

   trans
      Specifies op(``A``), the transposition operation applied to matrix ``A``. See :ref:`onemath_datatypes` for more details.

   unit_diag
      Specifies whether ``A`` is assumed to be unit triangular (all diagonal elements are 1). See :ref:`onemath_datatypes` for more details.

   m
      Specifices the number of rows of ``B``. The value of ``m`` must be at least zero.

   n
      Specifies the Number of columns of ``B``. The value of ``n`` must be at least zero.

   alpha
      Scaling factor for matrix-matrix product.

   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``m`` if ``left_right`` = ``side::left`` or ``lda``\ \*\ ``n`` if ``left_right`` = ``side::right``. See :ref:`matrix-storage` for more details.

   lda
      Leading dimension of ``A``. Must be at least ``m`` if ``left_right`` = ``side::left`` or at least ``n`` if ``left_right`` = ``side::right``. Must be positive.

   b
      Buffer holding input matrix ``B``. Must have size at least
      ``ldb``\ \*\ ``n`` if column major layout or at least ``ldb``\ \*\ ``m`` if row major layout is used. See :ref:`matrix-storage` for more details.

   ldb
      Leading dimension of matrix ``B``. It must be positive and at least ``m`` if column major layout or at least ``n`` if row major layout is used.

   beta
      Scaling factor for matrix ``C``.

   c
      Buffer holding input/output matrix ``C``. Size of the buffer must be at least
      ``ldc``\ \*\ ``n`` if column major layout or at least ``ldc``\ \*\ ``m`` if row major layout is used. See :ref:`matrix-storage` for more details.

   ldc
      Leading dimension of matrix ``C``. Must be at least ``m`` if column major layout or at least ``n`` if row major layout is used. Must be positive.

.. container:: section

   .. rubric:: Output Parameters

   c
      Output buffer overwritten by ``alpha``\ \*\ op(``A``)\ \*\ ``B`` + ``beta``\ \*\ ``C`` if ``left_right`` = ``side::left`` or ``alpha``\ \*\ ``B``\ \*\ op(``A``) + ``beta``\ \*\ ``C`` if ``left_right`` = ``side::right``.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`

        
.. _onemath_blas_trmm_usm:

trmm (USM Version)
------------------

**In-place API**
----------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event trmm(sycl::queue &queue,
                        oneapi::math::side left_right,
                        oneapi::math::uplo upper_lower,
                        oneapi::math::transpose transa,
                        oneapi::math::diag unit_diag,
                        std::int64_t m,
                        std::int64_t n,
                        value_or_pointer<T> alpha,
                        const T *a,
                        std::int64_t lda,
                        T *b,
                        std::int64_t ldb,
                        const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event trmm(sycl::queue &queue,
                        oneapi::math::side left_right,
                        oneapi::math::uplo upper_lower,
                        oneapi::math::transpose transa,
                        oneapi::math::diag unit_diag,
                        std::int64_t m,
                        std::int64_t n,
                        value_or_pointer<T> alpha,
                        const T *a,
                        std::int64_t lda,
                        T *b,
                        std::int64_t ldb,
                        const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section
   
   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   left_right
      Specifies whether ``A`` is on the left side of the
      multiplication (``side::left``) or on the right side
      (``side::right``). See :ref:`onemath_datatypes` for more details.

   upper_lower
      Specifies whether the matrix ``A`` is upper or lower
      triangular. See :ref:`onemath_datatypes` for more details.

   trans
      Specifies op(``A``), the transposition operation applied to
      ``A``. See :ref:`onemath_datatypes` for more details.

   unit_diag
      Specifies whether ``A`` is assumed to be unit triangular (all
      diagonal elements are 1). See :ref:`onemath_datatypes` for more details.

   m
      Specifies the number of rows of ``B``. The value of ``m`` must
      be at least zero.

   n
      Specifies the number of columns of ``B``. The value of ``n``
      must be at least zero.

   alpha
      Scaling factor for the matrix-matrix product. See :ref:`value_or_pointer` for more details.

   a
      Pointer to input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``m`` if ``left_right`` = ``side::left``, or
      ``lda``\ \*\ ``n`` if ``left_right`` = ``side::right``. See
      :ref:`matrix-storage` for
      more details.

   lda
      Leading dimension of ``A``. Must be at least ``m`` if
      ``left_right`` = ``side::left``, and at least ``n`` if
      ``left_right`` = ``side::right``. Must be positive.

   b
      Pointer to input/output matrix ``B``. Must have size at
      least ``ldb``\ \*\ ``n`` if column major layout is used to store
      matrices or at least ``ldb``\ \*\ ``m`` if row major layout is
      used to store matrices. See :ref:`matrix-storage` for more details.

   ldb
      Leading dimension of ``B``. It must be positive and at least
      ``m`` if column major layout is used to store matrices or at
      least ``n`` if row major layout is used to store matrices.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   b
      Pointer to the output matrix, overwritten by
      ``alpha``\ \*op(``A``)\*\ ``B`` or
      ``alpha``\ \*\ ``B``\ \*op(``A``).

.. container:: section

   .. rubric:: Notes

   If ``alpha`` = 0, matrix ``B`` is set to zero, and ``A`` and ``B``
   do not need to be initialized at entry.

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
      
**Out-of-place API**
--------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void trmm(sycl::queue &queue,
                 oneapi::math::side left_right,
                 oneapi::math::uplo upper_lower,
                 oneapi::math::transpose trans,
                 oneapi::math::diag unit_diag,
                 std::int64_t m,
                 std::int64_t n,
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
       void trmm(sycl::queue &queue,
                 oneapi::math::side left_right,
                 oneapi::math::uplo upper_lower,
                 oneapi::math::transpose trans,
                 oneapi::math::diag unit_diag,
                 std::int64_t m,
                 std::int64_t n,
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

   left_right
      Specifies whether ``A`` is on the left side of the multiplication
      (``side::left``) or on the right side (``side::right``). See :ref:`onemath_datatypes` for more details.

   upper_lower
      Specifies whether the matrix ``A`` is upper or lower triangular. See :ref:`onemath_datatypes` for more details.

   trans
      Specifies op(``A``), the transposition operation applied to matrix ``A``. See :ref:`onemath_datatypes` for more details.

   unit_diag
      Specifies whether ``A`` is assumed to be unit triangular (all diagonal elements are 1). See :ref:`onemath_datatypes` for more details.

   m
      Specifices the number of rows of ``B``. The value of ``m`` must be at least zero.

   n
      Specifies the Number of columns of ``B``. The value of ``n`` must be at least zero.

   alpha
      Scaling factor for matrix-matrix product. See :ref:`value_or_pointer` for more details.

   a
      Pointer to input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``m`` if ``left_right`` = ``side::left`` or ``lda``\ \*\ ``n`` if ``left_right`` = ``side::right``. See :ref:`matrix-storage` for more details.

   lda
      Leading dimension of ``A``. Must be at least ``m`` if ``left_right`` = ``side::left`` or at least ``n`` if ``left_right`` = ``side::right``. Must be positive.

   b
      Pointer to input matrix ``B``. Must have size at least
      ``ldb``\ \*\ ``n`` if column major layout or at least ``ldb``\ \*\ ``m`` if row major layout is used. See :ref:`matrix-storage` for more details.

   ldb
      Leading dimension of matrix ``B``. It must be positive and at least ``m`` if column major layout or at least ``n`` if row major layout is used.

   beta
      Scaling factor for matrix ``C``. See :ref:`value_or_pointer` for more details.

   c
      Pointer to input/output matrix ``C``. Must have size at least
      ``ldc``\ \*\ ``n`` if column major layout or at least ``ldc``\ \*\ ``m`` if row major layout is used. See :ref:`matrix-storage` for more details.

   ldc
      Leading dimension of matrix ``C``. Must be at least ``m`` if column major layout or at least ``n`` if row major layout is used. Must be positive.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   c
      Pointer to the output matrix, overwritten by ``alpha``\ \*\ op(``A``)\ \*\ ``B`` + ``beta``\ \*\ ``C`` if ``left_right`` = ``side::left`` or ``alpha``\ \*\ ``B``\ \*\ op(``A``) + ``beta``\ \*\ ``C`` if ``left_right`` = ``side::right``.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`

   **Parent topic:** :ref:`blas-level-3-routines`
