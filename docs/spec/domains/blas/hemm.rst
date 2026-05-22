.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_hemm:

hemm
====

Computes a matrix-matrix product where one input matrix is Hermitian
and one is general.

.. _onemath_blas_hemm_description:

.. rubric:: Description

The ``hemm`` routines compute a scalar-matrix-matrix product and add the
result to a scalar-matrix product, where one of the matrices in the
multiplication is Hermitian. The argument ``left_right`` determines
if the Hermitian matrix, ``A``, is on the left of the multiplication
(``left_right`` = ``side::left``) or on the right (``left_right`` =
``side::right``). Depending on ``left_right``, the operation is
defined as:

.. math::

      C \leftarrow alpha*A*B + beta*C

or

.. math::

      C \leftarrow alpha*B*A + beta*C

where:

``alpha`` and ``beta`` are scalars,

``A`` is a Hermitian matrix, either ``m``-by-``m`` or ``n``-by-``n``
matrices,

``B`` and ``C`` are ``m``-by-``n`` matrices.

``hemm`` supports the following precisions:

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_blas_hemm_buffer:

hemm (Buffer Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void hemm(sycl::queue &queue,
                 oneapi::math::side left_right,
                 oneapi::math::uplo upper_lower,
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
       void hemm(sycl::queue &queue,
                 oneapi::math::side left_right,
                 oneapi::math::uplo upper_lower,
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

   uplo
      Specifies whether ``A``'s data is stored in its upper or lower
      triangle. See :ref:`onemath_datatypes` for more details.

   m
      Specifies the number of rows of the matrix ``B`` and ``C``.

      The value of ``m`` must be at least zero.

   n
      Specifies the number of columns of the matrix ``B`` and ``C``.

      The value of ``n`` must be at least zero.

   alpha
      Scaling factor for the matrix-matrix product.

   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``m`` if ``A`` is on the left of the multiplication,
      or ``lda``\ \*\ ``n`` if ``A`` is on the right. See :ref:`matrix-storage`
      for more details.

   lda
      Leading dimension of ``A``. Must be at least ``m`` if ``A`` is on
      the left of the multiplication, or at least ``n`` if ``A`` is on
      the right. Must be positive.

   b
      Buffer holding input matrix ``B``. Must have size at least
      ``ldb``\ \*\ ``n`` if column major layout is
      used to store matrices or at least ``ldb``\ \*\ ``m`` if row
      major layout is used to store matrices. See :ref:`matrix-storage` for
      more details.

   ldb
      Leading dimension of ``B``. It must be positive and at least
      ``m`` if column major layout is used to store matrices or at
      least ``n`` if column major layout is used to store matrices.

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
      least ``n`` if column major layout is used to store matrices.

.. container:: section

   .. rubric:: Output Parameters

   c
      Output buffer, overwritten by ``alpha``\ \*\ ``A``\ \*\ ``B`` +
      ``beta``\ \*\ ``C`` (``left_right`` = ``side::left``) or
      ``alpha``\ \*\ ``B``\ \*\ ``A`` + ``beta``\ \*\ ``C``
      (``left_right`` = ``side::right``).

.. container:: section

   .. rubric:: Notes

   If ``beta`` = 0, matrix ``C`` does not need to be initialized before
   calling ``hemm``.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

.. _onemath_blas_hemm_usm:

hemm (USM Version)
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event hemm(sycl::queue &queue,
                        oneapi::math::side left_right,
                        oneapi::math::uplo upper_lower,
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
       sycl::event hemm(sycl::queue &queue,
                        oneapi::math::side left_right,
                        oneapi::math::uplo upper_lower,
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
      Specifies whether ``A`` is on the left side of the
      multiplication (``side::left``) or on the right side
      (``side::right``). See :ref:`onemath_datatypes` for more details.

   uplo
      Specifies whether ``A``'s data is stored in its upper or lower
      triangle. See :ref:`onemath_datatypes` for more details.

   m
      Specifies the number of rows of the matrix ``B`` and ``C``.

      The value of ``m`` must be at least zero.

   n
      Specifies the number of columns of the matrix ``B`` and ``C``.

      The value of ``n`` must be at least zero.

   alpha
      Scaling factor for the matrix-matrix product. See :ref:`value_or_pointer` for more details.

   a
      Pointer to input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``m`` if ``A`` is on the left of the
      multiplication, or ``lda``\ \*\ ``n`` if ``A`` is on the right.
      See :ref:`matrix-storage` for
      more details.

   lda
      Leading dimension of ``A``. Must be at least ``m`` if ``A`` is
      on the left of the multiplication, or at least ``n`` if ``A``
      is on the right. Must be positive.

   b
      Pointer to input matrix ``B``. Must have size at least
      ``ldb``\ \*\ ``n`` if column major layout is
      used to store matrices or at least ``ldb``\ \*\ ``m`` if row
      major layout is used to store matrices. See :ref:`matrix-storage` for
      more details.

   ldb
      Leading dimension of ``B``. It must be positive and at least
      ``m`` if column major layout is used to store matrices or at
      least ``n`` if column major layout is used to store matrices.

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
      least ``n`` if column major layout is used to store matrices.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   c
      Pointer to the output matrix, overwritten by
      ``alpha``\ \*\ ``A``\ \*\ ``B`` + ``beta``\ \*\ ``C``
      (``left_right`` = ``side::left``) or
      ``alpha``\ \*\ ``B``\ \*\ ``A`` + ``beta``\ \*\ ``C``
      (``left_right`` = ``side::right``).

.. container:: section

   .. rubric:: Notes

   If ``beta`` = 0, matrix ``C`` does not need to be initialized
   before calling ``hemm``.

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
