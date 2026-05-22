.. SPDX-FileCopyrightText: 2022 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_imatcopy:

imatcopy
========

Computes an in-place scaled matrix transpose or copy operation
using a general dense matrix.

.. _onemath_blas_imatcopy_description:

.. rubric:: Description

The ``imatcopy`` routine performs an in-place scaled
matrix copy or transposition.

The operation is defined as:

.. math::

      C \leftarrow \alpha * op(C)

where:

op(X) is one of op(X) = X, or op(X) = X\ :sup:`T`, or op(X) = X\ :sup:`H`,

``alpha`` is a scalar,

``C`` is a matrix to be transformed in place,

and ``C`` is ``m`` x ``n`` on input.

``imatcopy`` supports the following precisions:

   .. list-table::
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemath_blas_imatcopy_buffer:

imatcopy (Buffer Version)
-------------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void imatcopy(sycl::queue &queue,
                     oneapi::math::transpose trans,
                     std::int64_t m,
                     std::int64_t n,
                     T alpha,
                     sycl::buffer<T, 1> &matrix_in_out,
                     std::int64_t ld_in,
                     std::int64_t ld_out);
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void imatcopy(sycl::queue &queue,
                     oneapi::math::transpose trans,
                     std::int64_t m,
                     std::int64_t n,
                     T alpha,
                     sycl::buffer<T, 1> &matrix_in_out,
                     std::int64_t ld_in,
                     std::int64_t ld_out);
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   trans
      Specifies op(``C``), the transposition operation applied to the
      matrix ``C``. See :ref:`onemath_datatypes` for more details.

   m
      Number of rows of ``C`` on input. Must be at least zero.

   n
      Number of columns of ``C`` on input. Must be at least zero.

   alpha
      Scaling factor for the matrix transposition or copy.

   matrix_in_out
      Buffer holding the input/output matrix ``C``. Must have size as follows:

      .. list-table::
         :header-rows: 1
     
         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``trans`` = ``transpose::conjtrans``
         * - Column major
           - Size of array ``matrix_in_out`` must be at least max(``ld_in``, ``ld_out``) * ``n``
           - Size of array ``matrix_in_out`` must be at least max(``ld_in``*``n``, ``ld_out``*``m``)
         * - Row major
           - Size of array ``matrix_in_out`` must be at least max(``ld_in``, ``ld_out``) * ``m``
           - Size of array ``matrix_in_out`` must be at least max(``ld_in``*``m``, ``ld_out``*``n``)

   ld_in
      The leading dimension of the matrix ``C`` on input. It must be
      positive, and must be at least ``m`` if column major layout is
      used, and at least ``n`` if row-major layout is used.

   ld_out
      The leading dimension of the matrix ``C`` on output. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``trans`` = ``transpose::conjtrans``
         * - Column major
           - ``ld_out`` must be at least ``m``.
           - ``ld_out`` must be at least ``n``.
         * - Row major
           - ``ld_out`` must be at least ``n``.
           - ``ld_out`` must be at least ``m``.

.. container:: section

   .. rubric:: Output Parameters

   matrix_in_out
      Output buffer, overwritten by ``alpha`` * op(``C``).

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
      

.. _onemath_blas_imatcopy_usm:

imatcopy (USM Version)
----------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event imatcopy(sycl::queue &queue,
                            oneapi::math::transpose trans,
                            std::int64_t m,
                            std::int64_t n,
                            value_or_pointer<T> alpha,
                            T *matrix_in_out,
                            std::int64_t ld_in,
                            std::int64_t ld_out,
                            const std::vector<sycl::event> &dependencies = {});
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event imatcopy(sycl::queue &queue,
                            oneapi::math::transpose trans,
                            std::int64_t m,
                            std::int64_t n,
                            value_or_pointer<T> alpha,
                            T *matrix_in_out,
                            std::int64_t ld_in,
                            std::int64_t ld_out,
                            const std::vector<sycl::event> &dependencies = {});

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine will be executed.

   trans
      Specifies op(``C``), the transposition operation applied to the
      matrix ``C``. See :ref:`onemath_datatypes` for more details.

   m
      Number of rows for the matrix ``C`` on input. Must be at least zero.

   n
      Number of columns for the matrix ``C`` on input. Must be at least zero.

   alpha
      Scaling factor for the matrix transpose or copy operation. See :ref:`value_or_pointer` for more details.

   matrix_in_out
         Pointer to input/output matrix ``C``. Must have size as follows:

      .. list-table::
         :header-rows: 1
     
         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``trans`` = ``transpose::conjtrans``
         * - Column major
           - Size of array ``matrix_in_out`` must be at least max(``ld_in``, ``ld_out``) * ``n``
           - Size of array ``matrix_in_out`` must be at least max(``ld_in``*``n``, ``ld_out``*``m``)
         * - Row major
           - Size of array ``matrix_in_out`` must be at least max(``ld_in``, ``ld_out``) * ``m``
           - Size of array ``matrix_in_out`` must be at least max(``ld_in``*``m``, ``ld_out``*``n``)

   ld_in
      Leading dimension of the matrix ``C`` on input. If matrices are stored
      using column major layout, ``ld_in`` must be at least ``m``. If matrices
      are stored using row major layout, ``ld_in`` must be at least ``n``. 
      Must be positive.

   ld_out
      Leading dimension of the matrix ``C`` on output. Must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``trans`` = ``transpose::conjtrans``
         * - Column major
           - ``ld_out`` must be at least ``m``.
           - ``ld_out`` must be at least ``n``.
         * - Row major
           - ``ld_out`` must be at least ``n``.
           - ``ld_out`` must be at least ``m``.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   matrix_in_out
      Pointer to output matrix ``C`` overwritten by ``alpha`` * op(``C``).

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

