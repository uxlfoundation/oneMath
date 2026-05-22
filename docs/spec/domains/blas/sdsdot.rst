.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_sdsdot:

sdsdot
======

Computes a vector-vector dot product with double precision.

.. _onemath_blas_sdsdot_description:

.. rubric:: Description

The ``sdsdot`` routines perform a dot product between two vectors with
double precision:

.. math::

   result = sb + \sum_{i=1}^{n}X_iY_i

.. _onemath_blas_sdsdot_buffer:

sdsdot (Buffer Version)
-----------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void sdsdot(sycl::queue &queue,
                   std::int64_t n,
                   float sb,
                   sycl::buffer<float,1> &x,
                   std::int64_t incx,
                   sycl::buffer<float,1> &y,
                   std::int64_t incy,
                   sycl::buffer<float,1> &result)
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void sdsdot(sycl::queue &queue,
                   std::int64_t n,
                   float sb,
                   sycl::buffer<float,1> &x,
                   std::int64_t incx,
                   sycl::buffer<float,1> &y,
                   std::int64_t incy,
                   sycl::buffer<float,1> &result)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   n
      Number of elements in vectors ``x`` and ``y``.

   sb
      Single precision scalar to be added to the dot product.

   x
      Buffer holding input vector ``x``. The buffer must be of size
      at least (1 + (``n`` - 1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

   y
      Buffer holding input vector ``y``. The buffer must be of size
      at least (1 + (``n`` - 1)*abs(``incxy``)). See :ref:`matrix-storage` for
      more details.

   incy
      Stride of vector ``y``.

.. container:: section

   .. rubric:: Output Parameters

   result
      Buffer where the result (a scalar) will be stored. If ``n`` < 0
      the result is ``sb``.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

.. _onemath_blas_sdsdot_usm:

sdsdot (USM Version)
--------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event sdsdot(sycl::queue &queue,
                          std::int64_t n,
                          float sb,
                          const float *x,
                          std::int64_t incx,
                          const float *y,
                          std::int64_t incy,
                          float *result,
                          const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event sdsdot(sycl::queue &queue,
                          std::int64_t n,
                          float sb,
                          const float *x,
                          std::int64_t incx,
                          const float *y,
                          std::int64_t incy,
                          float *result,
                          const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   n
      Number of elements in vectors ``x`` and ``y``.

   sb
      Single precision scalar to be added to the dot product.

   x
      Pointer to the input vector ``x``. The array must be of size
      at least (1 + (``n`` - 1)*abs(``incx``)). See :ref:`matrix-storage`
      for more details.

   incx
      Stride of vector ``x``.

   y
      Pointer to the input vector ``y``. The array must be of size
      at least (1 + (``n`` - 1)*abs(``incxy``)). See :ref:`matrix-storage`
      for more details.

   incy
      Stride of vector ``y``.

   dependencies
      List of events to wait for before starting computation, if
      any. If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   result
      Pointer to where the result (a scalar) will be stored. If
      ``n`` < 0 the result is ``sb``.

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
      

   **Parent topic:** :ref:`blas-level-1-routines`
