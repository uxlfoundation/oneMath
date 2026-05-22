.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_rot:

rot
===

Performs rotation of points in the plane.

.. _onemath_blas_rot_description:

.. rubric:: Description

Given two vectors ``x`` and ``y`` of ``n`` elements, the ``rot`` routines
compute four scalar-vector products and update the input vectors with
the sum of two of these scalar-vector products as follows:

.. math::

   \left[\begin{array}{c}
      x\\y
   \end{array}\right]
   \leftarrow
   \left[\begin{array}{c}
      \phantom{-}c*x + s*y\\
      -s*x + c*y
   \end{array}\right]

If ``s`` is a complex type, the operation is defined as:

.. math::
   \left[\begin{array}{c}
      x\\y
   \end{array}\right]
   \leftarrow
   \left[\begin{array}{c}
       \phantom{-}c*x + s*y\\
       -conj(s)*x + c*y
   \end{array}\right]

``rot`` supports the following precisions.

.. list-table:: 
   :header-rows: 1

   * -  T
     -  Tc
     -  Ts
   * -  ``sycl::half``
     -  ``sycl::half``
     -  ``sycl::half``
   * -  ``oneapi::math::bfloat16``
     -  ``oneapi::math::bfloat16``
     -  ``oneapi::math::bfloat16``
   * -  ``float``
     -  ``float``
     -  ``float``
   * -  ``double``
     -  ``double``
     -  ``double``
   * -  ``std::complex<float>``
     -  ``float``
     -  ``std::complex<float>``
   * -  ``std::complex<double>``
     -  ``double``
     -  ``std::complex<double>``
   * -  ``std::complex<float>``
     -  ``float``
     -  ``float``
   * -  ``std::complex<double>``
     -  ``double``
     -  ``double``

.. _onemath_blas_rot_buffer:

rot (Buffer Version)
--------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void rot(sycl::queue &queue,
                std::int64_t n,
                sycl::buffer<T,1> &x,
                std::int64_t incx,
                sycl::buffer<T,1> &y,
                std::int64_t incy,
                Tc c,
                Ts s)
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void rot(sycl::queue &queue,
                std::int64_t n,
                sycl::buffer<T,1> &x,
                std::int64_t incx,
                sycl::buffer<T,1> &y,
                std::int64_t incy,
                Tc c,
                Ts s)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   n
      Number of elements in vector ``x``.

   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

   y
      Buffer holding input vector ``y``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incy``)). See :ref:`matrix-storage` for
      more details.

   incy
      Stride of vector ``y``.

   c
      Scaling factor.

   s
      Scaling factor.

.. container:: section

   .. rubric:: Output Parameters

   x
      Buffer holding updated buffer ``x``.

   y
      Buffer holding updated buffer ``y``.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

.. _onemath_blas_rot_usm:

rot (USM Version)
-----------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event rot(sycl::queue &queue,
                       std::int64_t n,
                       T *x,
                       std::int64_t incx,
                       T *y,
                       std::int64_t incy,
                       value_or_pointer<Tc> c,
                       value_or_pointer<Ts> s,
                       const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event rot(sycl::queue &queue,
                       std::int64_t n,
                       T *x,
                       std::int64_t incx,
                       T *y,
                       std::int64_t incy,
                       value_or_pointer<Tc> c,
                       value_or_pointer<Ts> s,
                       const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   n
      Number of elements in vector ``x``.

   x
      Pointer to input vector ``x``. The array holding input vector
      ``x`` must be of size at least (1 + (``n`` - 1)*abs(``incx``)).
      See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

   y
      Pointer to input vector ``y``. The array holding input vector
      ``y`` must be of size at least (1 + (``n`` - 1)*abs(``incy``)).
      See :ref:`matrix-storage` for
      more details.

   incy
      Stride of vector ``y``.

   c
      Scaling factor. See :ref:`value_or_pointer` for more details.

   s
      Scaling factor. See :ref:`value_or_pointer` for more details.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   x
      Pointer to the updated matrix ``x``.

   y
      Pointer to the updated matrix ``y``.

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
