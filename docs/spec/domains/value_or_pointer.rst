.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _value_or_pointer:

Scalar Arguments in BLAS
========================

.. container::

   The USM version of oneMath BLAS routines for DPC++ will accept either
   a scalar (for example ``float``) or pointer (``float*``) for parameters
   that represent a single fixed value (not a vector or matrix). These
   parameters are often named ``alpha`` or ``beta`` in BLAS.

   .. container:: section

      .. rubric:: Basic Use
         :name: basic-use
         :class: sectiontitle

      Users can call ``gemv`` with pointers:

        .. code-block::

          float *alpha_ptr = sycl::malloc_shared<float>(1, queue);
          float *beta_ptr = sycl::malloc_shared<float>(1, queue);
          // fill alpha_ptr and beta_ptr with desired values
          oneapi::math::blas::column_major::gemv(queue, trans, m, n, alpha_ptr, lda, x, incx, beta_ptr,
                                                y, incy).wait();

      or with literal values:

        .. code-block::

          oneapi::math::blas::column_major::gemv(queue, trans, m, n, 2, lda, x, incx, 2.7,
                                                y, incy).wait();

      Users can even mix scalar and pointer parameters in a single call:

        .. code-block::

          float *alpha_ptr = sycl::malloc_shared<float>(1, queue);
          oneapi::math::blas::column_major::gemv(queue, trans, m, n, alpha_ptr, lda, x, incx, 2.7,
                                             y, incy).wait();

      Pointers provided for scalar parameters may be SYCL-managed pointers
      to either device or host memory (for example pointers created with
      ``sycl::malloc_device``, ``sycl::malloc_shared``, or
      ``sycl::malloc_host``), or they may be raw pointers created with
      ``malloc`` or ``new``.

      For most users, this is all they need to know. A few details about how
      this is implemented are provided below.

   .. container:: section

      .. rubric:: Wrapper type
         :name: wrapper-time
         :class: sectiontitle

      The USM version of oneMath BLAS routines use a templated
      ``value_or_pointer<T>`` wrapper to enable either pointers or values
      to be passed to routines that take a scalar parameter.

      In general, users should not explicitly use this type in their
      code. There is no need to construct an object of type
      ``value_or_pointer`` in order to use the oneMath functions that
      include it in their function signatures. Instead, values and pointers
      in user code will be implicitly converted to this type when a user
      calls a oneMath function.

      The ``value_or_pointer<T>`` wrapper has two constructors, one that
      converts a value of type ``T`` (or anything convertible to ``T``) to
      ``value_or_pointer<T>``, and another that converts a pointer to ``T``
      to ``value_or_pointer<T>``. Internally, the oneMath functions can
      behave slightly differently depending on whether the underlying data
      is a value or a pointer, and if it points to host-side memory or
      device-side memory, but these uses should be transparent to users.

   .. container:: section

      .. rubric:: Dependencies
         :name: dependencies
         :class: sectiontitle

      For scalar parameters passed to oneMath BLAS routines as pointers, the
      timing of pointer dereferencing depends on whether it is a USM-managed
      pointer or a raw pointer.

      For a USM-managed pointer, it is dereferenced at kernel launch after
      the dependencies passed to the function have been resolved, so the
      value may be assigned asynchronously in another event passed as a
      dependency to the routine.

      A raw pointer (such as those allocated with ``malloc`` or ``new``) is
      dereferenced at the function call, so it must be valid when the
      function is called. In this case the data must be valid when the
      function is called and it may not be assigned asynchronously.


      **Parent topic:** :ref:`onemath_dense_linear_algebra`

