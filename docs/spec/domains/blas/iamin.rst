.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_iamin:

iamin
=====

Finds the index of the element with the smallest absolute value.

.. _onemath_blas_iamin_description:

.. rubric:: Description

The ``iamin`` routines return an index ``i`` such that ``x[i]`` has
the minimum absolute value of all elements in vector ``x`` (real
variants), or such that (\|Re(``x[i]``)\| + \|Im(``x[i]``)\|) is minimal
(complex variants).

The index is zero-based if ``base`` is set to ``oneapi::math::index_base::zero`` (default)
or one-based if it is set to ``oneapi::math::index_base::one``.

If either ``n`` or ``incx`` is not positive, the routine returns
``0``, regardless of the base of the index selected.

If more than one vector element is found with the same smallest
absolute value, the index of the first one encountered is returned.

If the vector contains ``NaN`` values, then the routine returns the
index of the first ``NaN``.

``iamin`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 


.. _onemath_blas_iamin_buffer:

iamin (Buffer Version)
----------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void iamin(sycl::queue &queue,
                  std::int64_t n,
                  sycl::buffer<T,1> &x,
                  std::int64_t incx,
                  sycl::buffer<std::int64_t,1> &result,
                  oneapi::math::index_base base = oneapi::math::index_base::zero)
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void iamin(sycl::queue &queue,
                  std::int64_t n,
                  sycl::buffer<T,1> &x,
                  std::int64_t incx,
                  sycl::buffer<std::int64_t,1> &result,
                  oneapi::math::index_base base = oneapi::math::index_base::zero)
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
      Stride of vector x.

   base
      Indicates how the output value is indexed. If omitted, defaults to zero-based
      indexing.

.. container:: section

   .. rubric:: Output Parameters

   result
      Buffer where the index ``i`` of the minimum element
      will be stored.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

.. _onemath_blas_iamin_usm:

iamin (USM Version)
-------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event iamin(sycl::queue &queue,
                         std::int64_t n,
                         const T *x,
                         std::int64_t incx,
                         std::int64_t *result,
                         oneapi::math::index_base base = oneapi::math::index_base::zero,
                         const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event iamin(sycl::queue &queue,
                         std::int64_t n,
                         const T *x,
                         std::int64_t incx,
                         std::int64_t *result,
                         oneapi::math::index_base base = oneapi::math::index_base::zero,
                         const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   n
      Number of elements in vector ``x``.

   x
      The pointer to input vector ``x``. The array holding input
      vector ``x`` must be of size at least (1 + (``n`` -
      1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector x.

   base
      Indicates how the output value is indexed. If omitted, defaults to zero-based
      indexing.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   result
      Pointer to where the index ``i`` of the minimum
      element will be stored.

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
