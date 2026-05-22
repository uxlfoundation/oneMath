.. SPDX-FileCopyrightText: 2024 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_sparse_set_dense_vector_data:

set_dense_vector_data
=====================

Reset the data of a ``dense_vector_handle_t`` object.

.. rubric:: Description and Assumptions

The ``oneapi::math::sparse::set_dense_vector_data`` function sets new data to the
``dense_vector_handle_t`` object.

In the case of buffers, the reference count of the provided buffer is
incremented which extends the lifetime of the underlying buffer until the
``dvhandle`` is destroyed with ``release_dense_vector`` or the data is reset
with ``set_dense_vector_data``.

In the case of USM, the object does not take ownership of the data.

Also see :ref:`onemath_sparse_init_dense_vector`.

.. _onemath_sparse_set_dense_vector_data_buffer:

set_dense_vector_data (Buffer version)
--------------------------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::sparse {

       template <typename dataType>
       void set_dense_vector_data (sycl::queue                                &queue,
                                   oneapi::math::sparse::dense_vector_handle_t dvhandle,
                                   std::int64_t                               size,
                                   sycl::buffer<dataType, 1>                  val);

   }

.. container:: section

   .. rubric:: Template parameters

   dataType
      See :ref:`supported template types<onemath_sparse_supported_types>`. Must
      be the same type as was used when creating the ``dense_vector_handle_t``.

.. container:: section

   .. rubric:: Input parameters

   queue
      The SYCL command queue which will be used for SYCL kernels execution.

   dvhandle
      Handle already initialized with :ref:`onemath_sparse_init_dense_vector`.

   size
      Number of elements of the provided data ``val``. Must be at least 0.

   val
      Buffer of length at least ``size``.

.. container:: section

   .. rubric:: Throws

   This function shall throw the following exceptions if the associated
   condition is detected. An implementation may throw additional
   implementation-specific exception(s) in case of error conditions not covered
   here.

   | :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
   | :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
   | :ref:`oneapi::math::uninitialized<onemath_exception_uninitialized>`
   | :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

.. _onemath_sparse_set_dense_vector_data_usm:

set_dense_vector_data (USM version)
-----------------------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::sparse {

       template <typename dataType>
       void set_dense_vector_data (sycl::queue                                &queue,
                                   oneapi::math::sparse::dense_vector_handle_t dvhandle,
                                   std::int64_t                               size,
                                   dataType                                   *val);

   }

.. container:: section

   .. rubric:: Template parameters

   dataType
      See :ref:`supported template types<onemath_sparse_supported_types>`. Must
      be the same type as was used when creating the ``dense_vector_handle_t``.

.. container:: section

   .. rubric:: Input parameters

   queue
      The SYCL command queue which will be used for SYCL kernels execution.

   dvhandle
      Handle already initialized with :ref:`onemath_sparse_init_dense_vector`.

   size
      Number of elements of the provided data ``val``. Must be at least 1.

   val
      USM pointer of length at least ``size``. The data must be accessible on
      the device. Using a USM pointer with a smaller allocated memory size is
      undefined behavior.

.. container:: section

   .. rubric:: Throws

   This function shall throw the following exceptions if the associated
   condition is detected. An implementation may throw additional
   implementation-specific exception(s) in case of error conditions not covered
   here.

   | :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
   | :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
   | :ref:`oneapi::math::uninitialized<onemath_exception_uninitialized>`
   | :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

**Parent topic:** :ref:`onemath_sparse_data_handles`
