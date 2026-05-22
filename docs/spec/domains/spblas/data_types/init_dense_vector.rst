.. SPDX-FileCopyrightText: 2024 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_sparse_init_dense_vector:

init_dense_vector
=================

Initializes a ``dense_vector_handle_t`` object with the provided data.

.. rubric:: Description and Assumptions

The ``oneapi::math::sparse::init_dense_vector`` function initializes the
``dense_vector_handle_t`` object with the provided data.

In the case of buffers, the reference count of the provided buffer is
incremented which extends the lifetime of the underlying buffer until the dense
vector handle is destroyed with ``release_dense_vector`` or the data is reset
with ``set_dense_vector_data``.

In the case of USM, the object does not take ownership of the data.

See :ref:`onemath_sparse_dense_vector_handle`.

.. _onemath_sparse_init_dense_vector_buffer:

init_dense_vector (Buffer version)
----------------------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::sparse {

       template <typename dataType>
       void init_dense_vector (sycl::queue                                &queue,
                               oneapi::math::sparse::dense_vector_handle_t *p_dvhandle,
                               std::int64_t                               size,
                               sycl::buffer<dataType, 1>                  val);

   }

.. container:: section

   .. rubric:: Template parameters

   dataType
      See :ref:`supported template types<onemath_sparse_supported_types>`.

.. container:: section

   .. rubric:: Input parameters

   queue
      The SYCL command queue which will be used for SYCL kernels execution.

   p_dvhandle
      The address of the ``p_dvhandle`` object to be initialized. Must only be
      called on an uninitialized ``dense_vector_handle_t`` object.

   size
      Number of elements of the provided data ``val``. Must be at least 1.

   val
      Buffer of length at least ``size``. Holds the data to initialize
      ``p_dvhandle`` with.

.. container:: section

   .. rubric:: Output parameters

   p_dvhandle
      On return, the address is updated to point to a newly allocated and
      initialized ``dense_vector_handle_t`` object that can be filled and used
      to perform sparse BLAS operations.

.. container:: section

   .. rubric:: Throws

   This function shall throw the following exceptions if the associated
   condition is detected. An implementation may throw additional
   implementation-specific exception(s) in case of error conditions not covered
   here.

   | :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
   | :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
   | :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
   | :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

.. _onemath_sparse_init_dense_vector_usm:

init_dense_vector (USM version)
-------------------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::sparse {

       template <typename dataType>
       void init_dense_vector (sycl::queue                                &queue,
                               oneapi::math::sparse::dense_vector_handle_t *p_dvhandle,
                               std::int64_t                               size,
                               dataType                                   *val);

   }

.. container:: section

   .. rubric:: Template parameters

   dataType
      See :ref:`supported template types<onemath_sparse_supported_types>`.

.. container:: section

   .. rubric:: Input parameters

   queue
      The SYCL command queue which will be used for SYCL kernels execution.

   p_dvhandle
      The address of the ``p_dvhandle`` object to be initialized. Must only be
      called on an uninitialized ``dense_vector_handle_t`` object.

   size
      Number of elements of the provided data ``val``. Must be at least 1.

   val
      USM pointer of length at least ``size``. Holds the data to initialize
      ``p_dvhandle`` with. The data must be accessible on the device. Using a
      USM pointer with a smaller allocated memory size is undefined behavior.

.. container:: section

   .. rubric:: Output parameters

   p_dvhandle
      On return, the address is updated to point to a newly allocated and
      initialized ``dense_vector_handle_t`` object that can be filled and used
      to perform sparse BLAS operations.

.. container:: section

   .. rubric:: Throws

   This function shall throw the following exceptions if the associated
   condition is detected. An implementation may throw additional
   implementation-specific exception(s) in case of error conditions not covered
   here.

   | :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
   | :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
   | :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
   | :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

**Parent topic:** :ref:`onemath_sparse_data_handles`
