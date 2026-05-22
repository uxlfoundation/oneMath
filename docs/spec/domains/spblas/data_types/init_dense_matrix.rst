.. SPDX-FileCopyrightText: 2024 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_sparse_init_dense_matrix:

init_dense_matrix
=================

Initializes a ``dense_matrix_handle_t`` object with the provided data.

.. rubric:: Description and Assumptions

The ``oneapi::math::sparse::init_dense_matrix`` function initializes the
``dense_matrix_handle_t`` object with the provided data.

In the case of buffers, the reference count of the provided buffer is
incremented which extends the lifetime of the underlying buffer until the dense
matrix handle is destroyed with ``release_dense_matrix`` or the data is reset
with ``set_dense_matrix_data``.

In the case of USM, the object does not take ownership of the data.

See :ref:`onemath_sparse_dense_matrix_handle`.

.. _onemath_sparse_init_dense_matrix_buffer:

init_dense_matrix (Buffer version)
----------------------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::sparse {

       template <typename dataType>
       void init_dense_matrix (sycl::queue                                &queue,
                               oneapi::math::sparse::dense_matrix_handle_t *p_dmhandle,
                               std::int64_t                               num_rows,
                               std::int64_t                               num_cols,
                               std::int64_t                               ld,
                               layout                                     dense_layout,
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

   p_dmhandle
      The address of the ``p_dmhandle`` object to be initialized. Must only be
      called on an uninitialized ``dense_matrix_handle_t`` object.

   num_rows
      Number of rows of the provided data ``val``. Must be at least 1.

   num_cols
      Number of columns of the provided data ``val``. Must be at least 1.

   ld
      Leading dimension of the provided data ``val``. Must be at least
      ``num_rows`` if column major layout is used or at least ``num_cols`` if
      row major layout is used.

   dense_layout
      Specify whether the data uses row major or column major.

   val
      Buffer of length at least ``ld*num_cols`` if column major is used or
      ``ld*num_rows`` if row major is used. Holds the data to initialize
      ``p_dmhandle`` with.

.. container:: section

   .. rubric:: Output parameters

   p_dmhandle
      On return, the address is updated to point to a newly allocated and
      initialized ``dense_matrix_handle_t`` object that can be filled and used
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

.. _onemath_sparse_init_dense_matrix_usm:

init_dense_matrix (USM version)
-------------------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::sparse {

       template <typename dataType>
       void init_dense_matrix (sycl::queue                                &queue,
                               oneapi::math::sparse::dense_matrix_handle_t *p_dmhandle,
                               std::int64_t                               num_rows,
                               std::int64_t                               num_cols,
                               std::int64_t                               ld,
                               layout                                     dense_layout,
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

   p_dmhandle
      The address of the ``p_dmhandle`` object to be initialized. Must only be
      called on an uninitialized ``dense_matrix_handle_t`` object.

   num_rows
      Number of rows of the provided data ``val``. Must be at least 1.

   num_cols
      Number of columns of the provided data ``val``. Must be at least 1.

   ld
      Leading dimension of the provided data ``val``. Must be at least
      ``num_rows`` if column major layout is used or at least ``num_cols`` if
      row major layout is used.

   dense_layout
      Specify whether the data uses row major or column major.

   val
      USM pointer of length at least ``ld*num_cols`` if column major is used or
      ``ld*num_rows`` if row major is used. Holds the data to initialize
      ``p_dmhandle`` with. The data must be accessible on the device. Using a
      USM pointer with a smaller allocated memory size is undefined behavior.

.. container:: section

   .. rubric:: Output parameters

   p_dmhandle
      On return, the address is updated to point to a newly allocated and
      initialized ``dense_matrix_handle_t`` object that can be filled and used
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
