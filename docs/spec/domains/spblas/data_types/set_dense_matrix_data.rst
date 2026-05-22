.. SPDX-FileCopyrightText: 2024 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_sparse_set_dense_matrix_data:

set_dense_matrix_data
=====================

Reset the data of a ``dense_matrix_handle_t`` object.

.. rubric:: Description and Assumptions

The ``oneapi::math::sparse::set_dense_matrix_data`` function sets new data to the
``dense_matrix_handle_t`` object with the provided data.

In the case of buffers, the reference count of the provided buffer is
incremented which extends the lifetime of the underlying buffer until the
``dmhandle`` is destroyed with ``release_dense_matrix`` or the data is reset
with ``set_dense_matrix_data``.

In the case of USM, the object does not take ownership of the data.

Also see :ref:`onemath_sparse_init_dense_matrix`.

.. _onemath_sparse_set_dense_matrix_data_buffer:

set_dense_matrix_data (Buffer version)
--------------------------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::sparse {

       template <typename dataType>
       void set_dense_matrix_data (sycl::queue                                &queue,
                                   oneapi::math::sparse::dense_matrix_handle_t dmhandle,
                                   std::int64_t                               num_rows,
                                   std::int64_t                               num_cols,
                                   std::int64_t                               ld,
                                   layout                                     dense_layout,
                                   sycl::buffer<dataType, 1>                  val);

   }

.. container:: section

   .. rubric:: Template parameters

   dataType
      See :ref:`supported template types<onemath_sparse_supported_types>`. Must
      be the same type as was used when creating the ``dense_matrix_handle_t``.

.. container:: section

   .. rubric:: Input parameters

   queue
      The SYCL command queue which will be used for SYCL kernels execution.

   dmhandle
      Handle already initialized with :ref:`onemath_sparse_init_dense_matrix`.

   num_rows
      Number of rows of the provided data ``val``. Must be at least 0.

   num_cols
      Number of columns of the provided data ``val``. Must be at least 0.

   ld
      Leading dimension of the provided data ``val``. Must be at least
      ``num_rows`` if column major layout is used or at least ``num_cols`` if
      row major layout is used.

   dense_layout
      Specify whether the data uses row major or column major.

   val
      Buffer of length at least ``ld*num_cols`` if column major is used or
      ``ld*num_rows`` if row major is used.

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

.. _onemath_sparse_set_dense_matrix_data_usm:

set_dense_matrix_data (USM version)
-----------------------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::sparse {

       template <typename dataType>
       void set_dense_matrix_data (sycl::queue                                &queue,
                                   oneapi::math::sparse::dense_matrix_handle_t dmhandle,
                                   std::int64_t                               num_rows,
                                   std::int64_t                               num_cols,
                                   std::int64_t                               ld,
                                   layout                                     dense_layout,
                                   dataType                                   *val);

   }

.. container:: section

   .. rubric:: Template parameters

   dataType
      See :ref:`supported template types<onemath_sparse_supported_types>`. Must
      be the same type as was used when creating the ``dense_matrix_handle_t``.

.. container:: section

   .. rubric:: Input parameters

   queue
      The SYCL command queue which will be used for SYCL kernels execution.

   dmhandle
      Handle already initialized with :ref:`onemath_sparse_init_dense_matrix`.

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
      ``ld*num_rows`` if row major is used. The data must be accessible on the
      device. Using a USM pointer with a smaller allocated memory size is
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
