.. SPDX-FileCopyrightText: 2024 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_sparse_init_coo_matrix:

init_coo_matrix
===============

Initializes a ``matrix_handle_t`` object with the provided COO data.

.. rubric:: Description and Assumptions

The ``oneapi::math::sparse::init_coo_matrix`` function initializes the
``matrix_handle_t`` object with the provided data.

In the case of buffers, the reference count of the provided buffer is
incremented which extends the lifetime of the underlying buffer until the sparse
matrix handle is destroyed with ``release_sparse_matrix`` or the data is reset
with ``set_coo_matrix_data``.

In the case of USM, the object does not take ownership of the data.

The ``oneapi::math::sparse::init_coo_matrix`` function defined below takes in the
number of non-zero elements in the sparse matrix as an argument. However, in
certain math operations where the output is a sparse matrix, e.g., sparse matrix
addition (sparse matrix + sparse matrix = sparse matrix), and multiplication of
two sparse matrices, the number of non-zero elements in the output sparse matrix
are not known in advance and must be calculated as part of the operation API.
Such APIs are currently not part of the oneMath Specification, but will be added
in the future. This behavior is currently left to be implementation-defined, but
may be clarified in the oneMath Specification in the future.

See :ref:`onemath_sparse_matrix_handle`.

.. _onemath_sparse_init_coo_matrix_buffer:

init_coo_matrix (Buffer version)
--------------------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::sparse {

       template <typename dataType, typename indexType>
       void init_coo_matrix (sycl::queue                          &queue,
                             oneapi::math::sparse::matrix_handle_t *p_smhandle,
                             std::int64_t                         num_rows,
                             std::int64_t                         num_cols,
                             std::int64_t                         nnz,
                             index_base                           index,
                             sycl::buffer<indexType, 1>           row_ind,
                             sycl::buffer<indexType, 1>           col_ind,
                             sycl::buffer<dataType, 1>            val);

   }

.. container:: section

   .. rubric:: Template parameters

   dataType
      See :ref:`supported template types<onemath_sparse_supported_types>`.

   indexType
      See :ref:`supported template types<onemath_sparse_supported_types>`.

.. container:: section

   .. rubric:: Input parameters

   queue
      The SYCL command queue which will be used for SYCL kernels execution.

   p_smhandle
      The address of the ``p_smhandle`` object to be initialized. Must only be
      called on an uninitialized ``matrix_handle_t`` object.

   num_rows
      Number of rows of the provided data ``val``. Must be at least 0.

   num_cols
      Number of columns of the provided data ``val``. Must be at least 0.

   nnz
      The number of explicit entries, also known as Number of Non-Zero elements.
      Must be at least 0.

   index
      Indicates how input arrays are indexed. The possible options are described
      in :ref:`onemath_enum_index_base` enum class.

   row_ind
      Buffer of length at least ``nnz`` containing the row indices in
      ``index``-based numbering. Refer to :ref:`onemath_sparse_coo` format for
      detailed description of ``row_ind``.

   col_ind
      Buffer of length at least ``nnz`` containing the column indices in
      ``index``-based numbering. Refer to :ref:`onemath_sparse_coo` format for
      detailed description of ``col_ind``.

   val
      Buffer of length at least ``nnz``. Contains the data of the input matrix
      which is not implicitly zero. The remaining input values are implicit
      zeros. Refer to :ref:`onemath_sparse_coo` format for detailed description
      of ``val``.

.. container:: section

   .. rubric:: Output parameters

   p_smhandle
      On return, the address is updated to point to a newly allocated and
      initialized ``matrix_handle_t`` object that can be filled and used to
      perform sparse BLAS operations.

.. container:: section

   .. rubric:: Notes

   - The parameters ``num_rows``, ``num_cols`` and ``nnz`` may be zero if and
     only if ``row_ind``, ``col_ind`` and ``val`` are zero-sized, otherwise they
     must be strictly greater than zero.

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

.. _onemath_sparse_init_coo_matrix_usm:

init_coo_matrix (USM version)
-----------------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::sparse {

       template <typename dataType, typename indexType>
       void init_coo_matrix (sycl::queue                          &queue,
                             oneapi::math::sparse::matrix_handle_t *p_smhandle,
                             std::int64_t                         num_rows,
                             std::int64_t                         num_cols,
                             std::int64_t                         nnz,
                             index_base                           index,
                             indexType                            *row_ind,
                             indexType                            *col_ind,
                             dataType                             *val);

   }

.. container:: section

   .. rubric:: Template parameters

   dataType
      See :ref:`supported template types<onemath_sparse_supported_types>`.

   indexType
      See :ref:`supported template types<onemath_sparse_supported_types>`.

.. container:: section

   .. rubric:: Input parameters

   queue
      The SYCL command queue which will be used for SYCL kernels execution.

   p_smhandle
      The address of the ``p_smhandle`` object to be initialized. Must only be
      called on an uninitialized ``matrix_handle_t`` object.

   num_rows
      Number of rows of the provided data ``val``. Must be at least 0.

   num_cols
      Number of columns of the provided data ``val``. Must be at least 0.

   nnz
      The number of explicit entries, also known as Number of Non-Zero elements.
      Must be at least 0.

   index
      Indicates how input arrays are indexed. The possible options are described
      in :ref:`onemath_enum_index_base` enum class.

   row_ind
      USM pointer of length at least ``nnz`` containing the row indices in
      ``index``-based numbering. Refer to :ref:`onemath_sparse_coo` format for
      detailed description of ``row_ind``. The data must be accessible on the
      device.

   col_ind
      USM pointer of length at least ``nnz`` containing the column indices in
      ``index``-based numbering. Refer to :ref:`onemath_sparse_coo` format for
      detailed description of ``col_ind``. The data must be accessible on the
      device.

   val
      USM pointer of length at least ``nnz``. Contains the data of the input
      matrix which is not implicitly zero. The remaining input values are
      implicit zeros. Refer to :ref:`onemath_sparse_coo` format for detailed
      description of ``val``. The data must be accessible on the device. Using a
      USM pointer with a smaller allocated memory size is undefined behavior.

.. container:: section

   .. rubric:: Output parameters

   p_smhandle
      On return, the address is updated to point to a newly allocated and
      initialized ``matrix_handle_t`` object that can be filled and used to
      perform sparse BLAS operations.

.. container:: section

   .. rubric:: Notes

   - The parameters ``num_rows``, ``num_cols`` and ``nnz`` may be zero if and
     only if ``row_ind``, ``col_ind`` and ``val`` are null pointers, otherwise
     they must be strictly greater than zero.

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
