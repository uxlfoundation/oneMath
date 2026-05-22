.. SPDX-FileCopyrightText: 2024 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_sparse_set_matrix_property:

set_matrix_property
===================

Matrix properties
-----------------

.. container:: section

   .. code:: cpp

      namespace oneapi::math::sparse {

          enum class matrix_property {
             symmetric,
             sorted,
             sorted_by_rows,
          };

      }

   Matrix properties are optional and "strong" guarantees. Unlike
   :ref:`onemath_sparse_matrix_view`, the user must ensure that the handle's data
   holds all the given properties. A property can be set as a hint for backends
   to optimize some operations. Multiple properties can be set to the same handle.

symmetric
^^^^^^^^^

Guarantees that the user-provided matrix data are symmetric, meaning the matrix
is square, the user data contain both lower and upper triangular regions, and
that its transpose is equal to itself.

sorted
^^^^^^

Guarantees that the user-provided matrix data is fully sorted. The table below
details the property for each matrix format:

   .. list-table::
      :header-rows: 1
      :widths: 20 80

      * - Matrix format
        - Description
      * - CSR
        - Guarantees that the column indices are sorted in ascending order for
          a given row.
      * - COO
        - Guarantees that the indices are sorted by rows then by columns within
          rows in ascending order.

sorted_by_rows
^^^^^^^^^^^^^^

Guarantees that the user-provided matrix data is partially sorted. The table below
details the property for each matrix format:

   .. list-table::
      :header-rows: 1
      :widths: 20 80

      * - Matrix format
        - Description
      * - CSR
        - Setting this property for a matrix using the CSR format will throw a
          :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
          exception.
      * - COO
        - Guarantees that the indices are sorted by rows in ascending order.
          The column indices within a row do not need to be sorted.

set_matrix_property
-------------------

Set a property to a ``matrix_handle_t`` object.

.. rubric:: Description and Assumptions

The ``oneapi::math::sparse::set_matrix_property`` function sets a property to a
matrix handle.

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::sparse {

       bool set_matrix_property (sycl::queue                          &queue,
                                 oneapi::math::sparse::matrix_handle_t smhandle,
                                 matrix_property                      property);

   }

.. container:: section

   .. rubric:: Input parameters

   queue
      The SYCL command queue which will be used for SYCL kernels execution.

   smhandle
      Initialized sparse matrix handle.

   property
      Matrix property to set.

.. container:: section

   .. rubric:: Return Values

   Return whether the property was set to the backend's handle. A backend may
   not have an equivalent property.

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
