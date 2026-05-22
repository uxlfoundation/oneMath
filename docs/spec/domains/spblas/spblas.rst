.. SPDX-FileCopyrightText: 2024 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_spblas:

Sparse BLAS
===========

.. container::

   Sparse BLAS routines provide basic operations on sparse vectors and matrices.

.. container:: tablenoborder

   .. list-table::
      :header-rows: 1

      * - Routines and Objects
        - Description
      * - :ref:`onemath_sparse_data_handles`
        - Matrix and vector handle types
      * - :ref:`onemath_sparse_spmm_header`
        - Compute the product of a sparse matrix with a dense matrix
      * - :ref:`onemath_sparse_spmv_header`
        - Compute the product of a sparse matrix with a dense vector
      * - :ref:`onemath_sparse_spsv_header`
        - Solve a triangular sparse linear system

.. toctree::
   :hidden:

   data_types/data_handles
   operations/spmm
   operations/spmv
   operations/spsv
   matrix_view
   supported-types

**Parent topic:** :ref:`onemath_sparse_linear_algebra`
