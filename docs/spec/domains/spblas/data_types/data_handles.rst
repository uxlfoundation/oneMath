.. SPDX-FileCopyrightText: 2024 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_sparse_data_handles:

Data handles
============

.. _onemath_sparse_dense_vector_handle:

Dense vector handle
-------------------

.. rubric:: Definition

.. code-block:: cpp

   namespace oneapi::math::sparse {

       struct dense_vector_handle;
       using dense_vector_handle_t = dense_vector_handle*;

   }

.. container:: section

   .. rubric:: Description

   Defines ``dense_vector_handle_t`` as an opaque pointer to the incomplete type
   ``dense_vector_handle``. Each backend may provide a different
   implementation of the type ``dense_vector_handle``.

   See related functions:

   - :ref:`onemath_sparse_init_dense_vector`
   - :ref:`onemath_sparse_set_dense_vector_data`
   - :ref:`onemath_sparse_release_dense_vector`

.. _onemath_sparse_dense_matrix_handle:

Dense matrix handle
-------------------

.. rubric:: Definition

.. code-block:: cpp

   namespace oneapi::math::sparse {

       struct dense_matrix_handle;
       using dense_matrix_handle_t = dense_matrix_handle*;

   }

.. container:: section

   .. rubric:: Description

   Defines ``dense_matrix_handle_t`` as an opaque pointer to the incomplete type
   ``dense_matrix_handle``. Each backend may provide a different
   implementation of the type ``dense_matrix_handle``.

   See related functions:

   - :ref:`onemath_sparse_init_dense_matrix`
   - :ref:`onemath_sparse_set_dense_matrix_data`
   - :ref:`onemath_sparse_release_dense_matrix`

.. _onemath_sparse_matrix_handle:

Sparse matrix handle
--------------------

.. rubric:: Definition

.. code-block:: cpp

   namespace oneapi::math::sparse {

       struct matrix_handle;
       using matrix_handle_t = matrix_handle*;

   }

.. container:: section

   .. rubric:: Description

   Defines ``matrix_handle_t`` as an opaque pointer to the incomplete type
   ``matrix_handle``. Each backend may provide a different
   implementation of the type ``matrix_handle``.

   See related functions:

   - :ref:`onemath_sparse_init_coo_matrix`
   - :ref:`onemath_sparse_init_csr_matrix`
   - :ref:`onemath_sparse_set_coo_matrix_data`
   - :ref:`onemath_sparse_set_csr_matrix_data`
   - :ref:`onemath_sparse_set_matrix_property`
   - :ref:`onemath_sparse_release_sparse_matrix`

   See a description of the supported :ref:`sparse formats<onemath_sparse_format_descriptions>`.

.. toctree::
   :hidden:

   init_dense_vector
   init_dense_matrix
   init_coo_matrix
   init_csr_matrix
   release_dense_vector
   release_dense_matrix
   release_sparse_matrix
   set_dense_vector_data
   set_dense_matrix_data
   set_coo_matrix_data
   set_csr_matrix_data
   set_matrix_property
   format-descriptions

**Parent topic:** :ref:`onemath_spblas`
