.. SPDX-FileCopyrightText: 2024 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_sparse_matrix_view_and_descr:

Matrix view
===========

.. _onemath_sparse_matrix_descriptor:

matrix_descr
------------

.. rubric:: Definition

.. code:: cpp

   namespace oneapi::math::sparse {

       enum class matrix_descr {
           general,
           symmetric,
           hermitian,
           triangular,
           diagonal,
       };

   }

.. container:: section

   .. rubric:: Description

   The matrix descriptor describes how an operation should interpret the data.

   .. list-table::
      :header-rows: 1
      :widths: 20 80

      * - Value
        - Description
      * - ``general``
        - General case, use complete data.
      * - ``symmetric``
        - View as symmetric, use given triangular part.
      * - ``hermitian``
        - View as hermitian, use given triangular part.
      * - ``triangular``
        - View as triangular, use given triangular part.
      * - ``diagonal``
        - View as diagonal, use only main diagonal values.

.. _onemath_sparse_matrix_view:

matrix_view
-----------

.. rubric:: Definition

.. code:: cpp

   namespace oneapi::math::sparse {

       struct matrix_view {
           matrix_descr type_view = matrix_descr::general;
           uplo uplo_view = uplo::lower;
           diag diag_view = diag::nonunit;

           matrix_view() = default;

           matrix_view(matrix_descr type_view);
       };

   }

.. container:: section

   .. rubric:: Description

   The matrix view holds information to specify which part of the matrix should
   be read without changing the matrix's data.

   See :ref:`onemath_sparse_matrix_descriptor`, :ref:`onemath_enum_uplo` and
   :ref:`onemath_enum_diag` for a description of the members.

   Each operation documents which combination of ``type_view``, ``uplo_view``
   and ``diag_view`` are valid.

   .. rubric:: Syntax

   .. code-block:: cpp

      namespace oneapi::math::sparse {

          matrix_view::matrix_view () = default;

      }

   .. container:: section

      .. rubric:: Default Constructor

      Initializes the ``matrix_view`` with the default values as shown in the class
      definition.

   .. rubric:: Syntax

   .. code-block:: cpp

      namespace oneapi::math::sparse {

          matrix_view::matrix_view(matrix_descr type_view);

      }

   .. container:: section

      .. rubric:: Constructor from a matrix_descr

      Initializes the ``matrix_view`` with the provided ``matrix_descr``. By default
      the other members are initialized to the same value as the default
      constructor.

**Parent topic:** :ref:`onemath_spblas`
