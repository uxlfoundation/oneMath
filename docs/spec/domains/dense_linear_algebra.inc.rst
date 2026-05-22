.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_dense_linear_algebra:

Dense Linear Algebra
---------------------

This section contains information about dense linear algebra routines:

:ref:`matrix-storage` provides information about dense matrix and vector storage formats that are used by oneMath :ref:`onemath_blas` and :ref:`onemath_lapack`.

:ref:`onemath_blas` provides vector, matrix-vector, and matrix-matrix routines for dense matrices and vector operations.

:ref:`value_or_pointer` describes some details of how scalar parameters (such as ``alpha`` and ``beta``) are handled so that users may pass either values or pointers for these parameters.

:ref:`onemath_lapack` provides more complex dense linear algebra routines, e.g., matrix factorization, solving dense systems of linear equations, least square problems, eigenvalue and singular value problems, and performing a number of related computational tasks.

.. toctree::
    :hidden:

    matrix-storage.rst
    value_or_pointer.rst
    blas/blas.rst
    lapack/lapack.rst
