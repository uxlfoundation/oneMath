.. SPDX-FileCopyrightText: 2024 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_sparse_linear_algebra:

Sparse Linear Algebra
----------------------

.. container::

   The |onemath_full_name| provides a C++ interface to a set of Sparse
   Linear Algebra routines using SYCL.

   :ref:`onemath_spblas` provides basic operations on sparse vectors and
   matrices. Most operations are split into three stages: query of the external
   workspace size, optimization stage and execution. For a given configuration,
   the first two stages would typically be called once for a set of input
   arguments and the execution stage may be called multiple times. During the
   optimization stage, the API may inspect the matrix properties including size,
   sparsity pattern and available parallelism, and may apply matrix format or
   structure changes to enable a more optimized algorithm. User-provided matrix
   data remain unmodified if such optimizations are made. In the execution
   stage, multiple routine calls can take advantage of the optimization stage
   data in order to improve performance. Each operation has a descriptor type
   that is used to carry information across the different stages.

.. toctree::
   :hidden:

   spblas/spblas.rst

