.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _blas-like-extensions:

BLAS-like Extensions
====================


.. container::


   |onemath_full_name| DPC++ provides additional routines to extend
   the functionality of the BLAS routines. These include routines
   to compute many independent vector-vector and matrix-matrix operations.

   The following table lists the BLAS-like extensions with their descriptions.


   .. container:: tablenoborder


      .. list-table:: 
         :header-rows: 1

         * -     Routines
           -     Description     
         * -     :ref:`onemath_blas_axpy_batch`
           -     Computes groups of vector-scalar products added to a vector.
         * -     :ref:`onemath_blas_gemm_batch`
           -     Computes groups of matrix-matrix products with general matrices.   
         * -     :ref:`onemath_blas_trsm_batch`
           -     Solves a triangular matrix equation for a group of matrices.   
         * -     :ref:`onemath_blas_gemmt` (deprecated)
           -     **gemmt has been deprecated. Use level 3** :ref:`onemath_blas_gemmtr` **instead.**

                 Computes a matrix-matrix product with general matrices, but updates
                 only the upper or lower triangular part of the result matrix.
         * -     :ref:`onemath_blas_gemm_bias`
           -     Computes a matrix-matrix product using general integer matrices with bias
         * -     :ref:`onemath_blas_imatcopy`
           -     Computes an in-place matrix transposition or copy.
         * -     :ref:`onemath_blas_omatcopy`
           -     Computes an out-of-place matrix transposition or copy.
         * -     :ref:`onemath_blas_omatcopy2`
           -     Computes a two-strided out-of-place matrix transposition or copy.
         * -     :ref:`onemath_blas_omatadd`
           -     Computes scaled matrix addition with possibly transposed arguments.
         * -     :ref:`onemath_blas_imatcopy_batch`
           -     Computes groups of in-place matrix transposition or copy operations.
         * -     :ref:`onemath_blas_omatcopy_batch`
           -     Computes groups of out-of-place matrix transposition or copy operations.
         * -     :ref:`onemath_blas_omatadd_batch`
           -     Computes groups of scaled matrix additions.
 




.. toctree::
    :hidden:

    axpy_batch
    axpby
    copy_batch
    dgmm_batch
    gemm_batch
    gemv_batch
    syrk_batch
    trsm_batch
    gemmt
    gemm_bias
    imatcopy
    omatcopy
    omatcopy2
    omatadd
    imatcopy_batch
    omatcopy_batch
    omatadd_batch

**Parent topic:** :ref:`onemath_blas`
