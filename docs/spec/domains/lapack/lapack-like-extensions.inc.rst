.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack-like-extensions-routines:

LAPACK-like Extensions Routines
===============================


.. container::


   |onemath_full_name| DPC++ provides additional routines to extend
   the functionality of the LAPACK routines. These include routines
   to compute many independent factorizations, linear equation solutions, and similar.
   The following table lists the LAPACK-like Extensions routine groups.


   .. container:: tablenoborder


      .. list-table:: 
         :header-rows: 1

         * -     Routines
           -     Scratchpad Size Routines
           -     Description     
         * -     :ref:`onemath_lapack_geqrf_batch`
           -     :ref:`onemath_lapack_geqrf_batch_scratchpad_size`
           -     Computes the QR factorizations of a batch of general matrices.
         * -     :ref:`onemath_lapack_getrf_batch`
           -     :ref:`onemath_lapack_getrf_batch_scratchpad_size`
           -     Computes the LU factorizations of a batch of general matrices.   
         * -     :ref:`onemath_lapack_getri_batch`
           -     :ref:`onemath_lapack_getri_batch_scratchpad_size`
           -     Computes the inverses of a batch of LU-factored general matrices.   
         * -     :ref:`onemath_lapack_getrs_batch`
           -     :ref:`onemath_lapack_getrs_batch_scratchpad_size`
           -     Solves systems of linear equations with a batch of LU-factored square coefficient matrices, with multiple right-hand sides.    
         * -     :ref:`onemath_lapack_orgqr_batch`
           -     :ref:`onemath_lapack_orgqr_batch_scratchpad_size`
           -     Generates the real orthogonal/complex unitary matrix :math:`Q_i` of the QR factorization formed by geqrf_batch.
         * -     :ref:`onemath_lapack_potrf_batch`
           -     :ref:`onemath_lapack_potrf_batch_scratchpad_size`
           -     Computes the Cholesky factorization of a batch of symmetric (Hermitian) positive-definite matrices.   
         * -     :ref:`onemath_lapack_potrs_batch`
           -     :ref:`onemath_lapack_potrs_batch_scratchpad_size`
           -     Solves systems of linear equations with a batch of Cholesky-factored symmetric (Hermitian) positive-definite coefficient matrices, with multiple right-hand sides.    
         * -     :ref:`onemath_lapack_ungqr_batch`
           -     :ref:`onemath_lapack_ungqr_batch_scratchpad_size`
           -     Generates the complex unitary matrix :math:`Q_i` with the QR factorization formed by geqrf_batch.



.. toctree::
    :hidden:

    geqrf_batch
    geqrf_batch_scratchpad_size
    getrf_batch
    getrf_batch_scratchpad_size
    getri_batch
    getri_batch_scratchpad_size
    getrs_batch
    getrs_batch_scratchpad_size
    orgqr_batch
    orgqr_batch_scratchpad_size
    potrf_batch
    potrf_batch_scratchpad_size
    potrs_batch
    potrs_batch_scratchpad_size
    ungqr_batch
    ungqr_batch_scratchpad_size
