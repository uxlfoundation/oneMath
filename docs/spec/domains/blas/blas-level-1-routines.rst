.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _blas-level-1-routines:

BLAS Level 1 Routines
=====================


.. container::


   BLAS Level 1 includes routines which perform
   vector-vector operations as described in the following table. 


   .. container:: tablenoborder


      .. list-table:: 
         :header-rows: 1

         * -     Routines
           -     Description     
         * -     :ref:`onemath_blas_asum`
           -     Sum of vector magnitudes      
         * -     :ref:`onemath_blas_axpy`
           -     Scalar-vector product      
         * -     :ref:`onemath_blas_copy`
           -     Copy vector      
         * -     :ref:`onemath_blas_dot`
           -     Dot product      
         * -     :ref:`onemath_blas_sdsdot`
           -     Dot product with double precision      
         * -     :ref:`onemath_blas_dotc`
           -     Dot product conjugated      
         * -     :ref:`onemath_blas_dotu`
           -     Dot product unconjugated      
         * -     :ref:`onemath_blas_nrm2`
           -     Vector 2-norm (Euclidean norm)      
         * -     :ref:`onemath_blas_rot`
           -     Plane rotation of points      
         * -     :ref:`onemath_blas_rotg`
           -     Generate Givens rotation of points      
         * -     :ref:`onemath_blas_rotm`
           -     Modified Givens plane rotation of points           
         * -     :ref:`onemath_blas_rotmg`
           -     Generate modified Givens plane rotation of points           
         * -     :ref:`onemath_blas_scal`
           -     Vector-scalar product      
         * -     :ref:`onemath_blas_swap`
           -     Vector-vector swap      
         * -     :ref:`onemath_blas_iamax`
           -     Index of the maximum absolute value element of a vector     
         * -     :ref:`onemath_blas_iamin`
           -     Index of the minimum absolute value element of a vector     

.. toctree::
    :hidden:

    asum
    axpy
    copy
    dot
    sdsdot
    dotc
    dotu
    nrm2
    rot
    rotg
    rotm
    rotmg
    scal
    swap
    iamax
    iamin


**Parent topic:** :ref:`onemath_blas`
