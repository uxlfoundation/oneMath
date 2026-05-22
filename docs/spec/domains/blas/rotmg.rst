.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_rotmg:

rotmg
=====

Computes the parameters for a modified Givens rotation.

.. _onemath_blas_rotmg_description:

.. rubric:: Description

Given Cartesian coordinates (``x1``, ``y1``) of an
input vector, the ``rotmg`` routines compute the components of a modified
Givens transformation matrix ``H`` that zeros the ``y``-component of
the resulting vector:

.. math::

      \begin{bmatrix}x1 \\ 0\end{bmatrix}=
      H
      \begin{bmatrix}x1\sqrt{d1} \\ y1\sqrt{d2}\end{bmatrix} 
      
``rotmg`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 

.. _onemath_blas_rotmg_buffer:

rotmg (Buffer Version)
----------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void rotmg(sycl::queue &queue,
                  sycl::buffer<T,1> &d1,
                  sycl::buffer<T,1> &d2,
                  sycl::buffer<T,1> &x1,
                  sycl::buffer<T,1>  y1,
                  sycl::buffer<T,1> &param)
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void rotmg(sycl::queue &queue,
                  sycl::buffer<T,1> &d1,
                  sycl::buffer<T,1> &d2,
                  sycl::buffer<T,1> &x1,
                  sycl::buffer<T,1>  y1,
                  sycl::buffer<T,1> &param)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   d1
      Buffer holding the scaling factor for the ``x``-coordinate of the
      input vector.

   d2
      Buffer holding the scaling factor for the ``y``-coordinate of the
      input vector.

   x1
      Buffer holding the ``x``-coordinate of the input vector.

   y1
      Scalar specifying the ``y``-coordinate of the input vector.

.. container:: section

   .. rubric:: Output Parameters

   d1
      Buffer holding the first diagonal element of the updated matrix.

   d2
      Buffer holding the second diagonal element of the updated matrix.

   x1
      Buffer holding the ``x``-coordinate of the rotated vector before
      scaling

   param
      Buffer holding an array of size 5.

      The elements of the ``param`` array are:

      ``param[0]`` contains a switch, ``flag``. The other array elements
      ``param[1-4]`` contain the components of the modified Givens 
      transformation matrix ``H``:
      h\ :sub:`11`, h\ :sub:`21`, h\ :sub:`12`, and
      h\ :sub:`22`, respectively.

      Depending on the values of ``flag``, the components of ``H`` are
      set as follows:

      | ``flag = -1.0``:

      .. math::
   
         H=\begin{bmatrix}h_{11} & h_{12} \\ h_{21} & h_{22}\end{bmatrix} 

      | ``flag = 0.0``:

      .. math::
   
         H=\begin{bmatrix}1.0 & h_{12} \\ h_{21} & 1.0\end{bmatrix} 

      | ``flag = 1.0``:

      .. math::
   
         H=\begin{bmatrix}h_{11} & 1.0 \\ -1.0 & h_{22}\end{bmatrix} 

      | ``flag = -2.0``:
      
      .. math::
   
         H=\begin{bmatrix}1.0 & 0.0 \\ 0.0 & 1.0\end{bmatrix} 

      In the last three cases, the matrix entries of 1.0, -1.0, and 0.0
      are assumed based on the value of ``flag`` and are not required to
      be set in the ``param`` vector.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

.. _onemath_blas_rotmg_usm:

rotmg (USM Version)
-------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event rotmg(sycl::queue &queue,
                         T *d1,
                         T *d2,
                         T *x1,
                         value_or_pointer<T>  y1,
                         T *param,
                         const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event rotmg(sycl::queue &queue,
                         T *d1,
                         T *d2,
                         T *x1,
                         value_or_pointer<T>  y1,
                         T *param,
                         const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   d1
      Pointer to the scaling factor for the ``x``-coordinate of the
      input vector.

   d2
      Pointer to the scaling factor for the ``y``-coordinate of the
      input vector.

   x1
      Pointer to the ``x``-coordinate of the input vector.

   y1
      Scalar specifying the ``y``-coordinate of the input vector. See :ref:`value_or_pointer` for more details.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   d1
      Pointer to the first diagonal element of the updated matrix.

   d2
      Pointer to the second diagonal element of the updated matrix.

   x1
      Pointer to the ``x``-coordinate of the rotated vector before
      scaling

   param
      Buffer holding an array of size 5.

      The elements of the ``param`` array are:

      ``param[0]`` contains a switch, ``flag``. The other array elements
      ``param[1-4]`` contain the components of the modified Givens 
      transformation matrix ``H``:
      h\ :sub:`11`, h\ :sub:`21`, h\ :sub:`12`, and
      h\ :sub:`22`, respectively.

      Depending on the values of ``flag``, the components of ``H``
      are set as follows:

      | ``flag = -1.0``:

      .. math::
   
         H=\begin{bmatrix}h_{11} & h_{12} \\ h_{21} & h_{22}\end{bmatrix} 

      | ``flag = 0.0``:

      .. math::
   
         H=\begin{bmatrix}1.0 & h_{12} \\ h_{21} & 1.0\end{bmatrix} 

      | ``flag = 1.0``:

      .. math::
   
         H=\begin{bmatrix}h_{11} & 1.0 \\ -1.0 & h_{22}\end{bmatrix} 

      | ``flag = -2.0``:
      
      .. math::
   
         H=\begin{bmatrix}1.0 & 0.0 \\ 0.0 & 1.0\end{bmatrix} 

      In the last three cases, the matrix entries of 1.0, -1.0, and 0.0
      are assumed based on the value of ``flag`` and are not required to
      be set in the ``param`` vector.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

   **Parent topic:** :ref:`blas-level-1-routines`
