.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_blas_rotm:

rotm
====

Performs modified Givens rotation of points in the plane.

.. _onemath_blas_rotm_description:

.. rubric:: Description

Given two vectors ``x`` and ``y``, each vector element of these
vectors is replaced as follows:

.. math::

      \begin{bmatrix}x_i \\ y_i\end{bmatrix}=
      H
      \begin{bmatrix}x_i \\ y_i\end{bmatrix} 

for ``i`` from 1 to ``n``, where ``H`` is a modified Givens
transformation matrix.

``rotm`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 

.. _onemath_blas_rotm_buffer:

rotm (Buffer Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       void rotm(sycl::queue &queue,
                 std::int64_t n,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx,
                 sycl::buffer<T,1> &y,
                 std::int64_t incy,
                 sycl::buffer<T,1> &param)
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       void rotm(sycl::queue &queue,
                 std::int64_t n,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx,
                 sycl::buffer<T,1> &y,
                 std::int64_t incy,
                 sycl::buffer<T,1> &param)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   n
      Number of elements in vector ``x``.

   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

   y
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incy``)). See :ref:`matrix-storage` for
      more details.

   incy
      Stride of vector ``y``.

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

   .. rubric:: Output Parameters

   x
      Buffer holding updated buffer ``x``.

   y
      Buffer holding updated buffer ``y``.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
       
   
   :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`
       

   :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
       

   :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
       

   :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
      

.. _onemath_blas_rotm_usm:

rotm (USM Version)
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::blas::column_major {
       sycl::event rotm(sycl::queue &queue,
                        std::int64_t n,
                        T *x,
                        std::int64_t incx,
                        T *y,
                        std::int64_t incy,
                        const T *param,
                        const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::math::blas::row_major {
       sycl::event rotm(sycl::queue &queue,
                        std::int64_t n,
                        T *x,
                        std::int64_t incx,
                        T *y,
                        std::int64_t incy,
                        const T *param,
                        const std::vector<sycl::event> &dependencies = {})
   }
   
.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   n
      Number of elements in vector ``x``.

   x
      Pointer to the input vector ``x``. The array holding the vector
      ``x`` must be of size at least (1 + (``n`` - 1)*abs(``incx``)).
      See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

   yparam
      Pointer to the input vector ``y``. The array holding the vector
      ``y`` must be of size at least (1 + (``n`` - 1)*abs(``incy``)).
      See :ref:`matrix-storage` for
      more details.

   incy
      Stride of vector ``y``.

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
   
   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   x
      Pointer to the updated array ``x``.

   y
      Pointer to the updated array ``y``.

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
