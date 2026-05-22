.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_lapack_sytrf:

sytrf
=====

Computes the Bunch-Kaufman factorization of a symmetric matrix.

.. container:: section

  .. rubric:: Description
      
``sytrf`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``float`` 
        * -  ``double`` 
        * -  ``std::complex<float>`` 
        * -  ``std::complex<double>`` 

The routine computes the factorization of a real/complex symmetric
matrix :math:`A` using the Bunch-Kaufman diagonal pivoting method. The
form of the factorization is:

-  if ``upper_lower=uplo::upper``, :math:`A` = :math:`UDU^{T}`

-  if ``upper_lower=uplo::lower``, :math:`A` = :math:`LDL^{T}`

where :math:`A` is the input matrix, :math:`U` and :math:`L` are products of
permutation and triangular matrices with unit diagonal (upper
triangular for :math:`U` and lower triangular for :math:`L`), and :math:`D` is a
symmetric block-diagonal matrix with :math:`1 \times 1` and :math:`2 \times 2` diagonal
blocks. :math:`U` and :math:`L` have :math:`2 \times 2` unit diagonal blocks
corresponding to the :math:`2 \times 2` blocks of :math:`D`.

sytrf (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      void sytrf(cl::sycl::queue &queue, oneapi::math::uplo upper_lower, std::int64_t n, cl::sycl::buffer<T,1> &a, std::int64_t lda, cl::sycl::buffer<int_64,1> &ipiv, cl::sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

upper_lower
   Indicates whether the upper or lower triangular part of    :math:`A` is stored and how :math:`A` is factored:

      If   ``upper_lower=uplo::upper``, the buffer ``a`` stores the upper triangular   part of the matrix :math:`A`, and :math:`A` is factored as :math:`UDU^T`.

      If ``upper_lower=uplo::lower``, the buffer ``a`` stores   the lower triangular part of the matrix :math:`A`, and :math:`A` is factored   as :math:`LDL^T`.

n
   The order of matrix :math:`A` (:math:`0 \le n`).

a
   The buffer ``a``, size :math:`\max(1,lda \cdot n)`. The buffer ``a``    contains either the upper or the lower triangular part of the matrix   :math:`A` (see ``upper_lower``). The second dimension of ``a`` must be at   least :math:`\max(1, n)`.

lda
   The leading dimension of ``a``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_sytrf_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters
      
a
   The upper or lower triangular part of a is overwritten by    details of the block-diagonal matrix :math:`D` and the multipliers used   to obtain the factor :math:`U` (or :math:`L`).

ipiv
   Buffer, size at least :math:`\max(1, n)`. Contains details of    the interchanges and the block structure of :math:`D`. If   :math:`\text{ipiv}(i)=k>0`, then :math:`d_{ii}` is a :math:`1 \times 1` block, and the   :math:`i`-th row and column of :math:`A` was interchanged with the :math:`k`-th   row and column.

      If ``upper_lower=oneapi::math::uplo::upper``   and :math:`\text{ipiv}(i)=\text{ipiv}(i-1)=-m<0`, then :math:`D` has a :math:`2 \times 2` block in   rows/columns :math:`i` and :math:`i`-1, and (:math:`i-1`)-th row and column of   :math:`A` was interchanged with the :math:`m`-th row and   column.

      If ``upper_lower=oneapi::math::uplo::lower`` and   :math:`\text{ipiv}(i)=\text{ipiv}(i+1)=-m<0`, then :math:`D` has a :math:`2 \times 2` block in   rows/columns :math:`i` and :math:`i+1`, and (:math:`i+1`)-th row and column   of :math:`A` was interchanged with the :math:`m`-th row and column.

scratchpad
   Buffer holding scratchpad memory to be used by routine for storing intermediate results.

.. container:: section

  .. rubric:: Throws
         
This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`

:ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`

:ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`

:ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

:ref:`oneapi::math::lapack::invalid_argument<onemath_lapack_exception_invalid_argument>`

:ref:`oneapi::math::lapack::computation_error<onemath_lapack_exception_computation_error>`

   Exception is thrown in case of problems during calculations. The ``info`` code of the problem can be obtained by `info()` method of exception object:

   If :math:`\text{info}=-i`, the :math:`i`-th parameter had an illegal value.

   If :math:`\text{info}=i`, :math:`d_{ii}` is 0. The factorization has been completed, but :math:`D` is exactly singular. Division by 0 will occur if you use :math:`D` for solving a system of linear equations.

   If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should not be less than value return by `detail()` method of exception object.

sytrf (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::lapack {
      cl::sycl::event sytrf(cl::sycl::queue &queue, oneapi::math::uplo upper_lower, std::int64_t n, T *a, std::int64_t lda, int_64 *ipiv, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

upper_lower
   Indicates whether the upper or lower triangular part of    :math:`A` is stored and how :math:`A` is factored:

      If   ``upper_lower=uplo::upper``, the array ``a`` stores the upper triangular   part of the matrix :math:`A`, and :math:`A` is factored as :math:`UDU^T`.

      If ``upper_lower=uplo::lower``, the array ``a`` stores   the lower triangular part of the matrix :math:`A`, and :math:`A` is factored   as :math:`LDL^T`.

n
   The order of matrix :math:`A` (:math:`0 \le n`).

a
   The pointer to :math:`A`, size :math:`\max(1,\text{lda} \cdot n)`, containing either the upper or the lower triangular part of the matrix   :math:`A` (see ``upper_lower``). The second dimension of ``a`` must be at   least :math:`\max(1, n)`.

lda
   The leading dimension of ``a``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemath_lapack_sytrf_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

a
   The upper or lower triangular part of a is overwritten by    details of the block-diagonal matrix :math:`D` and the multipliers used   to obtain the factor :math:`U` (or :math:`L`).

ipiv
   Pointer to array of size at least :math:`\max(1, n)`. Contains details of    the interchanges and the block structure of :math:`D`. If   :math:`\text{ipiv}(i)=k>0`, then :math:`d_{ii}` is a :math:`1 \times 1` block, and the   :math:`i`-th row and column of :math:`A` was interchanged with the :math:`k`-th   row and column.

      If ``upper_lower=oneapi::math::uplo::upper``   and :math:`\text{ipiv}(i)=\text{ipiv}(i-1)=-m<0`, then :math:`D` has a :math:`2 \times 2` block in   rows/columns :math:`i` and :math:`i-1`, and (:math:`i-1`)-th row and column of   :math:`A` was interchanged with the :math:`m`-th row and   column.
      
      If ``upper_lower=oneapi::math::uplo::lower`` and   :math:`\text{ipiv}(i)=\text{ipiv}(i+1)=-m<0`, then :math:`D` has a :math:`2 \times 2` block in   rows/columns :math:`i` and :math:`i+1`, and (:math:`i+1`)-th row and column   of :math:`A` was interchanged with the :math:`m`-th row and column.

scratchpad
   Pointer to scratchpad memory to be used by routine for storing intermediate results.

.. container:: section

  .. rubric:: Throws
         
This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`

:ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`

:ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`

:ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

:ref:`oneapi::math::lapack::invalid_argument<onemath_lapack_exception_invalid_argument>`

:ref:`oneapi::math::lapack::computation_error<onemath_lapack_exception_computation_error>`

   Exception is thrown in case of problems during calculations. The ``info`` code of the problem can be obtained by `info()` method of exception object:

   If :math:`\text{info}=-i`, the :math:`i`-th parameter had an illegal value.

   If :math:`\text{info}=i`, :math:`d_{ii}` is 0. The factorization has been completed, but :math:`D` is exactly singular. Division by 0 will occur if you use :math:`D` for solving a system of linear equations.

   If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should not be less than value return by `detail()` method of exception object.

.. container:: section

  .. rubric:: Return Values
         
Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemath_lapack-linear-equation-routines`

