.. SPDX-FileCopyrightText: 2024 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_sparse_spmm_header:

spmm
====

Computes a sparse matrix by dense matrix product.

.. rubric:: Description and Assumptions

The ``oneapi::math::sparse::spmm`` routine computes a sparse matrix by dense
matrix product defined as:

.. math::

   C \leftarrow \alpha \cdot \text{op}(A) \cdot \text{op}(B) + \beta \cdot C


| where:
|   :math:`\alpha` and :math:`\beta` are scalars,
|   :math:`C` is a dense matrix of size ``m``-by-``n``,
|   :math:`\text{op}(A)` is a transformed sparse matrix of size ``m``-by-``k``,
|   :math:`\text{op}(B)` is a transformed dense matrix of size ``k``-by-``n``,
|   :math:`\text{op}()` is the transform operation using the following description:

.. math::

    \text{op}(A) = \begin{cases} A,& \text{oneapi::math::transpose::nontrans}\\
                                 A^\mathsf{T},& \text{oneapi::math::transpose::trans}\\
                                 A^\mathsf{H},& \text{oneapi::math::transpose::conjtrans}
                   \end{cases}

.. _onemath_sparse_spmm_descr:

spmm_descr
----------

.. rubric:: Definition

.. code-block:: cpp

   namespace oneapi::math::sparse {

       struct spmm_descr;
       using spmm_descr_t = spmm_descr*;

   }

.. container:: section

   .. rubric:: Description

   Defines ``spmm_descr_t`` as an opaque pointer to the incomplete type
   ``spmm_descr``. Each backend may provide a different implementation of the
   type ``spmm_descr``. The ``spmm_descr_t`` object persists through the various
   stages of the spmm operation to house relevant state, optimizations and
   workspaces.

.. _onemath_sparse_init_spmm_descr:

init_spmm_descr
---------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::sparse {

       void init_spmm_descr (sycl::queue                       &queue,
                             oneapi::math::sparse::spmm_descr_t *p_spmm_descr);

   }

.. container:: section

   .. rubric:: Input parameters

   queue
      The SYCL command queue which will be used for SYCL kernels execution.

   p_spmm_descr
      The address of the ``p_spmm_descr`` object to be initialized. Must only be
      called on an uninitialized ``spmm_descr_t`` object.

.. container:: section

   .. rubric:: Output parameters

   p_spmm_descr
      On return, the address is updated to point to a newly allocated and
      initialized ``spmm_descr_t`` object that can be used to perform spmm.

.. container:: section

   .. rubric:: Throws

   This function shall throw the following exceptions if the associated
   condition is detected. An implementation may throw additional
   implementation-specific exception(s) in case of error conditions not covered
   here.

   | :ref:`oneapi::math::host_bad_alloc<onemath_exception_host_bad_alloc>`
   | :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
   | :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
   | :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

.. _onemath_sparse_release_spmm_descr:

release_spmm_descr
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::sparse {

       sycl::event release_spmm_descr (sycl::queue                       &queue,
                                       oneapi::math::sparse::spmm_descr_t spmm_descr,
                                       const std::vector<sycl::event>    &dependencies = {});

   }

.. container:: section

   .. rubric:: Input parameters

   queue
      The SYCL command queue which will be used for SYCL kernels execution.

   spmm_descr
      Descriptor initialized with ``init_spmm_descr``.

   dependencies
      List of events to depend on before starting asynchronous tasks that access
      data on the device. Defaults to no dependencies.

.. container:: section

   .. rubric:: Return Values

   Output event that can be waited upon or added as a dependency for the
   completion of the function.

.. container:: section

   .. rubric:: Throws

   This function shall throw the following exceptions if the associated
   condition is detected. An implementation may throw additional
   implementation-specific exception(s) in case of error conditions not covered
   here.

   | :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
   | :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
   | :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

.. _onemath_sparse_spmm_alg:

spmm_alg
--------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::sparse {

       enum class spmm_alg {
           default_alg,
           no_optimize_alg,
           coo_alg1,
           coo_alg2,
           coo_alg3,
           coo_alg4,
           csr_alg1,
           csr_alg2,
           csr_alg3,
       };

   }

.. container:: section

   .. rubric:: Description

   These algorithm enums are provided in case backends would like to implement
   various different algorithms for the operation. Behavior of the algorithms
   (e.g., bitwise reproducibility, atomics usage) and the preconditions to using
   specific algorithms (e.g. sortedness of matrix arrays) is
   implementation-defined and must be documented in the library implementing the
   oneAPI specification.

.. _onemath_sparse_spmm:

spmm
----

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::sparse {

       void spmm_buffer_size(
           sycl::queue                                &queue,
           oneapi::math::transpose                     opA,
           oneapi::math::transpose                     opB,
           const void*                                alpha,
           oneapi::math::sparse::matrix_view           A_view,
           oneapi::math::sparse::matrix_handle_t       A_handle,
           oneapi::math::sparse::dense_matrix_handle_t B_handle,
           const void*                                beta,
           oneapi::math::sparse::dense_matrix_handle_t C_handle,
           oneapi::math::sparse::spmm_alg              alg,
           oneapi::math::sparse::spmm_descr_t          spmm_descr,
           std::size_t                                &temp_buffer_size);

       void spmm_optimize(
           sycl::queue                                &queue,
           oneapi::math::transpose                     opA,
           oneapi::math::transpose                     opB,
           const void*                                alpha,
           oneapi::math::sparse::matrix_view           A_view,
           oneapi::math::sparse::matrix_handle_t       A_handle,
           oneapi::math::sparse::dense_matrix_handle_t B_handle,
           const void*                                beta,
           oneapi::math::sparse::dense_matrix_handle_t C_handle,
           oneapi::math::sparse::spmm_alg              alg,
           oneapi::math::sparse::spmm_descr_t          spmm_descr,
           sycl::buffer<std::uint8_t, 1>              workspace);

       sycl::event spmm_optimize(
           sycl::queue                                &queue,
           oneapi::math::transpose                     opA,
           oneapi::math::transpose                     opB,
           const void*                                alpha,
           oneapi::math::sparse::matrix_view           A_view,
           oneapi::math::sparse::matrix_handle_t       A_handle,
           oneapi::math::sparse::dense_matrix_handle_t B_handle,
           const void*                                beta,
           oneapi::math::sparse::dense_matrix_handle_t C_handle,
           oneapi::math::sparse::spmm_alg              alg,
           oneapi::math::sparse::spmm_descr_t          spmm_descr,
           void*                                      workspace,
           const std::vector<sycl::event>             &dependencies = {});

       sycl::event spmm(
           sycl::queue                                &queue,
           oneapi::math::transpose                     opA,
           oneapi::math::transpose                     opB,
           const void*                                alpha,
           oneapi::math::sparse::matrix_view           A_view,
           oneapi::math::sparse::matrix_handle_t       A_handle,
           oneapi::math::sparse::dense_matrix_handle_t B_handle,
           const void*                                beta,
           oneapi::math::sparse::dense_matrix_handle_t C_handle,
           oneapi::math::sparse::spmm_alg              alg,
           oneapi::math::sparse::spmm_descr_t          spmm_descr,
           const std::vector<sycl::event>             &dependencies = {});

   }

.. container:: section

   .. rubric:: Notes

   - ``spmm_buffer_size`` and ``spmm_optimize`` must be called at least once
     before ``spmm`` with the same arguments. ``spmm`` can then be called
     multiple times. Calling ``spmm_optimize`` on the same descriptor can reset
     some of the descriptor's data such as the ``workspace``.
   - In the general case, not calling the functions in the order specified above
     is undefined behavior. Not calling ``spmm_buffer_size`` or
     ``spmm_optimize`` at least once with a given descriptor will throw an
     :ref:`oneapi::math::uninitialized<onemath_exception_uninitialized>`
     exception. Calling ``spmm`` with arguments not matching ``spmm_optimize``
     will throw a
     :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
     exception, unless stated otherwise.
   - The data of the dense handles ``B_handle`` and ``C_handle`` and the scalars
     ``alpha`` and ``beta`` can be reset before each call to ``spmm``. Changing
     the data of the sparse handle ``A_handle`` is undefined behavior.
   - The data must be available on the device when calling ``spmm_optimize`` by
     using event dependencies if needed.
   - ``spmm_optimize`` and ``spmm`` are asynchronous.
   - The algorithm defaults to ``spmm_alg::default_alg`` if a backend does not
     support the provided algorithm.
   - The container type of all the handles and ``workspace`` must be consistent
     and use either USM pointers or SYCL buffers.

   .. rubric:: Input Parameters

   queue
      The SYCL command queue which will be used for SYCL kernels execution.

   opA
      Specifies operation ``op()`` on the input matrix A. The possible options
      are described in :ref:`onemath_enum_transpose` enum class.

   opB
      Specifies operation ``op()`` on the input matrix B. The possible options
      are described in :ref:`onemath_enum_transpose` enum class.

   alpha
      Host or USM pointer representing :math:`\alpha`. The USM allocation can be
      on the host or device. The requirements are:

      * Must use the same kind of memory as ``beta``.
      * Must be a host pointer if SYCL buffers are used.
      * Must be of the same type as the handles' data type.

   A_view
      Specifies which part of the handle should be read as described by
      :ref:`onemath_sparse_matrix_view`. The ``type_view`` field must be
      ``matrix_descr::general`` and the ``uplo_view`` and ``diag_view`` fields
      are ignored.

   A_handle
      Sparse matrix handle object representing :math:`A`.

   B_handle
      Dense matrix handle object representing :math:`B`.

   beta
      Host or USM pointer representing :math:`\beta`. The USM allocation can be
      on the host or device. The requirements are:

      * Must use the same kind of memory as ``alpha``.
      * Must be a host pointer if SYCL buffers are used.
      * Must be of the same type as the handles' data type.

   C_handle
      Dense matrix handle object representing :math:`C`.

   alg
      Specifies the :ref:`spmm algorithm<onemath_sparse_spmm_alg>` to use.

   spmm_descr
      Initialized :ref:`spmm descriptor<onemath_sparse_spmm_descr>`.

   temp_buffer_size
      Output buffer size in bytes.

   workspace
      | Workspace buffer or USM pointer, must be at least of size
        ``temp_buffer_size`` bytes and the address aligned on the size of the
        handles' data type.
      | If it is a buffer, its lifetime is extended until the :ref:`spmm
        descriptor<onemath_sparse_spmm_descr>` is released or the workspace is
        reset by ``spmm_optimize``. The workspace cannot be a sub-buffer.
      | If it is a USM pointer, it must not be free'd until the corresponding
        ``spmm`` has completed. The data must be accessible on the device.

   dependencies
      List of events to depend on before starting asynchronous tasks that access
      data on the device. Ignored if buffers are used. Defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   temp_buffer_size
      Output buffer size in bytes. A temporary workspace of at least this size
      must be allocated to perform the specified spmm.

   C_handle
      Dense matrix handle object representing :math:`C`, result of the ``spmm``
      operation.

.. container:: section

   .. rubric:: Return Values

   Output event that can be waited upon or added as a dependency for the
   completion of the function. May be an empty event if buffers are used.

.. container:: section

   .. rubric:: Throws

   These functions shall throw the following exceptions if the associated
   condition is detected. An implementation may throw additional
   implementation-specific exception(s) in case of error conditions not covered
   here.

   | :ref:`oneapi::math::computation_error<onemath_exception_computation_error>`
   | :ref:`oneapi::math::device_bad_alloc<onemath_exception_device_bad_alloc>`
   | :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
   | :ref:`oneapi::math::unimplemented<onemath_exception_unimplemented>`
   | :ref:`oneapi::math::uninitialized<onemath_exception_uninitialized>`
   | :ref:`oneapi::math::unsupported_device<onemath_exception_unsupported_device>`

**Parent topic:** :ref:`onemath_spblas`
