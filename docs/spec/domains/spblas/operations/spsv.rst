.. SPDX-FileCopyrightText: 2024 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_sparse_spsv_header:

spsv
====

Solves a system of linear equations where the coefficients are described by a
triangular sparse matrix.

.. rubric:: Description and Assumptions

The ``oneapi::math::sparse::spsv`` routine solves a system of linear equations
for a square matrix:

.. math::

   \text{op}(A) \cdot y \leftarrow \alpha \cdot x


| where:
|   :math:`\alpha` is a scalar,
|   :math:`x` and :math:`y` are dense vectors of size ``m``,
|   :math:`\text{op}(A)` is a transformed sparse matrix of size ``m``-by-``m``,
|   :math:`\text{op}()` is the transform operation using the following description:

.. math::

    \text{op}(A) = \begin{cases} A,& \text{oneapi::math::transpose::nontrans}\\
                                 A^\mathsf{T},& \text{oneapi::math::transpose::trans}\\
                                 A^\mathsf{H},& \text{oneapi::math::transpose::conjtrans}
                   \end{cases}

.. _onemath_sparse_spsv_descr:

spsv_descr
----------

.. rubric:: Definition

.. code-block:: cpp

   namespace oneapi::math::sparse {

       struct spsv_descr;
       using spsv_descr_t = spsv_descr*;

   }

.. container:: section

   .. rubric:: Description

   Defines ``spsv_descr_t`` as an opaque pointer to the incomplete type
   ``spsv_descr``. Each backend may provide a different implementation of the
   type ``spsv_descr``. The `spsv_descr_t` object persists through the various
   stages of the spsv operation to house relevant state, optimizations and
   workspaces.

.. _onemath_sparse_init_spsv_descr:

init_spsv_descr
---------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::sparse {

       void init_spsv_descr (sycl::queue                       &queue,
                             oneapi::math::sparse::spsv_descr_t *p_spsv_descr);

   }

.. container:: section

   .. rubric:: Input parameters

   queue
      The SYCL command queue which will be used for SYCL kernels execution.

   p_spsv_descr
      The address of the ``p_spsv_descr`` object to be initialized. Must only be
      called on an uninitialized ``spsv_descr_t`` object.

.. container:: section

   .. rubric:: Output parameters

   p_spsv_descr
      On return, the address is updated to point to a newly allocated and
      initialized ``spsv_descr_t`` object that can be used to perform spsv.

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

.. _onemath_sparse_release_spsv_descr:

release_spsv_descr
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::sparse {

       sycl::event release_spsv_descr (sycl::queue                       &queue,
                                       oneapi::math::sparse::spsv_descr_t spsv_descr,
                                       const std::vector<sycl::event>    &dependencies = {});

   }

.. container:: section

   .. rubric:: Input parameters

   queue
      The SYCL command queue which will be used for SYCL kernels execution.

   spsv_descr
      Descriptor initialized with ``init_spsv_descr``.

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

.. _onemath_sparse_spsv_alg:

spsv_alg
--------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::sparse {

       enum class spsv_alg {
           default_alg,
           no_optimize_alg,
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

.. _onemath_sparse_spsv:

spsv
----

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::sparse {

       void spsv_buffer_size(
           sycl::queue                                &queue,
           oneapi::math::transpose                     opA,
           const void*                                alpha,
           oneapi::math::sparse::matrix_view           A_view,
           oneapi::math::sparse::matrix_handle_t       A_handle,
           oneapi::math::sparse::dense_vector_handle_t x_handle,
           oneapi::math::sparse::dense_vector_handle_t y_handle,
           oneapi::math::sparse::spsv_alg              alg,
           oneapi::math::sparse::spsv_descr_t          spsv_descr,
           std::size_t                                &temp_buffer_size);

       void spsv_optimize(
           sycl::queue                                &queue,
           oneapi::math::transpose                     opA,
           const void*                                alpha,
           oneapi::math::sparse::matrix_view           A_view,
           oneapi::math::sparse::matrix_handle_t       A_handle,
           oneapi::math::sparse::dense_vector_handle_t x_handle,
           oneapi::math::sparse::dense_vector_handle_t y_handle,
           oneapi::math::sparse::spsv_alg              alg,
           oneapi::math::sparse::spsv_descr_t          spsv_descr,
           sycl::buffer<std::uint8_t, 1>              workspace);

       sycl::event spsv_optimize(
           sycl::queue                                &queue,
           oneapi::math::transpose                     opA,
           const void*                                alpha,
           oneapi::math::sparse::matrix_view           A_view,
           oneapi::math::sparse::matrix_handle_t       A_handle,
           oneapi::math::sparse::dense_vector_handle_t x_handle,
           oneapi::math::sparse::dense_vector_handle_t y_handle,
           oneapi::math::sparse::spsv_alg              alg,
           oneapi::math::sparse::spsv_descr_t          spsv_descr,
           void*                                      workspace,
           const std::vector<sycl::event>             &dependencies = {});

       sycl::event spsv(
           sycl::queue                                &queue,
           oneapi::math::transpose                     opA,
           const void*                                alpha,
           oneapi::math::sparse::matrix_view           A_view,
           oneapi::math::sparse::matrix_handle_t       A_handle,
           oneapi::math::sparse::dense_vector_handle_t x_handle,
           oneapi::math::sparse::dense_vector_handle_t y_handle,
           oneapi::math::sparse::spsv_alg              alg,
           oneapi::math::sparse::spsv_descr_t          spsv_descr,
           const std::vector<sycl::event>             &dependencies = {});

   }

.. container:: section

   .. rubric:: Notes

   - ``spsv_buffer_size`` and ``spsv_optimize`` must be called at least once
     before ``spsv`` with the same arguments. ``spsv`` can then be called
     multiple times. Calling ``spsv_optimize`` on the same descriptor can reset
     some of the descriptor's data such as the ``workspace``.
   - In the general case, not calling the functions in the order specified above
     is undefined behavior. Not calling ``spsv_buffer_size`` or
     ``spsv_optimize`` at least once with a given descriptor will throw a
     :ref:`oneapi::math::uninitialized<onemath_exception_uninitialized>`
     exception. Calling ``spsv`` with arguments not matching ``spsv_optimize``
     will throw an
     :ref:`oneapi::math::invalid_argument<onemath_exception_invalid_argument>`
     exception, unless stated otherwise.
   - The data of the dense handle ``x_handle`` and scalar ``alpha`` can be reset
     before each call to ``spsv``. Changing the data of the sparse handle
     ``A_handle`` is undefined behavior.
   - The data must be available on the device when calling ``spsv_optimize`` by
     using event dependencies if needed.
   - ``spsv_optimize`` and ``spsv`` are asynchronous.
   - The algorithm defaults to ``spsv_alg::default_alg`` if a backend does not
     support the provided algorithm.
   - The container type of all the handles and ``workspace`` must be consistent
     and use either USM pointers or SYCL buffers.

   .. rubric:: Input Parameters

   queue
      The SYCL command queue which will be used for SYCL kernels execution.

   opA
      Specifies operation ``op()`` on the input matrix. The possible options are
      described in :ref:`onemath_enum_transpose` enum class.

   alpha
      Host or USM pointer representing :math:`\alpha`. The USM allocation can be
      on the host or device. Must be a host pointer if SYCL buffers are used.
      Must be of the same type as the handles' data type.

   A_view
      Specifies which part of the handle should be read as described by
      :ref:`onemath_sparse_matrix_view`. The ``type_view`` field must be
      ``matrix_descr::triangular``. The ``diag_view`` field can be
      ``diag::nonunit`` or ``diag::unit``.

   A_handle
      Sparse matrix handle object representing :math:`A`.

   x_handle
      Dense vector handle object representing :math:`x`.

   y_handle
      Dense vector handle object representing :math:`y`.

   alg
      Specifies the :ref:`spsv algorithm<onemath_sparse_spsv_alg>` to use.

   spsv_descr
      Initialized :ref:`spsv descriptor<onemath_sparse_spsv_descr>`.

   temp_buffer_size
      Output buffer size in bytes.

   workspace
      | Workspace buffer or USM pointer, must be at least of size
        ``temp_buffer_size`` bytes and the address aligned on the size of the
        handles' data type.
      | If it is a buffer, its lifetime is extended until the :ref:`spsv
        descriptor<onemath_sparse_spsv_descr>` is released or the workspace is
        reset by ``spsv_optimize``. The workspace cannot be a sub-buffer.
      | If it is a USM pointer, it must not be free'd until the corresponding
        ``spsv`` has completed. The data must be accessible on the device.

   dependencies
      List of events to depend on before starting asynchronous tasks that access
      data on the device. Ignored if buffers are used. Defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   temp_buffer_size
      Output buffer size in bytes. A temporary workspace of at least this size
      must be allocated to perform the specified spsv.

   y_handle
      Dense vector handle object representing :math:`y`, result of the ``spsv``
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
