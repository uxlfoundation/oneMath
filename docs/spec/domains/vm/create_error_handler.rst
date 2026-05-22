.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_vm_create_error_handler:

create_error_handler
====================


.. container::


   Creates an error handler for VM functions that support computational error handling.


   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      Buffer API:


      .. code-block:: cpp


            namespace oneapi::math::vm {

            oneapi::math::vm::error_handler<T> create_error_handler(
                sycl::buffer<oneapi::math::vm::status, 1> & status_array,
                int64_t length = 1,
                oneapi::math::vm::status status = oneapi::math::vm::status::not_defined,
                T fixup = {},
                bool copysign = false);

            } // namespace oneapi::math::vm


      USM API:


      .. code-block:: cpp


            namespace oneapi::math::vm {

            oneapi::math::vm::error_handler<T> create_error_handler(
                oneapi::math::vm::status* status_array,
                int64_t length = 1,
                oneapi::math::vm::status status = oneapi::math::vm::status::not_defined,
                T fixup = {},
                bool copysign = false);

            } // namespace oneapi::math::vm


    ``create_error_handler`` supports the following precisions.


    .. list-table::
        :header-rows: 1

        * - T
        * - ``float``
        * - ``double``
        * - ``std::complex<float>``
        * - ``std::complex<double>``


   .. container:: section


      .. rubric:: Description
         :class: sectiontitle


      `create_error_handler` creates an computational error handler to be passed to VM
      functions that support computational error handling.

      A VM computational error handler supports three modes:

      - **Single status mode**: all computational errors that happened during the execution
        of a function are being written into one single status variable.

        After the execution, the single value is either un-changed if no
        errors happened or contains bitwise-OR of initial value and non-success status codes
        occurred during computation.

        To enable this mode, ``status_array`` must point to any ``status``-type array or buffer
        of 1 or more elements and ``length`` must be 1.

      - **Multiple status mode**: each non-successful status code  is saved in ``status_array``
        at the same index as the argument causing the non-success status code.

        Success status codes are not written to ``status_array``. This means the array
        needs to be allocated and initialized before function execution.

        To enable this mode, ``status_array`` must have at least the same length
        as the argument and result vectors, and ``length`` must be set to this
        length.

      - **Fixup mode**: for all arguments that caused a specific error status,
        results are overwritten by a user-defined value.

        To enable this mode, the target ``status`` and ``fixup`` values
        must be set. The fixup value is written to results for each argument
        for which calculation resulted in the ``status`` status code.

        To fix multiple error status codes, ``status`` can be provided with bitwise-OR of status codes.

        If ``copysign`` is set to true then the sign of ``fixup`` is set to
        the same sign as the argument that caused the status code – a suitable
        option for symmetric math functions.


      The following table lists the possible computational status code values.


      .. container:: tablenoborder


         .. list-table::
            :header-rows: 1

            * - Status code
              - Description
            * - Successful Execution
              -
            * - ``oneapi::math::vm::status::success``
              - VM function execution completed successfully
            * - ``oneapi::math::vm::status::not_defined``
              - VM status not defined
            * - Warnings
              -
            * - ``oneapi::math::vm::status::accuracy_warning``
              - VM function execution completed successfully in a different
                accuracy mode
            * - Computational Errors
              -
            * - ``oneapi::math::vm::status::errdom``
              - Values are out of a range of definition producing invalid
                (QNaN) result
            * - ``oneapi::math::vm::status::sing``
              - Values cause divide-by-zero (singularity) computational errors and produce
                and invalid (QNaN or Inf) result
            * - ``oneapi::math::vm::status::overflow``
              - An overflow happened during the calculation process
            * - ``oneapi::math::vm::status::underflow``
              - An underflow happened during the calculation process


      Notes:

      - ``status_array`` must be allocated and initialized before calling VM
        functions in multiple status error handling mode.

        The array should be large enough to contain ``n`` status codes, where
        ``n`` is the same as the input/output vector size for the VM function.

      - If no arguments are passed to ``create_error_handler``, then an empty
        object is created with all three error handling modes disabled.

        In this case, the VM math functions set the global status code only.


   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      status_array
         Array to store status codes (should be a buffer for buffer
         API).


      length
         Length of the errarray. This is an optional argument, default
         value is ``1``.


      status_code
         Status code to match and fix the results. This is an optional argument,
         default value is ``oneapi::math::vm::status::not_defined``.


      fixup
         Fixup value for results. This is an optional argument, default
         value is ``0.0``.


      copysign
         Flag for setting the fixup value's sign the same as the
         argument’s. This is an optional argument, default value
         ``false``.


   .. container:: section


      .. rubric:: Output Parameters
         :class: sectiontitle


      return value
         Specifies the error handler object to be created.


.. container:: familylinks


   .. container:: parentlink

      **Parent topic:** :ref:`onemath_vm_service_functions`
