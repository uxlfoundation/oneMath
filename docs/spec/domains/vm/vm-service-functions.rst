.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_vm_service_functions:

VM Service Functions
====================


.. container::


   The VM Service functions enable you to set/get the accuracy mode and
   error code. These functions are available both in the Fortran and C
   interfaces. The table below lists available VM Service functions and
   their short description.


   .. container:: tablenoborder


      .. list-table::
         :header-rows: 1

         * - Function Short Name
           - Description
         * - :ref:`onemath_vm_setmode`
           - Sets the VM mode for given queue
         * - :ref:`onemath_vm_get_mode`
           - Gets the VM mode for given queue
         * - :ref:`onemath_vm_set_status`
           - Sets the VM status code for given queue
         * - :ref:`onemath_vm_get_status`
           - Gets the VM status code for given queue
         * - :ref:`onemath_vm_clear_status`
           - Clears the VM status code for given queue
         * - :ref:`onemath_vm_create_error_handler`
           - Creates the local VM error handler for a function


**Parent topic:** :ref:`onemath_vm`

.. toctree::
    :hidden:

    setmode
    get_mode
    set_status
    get_status
    clear_status
    create_error_handler
