.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_vm_setmode:

set_mode
========


.. container::


   Sets a new mode for VM functions according to the ``mode`` parameter
   and returns the previous VM mode.


   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle

      .. code-block:: cpp


            namespace oneapi::math::vm {

                oneapi::math::vm::mode set_mode(
                    sycl::queue& exec_queue,
                    oneapi::math::vm::mode new_mode);

            } // namespace oneapi::math::vm


      .. rubric:: Description
         :class: sectiontitle


      The set_mode function sets a new mode for VM functions according
      to the ``new_mode`` parameter and returns the previous VM mode.
      The mode change has a global effect on all the VM functions within
      a queue.

      The ``mode`` parameter is designed to control accuracy for a given queue.

      .. list-table::
         :header-rows: 1

         * - Value of mode
           - Description
         * - Accuracy Control
           -
         * - ``oneapi::math::vm::mode::ha``
           - High accuracy versions of VM functions.
         * - ``oneapi::math::vm::mode::la``
           - Low accuracy versions of VM functions.
         * - ``oneapi::math::vm::mode::ep``
           - Enhanced performance accuracy versions of VM functions.
         * - ``oneapi::math::vm::mode::not_defined``
           - VM mode not defined. This has no effect.

      The assumed value of the ``mode`` parameter for a new queue, if ``set_mode``
      is not called is ``oneapi::math::vm::mode::ha``.



.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   exec_queue
      The queue where the routine should be executed.


   new_mode
      Specifies the VM mode to be set.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   return value (old_mode)
      Specifies the former VM mode.


.. container:: familylinks


   .. container:: parentlink

      **Parent topic:** :ref:`onemath_vm_service_functions`


