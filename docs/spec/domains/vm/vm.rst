.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_vm:

Vector Math
------------

.. container::

   Application programs that improve performance with VM include
   nonlinear programming software, computation of integrals, financial
   calculations, computer graphics, and many others.


   VM functions fall into the following groups according to the
   operations they perform:


   - :ref:`onemath_vm_mathematical_functions` compute values of mathematical
     functions, such as sine, cosine, exponential, or logarithm, on vectors
     stored contiguously in memory.


   - :ref:`onemath_vm_service_functions` set/get the accuracy modes and the
     error codes, and create error handlers for mathematical functions.


   The VM mathematical functions take an input vector as an argument,
   compute values of the respective function element-wise, and return
   the results in an output vector. All the VM mathematical functions
   can perform in-place operations, where the input and output arrays
   are at the same memory locations.

   - :ref:`onemath_vm_special_value_notations`

.. toctree::
    :hidden:

    special-value-notations.rst
    vm-mathematical-functions.rst
    vm-service-functions.rst
    vm-exceptions.rst
    bibliography.rst

**Parent topic:** :ref:`onemath_vm_inc`