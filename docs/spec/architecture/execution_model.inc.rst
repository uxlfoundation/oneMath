.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_execution_model:

Execution Model
---------------

This section describes the execution environment common to all oneMath functionality. The execution environment includes how data is provided to computational routines in :ref:`onemath_queues`, support for several devices in :ref:`onemath_device_usage`, synchronous and asynchronous execution models in :ref:`onemath_asynchronous_synchronous_execution` and :ref:`onemath_host_thread_safety`.

.. _onemath_queues:

Use of Queues
+++++++++++++

The ``sycl::queue`` defined in the oneAPI DPC++ specification is used to specify the device and features enabled on that device on which a task will be enqueued.  There are two forms of computational routines in oneMath: class based :ref:`onemath_member_functions` and standalone :ref:`onemath_nonmember_functions`.  As these may interact with the ``sycl::queue`` in different ways, we provide a section for each one to describe assumptions.


.. _onemath_nonmember_functions:

Non-Member Functions
********************

Each oneMath non-member computational routine takes a ``sycl::queue`` reference as its first parameter::

    math::domain::routine(sycl::queue &q, ...);

All computation performed by the routine shall be done on the hardware device(s) associated with this queue, with possible aid from the host, unless otherwise specified.
In the case of an ordered queue, all computation shall also be ordered with respect to other kernels as if enqueued on that queue.

A particular oneMath implementation may not support the execution of a given oneMath routine on the specified device(s). In this case, the implementation may either perform the computation on the host or throw an exception.  See :ref:`onemath_exceptions` for the possible exceptions.


.. _onemath_member_functions:

Member Functions
****************

oneMath class-based APIs, such as those in the RNG and DFT domains, require a ``sycl::queue`` as an argument to the constructor or another setup routine.
The execution requirements for computational routines from the previous section also apply to computational class methods.

.. _onemath_device_usage:

Device Usage
++++++++++++

oneMath itself does not currently provide any interfaces for controlling device usage: for instance, controlling the number of cores used on the CPU, or the number of execution units on a GPU. However, such functionality may be available by partitioning a ``sycl::device`` instance into subdevices, when supported by the device.

When given a queue associated with such a subdevice, a oneMath implementation shall only perform computation on that subdevice.

.. _onemath_asynchronous_synchronous_execution:

Asynchronous Execution
++++++++++++++++++++++
The oneMath API is designed to allow asynchronous execution of computational routines, to facilitate concurrent usage of multiple devices in the system. Each computational routine enqueues work to be performed on the selected device, and may (but is not required to) return before execution completes.

Hence, it is the calling application's responsibility to ensure that any inputs are valid until computation is complete, and likewise to wait for computation completion before reading any outputs. This can be done automatically when using DPC++ buffers, or manually when using Unified Shared Memory (USM) pointers, as described in the sections below.

Unless otherwise specified, asynchronous execution is *allowed*, but not *guaranteed*, by any oneMath computational routine, and may vary between implementations and/or versions. oneMath implementations must clearly document whether execution is guaranteed to be asynchronous for each supported routine. Regardless, calling applications shall not launch any oneMath computational routine with a dependency on a future oneMath API call, even if this computational routine executes asynchronously (i.e. a oneMath implementation may assume no antidependencies are present). This guarantee allows oneMath implementations to reserve resources for execution without risking deadlock.

.. _onemath_synchronization_with_buffers:

Synchronization When Using Buffers
***********************************

``sycl::buffer`` objects automatically manage synchronization between kernel launches linked by a data dependency (either read-after-write, write-after-write, or write-after-read).

oneMath routines are not required to perform any additional synchronization of ``sycl::buffer`` arguments.

.. _onemath_synchronization_with_usm:

Synchronization When Using USM APIs
***********************************

When USM pointers are used as input to, or output from, a oneMath routine, it becomes the calling application's responsibility to manage possible asynchronicity.

To help the calling application, all oneMath routines with at least one USM pointer argument also take an optional reference to a list of *input events*, of type ``std::vector<sycl::event>``, and have a return value of type ``sycl::event`` representing computation completion::

    sycl::event math::domain::routine(..., const std::vector<sycl::event> &in_events = {});

The routine shall ensure that all input events (if the list is present and non-empty) have occurred before any USM pointers are accessed. Likewise, the routine's output event shall not be complete until the routine has finished accessing all USM pointer arguments.

For class methods, "argument" includes any USM pointers previously provided to the object via the class constructor or other class methods.

.. _onemath_host_thread_safety:

Host Thread Safety
++++++++++++++++++

All oneMath member and non-member functions shall be *host thread safe*. That is, they may be safely called simultaneously from concurrent host threads. However, oneMath objects in class-based APIs may not be shared between concurrent host threads unless otherwise specified.
