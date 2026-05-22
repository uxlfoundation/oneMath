.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_memory_model:

Memory Model
-------------

The oneMath memory model shall follow directly from the oneAPI memory model.  Mainly, oneMath shall support two modes of encapsulating data for consumption on the device: the buffer memory abstraction model and the pointer-based memory model using Unified Shared Memory (USM).  These two paradigms shall also support both synchronous and asynchronous execution models as described in :ref:`onemath_asynchronous_synchronous_execution`.


.. _onemath_buffers:

The Buffer Memory Model
+++++++++++++++++++++++

The SYCL 1.2.1 specification defines the buffer container templated on the provided data type which encapsulates the data in a SYCL application across both host and devices.  It provides the concept of accessors as the mechanism to access the buffer data with different modes to read and or write into that data. These accessors allow SYCL to create and manage the data dependencies in the SYCL graph that order the kernel executions.  With the buffer model, all data movement is handled by the SYCL runtime supporting both synchronous and asynchronous execution.

oneMath provides APIs where buffers (in particular 1D buffers, ``sycl::buffer<T,1>``) contain the memory for all non scalar input and output data arguments. See :ref:`onemath_synchronization_with_buffers` for details on how oneMath routines manage any data dependencies with buffer arguments.  Any higher dimensional buffer must be converted to a 1D buffer prior to use in oneMath APIs, e.g., via ``buffer::reinterpret``.



.. _onemath_usm:

Unified Shared Memory Model
+++++++++++++++++++++++++++

While the buffer model is powerful and elegantly expresses data dependencies, it can be a burden for programmers to replace all pointers and arrays by buffers in their C++ applications. DPC++ also provides pointer-based addressing for device-accessible data, using the Unified Shared Memory (USM) model. Correspondingly, oneMath provides USM APIs in which non-scalar input and output data arguments are passed by USM pointer.

USM devices and system configurations vary in their ability to share data between devices and between a device and the host. oneMath implementations may only assume that user-provided USM pointers are accessible by the device associated with the user-provided queue. In particular, an implementation must not assume that USM pointers can be accessed by any other device, or by the host, without querying the DPC++ runtime. An implementation must accept any device-accessible USM pointer regardless of how it was created (`sycl::malloc_device`, `sycl::malloc_shared`, etc.).

Unlike buffers, USM pointers cannot automatically manage data dependencies between kernels. Users may use in-order queues to ensure ordered execution, or explicitly manage dependencies with ``sycl::event`` objects. To support the second use case, oneMath USM APIs accept input events (prerequisites before computation can begin) and return an output event (indicating computation is complete). See :ref:`onemath_synchronization_with_usm` for details.

