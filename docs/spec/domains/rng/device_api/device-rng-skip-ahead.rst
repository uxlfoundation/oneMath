.. SPDX-FileCopyrightText: 2023 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_device_rng_skip_ahead:

skip_ahead
==========

.. rubric:: Description

Proceed state of engine by the skip-ahead method.

The ``skip_ahead`` function supports the following interfaces to apply the skip-ahead method:

- Common interface
- Interface with a partitioned number of skipped elements


skip_ahead
----------

.. rubric:: Common Interface


.. code-block:: cpp

   namespace oneapi::math::rng::device {
     template<typename Engine>
     void skip_ahead (Engine& engine, std::uint64_t num_to_skip)
   }

.. container:: section

    .. rubric:: Template Parameters

    Engine
        Object of engine class, which supports the block-splitting method.

.. container:: section

    .. rubric:: Input Parameters

    engine
        Engine which state would be skipped.

    num_to_skip
        Number of skipped elements.


.. rubric:: Interface with a partitioned number of skipped elements

.. code-block:: cpp

   namespace oneapi::math::rng::device {
     template<typename Engine>
     void skip_ahead (Engine& engine, std::initializer_list<std::uint64_t> num_to_skip)
   }

.. container:: section

    .. rubric:: Template Parameters

    Engine
        Object of engine class, which supports the block-splitting method.

.. container:: section

    .. rubric:: Input Parameters

    engine
        Engine which state would be skipped.

    num_to_skip
        Partitioned number of skipped elements. The total number of skipped elements 
        would be: :math:`num\_to\_skip[0] + num\_to\_skip[1] \cdot 2^{64} + ... + 
        num\_to\_skip[1] \cdot 2^{64 (n - 1)}`, where `n` is a number of elements in `num_to_skip` list.

**Parent topic:** :ref:`onemath_device_rng_service_routines`
