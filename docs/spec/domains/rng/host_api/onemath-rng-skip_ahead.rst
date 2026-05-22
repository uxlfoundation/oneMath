.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_skip_ahead:

skip_ahead
==========

Proceed state of engine by the skip-ahead method.

.. _onemath_rng_skip_ahead_description:

.. rubric:: Description and Assumptions

oneapi::math::rng::skip_ahead function changes the current state of the engine so that with the further call of the generator the output subsequence begins with the specified offset see `Figure "Block-Splitting Method" <#rng_block_splitting>`__.

.. container:: figtop
    :name: rng_block_splitting

    Block-Splitting Method

    |image0|

.. _onemath_rng_skip_ahead_common:

skip_ahead
----------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    template<typename EngineType>
    void skip_ahead(EngineType& engine, std::uint64_t num_to_skip);
    }

.. container:: section

    .. rubric:: Template Parameters

    EngineType
        Type of engine. Note: may not be supported by all available engine classes.

.. container:: section

    .. rubric:: Input Parameters

    engine
        Engine which state would be skipped.

    num_to_skip
        Number of elements to skip in the engine's sequence.

.. rubric:: Example

.. code-block:: cpp

    // Creating 3 identical engines
    oneapi::math::rng::mcg31m1 engine_1(queue, seed);
    oneapi::math::rng::mcg31m1 engine_2(engine_1);
    oneapi::math::rng::mcg31m1 engine_3(engine_2);

    // Skipping ahead by 7 elements the 2nd engine
    oneapi::math::rng::skip_ahead(engine_2, 7);

    // Skipping ahead by 14 elements the 3rd engine
    oneapi::math::rng::skip_ahead(engine_3, 14);


.. _onemath_rng_skip_ahead_ex:

skip_ahead (Interface with a partitioned number of skipped elements)
--------------------------------------------------------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    template<typename EngineType>
    void oneapi::math::rng::skip_ahead(EngineType& engine, std::initializer_list<std::uint64_t> num_to_skip);
    }

.. container:: section

    .. rubric:: Template Parameters

    EngineType
        Type of engine. Note: may not be supported by all available engine classes.

.. container:: section

    .. rubric:: Input Parameters

    engine
        Engine which state would be skipped.

    num_to_skip
        Partitioned number of elements to skip in the engine's sequence. The total number of skipped elements would be: :math:`num\_to\_skip[0] + num\_to\_skip[1] \cdot 2^{64} + ... + num\_to\_skip[1] \cdot 2^{64 (n - 1)}`, where n is a number of elements in num_to_skip list.

.. rubric:: Example with Partitioned Numer of Elements

.. code-block:: cpp

    // Creating the first engine
    oneapi::math::rng::mrg32k3a engine_1(queue, seed);

    // To skip 2^64 elements in the random stream number of skipped elements should be
    // represented as num_to_skip = 2^64 = 0 + 1 * 2^64
    std::initializer_list<std::uint64_t> num_to_skip = {0, 1};

    // Creating the 2nd engine based on 1st. Skipping by 2^64
    oneapi::math::rng::mrg32k3a engine_2(engine_1);
    oneapi::math::rng::skip_ahead(engine_2, num_to_skip);


**Parent topic:** :ref:`onemath_rng_service_routines`

.. |image0| image:: ../../equations/rng-skip-ahead.png
