.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_generate:

generate
========

Entry point to obtain random numbers from a given engine with proper statistics of a given distribution.

.. _onemath_rng_generate_description:

.. rubric:: Description and Assumptions

oneapi::math::rng::generate function produces random numbers sequence from the given engine object and applied transformation from a given distribution object.

.. _onemath_rng_generate_buffer:

generate (Buffer version)
-------------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    template<typename DistrType, typename EngineType>
    void generate (const DistrType& distr, EngineType& engine, std::int64_t n, sycl::buffer<typename DistrType::result_type, 1>& r);
    }

.. container:: section

    .. rubric:: Template Parameters

    DistrType
        Type of distribution which is used for random number generation.

    EngineType
        Type of engine which is used for random number generation.

.. container:: section

    .. rubric:: Input Parameters

    distr
        Distribution object. See :ref:`onemath_rng_distributions` for details.

    engine
        Engine object. See :ref:`onemath_rng_engines_basic_random_number_generators` for details.

    n
        Number of random values to be generated.

.. container:: section

    .. rubric:: Output Parameters

    r
        sycl::buffer of generated values.

.. container:: section

    .. rubric:: Throws

    oneapi::math::invalid_argument
        Exception is thrown when n > r.get_count(), or n < 0

.. _onemath_rng_generate_usm:

generate (USM version)
----------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    template<typename DistrType, typename EngineType>
    sycl::event generate (const DistrType& distr, EngineType& engine, std::int64_t n, typename DistrType::result_type* r, const std::vector<sycl::event> & dependencies);
    }

.. container:: section

    .. rubric:: Template Parameters

    DistrType
        Type of distribution which is used for random number generation.

    EngineType
        Type of engine which is used for random number generation.

.. container:: section

    .. rubric:: Input Parameters

    distr
        Distribution object. See :ref:`onemath_rng_distributions` for details.

    engine
        Engine object. See :ref:`onemath_rng_engines_basic_random_number_generators` for details.

    n
        Number of random values to be generated.

    dependencies
        Optional parameter. List of events to wait for before starting computation, if any.

.. container:: section

    .. rubric:: Output Parameters

    r
        pointer to generated values.

.. container:: section

    .. rubric:: Throws

    oneapi::math::invalid_argument
        Exception is thrown when r == nullptr, or n < 0

.. container:: section

    .. rubric:: Return Value

    Output event to wait on to ensure computation is complete.


**Parent topic:** :ref:`onemath_rng_generate_routine`
