.. SPDX-FileCopyrightText: 2023 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_device_rng_generate:

generate
========

.. rubric:: Description

Entry point to obtain random numbers from a given engine with proper statistics of a given distribution.

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::rng::device {
     template<typename Distr, typename Engine>
     auto generate(Distr& distr, Engine& engine) ->
       typename std::conditional<Engine::vec_size == 1, typename Distr::result_type,
                                 sycl::vec<typename Distr::result_type, Engine::vec_size>>::type
   }

.. container:: section

    .. rubric:: Template Parameters

    Distr
        Type of distribution which is used for random number generation.

    Engine
        Type of engine which is used for random number generation.

.. container:: section

    .. rubric:: Input Parameters

    distr
        Distribution object. See :ref:`onemath_device_rng_distributions` for details.

    engine
        Engine object. See :ref:`onemath_device_rng_engines` for details.

.. container:: section

    .. rubric:: Return Value

    Returns `Distr::result_type` if `Engine::vec_size == 1` or `sycl::vec<typename Distr::result_type, Engine::vec_size>` 
    with generated random numbers.

**Parent topic:** :ref:`onemath_device_rng_generate_routines`
