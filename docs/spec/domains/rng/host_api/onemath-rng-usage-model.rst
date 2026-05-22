.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_usage_model:

oneMath RNG Host Usage Model
============================


.. rubric:: Description

A typical algorithm for random number generators is as follows:

1. Create and initialize the object for basic random number generator.

    -  Use the `skip_ahead` or `leapfrog` function if it is required (used in parallel with random number generation for Host and CPU devices).

2. Create and initialize the object for distribution generator.

3. Call the generate routine to get random numbers with appropriate statistical distribution.

The following example demonstrates random numbers generation with PHILOX4X32X10 basic generator (engine).

Buffer-based example
--------------------

.. code-block:: cpp

    #include "oneapi/math/rng.hpp"

    int main() {
        sycl::queue q;

        // Create the random number generator object
        oneapi::math::rng::philox4x32x10 engine(q, seed);
        // Create the distribution object
        oneapi::math::rng::gaussian<double> distr(5.0, 2.0);
        // Fill the SYCL buffer with random numbers
        oneapi::math::rng::generate(distr, engine, n, sycl_buffer);

        // ...
    }


USM-based example
-----------------

.. code-block:: cpp

    #include "oneapi/math/rng.hpp"

    int main() {
        sycl::queue q;

        // Create the random number generator object
        oneapi::math::rng::philox4x32x10 engine(q, seed);
        // Create the distribution object
        oneapi::math::rng::gaussian<double> distr(5.0, 2.0);
        // Fill the USM memory under the pointer with random numbers
        auto event = oneapi::math::rng::generate(distr, engine, n, usm_ptr);
        // ...
        // wait until generation is finalized
        event.wait();
        // ...
    }

**Parent topic:** :ref:`onemath_rng_manual_offload_routines`
