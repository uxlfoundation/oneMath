.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_nondeterministic:

nondeterministic
================

Non-deterministic random number generator.

.. _onemath_rng_nondeterministic_description:

.. rubric:: Description

Implementation defined generator with non-deterministic source of randomness (for example, a hardware device).

.. _onemath_rng_nondeterministic_description_syntax:

class nondeterministic
----------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    class nondeterministic {
    public:
        nondeterministic(sycl::queue queue);

        nondeterministic(const nondeterministic& other);

        nondeterministic(nondeterministic&& other);

        nondeterministic& operator=(const nondeterministic& other);

        nondeterministic& operator=(nondeterministic&& other);

        ~nondeterministic();
    };
    }

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `nondeterministic(sycl::queue queue)`_
          - Constructor for the particular device
        * - `nondeterministic(const nondeterministic& other)`_
          - Copy constructor
        * - `nondeterministic(nondeterministic&& other)`_
          - Move constructor
        * - `nondeterministic& operator=(const nondeterministic& other)`_
          - Copy assignment operator
        * - `nondeterministic& operator=(nondeterministic&& other)`_
          - Move assignment operator

.. container:: section

    .. rubric:: Constructors

    .. _`nondeterministic(sycl::queue queue)`:

    .. code-block:: cpp
    
        nondeterministic::nondeterministic(sycl::queue queue)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.


    .. _`nondeterministic(const nondeterministic& other)`:

    .. code-block:: cpp
    
        nondeterministic::nondeterministic(const nondeterministic& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``nondeterministic`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`nondeterministic(nondeterministic&& other)`:

    .. code-block:: cpp

        nondeterministic::nondeterministic(nondeterministic&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``nondeterministic`` object. The ``queue`` and state of the other engine is moved to the current engine.

    .. _`nondeterministic& operator=(const nondeterministic& other)`:

    .. code-block:: cpp

        nondeterministic::nondeterministic& operator=(const nondeterministic& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``nondeterministic`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`nondeterministic& operator=(nondeterministic&& other)`:

    .. code-block:: cpp

        nondeterministic::nondeterministic& operator=(nondeterministic&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``nondeterministic`` r-value object. The ``queue`` and state of the other engine is moved to the current engine.

**Parent topic:** :ref:`onemath_rng_engines_basic_random_number_generators`
