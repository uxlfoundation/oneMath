.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_mcg59:

mcg59
=====

The 59-bit multiplicative congruential pseudorandom number generator.

.. _onemath_rng_mcg59_description:

.. rubric:: Description

The mcg59 engine is a 59-bit multiplicative congruential generator from NAG Numerical Libraries :ref:`NAG <onemath_rng_bibliography>`. The mcg59 generator belongs to linear congruential generators with the period length of approximately :math:`2^{57}`.

.. container:: section

    .. rubric:: Generation algorithm

    :math:`x_n=ax_{n-1}(mod \ m)`

    :math:`u_n = x_n / m`

    :math:`a = 13^{13}, m=2^{59}`

.. _onemath_rng_mcg59_description_syntax:

class mcg59
-----------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    class mcg59 {
    public:
        static constexpr std::uint64_t default_seed = 1;

        mcg59(sycl::queue queue, std::uint64_t seed = default_seed);

        mcg59(const mcg59& other);

        mcg59(mcg59&& other);

        mcg59& operator=(const mcg59& other);

        mcg59& operator=(mcg59&& other);

        ~mcg59();
    };
    }

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `mcg59(sycl::queue queue, std::uint64_t seed = default_seed)`_
          - Constructor for common seed initialization of the engine
        * - `mcg59(const mcg59& other)`_
          - Copy constructor
        * - `mcg59(mcg59&& other)`_
          - Move constructor
        * - `mcg59& operator=(const mcg59& other)`_
          - Copy assignment operator
        * - `mcg59& operator=(mcg59&& other)`_
          - Move assignment operator

.. container:: section

    .. rubric:: Constructors

    .. _`mcg59(sycl::queue queue, std::uint64_t seed = default_seed)`:

    .. code-block:: cpp
    
        mcg59::mcg59(sycl::queue queue, std::uint64_t seed = default_seed)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state, assume :math:`x_0 = seed \ mod \ 2^{59}`, if :math:`x_0 = 0`, assume :math:`x_0 = 1`.

    .. _`mcg59(const mcg59& other)`:

    .. code-block:: cpp
    
        mcg59::mcg59(const mcg59& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mcg59`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`mcg59(mcg59&& other)`:

    .. code-block:: cpp

        mcg59::mcg59(mcg59&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mcg59`` object. The ``queue`` and state of the other engine is moved to the current engine.

    .. _`mcg59& operator=(const mcg59& other)`:

    .. code-block:: cpp

        mcg59::mcg59& operator=(const mcg59& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mcg59`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`mcg59& operator=(mcg59&& other)`:

    .. code-block:: cpp

        mcg59::mcg59& operator=(mcg59&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mcg59`` r-value object. The ``queue`` and state of the other engine is moved to the current engine.

**Parent topic:** :ref:`onemath_rng_engines_basic_random_number_generators`
