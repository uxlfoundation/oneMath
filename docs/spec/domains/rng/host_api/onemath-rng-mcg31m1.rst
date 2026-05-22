.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_mcg31m1:

mcg31m1
=======

The 31-bit multiplicative congruential pseudorandom number generator MCG(1132489760, 231 -1).

.. _onemath_rng_mcg31m1_description:

.. rubric:: Description

The mcg31m1 engine is a 31-bit multiplicative congruential generator :ref:`[L'Ecuyer99] <onemath_rng_bibliography>`. The mcg31m1 generator belongs to linear congruential generators with the period length of approximately :math:`2^{31}`. Such generators are still used as default random number generators in various software systems, mainly due to the simplicity of the portable versions implementation, speed, and compatibility with the earlier systems versions. However, their period length does not meet the requirements for modern basic generators. Still, the mcg31m1 generator possesses good statistic properties and you may successfully use it to generate random numbers of different distributions for small samplings.

.. container:: section

    .. rubric:: Generation algorithm

    :math:`x_n=ax_{n-1}(mod \ m)`

    :math:`u_n = x_n / m`

    :math:`a = 1132489760, m=2^{31} - 1`

.. _onemath_rng_mcg31m1_description_syntax:

class mcg31m1
-------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    class mcg31m1 {
    public:
        static constexpr std::uint32_t default_seed = 1;

        mcg31m1(sycl::queue queue, std::uint32_t seed = default_seed);

        mcg31m1(const mcg31m1& other);

        mcg31m1(mcg31m1&& other);

        mcg31m1& operator=(const mcg31m1& other);

        mcg31m1& operator=(mcg31m1&& other);

        ~mcg31m1();
    };
    }

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `mcg31m1(sycl::queue queue, std::uint32_t seed = default_seed)`_
          - Constructor for common seed initialization of the engine
        * - `mcg31m1(const mcg31m1& other)`_
          - Copy constructor
        * - `mcg31m1(mcg31m1&& other)`_
          - Move constructor
        * - `mcg31m1& operator=(const mcg31m1& other)`_
          - Copy assignment operator
        * - `mcg31m1& operator=(mcg31m1&& other)`_
          - Move assignment operator

.. container:: section

    .. rubric:: Constructors

    .. _`mcg31m1(sycl::queue queue, std::uint32_t seed = default_seed)`:

    .. code-block:: cpp

        mcg31m1::mcg31m1(sycl::queue queue, std::uint32_t seed = default_seed)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid sycl::queue object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state, assume :math:`x_0 = seed \ mod \ 0x7FFFFFFF`, if :math:`x_0 = 0`, assume :math:`x_0 = 1`.

    .. _`mcg31m1(const mcg31m1& other)`:

    .. code-block:: cpp
    
        mcg31m1::mcg31m1(const mcg31m1& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mcg31m1`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`mcg31m1(mcg31m1&& other)`:

    .. code-block:: cpp

        mcg31m1::mcg31m1(mcg31m1&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mcg31m1`` object. The ``queue`` and state of the other engine is moved to the current engine.

    .. _`mcg31m1& operator=(const mcg31m1& other)`:

    .. code-block:: cpp

        mcg31m1::mcg31m1& operator=(const mcg31m1& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mcg31m1`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`mcg31m1& operator=(mcg31m1&& other)`:

    .. code-block:: cpp

        mcg31m1::mcg31m1& operator=(mcg31m1&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mcg31m1`` r-value object. The ``queue`` and state of the other engine is moved to the current engine.

**Parent topic:** :ref:`onemath_rng_engines_basic_random_number_generators`
