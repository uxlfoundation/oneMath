.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_ars5:

ars5
====

The ars5 counter-based pseudorandom number generator.

.. _onemath_rng_ars5_description:

.. rubric:: Description

The ars5 engine is a keyed family of counter-based BRNG. The state consists of a 128-bit integer counter :math:`c` and a 128-bit key :math:`k`. The BRNG is based on the AES encryption algorithm [:ref:`FIPS-197 <onemath_rng_bibliography>`].

.. container:: section

    .. rubric:: Generation algorithm

    The generator has a 32-bit integer output obtained in the following way [:ref:`Salmon11 <onemath_rng_bibliography>`]:

    1. The i-th number is defined by the following formula :math:`r_i=(f(i/ 4) >> ((i \ mod \ 4) * 32)) \ mod \ 2 ^ {32}`
    2. Function :math:`f(c)` takes a 128-bit argument and returns a 128-bit number. The returned number is obtained as follows:
        2.1. :math:`c_0 = c \oplus k` and :math:`k_0 = k`.

        2.2. The following recurrence is calculated N = 5 times:

        :math:`c_{i+1} = SubBytes(c)`

        :math:`c_{i+1} = ShiftRows(c_{i+1})`

        :math:`c_{i+1} = MixColumns(c_{i+1})`, this step is omitted if :math:`i+1 = N`

        :math:`c_{i+1} = AddRoundKey(c_{i+1}, k_j)`

        :math:`Lo(k_{i+1}) = Lo(k) + 0x9E3779B97F4A7C15`

        :math:`Hi(k_{i+1}) = Hi(k) + 0xBB67AE8584CAA73B`

        Specification for :math:`SubBytes, ShiftRows, MixColumns, AddRoundKey` functions can be found in [:ref:`FIPS-197 <onemath_rng_bibliography>`].

        2.3. Put :math:`f(c) = c_N`, where :math:`N = 10`

    3. Real output: :math:`u_n=(int)r_n / 2^{32} + 1/2`

.. _onemath_rng_ars5_description_syntax:

class ars5
----------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    class ars5 {
    public:
            static constexpr std::uint64_t default_seed = 0;

            ars5(sycl::queue queue, std::uint64_t seed = default_seed);

            ars5(sycl::queue queue, std::initializer_list<std::uint64_t> seed);

            ars5(const ars5& other);

            ars5(ars5&& other);

            ars5& operator=(const ars5& other);

            ars5& operator=(ars5&& other);

            ~ars5();
    };
    }

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `ars5(sycl::queue queue, std::uint64_t seed)`_
          - Constructor for common seed initialization of the engine
        * - `ars5(sycl::queue queue, std::initializer_list<std::uint64_t> seed)`_
          - Constructor for extended seed initialization of the engine
        * - `ars5(const ars5& other)`_
          - Copy constructor
        * - `ars5(ars5&& other)`_
          - Move constructor
        * - `ars5& operator=(const ars5& other)`_
          - Copy assignment operator
        * - `ars5& operator=(ars5&& other)`_
          - Move assignment operator

.. container:: section

    .. rubric:: Constructors

    .. _`ars5(sycl::queue queue, std::uint64_t seed)`:

    .. code-block:: cpp

      ars5::ars5(sycl::queue queue, std::uint64_t seed)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid sycl::queue object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state, assume :math:`k = seed, c = 0`, where :math:`k` is 128-bit key, :math:`c` is 128-bit counter.

    .. _`ars5(sycl::queue queue, std::initializer_list<std::uint64_t> seed)`:

    .. code-block:: cpp

      ars5::ars5(sycl::queue queue, std::initializer_list<std::uint64_t> seed)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state, assume
            if :math:`n = 0: k = 0, c = 0`

            if :math:`n = 1: k = seed[0], c = 0`

            if :math:`n = 2: k = seed[0] + seed[1] \cdot 2^{64}, c = 0`

            if :math:`n = 3: k = seed[0] + seed[1] \cdot 2^{64}, c = seed[2]`

            if :math:`n = 4: k = seed[0] + seed[1] \cdot 2^{64}, c = seed[2] + seed[3] \cdot 2^{64}`

            for :math:`n > 4` following arguments are ignored

    .. _`ars5(const ars5& other)`:

    .. code-block:: cpp

      ars5::ars5(const ars5& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``ars5`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`ars5(ars5&& other)`:

    .. code-block:: cpp

      ars5::ars5(ars5&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``ars5`` r-value object. The ``queue`` and state of the other engine is moved to the current engine.


    .. _`ars5& operator=(const ars5& other)`:

    .. code-block:: cpp

        ars5::ars5& operator=(const ars5& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``ars5`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`ars5& operator=(ars5&& other)`:

    .. code-block:: cpp

        ars5::ars5& operator=(ars5&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``ars5`` r-value object. The ``queue`` and state of the other engine is moved to the current engine.

**Parent topic:** :ref:`onemath_rng_engines_basic_random_number_generators`
