.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_wichmann_hill:

wichmann_hill
=============

The wichmann_hill engine is the set of 273 Wichmann-Hill's combined multiplicative congruential generators from NAG Numerical Libraries :ref:`[NAG] <onemath_rng_bibliography>`.

.. _onemath_rng_wichmann_hill_description:

.. rubric:: Description

The set of 372 different basic pseudorandom number generators wichmann_hill is the second basic generator in the NAG libraries.

.. container:: section

    .. rubric:: Generation algorithm

    :math:`x_n=a_{1, j} x_{n-1} (mod \ m_{1, j})`

    :math:`y_n = a_{2, j} y_{n-1} (mod \ m_{2, j})`

    :math:`z_n = a_{3, j} z_{n-1} (mod \ m_{3, j})`

    :math:`w_n = a_{4, j} w_{n-1} (mod \ m_{4, j})`

    :math:`u_n = (x_n / m_{1, j} + y_n / m_{2, j} + z_n / m_{3, j} + w_n / m_{4, j}) mod \ 1`

    The constants :math:`a_{i, j}` range from 112 to 127, the constants :math:`m_{i, j}` are prime numbers ranging from 16718909 to 16776917, close to :math:`2 ^ {24}`.


.. _onemath_rng_wichmann_hill_description_syntax:

class wichmann_hill
-------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    class wichmann_hill {
    public:
        static constexpr std::uint32_t default_seed = 1;

        wichmann_hill(sycl::queue queue, std::uint32_t seed = default_seed);

        wichmann_hill(sycl::queue queue, std::uint32_t seed, std::uint32_t engine_idx);

        wichmann_hill(sycl::queue queue, std::initializer_list<std::uint32_t> seed);

        wichmann_hill(sycl::queue queue, std::initializer_list<std::uint32_t> seed, std::uint32_t engine_idx);

        wichmann_hill(const wichmann_hill& other);

        wichmann_hill(wichmann_hill&& other);

        wichmann_hill& operator=(const wichmann_hill& other);

        wichmann_hill& operator=(wichmann_hill&& other);

        ~wichmann_hill();
    };
    }

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `wichmann_hill(sycl::queue queue, std::uint32_t seed = default_seed)`_
          - Constructor for common seed initialization of the engine (for this case multiple generators of the set would be used)
        * - `wichmann_hill(sycl::queue queue, std::uint32_t seed, std::uint32_t engine_idx)`_
          - Constructor for common seed initialization of the engine (for this case single generator of the set would be used)
        * - `wichmann_hill(sycl::queue& queue, std::initializer_list<std::uint32_t> seed)`_
          - Constructor for extended seed initialization of the engine (for this case multiple generators of the set would be used)
        * - `wichmann_hill(sycl::queue& queue, std::initializer_list<std::uint32_t> seed, std::uint32_t engine_idx)`_
          - Constructor for extended seed initialization of the engine (for this case single generator of the set would be used)
        * - `wichmann_hill(const wichmann_hill& other)`_
          - Copy constructor
        * - `wichmann_hill(wichmann_hill&& other)`_
          - Move constructor
        * - `wichmann_hill& operator=(const wichmann_hill& other)`_
          - Copy assignment operator
        * - `wichmann_hill& operator=(wichmann_hill&& other)`_
          - Move assignment operator

.. container:: section

    .. rubric:: Constructors

    .. _`wichmann_hill(sycl::queue queue, std::uint32_t seed = default_seed)`:

    .. code-block:: cpp

        wichmann_hill::wichmann_hill(sycl::queue queue, std::uint32_t seed = default_seed)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state. Assume :math:`x_0=seed \ mod \ m_1, y_0 = z_0 = w_0 = 1`. If :math:`x_0 = 0`, assume :math:`x_0 = 1`.

    .. _`wichmann_hill(sycl::queue queue, std::uint32_t seed, std::uint32_t engine_idx)`:

    .. code-block:: cpp

        wichmann_hill::wichmann_hill(sycl::queue queue, std::uint32_t seed, std::uint32_t engine_idx)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state. Assume :math:`x_0=seed \ mod \ m_1, y_0 = z_0 = w_0 = 1`. If :math:`x_0 = 0`, assume :math:`x_0 = 1`.

        engine_idx
            The index of the set 1, ..., 273.

    .. container:: section

        .. rubric:: Throws

        oneapi::math::invalid_argument
            Exception is thrown when :math:`idx > 273`

    .. _`wichmann_hill(sycl::queue& queue, std::initializer_list<std::uint32_t> seed)`:

    .. code-block:: cpp

        wichmann_hill::wichmann_hill(sycl::queue& queue, std::initializer_list<std::uint32_t> seed)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state, assume:
            if :math:`n = 0: x_{0} = y_{0} = z_{0} = w_{0} = 1`

            if :math:`n = 1: x_{0} = seed[0] \ mod \ m_1, y_{0} = z_{0} = w_{0} = 1`. If :math:`x_0 = 0`, assume :math:`x_0 = 1`.

            if :math:`n = 2: x_{0} = seed[0] \ mod \ m_1, y_{0} = seed[1] \ mod \ m_2, z_{0} = w_{0} = 1`.

            if :math:`n = 3: x_{0} = seed[0] \ mod \ m_1, y_{0} = seed[1] \ mod \ m_2, z_{0} = seed[2] \ mod \ m_3, w_{0} = 1`.

            if :math:`n \geqslant 4: x_{0} = seed[0] \ mod \ m_1, y_{0} = seed[1] \ mod \ m_2`

                :math:`z_{0} = seed[2] \ mod \ m_3, w_{0} = seed[3] \ mod \ m_4`.


    .. _`wichmann_hill(sycl::queue& queue, std::initializer_list<std::uint32_t> seed, std::uint32_t engine_idx)`:

    .. code-block:: cpp

        wichmann_hill::wichmann_hill(sycl::queue& queue, std::initializer_list<std::uint32_t> seed, std::uint32_t engine_idx)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state, assume:
            if :math:`n = 0: x_{0} = y_{0} = z_{0} = w_{0} = 1`

            if :math:`n = 1: x_{0} = seed[0] \ mod \ m_1, y_{0} = z_{0} = w_{0} = 1`. If :math:`x_0 = 0`, assume :math:`x_0 = 1`.

            if :math:`n = 2: x_{0} = seed[0] \ mod \ m_1, y_{0} = seed[1] \ mod \ m_2, z_{0} = w_{0} = 1`.

            if :math:`n = 3: x_{0} = seed[0] \ mod \ m_1, y_{0} = seed[1] \ mod \ m_2, z_{0} = seed[2] \ mod \ m_3, w_{0} = 1`.

            if :math:`n \geqslant 4: x_{0} = seed[0] \ mod \ m_1, y_{0} = seed[1] \ mod \ m_2`

                :math:`z_{0} = seed[2] \ mod \ m_3, w_{0} = seed[3] \ mod \ m_4`.

        engine_idx
            The index of the set 1, ..., 273.

    .. _`wichmann_hill(const wichmann_hill& other)`:

    .. code-block:: cpp

        wichmann_hill::wichmann_hill(const wichmann_hill& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``wichmann_hill`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`wichmann_hill(wichmann_hill&& other)`:

    .. code-block:: cpp

        wichmann_hill::wichmann_hill(wichmann_hill&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``wichmann_hill`` object. The ``queue`` and state of the other engine is moved to the current engine.

    .. _`wichmann_hill& operator=(const wichmann_hill& other)`:

    .. code-block:: cpp

        wichmann_hill::wichmann_hill& operator=(const wichmann_hill& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``wichmann_hill`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`wichmann_hill& operator=(wichmann_hill&& other)`:

    .. code-block:: cpp

        wichmann_hill::wichmann_hill& operator=(wichmann_hill&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``wichmann_hill`` r-value object. The ``queue`` and state of the other engine is moved to the current engine.

**Parent topic:** :ref:`onemath_rng_engines_basic_random_number_generators`
