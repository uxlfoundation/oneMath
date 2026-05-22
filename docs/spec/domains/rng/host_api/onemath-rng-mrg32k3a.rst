.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_mrg32k3a:

mrg32k3a
========

The combined multiple recursive pseudorandom number generator MRG32k3a.

.. _onemath_rng_mrg32k3a_description:

.. rubric:: Description

MRG32k3a engine is a 32-bit combined multiple recursive generator with two components of order 3 :ref:`[L'Ecuyer99a] <onemath_rng_bibliography>`. MRG32k3a combined generator meets the requirements for modern RNGs, such as good multidimensional uniformity, or a long period (:math:`p \approx 2^{191}`).

.. container:: section

    .. rubric:: Generation algorithm


    :math:`x_n=a_{11} x_{n-1} + a_{12} x_{n-2} + a_{13} x_{n-3}(mod \ m_{1})`

    :math:`y_n = a_{21} y_{n-1} + a_{22} y_{n-2} + a_{23} (mod \ m_2)`

    :math:`z_n = x_n - y_n (mod \ m_{1})`

    :math:`u_n = z_n / m_1`

    :math:`a_{11} = 0, a_{12} = 1403580, a_{13} = -810728, m_1 = 2^{32} - 209`

    :math:`a_{21} = 527612, a_{22} = 0, a_{23} = -1370589, m_2 = 2^{32} - 22853`


.. _onemath_rng_mrg32k3a_description_syntax:

class mrg32k3a
--------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    class mrg32k3a {
    public:
        static constexpr std::uint32_t default_seed = 1;

        mrg32k3a(sycl::queue queue, std::uint32_t seed = default_seed);

        mrg32k3a(sycl::queue queue, std::initializer_list<std::uint32_t> seed);

        mrg32k3a(const mrg32k3a& other);

        mrg32k3a(mrg32k3a&& other);

        mrg32k3a& operator=(const mrg32k3a& other);

        mrg32k3a& operator=(mrg32k3a&& other);

        ~mrg32k3a();
    };
    }

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `mrg32k3a(sycl::queue queue, std::uint32_t seed = default_seed)`_
          - Constructor for common seed initialization of the engine
        * - `mrg32k3a(sycl::queue queue, std::initializer_list<std::uint32_t> seed)`_
          - Constructor for extended seed initialization of the engine
        * - `mrg32k3a(const mrg32k3a& other)`_
          - Copy constructor
        * - `mrg32k3a(mrg32k3a&& other)`_
          - Move constructor
        * - `mrg32k3a& operator=(const mrg32k3a& other)`_
          - Copy assignment operator
        * - `mrg32k3a& operator=(mrg32k3a&& other)`_
          - Move assignment operator

.. container:: section

    .. rubric:: Constructors

    .. _`mrg32k3a(sycl::queue queue, std::uint32_t seed = default_seed)`:

    .. code-block:: cpp
    
        mrg32k3a::mrg32k3a(sycl::queue queue, std::uint32_t seed = default_seed)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue object``, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state, assume :math:`x_{-3} = seed \ mod \ m_1, x_{-2} = x_{-1} = y_{-3} = y_{-2} = y_{-1} = 1`.

    .. _`mrg32k3a(sycl::queue queue, std::initializer_list<std::uint32_t> seed)`:

    .. code-block:: cpp
    
        mrg32k3a::mrg32k3a(sycl::queue queue, std::initializer_list<std::uint32_t> seed)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state, assume
            if :math:`n = 0: x_{-3} = x_{-2} = x_{-1} = y_{-3} = y_{-2} = y_{-1} = 1`

            if :math:`n = 1: x_{-3} = seed[0] \ mod \ m_1, x_{-2} = x_{-1} = y_{-3} = y_{-2} = y_{-1} = 1`

            if :math:`n = 2: x_{-3} = seed[0] \ mod \ m_1, x_{-2} = seed[1] \ mod \ m_1, x_{-1} = y_{-3} = y_{-2} = y_{-1} = 1`

            if :math:`n = 3: x_{-3} = seed[0] \ mod \ m_1, x_{-2} = seed[1] \ mod \ m_1, x_{-1} = seed[2] \ mod \ m_1`

                :math:`y_{-3} = y_{-2} = y_{-1} = 1`

            if :math:`n = 4: x_{-3} = seed[0] \ mod \ m_1, x_{-2} = seed[1] \ mod \ m_1, x_{-1} = seed[2] \ mod \ m_1`
            
                :math:`y_{-3} = seed[3] \ mod \ m_2, y_{-2} = y_{-1} = 1`

            if :math:`n = 5: x_{-3} = seed[0] \ mod \ m_1, x_{-2} = seed[1] \ mod \ m_1, x_{-1} = seed[2] \ mod \ m_1`

                :math:`y_{-3} = seed[3] \ mod \ m_2, y_{-2} = seed[4] \ mod \ m_2, y_{-1} = 1`

            if :math:`n \geqslant 6: x_{-3} = seed[0] \ mod \ m_1, x_{-2} = seed[1] \ mod \ m_1, x_{-1} = seed[2] \ mod \ m_1`

                :math:`y_{-3} = seed[3] \ mod \ m_2, y_{-2} = seed[4] \ mod \ m_2, y_{-1} = seed[5] \ mod \ m_2`

            if the values prove to be :math:`x_{-3} = x_{-2} = x_{-1} = 0`, assume :math:`x_{-3} = 1`

            if the values prove to be :math:`y_{-3} = y_{-2} = y_{-1} = 0`, assume :math:`y_{-3} = 1`

    .. _`mrg32k3a(const mrg32k3a& other)`:

    .. code-block:: cpp
    
        mrg32k3a::mrg32k3a(const mrg32k3a& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mrg32k3a`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`mrg32k3a(mrg32k3a&& other)`:

    .. code-block:: cpp

        mrg32k3a::mrg32k3a(mrg32k3a&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mrg32k3a`` object. The ``queue`` and state of the other engine is moved to the current engine.

    .. _`mrg32k3a& operator=(const mrg32k3a& other)`:

    .. code-block:: cpp

        mrg32k3a::mrg32k3a& operator=(const mrg32k3a& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mrg32k3a`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`mrg32k3a& operator=(mrg32k3a&& other)`:

    .. code-block:: cpp

        mrg32k3a::mrg32k3a& operator=(mrg32k3a&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mrg32k3a`` r-value object. The ``queue`` and state of the other engine is moved to the current engine.

**Parent topic:** :ref:`onemath_rng_engines_basic_random_number_generators`
