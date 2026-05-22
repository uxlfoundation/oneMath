.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_mt19937:

mt19937
=======

Mersenne Twister pseudorandom number generator.

.. _onemath_rng_mt19937_description:

.. rubric:: Description

The Mersenne Twister pseudorandom number generator, mt19937, is a modification of twisted generalized feedback shift register generator [:ref:`Matsumoto98 <onemath_rng_bibliography>`]. MT19937 has the period length of :math:`2^{19937} - 1` and is 623-dimensionally equidistributed with up to 32-bit accuracy. These properties make the generator applicable for simulations in various fields of science and engineering. The state of the generator is represented by 624 32-bit unsigned integer numbers.

.. container:: section

    .. rubric:: Generation algorithm

    :math:`x_{n}=x_{n-(624-397)} \oplus ( (x_{n-624} \& 0x80000000) | (x_{n - 624 + 1} \& 0x7FFFFFFF) )A`

    :math:`y_{n} = x_{n}`

    :math:`y_{n} = y_{n} \oplus (y_{n} >> 11)`

    :math:`y_{n} = y_{n} \oplus ( (y_{n} << 7) \& 0x9D2C5680)`

    :math:`y_{n} = y_{n} \oplus ( (y_{n} << 15) \& 0xEFC60000)`

    :math:`y_{n} = y_{n} \oplus (y_{n} >> 18)`

    :math:`u_{n} = y_{n} / 2^{32}`


.. container:: section

    Matrix :math:`A_j(32x32)` has the following format:

.. math::

    \left [ \begin{array}{ccccc} 0 & 1 & 0 & ... & ... \\ 0 & 0 & ... & 0 & ... \\ ... & ... & ... & ... & ... \\ 0 & ... & 0 & 0 & 1 \\ a_{31} & a_{30} & ... & ... & a_{0} \end{array}\right ]

.. container:: section

    Where the 32-bit vector :math:`a = a_{31}..a_{0}` has the value :math:`a=0x9908B0DF`.

.. _onemath_rng_mt19937_description_syntax:

class mt19937
--------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    class mt19937 {
    public:
        static constexpr std::uint32_t default_seed = 1;

        mt19937(sycl::queue queue, std::uint32_t seed = default_seed);

        mt19937(sycl::queue queue, std::initializer_list<std::uint32_t> seed);

        mt19937(const mt19937& other);

        mt19937(mt19937&& other);

        mt19937& operator=(const mt19937& other);

        mt19937& operator=(mt19937&& other);

        ~mt19937();
    };
    }

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `mt19937(sycl::queue queue, std::uint32_t seed = default_seed)`_
          - Constructor for common seed initialization of the engine
        * - `mt19937(sycl::queue queue, std::initializer_list<std::uint32_t> seed)`_
          - Constructor for extended seed initialization of the engine
        * - `mt19937(const mt19937& other)`_
          - Copy constructor
        * - `mt19937(mt19937&& other)`_
          - Move constructor
        * - `mt19937& operator=(const mt19937& other)`_
          - Copy assignment operator
        * - `mt19937& operator=(mt19937&& other)`_
          - Move assignment operator

.. container:: section

    .. rubric:: Constructors

    .. _`mt19937(sycl::queue queue, std::uint32_t seed = default_seed)`:

    .. code-block:: cpp
    
        mt19937::mt19937(sycl::queue queue, std::uint32_t seed = default_seed)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state. The initialization algorithm described in [:ref:`MT2203 <onemath_rng_bibliography>`].

    .. _`mt19937(sycl::queue queue, std::initializer_list<std::uint32_t> seed)`:

    .. code-block:: cpp
    
        mt19937::mt19937(sycl::queue queue, std::initializer_list<std::uint32_t> seed)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state. The initialization algorithm described in [:ref:`MT2203 <onemath_rng_bibliography>`].

    .. _`mt19937(const mt19937& other)`:

    .. code-block:: cpp
    
        mt19937::mt19937(const mt19937& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mt19937`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`mt19937(mt19937&& other)`:

    .. code-block:: cpp

        mt19937::mt19937(mt19937&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mt19937`` object. The ``queue`` and state of the other engine is moved to the current engine.

    .. _`mt19937& operator=(const mt19937& other)`:

    .. code-block:: cpp

        mt19937::mt19937& operator=(const mt19937& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mt19937`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`mt19937& operator=(mt19937&& other)`:

    .. code-block:: cpp

        mt19937::mt19937& operator=(mt19937&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mt19937`` r-value object. The ``queue`` and state of the other engine is moved to the current engine.

**Parent topic:** :ref:`onemath_rng_engines_basic_random_number_generators`
