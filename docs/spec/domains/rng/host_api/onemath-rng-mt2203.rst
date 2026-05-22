.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_mt2203:

mt2203
======

The mt2203 engine is the set of Mersenne Twister pseudorandom number generators MT2203 [:ref:`Matsumoto98 <onemath_rng_bibliography>`], [:ref:`Matsumoto00 <onemath_rng_bibliography>`].
Coefficients are obtained by `the following tool`_. The amount of generators in the set (denoted `n_engines`) is implementation defined.

.. _onemath_rng_mt2203_description:

.. rubric:: Description

The set of basic pseudorandom number generators MT2203 is a natural addition to the MT19937 generator.
MT2203 generators are intended for use in large scale Monte Carlo simulations performed on multi-processor computer systems.

.. container:: section

    .. rubric:: Generation algorithm

    For :math:`j = 1,..., n_engines`:

    :math:`x_{n, j} =x_{n - (69 - 34), j} \oplus ( (x_{n - 69, j} \& 0xFFFFFFE0) | (x_{n + 69 + 1, j} \& 0x1F) ) A_{j}`

    :math:`y_{n, j} = x_{n, j}`

    :math:`y_{n, j} = y_{n, j} \oplus (y_{n, j} >> 12)`

    :math:`y_{n, j} = y_{n, j} \oplus ( (y_{n, j} << 7) \& b_j)`

    :math:`y_{n, j} = y_{n, j} \oplus ( (y_{n, j} << 15) \& c_j)`

    :math:`y_{n, j} = y_{n, j} \oplus (y_{n, j} >> 18)`


.. container:: section

    Matrix :math:`A_j(32x32)` has the following format:

.. math::

    \left [ \begin{array}{ccccc} 0 & 1 & 0 & ... & ... \\ 0 & 0 & ... & 0 & ... \\ ... & ... & ... & ... & ... \\ 0 & ... & 0 & 0 & 1 \\ a_{31, j} & a_{30, j} & ... & ... & a_{0, 1} \end{array}\right ]


.. _onemath_rng_mt2203_description_syntax:

class mt2203
------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    class mt2203 {
    public:
        static constexpr std::uint32_t default_seed = 1;

        mt2203(sycl::queue queue, std::uint32_t seed = default_seed);

        mt2203(sycl::queue queue, std::uint32_t seed, std::uint32_t engine_idx);

        mt2203(sycl::queue queue, std::initializer_list<std::uint32_t> seed);

        mt2203(sycl::queue queue, std::initializer_list<std::uint32_t> seed, std::uint32_t engine_idx);

        mt2203(const mt2203& other);

        mt2203(mt2203&& other);

        mt2203& operator=(const mt2203& other);

        mt2203& operator=(mt2203&& other);

        ~mt2203();
    };
    }

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `mt2203(sycl::queue queue, std::uint32_t seed = default_seed)`_
          - Constructor for common seed initialization of the engine (for this case multiple generators of the set would be used)
        * - `mt2203(sycl::queue queue, std::uint32_t seed, std::uint32_t engine_idx)`_
          - Constructor for common seed initialization of the engine (for this case single generator of the set would be used)
        * - `mt2203(sycl::queue queue, std::initializer_list<std::uint32_t> seed)`_
          - Constructor for extended seed initialization of the engine (for this case multiple generators of the set would be used)
        * - `mt2203(sycl::queue queue, std::initializer_list<std::uint32_t> seed, std::uint32_t engine_idx)`_
          - Constructor for extended seed initialization of the engine (for this case single generator of the set would be used)
        * - `mt2203(const mt2203& other)`_
          - Copy constructor
        * - `mt2203(mt2203&& other)`_
          - Move constructor
        * - `mt2203& operator=(const mt2203& other)`_
          - Copy assignment operator
        * - `mt2203& operator=(mt2203&& other)`_
          - Move assignment operator

.. container:: section

    .. rubric:: Constructors

    .. _`mt2203(sycl::queue queue, std::uint32_t seed = default_seed)`:

    .. code-block:: cpp

        mt2203::mt2203(sycl::queue queue, std::uint32_t seed = default_seed)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state. The initialization algorithm described in [:ref:`MT2203 <onemath_rng_bibliography>`].

    .. _`mt2203(sycl::queue queue, std::uint32_t seed, std::uint32_t engine_idx)`:

    .. code-block:: cpp

        mt2203::mt2203(sycl::queue queue, std::uint32_t seed, std::uint32_t engine_idx)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state. The initialization algorithm described in [:ref:`MT2203 <onemath_rng_bibliography>`].

        engine_idx
            The index of the set 1, ..., n_engines.

    .. container:: section

        .. rubric:: Throws

        oneapi::math::invalid_argument
            Exception is thrown when :math:`idx > n_engines`

    .. _`mt2203(sycl::queue queue, std::initializer_list<std::uint32_t> seed)`:

    .. code-block:: cpp

        mt2203::mt2203(sycl::queue queue, std::initializer_list<std::uint32_t> seed)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state. The initialization algorithm described in [:ref:`MT2203 <onemath_rng_bibliography>`].

    .. _`mt2203(sycl::queue queue, std::initializer_list<std::uint32_t> seed, std::uint32_t engine_idx)`:

    .. code-block:: cpp

        mt2203::mt2203(sycl::queue queue, std::initializer_list<std::uint32_t> seed, std::uint32_t engine_idx)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state. The initialization algorithm described in [:ref:`MT2203 <onemath_rng_bibliography>`].

        engine_idx
            The index of the set 1, ..., n_engines.

    .. _`mt2203(const mt2203& other)`:

    .. code-block:: cpp

        mt2203::mt2203(const mt2203& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mt2203`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`mt2203(mt2203&& other)`:

    .. code-block:: cpp

        mt2203::mt2203(mt2203&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mt2203`` object. The ``queue`` and state of the other engine is moved to the current engine.

    .. _`mt2203& operator=(const mt2203& other)`:

    .. code-block:: cpp

        mt2203::mt2203& operator=(const mt2203& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mt2203`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`mt2203& operator=(mt2203&& other)`:

    .. code-block:: cpp

        mt2203::mt2203& operator=(mt2203&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``mt2203`` r-value object. The ``queue`` and state of the other engine is moved to the current engine.

**Parent topic:** :ref:`onemath_rng_engines_basic_random_number_generators`

.. _`the following tool`: https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/DC/dc.html
