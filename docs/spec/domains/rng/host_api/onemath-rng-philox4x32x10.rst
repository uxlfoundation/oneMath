.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_philox4x32x10:

philox4x32x10
=============

The Philox4x32x10 counter-based pseudorandom number generator.

.. _onemath_rng_philox4x32x10_description:

.. rubric:: Description

The Philox4x32x10 engine is a keyed family of generator of counter-based BRNG. The state consists of 128-bit integer counter :math:`c` and two 32-bits keys :math:`k_0` and :math:`k_1`.

.. container:: section

    .. rubric:: Generation algorithm

    The generator has 32-bit integer output obtained in the following way [:ref:`Salmon11 <onemath_rng_bibliography>`]:

    1. :math:`c_n=c_{n-1} + 1`
    2. :math:`\omega_n = f(c_n)`, where :math:`f` is a function that takes 128-bit argument and returns a 128-bit number. The returned number is obtained as follows:
        2.1. The argument :math:`c` is interpreted as four 32-bit numbers :math:`c = \overline{L_1 R_1 L_0 R_0}`, where :math:`\overline{A B C D} = A \cdot  2^{96} + B \cdot  2^{64} + C \cdot  2^{32} + D`, put :math:`k_0^0 =k_0, k_1^0=k_1`.
        
        2.2. The following recurrence is calculated:

        :math:`L_1^{i+1} =mullo(R_1^i, 0xD2511F53)`

        :math:`R_1^{i+1} =mulhi(R_0^i, 0xCD9E8D57) \oplus k_0^i \oplus L_0^i`

        :math:`L_0^{i+1} =mullo(R_0^i, 0xCD9E8D57)`

        :math:`R_0^{i+1} =mulhi(R_1^i, 0xD2511F53) \oplus k_1^i \oplus L_1^i`

        :math:`k_0^{i+1} =k_0^i + 0xBB67AE85`

        :math:`k_1^{i+1} =k_1^i + 0x9E3779B9`, where :math:`mulhi(a, b)` and :math:`mullo(a, b)` are high and low parts of the :math:`a \cdot b` product respectively.

        2.3. Put :math:`f(c) = \overline{L_1^N R_1^N L_0^N R_0^N}`, where :math:`N = 10`

    3. Integer output: :math:`r_{4n + k} = \omega_n(k)`, where :math:`\omega_n(k)` is the k-th 32-bit integer in quadruple :math:`\omega_n, k = 0, 1, 2, 3`
    4. Real output: :math:`u_n=(int)r_n / 2^{32} + 1/2`

.. _onemath_rng_philox4x32x10_description_syntax:

class philox4x32x10
-------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    class philox4x32x10 {
    public:
    static constexpr std::uint64_t default_seed = 0;

    philox4x32x10(sycl::queue queue, std::uint64_t seed = default_seed);

    philox4x32x10(sycl::queue queue, std::initializer_list<std::uint64_t> seed);

    philox4x32x10(const philox4x32x10& other);

    philox4x32x10(philox4x32x10&& other);

    philox4x32x10& operator=(const philox4x32x10& other);

    philox4x32x10& operator=(philox4x32x10&& other);

    ~philox4x32x10();
    };
    }

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `philox4x32x10(sycl::queue queue, std::uint64_t seed = default_seed)`_
          - Constructor for common seed initialization of the engine
        * - `philox4x32x10(sycl::queue queue, std::initializer_list<std::uint64_t> seed)`_
          - Constructor for extended seed initialization of the engine
        * - `philox4x32x10(const philox4x32x10& other)`_
          - Copy constructor
        * - `philox4x32x10(philox4x32x10&& other)`_
          - Move constructor
        * - `philox4x32x10& operator=(const philox4x32x10& other)`_
          - Copy assignment operator
        * - `philox4x32x10& operator=(philox4x32x10&& other)`_
          - Move assignment operator

.. container:: section

    .. rubric:: Constructors

    .. _`philox4x32x10(sycl::queue queue, std::uint64_t seed = default_seed)`:

    .. code-block:: cpp

      philox4x32x10::philox4x32x10(sycl::queue queue, std::uint64_t seed = default_seed)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state, assume :math:`k = seed, c = 0`, where :math:`k` is a 64-bit key, :math:`c` is a 128-bit counter.

    .. _`philox4x32x10(sycl::queue queue, std::initializer_list<std::uint64_t> seed)`:

    .. code-block:: cpp

      philox4x32x10::philox4x32x10(sycl::queue queue, std::initializer_list<std::uint64_t> seed)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state, assume
            if :math:`n = 0: k = 0, c = 0`

            if :math:`n = 1: k = seed[0], c = 0`

            if :math:`n = 2: k = seed[0], c = seed[1]`

            if :math:`n = 3: k = seed[0], c = seed[1] + seed[2] \cdot 2^{64}`

            for :math:`n > 3` following arguments are ignored

    .. _`philox4x32x10(const philox4x32x10& other)`:

    .. code-block:: cpp

      philox4x32x10::philox4x32x10(const philox4x32x10& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``philox4x32x10`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.
  
    .. _`philox4x32x10(philox4x32x10&& other)`:

    .. code-block:: cpp

      philox4x32x10::philox4x32x10(philox4x32x10&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``philox4x32x10`` r-value object. The ``queue`` and state of the other engine is moved to the current engine.

    .. _`philox4x32x10& operator=(const philox4x32x10& other)`:

    .. code-block:: cpp

      philox4x32x10::philox4x32x10& operator=(const philox4x32x10& other)
  
    .. container:: section

        .. rubric:: Input Parameters

        other
          Valid ``philox4x32x10`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`philox4x32x10& operator=(philox4x32x10&& other)`:

    .. code-block:: cpp

      philox4x32x10::philox4x32x10& operator=(philox4x32x10&& other)
  
    .. container:: section

        .. rubric:: Input Parameters

        other
          Valid ``philox4x32x10`` r-value object. The ``queue`` and state of the other engine is moved to the current engine.

**Parent topic:** :ref:`onemath_rng_engines_basic_random_number_generators`
