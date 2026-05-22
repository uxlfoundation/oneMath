.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_sfmt19937:

sfmt19937
=========

The SIMD-oriented Mersenne Twister pseudorandom number generator.

.. _onemath_rng_sfmt19937_description:

.. rubric:: Description

SIMD-oriented Fast Mersenne Twister pseudorandom number generator SFMT19937 [:ref:`Saito08 <onemath_rng_bibliography>`] with a period length equal to :math:`2 ^ {19937}-1` of the produced sequence. The state of the engine contains the array of 156 128-bit integers.

.. container:: section

    .. rubric:: Generation algorithm

    :math:`w_n = w_0 A \oplus w_M B \oplus w_{n-2} C \oplus w_{n-1} D`

    Where :math:`w_0, w_M, w_{n-2}, ...` are the 128-bit integers, and :math:`wA, wB, wC, wD` operations are defined as follows:

    :math:`wA = (w << 8) \oplus w`, left shift of 128-bit integer :math:`w` by :math:`a` followed by exclusive-or operation

    :math:`wB = (w >> 8) \& mask`, right shift of each 32-bit integer in quadruple :math:`w` by and-operator with quadruple of 32-bit masks :math:`mask = (0xBFFFFFF6, 0xDFFAFFFF, 0xDDFECB7F, 0xDFFFFFEF)`

    :math:`wC = (w >> 8) \oplus w`, right shift of 128-bit integer :math:`w`

    :math:`wD = (w << 8)`, left shift of each 32-bit integer in quadruple :math:`w`

    Integer output: :math:`r_{4n+k} = w_{n}(k)`, where :math:`w_{n}(k)` is the k-th 32-bit integer in quadruple :math:`w_{n}, k = 0, 1, 2, 3`

    :math:`u_n = (int) r_n / 2^{32} + 1/2`


.. _onemath_rng_sfmt19937_description_syntax:

class sfmt19937
---------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    class sfmt19937 {
    public:
        static constexpr std::uint32_t default_seed = 1;

        sfmt19937(sycl::queue queue, std::uint32_t seed = default_seed);

        sfmt19937(sycl::queue queue, std::initializer_list<std::uint32_t> seed);

        sfmt19937(const sfmt19937& other);

        sfmt19937(sfmt19937&& other);

        sfmt19937& operator=(const sfmt19937& other);

        sfmt19937& operator=(sfmt19937&& other);

        ~sfmt19937();
    };
    }

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `sfmt19937(sycl::queue queue, std::uint32_t seed = default_seed)`_
          - Constructor for common seed initialization of the engine
        * - `sfmt19937(sycl::queue queue, std::initializer_list<std::uint32_t> seed)`_
          - Constructor for extended seed initialization of the engine
        * - `sfmt19937(const sfmt19937& other)`_
          - Copy constructor
        * - `sfmt19937(sfmt19937&& other)`_
          - Move constructor
        * - `sfmt19937& operator=(const sfmt19937& other)`_
          - Copy assignment operator
        * - `sfmt19937& operator=(sfmt19937&& other)`_
          - Move assignment operator

.. container:: section

    .. rubric:: Constructors

    .. _`sfmt19937(sycl::queue queue, std::uint32_t seed = default_seed)`:

    .. code-block:: cpp
    
        sfmt19937::sfmt19937(sycl::queue queue, std::uint32_t seed = default_seed)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state. The initialization algorithm described in [:ref:`Saito08 <onemath_rng_bibliography>`].

    .. _`sfmt19937(sycl::queue queue, std::initializer_list<std::uint32_t> seed)`:

    .. code-block:: cpp
    
        sfmt19937::sfmt19937(sycl::queue queue, std::initializer_list<std::uint32_t> seed)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        seed
            The initial conditions of the generator state. The initialization algorithm described in [:ref:`Saito08 <onemath_rng_bibliography>`].

    .. _`sfmt19937(const sfmt19937& other)`:

    .. code-block:: cpp
    
        sfmt19937::sfmt19937(const sfmt19937& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``sfmt19937`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`sfmt19937(sfmt19937&& other)`:

    .. code-block:: cpp

        sfmt19937::sfmt19937(sfmt19937&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``sfmt19937`` object. The ``queue`` and state of the other engine is moved to the current engine.

    .. _`sfmt19937& operator=(const sfmt19937& other)`:

    .. code-block:: cpp

        sfmt19937::sfmt19937& operator=(const sfmt19937& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``sfmt19937`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`sfmt19937& operator=(sfmt19937&& other)`:

    .. code-block:: cpp

        sfmt19937::sfmt19937& operator=(sfmt19937&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``sfmt19937`` r-value object. The ``queue`` and state of the other engine is moved to the current engine.

**Parent topic:** :ref:`onemath_rng_engines_basic_random_number_generators`
