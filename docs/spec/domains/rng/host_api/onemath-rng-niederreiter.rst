.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_niederreiter:

niederreiter
============

The niederreiter generator is a 32-bit Gray code-based quasi-random number generator.

.. _onemath_rng_niederreiter_description:

.. rubric:: Description

According to results of Bratley, Fox and Niederreiter :ref:`[Bratley92] <onemath_rng_bibliography>` Niederreiter sequences have the best known theoretical asymptotic properties. The default dimension of quasi-random vectors can vary from 1 to 318 inclusive. It is also allowed to register user-defined parameters (irreducible polynomials).

.. container:: section

    .. rubric:: Generation algorithm


    :math:`x_n=x_{n_1} \oplus v_c`

    :math:`u_n=x_n / 2 ^ {32}`

    The value :math:`c` is the right-most zero bit in :math:`n-1`; :math:`x_n` is s-dimensional vector of 32-bit values. The s-dimensional vectors (calculated during engine initialization) :math:`v_i, i = 1, 32` are called direction numbers. The vector :math:`u_n` is the generator output normalized to the unit hypercube :math:`(0, 1) ^ s`.

.. _onemath_rng_niederreiter_description_syntax:

class niederreiter
------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    class niederreiter {
    public:
        static constexpr std::uint32_t default_dimensions_number = 1;

        niederreiter(sycl::queue queue, std::uint32_t dimensions = default_dimensions_number);

        niederreiter(sycl::queue queue, std::vector<std::uint32_t>& irred_polynomials);

        niederreiter(const niederreiter& other);

        niederreiter(niederreiter&& other);

        niederreiter& operator=(const niederreiter& other);

        niederreiter& operator=(niederreiter&& other);

        ~niederreiter();
    };
    }

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `niederreiter(sycl::queue queue, std::uint32_t dimensions = default_dimensions_number)`_
          - Constructor with specified number of dimensions. The value should be :math:`1..318`.
        * - `niederreiter(sycl::queue queue, std::vector<std::uint32_t>& irred_polynomials)`_
          - Constructor for extended use-case, when it's needed to use the number of dimensions greater than 318 or obtain another sequence.
        * - `niederreiter(const niederreiter& other)`_
          - Copy constructor
        * - `niederreiter(niederreiter&& other)`_
          - Move constructor
        * - `niederreiter& operator=(const niederreiter& other)`_
          - Copy assignment operator
        * - `niederreiter& operator=(niederreiter&& other)`_
          - Move assignment operator

.. container:: section

    .. rubric:: Constructors

    .. _`niederreiter(sycl::queue queue, std::uint32_t dimensions = default_dimensions_number)`:

    .. code-block:: cpp
    
        niederreiter::niederreiter(sycl::queue queue, std::uint32_t dimensions = default_dimensions_number)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        dimensions
            Number of dimensions. If :math:`dimen < 1` or :math:`dimen > 318`, assume :math:`dimen = 1`.

    .. _`niederreiter(sycl::queue queue, std::vector<std::uint32_t>& irred_polynomials)`:

    .. code-block:: cpp
    
        niederreiter::niederreiter(sycl::queue queue, std::vector<std::uint32_t>& irred_polynomials)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        irred_polynomials
            If you want to generate quasi-random vectors of greater dimension or obtain another sequence, you can register a set of your own irreducible polynomials. The number of dimensions corresponds to the length of the vector.

    .. _`niederreiter(const niederreiter& other)`:

    .. code-block:: cpp
    
        niederreiter::niederreiter(const niederreiter& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``niederreiter`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`niederreiter(niederreiter&& other)`:

    .. code-block:: cpp

        niederreiter::niederreiter(niederreiter&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``niederreiter`` object. The ``queue`` and state of the other engine is moved to the current engine.

    .. _`niederreiter& operator=(const niederreiter& other)`:

    .. code-block:: cpp

        niederreiter::niederreiter& operator=(const niederreiter& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``niederreiter`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`niederreiter& operator=(niederreiter&& other)`:

    .. code-block:: cpp

        niederreiter::niederreiter& operator=(niederreiter&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``niederreiter`` r-value object. The ``queue`` and state of the other engine is moved to the current engine.

**Parent topic:** :ref:`onemath_rng_engines_basic_random_number_generators`
