.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_sobol:

sobol
=====

The sobol is a 32-bit Gray code-based quasi-random number generator.

.. _onemath_rng_sobol_description:

.. rubric:: Description

Bratley and Fox :ref:`[Bratley88] <onemath_rng_bibliography>` provide an implementation of the SOBOL quasi-random number generator. The default dimensions of quasi-random vectors can vary from 1 to 40 inclusive. It is also allowed to register user-defined parameters (direction numbers).

.. container:: section

    .. rubric:: Generation algorithm


    :math:`x_n=x_{n_1} \oplus v_c`

    :math:`u_n=x_n / 2 ^ {32}`

    The value :math:`c` is the right-most zero bit in :math:`n-1`; :math:`x_n` is s-dimensional vector of 32-bit values. The s-dimensional vectors (calculated during engine initialization) :math:`v_i, i = 1, 32` are called direction numbers. The vector :math:`u_n` is the generator output normalized to the unit hypercube :math:`(0, 1) ^ s`.

.. _onemath_rng_sobol_description_syntax:

class sobol
-----------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    class sobol {
    public:
        static constexpr std::uint32_t default_dimensions_number = 1;

        sobol(sycl::queue queue, std::uint32_t dimensions = default_dimensions_number);

        sobol(sycl::queue queue, std::vector<std::uint32_t>& direction_numbers);

        sobol(const sobol& other);

        sobol(sobol&& other);

        sobol& operator=(const sobol& other);

        sobol& operator=(sobol&& other);

        ~sobol();
    };
    }

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `sobol(sycl::queue queue, std::uint32_t dimensions = default_dimensions_number)`_
          - Constructor with specified number of dimensions. The value should be :math:`1..40`.
        * - `sobol(sycl::queue queue, std::vector<std::uint32_t>& direction_numbers)`_
          - Constructor for extended use-case, when it's needed to use the number of dimensions greater than 40 or obtain another sequence.
        * - `sobol(const sobol& other)`_
          - Copy constructor
        * - `sobol(sobol&& other)`_
          - Move constructor
        * - `sobol& operator=(const sobol& other)`_
          - Copy assignment operator
        * - `sobol& operator=(sobol&& other)`_
          - Move assignment operator

.. container:: section

    .. rubric:: Constructors

    .. _`sobol(sycl::queue queue, std::uint32_t dimensions = default_dimensions_number)`:

    .. code-block:: cpp
    
        sobol::sobol(sycl::queue queue, std::uint32_t dimensions = default_dimensions_number)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        dimensions
            Number of dimensions. If :math:`dimen < 1` or :math:`dimen > 40`, assume :math:`dimen = 1`.

    .. _`sobol(sycl::queue queue, std::vector<std::uint32_t>& direction_numbers)`:

    .. code-block:: cpp
    
        sobol::sobol(sycl::queue queue, std::vector<std::uint32_t>& direction_numbers)

    .. container:: section

        .. rubric:: Input Parameters

        queue
            Valid ``sycl::queue`` object, calls of the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` routine submits kernels in this queue to obtain random numbers from a given engine.

        direction_numbers
            If you want to generate quasi-random vectors of greater dimension or obtain another sequence, you can register a set of your own direction_numbers. The number of dimensions corresponds to direction_numbers.size() / 32.

    .. _`sobol(const sobol& other)`:

    .. code-block:: cpp
    
        sobol::sobol(const sobol& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``sobol`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`sobol(sobol&& other)`:

    .. code-block:: cpp

        sobol::sobol(sobol&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``sobol`` object. The ``queue`` and state of the other engine is moved to the current engine.

    .. _`sobol& operator=(const sobol& other)`:

    .. code-block:: cpp

        sobol::sobol& operator=(const sobol& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``sobol`` object. The ``queue`` and state of the other engine is copied and applied to the current engine.

    .. _`sobol& operator=(sobol&& other)`:

    .. code-block:: cpp

        sobol::sobol& operator=(sobol&& other)

    .. container:: section

        .. rubric:: Input Parameters

        other
            Valid ``sobol`` r-value object. The ``queue`` and state of the other engine is moved to the current engine.

**Parent topic:** :ref:`onemath_rng_engines_basic_random_number_generators`
