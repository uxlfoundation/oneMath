.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_uniform_discrete:

uniform (discrete)
==================

Class is used for generation of uniformly distributed integer types random numbers.

.. _onemath_rng_uniform_discrete_description:

.. rubric:: Description

The class object is used in the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` function to provide random numbers uniformly distributed over the interval :math:`[a, b)`, where :math:`a`, :math:`b` are the left and right bounds of the
interval, respectively, and :math:`a, b \in R; a < b`.

The probability distribution is given by:

.. math::

    P(X = k) = \frac{1}{b - a}, k \in \{a, a + 1, ... , b - 1\}

The cumulative distribution function is as follows:

.. math::

    F_{a, b}(x) = \left\{ \begin{array}{rcl} 0, x < a \\ \frac{\lfloor x - a + 1 \rfloor}{b-a}, a \leq x < b, x \in R \\ 1, x \ge b \end{array}\right.


.. _onemath_rng_uniform_discrete_syntax:

class uniform
-------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    template<typename Method = uniform_method::by_default>
    class uniform<std::int32_t, Method> {
    public:
        using method_type = Method;
        using result_type = std::int32_t;
        uniform();
        explicit uniform(std::int32_t a, std::int32_t b);
        std::int32_t a() const;
        std::int32_t b() const;
    };
    }

.. container:: section

    .. rubric:: Template parameters

    .. container:: section

        typename Method = oneapi::math::rng::uniform_method::by_default
            Transformation method, which will be used for generation. Supported types:

                * ``oneapi::math::rng::uniform_method::by_default``
                * ``oneapi::math::rng::uniform_method::standard``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_rng_distributions_template_parameter_onemath_rng_method_values>`.

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `uniform()`_
          - Default constructor
        * - `explicit uniform(std::int32_t a, std::int32_t b)`_
          - Constructor with parameters
        * - `std::int32_t a() const`_
          - Method to obtain left bound `a`
        * - `std::int32_t b() const`_
          - Method to obtain right bound `b`

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            result_type = std::int32_t

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`uniform()`:

        .. code-block:: cpp

            uniform()

        .. container:: section

            .. rubric:: Description

            Default constructor for distribution, parameters set as `a` = 0, `b` = std::numeric_limits<std::int32_t>::max().

    .. container:: section

        .. _`explicit uniform(std::int32_t a, std::int32_t b)`:

        .. code-block:: cpp

            uniform(std::int32_t a, std::int32_t b)

        .. container:: section

            .. rubric:: Description

            Constructor with parameters. `a` is a left bound, `b` is a right bound, assume :math:`a < b`.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when :math:`a \ge b`

.. container:: section

    .. rubric:: Characteristics

    .. container:: section

        .. _`std::int32_t a() const`:

        .. code-block:: cpp

            a() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `a` - left bound.

    .. container:: section

        .. _`std::int32_t b() const`:

        .. code-block:: cpp

            b() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `b` - right bound.

**Parent topic:** :ref:`onemath_rng_distributions`
