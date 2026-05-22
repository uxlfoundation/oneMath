.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_uniform_continuous:

uniform (continuous)
====================

Class is used for generation of uniformly distributed real types random numbers.

.. _onemath_rng_uniform_continuous_description:

.. rubric:: Description

The class object is used in the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` function to provide random numbers uniformly distributed over the interval :math:`[a, b)`, where :math:`a`, :math:`b` are the left and right bounds of the
interval, respectively, and :math:`a, b \in R; a < b`

The probability distribution is given by:

.. math::

    f_{a, b}(x) = \left\{ \begin{array}{rcl} \frac{1}{b-a}, x \in [a, b) \\ 0, x \notin [a, b) \end{array}\right.

The cumulative distribution function is as follows:

.. math::

    F_{a, b}(x) = \left\{ \begin{array}{rcl} 0, x < a \\ \frac{x - a}{b - a}, a \leq x < b, x \in R \\ 1, x \ge b \end{array}\right.


.. _onemath_rng_uniform_continuous_syntax:

class uniform
-------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    template<typename RealType = float, typename Method = uniform_method::by_default>
    class uniform {
    public:
        using method_type = Method;
        using result_type = RealType;
        uniform();
        explicit uniform(RealType a, RealType b);
        RealType a() const;
        RealType b() const;
    };
    }

.. container:: section

    .. rubric:: Template parameters

    .. container:: section

        typename RealType
            Type of the produced values. Supported types:
                * ``float``
                * ``double``

    .. container:: section

        typename Method = oneapi::math::rng::uniform_method::by_default
            Transformation method, which will be used for generation. Supported types:

                * ``oneapi::math::rng::uniform_method::by_default``
                * ``oneapi::math::rng::uniform_method::standard``
                * ``oneapi::math::rng::uniform_method::accurate``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_rng_distributions_template_parameter_onemath_rng_method_values>`

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `uniform()`_
          - Default constructor
        * - `explicit uniform(RealType a, RealType b)`_
          - Constructor with parameters
        * - `RealType a() const`_
          - Method to obtain left bound `a`
        * - `RealType b() const`_
          - Method to obtain right bound `b`

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            uniform::method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            uniform::result_type = RealType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`uniform()`:

        .. code-block:: cpp

            uniform::uniform()

        .. container:: section

            .. rubric:: Description

            Default constructor for distribution, parameters set as `a` = 0.0, `b` = 1.0.

    .. container:: section

        .. _`explicit uniform(RealType a, RealType b)`:

        .. code-block:: cpp

            explicit uniform::uniform(RealType a, RealType b)

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

        .. _`RealType a() const`:

        .. code-block:: cpp

            RealType uniform::a() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `a` - left bound.

    .. container:: section

        .. _`RealType b() const`:

        .. code-block:: cpp

            RealType uniform::b() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `b` - right bound.

**Parent topic:** :ref:`onemath_rng_distributions`
