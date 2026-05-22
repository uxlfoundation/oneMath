.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_negative_binomial:

negative_binomial
=================

Class is used for generation of negative binomially distributed integer types random numbers.

.. _onemath_rng_negative_binomial_description:

.. rubric:: Description

The class object is used in the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` function to provide random numbers negative binomially distributed with distribution parameters :math:`a` and :math:`p`, where :math:`p, a \in R; 0 \leq p \leq 1, a > 0`.

The probability distribution is given by:

.. math::

    P(X = k) = C^k_{a + k - 1}p^a(1 - p)^{k}, k \in \{0, 1, 2, ...\}

The cumulative distribution function is as follows:

.. math::

    F_{a, p}(x) = \left\{ \begin{array}{rcl} \sum^{\lfloor x \rfloor}_{k = 0} C^k_{a + k - 1}p^a(1 - p)^{k}, x \ge 0, x \in R \\ 0, x < 0 \end{array}\right.


.. _onemath_rng_negative_binomial_syntax:

class negative_binomial
-----------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    template<typename IntType = std::int32_t, typename Method = negative_binomial_method::by_default>
    class negative_binomial {
    public:
        using method_type = Method;
        using result_type = IntType;
        negative_binomial();
        explicit negative_binomial(double a, double p);
        double a() const;
        double p() const;
    };
    }

.. container:: section

    .. rubric:: Template parameters

    .. container:: section

        typename IntType
            Type of the produced values. Supported types:
                * ``std::int32_t``
                * ``std::uint32_t``

    .. container:: section

        typename Method = oneapi::math::rng::negative_binomial_method::by_default
            Transformation method, which will be used for generation. Supported types:

                * ``oneapi::math::rng::negative_binomial_method::by_default``
                * ``oneapi::math::rng::negative_binomial_method::nbar``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_rng_distributions_template_parameter_onemath_rng_method_values>`.

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `negative_binomial()`_
          - Default constructor
        * - `explicit negative_binomial(double a, double p)`_
          - Constructor with parameters
        * - `double a() const`_
          - Method to obtain the first distribution parameter `a`
        * - `double p() const`_
          - Method to obtain the second distribution parameter `p`

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            negative_binomial::method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            negative_binomial::result_type = IntType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`negative_binomial()`:

        .. code-block:: cpp

            negative_binomial::negative_binomial()

        .. container:: section

            .. rubric:: Description

            Default constructor for distribution, parameters set as `a` = 0.1, `p` = 0.5.

    .. container:: section

        .. _`explicit negative_binomial(double a, double p)`:

        .. code-block:: cpp

            explicit negative_binomial::negative_binomial(double a, double p)

        .. container:: section

            .. rubric:: Description

            Constructor with parameters. `a` is the first distribution parameter, `p` is the second distribution parameter.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when :math:`p \ge 1.0`, or :math:`p \leq 0.0`, or :math:`a \leq 0.0`

.. container:: section

    .. rubric:: Characteristics

    .. container:: section

        .. _`double a() const`:

        .. code-block:: cpp

            double negative_binomial::a() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `a` - the first distribution parameter.

    .. container:: section

        .. _`double p() const`:

        .. code-block:: cpp

            double negative_binomial::p() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `p` - the second distribution parameter.

**Parent topic:** :ref:`onemath_rng_distributions`
