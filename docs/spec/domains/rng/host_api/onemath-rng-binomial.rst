.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_binomial:

binomial
========

Class is used for generation of binomially distributed integer types random numbers.

.. _onemath_rng_binomial_description:

.. rubric:: Description

The class object is used in the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` function to provide random numbers binomially distributed with a number of independent Bernoulli trials :math:`m`, and with probability :math:`p` of a single trial success, where :math:`p \in R; 0 \leq p \leq 1, m \in N`.

A binomially distributed variate represents the number of successes in :math:`m` independent Bernoulli trials with probability of a single trial success :math:`p`.

The probability distribution is given by:

.. math::

    P(X = k) = C^k_{m}p^k(1 - p)^{m - k}, k \in \{0, 1, ... , m\}

The cumulative distribution function is as follows:

.. math::

    F_{m, p}(x) = \left\{ \begin{array}{rcl} 0, x < 0 \\ \sum^{\lfloor x \rfloor}_{k = 0} C^k_{m}p^k(1 - p)^{m - k}, 0 \leq x < m, x \in R \\ 1, x \ge m \end{array}\right.


.. _onemath_rng_binomial_syntax:

class binomial
--------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    template<typename IntType = std::int32_t, typename Method = binomial_method::by_default>
    class binomial {
    public:
        using method_type = Method;
        using result_type = IntType;
        binomial();
        explicit binomial(std::int32_t ntrial, double p);
        std::int32_t ntrial() const;
        double p() const;
    };
    }

.. container:: section

    .. rubric:: Template parameters

    .. container:: section

        typename IntType
            Type of the produced values. Supported types:
                * ``std::int32_t``

    .. container:: section

        typename Method = oneapi::math::rng::binomial_method::by_default
            Transformation method, which will be used for generation. Supported types:

                * ``oneapi::math::rng::binomial_method::by_default``
                * ``oneapi::math::rng::binomial_method::btpe``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_rng_distributions_template_parameter_onemath_rng_method_values>`.

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `binomial()`_
          - Default constructor
        * - `explicit binomial(std::int32_t ntrial, double p)`_
          - Constructor with parameters
        * - `std::int32_t ntrial() const`_
          - Method to obtain number of independent trials `m`
        * - `double p() const`_
          - Method to obtain success probability of a single trial `p`

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            binomial::method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            binomial::result_type = IntType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`binomial()`:

        .. code-block:: cpp

            binomial::binomial()

        .. container:: section

            .. rubric:: Description

            Default constructor for distribution, parameters set as `m` = 5, `p` = 0.5.

    .. container:: section

        .. _`explicit binomial(std::int32_t ntrial, double p)`:

        .. code-block:: cpp

            explicit binomial::binomial(std::int32_t ntrial, double p)

        .. container:: section

            .. rubric:: Description

            Constructor with parameters. `ntrial` is the number of independent trials, `p` is the success probability of a single trial.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when :math:`p > 1.0`, or :math:`p < 0.0`, or :math:`ntrial < 1`

.. container:: section

    .. rubric:: Characteristics

    .. container:: section

        .. _`std::int32_t ntrial() const`:

        .. code-block:: cpp

            std::int32_t binomial::ntrial() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `m` - number of independent trials.

    .. container:: section

        .. _`double p() const`:

        .. code-block:: cpp

            double binomial::p() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `p` - success probability of a single trial.

**Parent topic:** :ref:`onemath_rng_distributions`
