.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_multinomial:

multinomial
===========

Class is used for generation of multinomially distributed integer types random numbers.

.. _onemath_rng_multinomial_description:

.. rubric:: Description

The class object is used in the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` function to provide `n` random numbers multinomially distributed, with independent trials :math:`(ntrial, m)` and possible mutually exclusive outcomes :math:`k`, with corresponding probabilities :math:`p_i`, where :math:`p_i \in R; 0 \leq p_i \leq 1; m, k \in N`.

The probability distribution is given by:

.. math::

    P(X_1 = x_1, ..., X_k = x_k) = \frac{m!}{\Pi^k_{i = 1}x_i!}\Pi^k_{i = 1}p_i^{x_i}, 0 \leq x_i \leq m, \sum^{k}_{i = 1}x_i = m

.. _onemath_rng_multinomial_syntax:

class multinomial
-----------------

Let ``SequenceContainerOrView`` be a type that can be one of C++ Sequence containers or C++ Views (``span``, ``mdspan``).
It's implementation defined which type ``SequenceContainerOrView`` represents.

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    template<typename IntType = std::int32_t, typename Method = multinomial_method::by_default>
    class multinomial {
    public:
        using method_type = Method;
        using result_type = IntType;
        explicit multinomial(double ntrial, SequenceContainerOrView<double> p);
        std::int32_t ntrial() const;
        SequenceContainerOrView<double> p() const;
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

        typename Method = oneapi::math::rng::multinomial_method::by_default
            Transformation method, which will be used for generation. Supported types:

                * ``oneapi::math::rng::multinomial_method::by_default``
                * ``oneapi::math::rng::multinomial_method::poisson_icdf_based``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_rng_distributions_template_parameter_onemath_rng_method_values>`.

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `explicit multinomial(double ntrial, SequenceContainerOrView<double> p)`_
          - Constructor with parameters
        * - `std::int32_t ntrial() const`_
          - Method to obtain number of independent trials
        * - `SequenceContainerOrView<double> p() const`_
          - Method to obtain a probability parameter of possible outcomes

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            multinomial::method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines the transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            multinomial::result_type = IntType

        .. container:: section

            .. rubric:: Description

            The type which defines the type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`explicit multinomial(double ntrial, SequenceContainerOrView<double> p)`:

        .. code-block:: cpp

            explicit multinomial::multinomial(double ntrial, SequenceContainerOrView<double> p)

        .. container:: section

            .. rubric:: Description

            Constructor with parameters. ``ntrial`` is a number of independent trials, ``p`` is a probability parameter.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when :math:`ntrial < 0`, or :math:`p.size() < 1`

.. container:: section

    .. rubric:: Characteristics

    .. container:: section

        .. _`std::int32_t ntrial() const`:

        .. code-block:: cpp

            std::int32_t multinomial::ntrial() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `ntrial`.

    .. container:: section

        .. _`SequenceContainerOrView<double> p() const`:

        .. code-block:: cpp

            SequenceContainerOrView<double> multinomial::p() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `p`.

**Parent topic:** :ref:`onemath_rng_distributions`
