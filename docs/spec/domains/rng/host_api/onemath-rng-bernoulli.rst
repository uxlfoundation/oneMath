.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_bernoulli:

bernoulli
=========

Class is used for generation of Bernoulli distributed integer types random numbers.

.. _onemath_rng_bernoulli_description:

.. rubric:: Description

The class object is used in the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` function to provide random numbers Bernoulli distributed with probability :math:`p` of a single trial success, where :math:`p \in R; 0 \leq p; p \leq 1`.

The probability distribution is given by:

.. math::

    P(X = 1) = p

.. math::

    P(X = 0) = 1 - p

The cumulative distribution function is as follows:

.. math::

    F_{p}(x) = \left\{ \begin{array}{rcl} 0, x < 0 \\ 1 - p, 0 \leq x < 1, x \in R \\ 1, x \ge 1 \end{array}\right.

.. _onemath_rng_bernoulli_syntax:

class bernoulli
---------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    template<typename IntType = std::int32_t, typename Method = bernoulli_method::by_default>
    class bernoulli {
    public:
        using method_type = Method;
        using result_type = IntType;
        bernoulli();
        explicit bernoulli(float p);
        float p() const;
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

        typename Method = oneapi::math::rng::bernoulli_method::by_default
            Transformation method, which will be used for generation. Supported types:

                * ``oneapi::math::rng::bernoulli_method::by_default``
                * ``oneapi::math::rng::bernoulli_method::icdf``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_rng_distributions_template_parameter_onemath_rng_method_values>`.

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `bernoulli()`_
          - Default constructor
        * - `explicit bernoulli(float p)`_
          - Constructor with parameters
        * - `float p() const`_
          - Method to obtain probability `p`

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            bernoulli::method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            bernoulli::result_type = IntType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`bernoulli()`:

        .. code-block:: cpp

            bernoulli::bernoulli()

        .. container:: section

            .. rubric:: Description

            Default constructor for distribution, parameters set as `p` = 0.5f.

    .. container:: section

        .. _`explicit bernoulli(float p)`:

        .. code-block:: cpp

            explicit bernoulli::bernoulli(float p)

        .. container:: section

            .. rubric:: Description

            Constructor with parameters. `p` is a probability.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when `p > 1.0f`, or `p < 0.0f`

.. container:: section

    .. rubric:: Characteristics

    .. container:: section

        .. _`float p() const`:

        .. code-block:: cpp

            float p() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `p` - probability.

**Parent topic:** :ref:`onemath_rng_distributions`
