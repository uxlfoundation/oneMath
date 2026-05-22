.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_poisson_v:

poisson_v
=========

Class is used for generation of Poisson distributed integer types random numbers with varying mean.

.. _onemath_rng_poisson_v_description:

.. rubric:: Description

The class object is used in the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` function to provide
n random numbers Poisson distributed, with distribution parameter :math:`\lambda_i`, where :math:`\lambda_i \in R; \lambda_i > 0; i = 1, ... , n`.

The probability distribution is given by:

.. math::

    P(X_i = k) = \frac{\lambda_i^ke^{-\lambda_i}}{k!}, k \in \{0, 1, 2, ... \}.

The cumulative distribution function is as follows:

.. math::

    F_{\lambda_i}(x) = \left\{ \begin{array}{rcl} \sum_{k = 0}^{\lfloor x \rfloor} \frac{\lambda_i^ke^{-\lambda_i}}{k!}, x \ge 0 \\ 0, x < 0 \end{array}\right.

.. _onemath_rng_poisson_v_syntax:

class poisson_v
---------------

Let ``SequenceContainerOrView`` be a type that can be one of C++ Sequence containers or C++ Views (``span``, ``mdspan``).
It's implementation defined which type ``SequenceContainerOrView`` represents.

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    template<typename IntType = std::int32_t, typename Method = poisson_v_method::by_default>
    class poisson_v {
    public:
        using method_type = Method;
        using result_type = IntType;
        explicit poisson_v(SequenceContainerOrView<double> lambda);
        SequenceContainerOrView<double> lambda() const;
    };
    }

.. container:: section

    .. rubric:: Template parameters

    .. container:: section

        typename IntType
            Type of the produced values. Supported types:
                * ``std::int32_t``

    .. container:: section

        typename Method = oneapi::math::rng::poisson_v_method::by_default
            Transformation method, which will be used for generation. Supported types:

                * ``oneapi::math::rng::poisson_v_method::by_default``
                * ``oneapi::math::rng::poisson_v_method::gaussian_icdf_based``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_rng_distributions_template_parameter_onemath_rng_method_values>`.

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `explicit poisson_v(SequenceContainerOrView<double> lambda)`_
          - Constructor with parameters
        * - `SequenceContainerOrView<double> lambda() const`_
          - Method to obtain distribution parameter

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            poisson_v::method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            poisson_v::result_type = IntType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`explicit poisson_v(SequenceContainerOrView<double> lambda)`:

        .. code-block:: cpp

            explicit poisson_v::poisson_v(SequenceContainerOrView<double> lambda)

        .. container:: section

            .. rubric:: Description

            Constructor with parameters. ``lambda`` is a distribution parameter.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when :math:`lambda.size() \leq 1`

.. container:: section

    .. rubric:: Characteristics

    .. container:: section

        .. _`SequenceContainerOrView<double> lambda() const`:

        .. code-block:: cpp

            SequenceContainerOrView<double> poisson_v::lambda() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter ``lambda``.

**Parent topic:** :ref:`onemath_rng_distributions`
