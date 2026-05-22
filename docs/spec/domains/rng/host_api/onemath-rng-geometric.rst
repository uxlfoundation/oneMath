.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_geometric:

geometric
=========

Class is used for generation of geometrically distributed integer types random numbers.

.. _onemath_rng_geometric_description:

.. rubric:: Description

The class object is used in the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` function to provide random numbers geometrically distributed with probability :math:`p` of a single success trial, where :math:`p \in R; 0 < p < 1`.

The probability distribution is given by:

.. math::

     P(X = k) = p * (1 - p)^k, k = \{0, 1, 2, ... \}.

The cumulative distribution function is as follows:

.. math::

     F_{p}(x) = \left\{ \begin{array}{rcl} 0, x < 0 \\ 1 - (1 - p)^{\lfloor x + 1 \rfloor}, x \ge 0 \end{array}\right.

.. _onemath_rng_geometric_syntax:

class geometric
---------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    template<typename IntType = std::int32_t, typename Method = geometric_method::by_default>
    class geometric {
    public:
        using method_type = Method;
        using result_type = IntType;
        geometric();
        explicit geometric(float p);
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

        typename Method = oneapi::math::rng::geometric_method::by_default
            Transformation method, which will be used for generation. Supported types:

                * ``oneapi::math::rng::geometric_method::by_default``
                * ``oneapi::math::rng::geometric_method::icdf``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_rng_distributions_template_parameter_onemath_rng_method_values>`.

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `geometric()`_
          - Default constructor
        * - `explicit geometric(float p)`_
          - Constructor with parameters
        * - `float p() const`_
          - Method to obtain probability value

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            geometric::method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            geometric::result_type = IntType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`geometric()`:

        .. code-block:: cpp

            geometric::geometric()

        .. container:: section

            .. rubric:: Description

            Default constructor for distribution, parameters set as `p` = 0.5.

    .. container:: section

        .. _`explicit geometric(float p)`:

        .. code-block:: cpp

            explicit geometric::geometric(float p)

        .. container:: section

            .. rubric:: Description

            Constructor with parameters. `p` is a probability value.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when :math:`p \ge 1.0f`, or :math:`p \leq 0.0f`

.. container:: section

    .. rubric:: Characteristics

    .. container:: section

        .. _`float p() const`:

        .. code-block:: cpp

            float geometric::p() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `p` - probability value.

**Parent topic:** :ref:`onemath_rng_distributions`
