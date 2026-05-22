.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_rayleigh:

rayleigh
========

Class is used for generation of Rayleigh distributed real types random numbers.

.. _onemath_rng_rayleigh_description:

.. rubric:: Description

The class object is used in the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` function to provide random numbers Rayleigh distributed with displacement :math:`a`, and scalefactor :math:`(b, \beta)`, where :math:`a, \beta \in R; \beta > 0`.

The Rayleigh distribution is a special case of the :ref:`onemath_rng_weibull` distribution, where the shape parameter `\alpha` = 2 .

The probability distribution is given by:

.. math::

     f_{a, \beta}(x) = \left\{ \begin{array}{rcl} \frac{2(x - a)}{\beta^2}exp(-\frac{(x - a)^2)}{\beta^2}), x \ge a \\ 0, x < a \end{array}\right.

The cumulative distribution function is as follows:

.. math::

    F_{a, \beta}(x) = \left\{ \begin{array}{rcl} 1 - exp(-\frac{(x - a)^2)}{\beta^2}), x \ge a \\ 0, x < a \end{array}\right.

.. _onemath_rng_rayleigh_syntax:

class rayleigh
--------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    template<typename RealType = float, typename Method = rayleigh_method::by_default>
    class rayleigh {
    public:
        using method_type = Method;
        using result_type = RealType;
        rayleigh();
        explicit rayleigh(RealType a, RealType b);
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

        typename Method = oneapi::math::rng::rayleigh_method::by_default
            Transformation method, which will be used for generation. Supported types:

                * ``oneapi::math::rng::rayleigh_method::by_default``
                * ``oneapi::math::rng::rayleigh_method::icdf``
                * ``oneapi::math::rng::rayleigh_method::icdf_accurate``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_rng_distributions_template_parameter_onemath_rng_method_values>`.

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `rayleigh()`_
          - Default constructor
        * - `explicit rayleigh(RealType a, RealType b)`_
          - Constructor with parameters
        * - `RealType a() const`_
          - Method to obtain displacement value
        * - `RealType b() const`_
          - Method to obtain scalefactor value

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            rayleigh::method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            rayleigh::result_type = RealType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`rayleigh()`:

        .. code-block:: cpp

            rayleigh::rayleigh()

        .. container:: section

            .. rubric:: Description

            Default constructor for distribution, parameters set as `a` = 0.0, and `b` = 1.0.

    .. container:: section

        .. _`explicit rayleigh(RealType a, RealType b)`:

        .. code-block:: cpp

            explicit rayleigh::rayleigh(RealType a, RealType b)

        .. container:: section

            .. rubric:: Description

            Constructor with parameters. `a` is a displacement value, `b` is a scalefactor value.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when :math:`b \leq` static_cast<RealType>(0.0)

.. container:: section

    .. rubric:: Characteristics

    .. container:: section

        .. _`RealType a() const`:

        .. code-block:: cpp

            RealType rayleigh::a() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `a` - displacement value.

    .. container:: section

        .. _`RealType b() const`:

        .. code-block:: cpp

            RealType rayleigh::b() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `b` - scalefactor value.

**Parent topic:** :ref:`onemath_rng_distributions`
