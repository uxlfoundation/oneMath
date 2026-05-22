.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_cauchy:

cauchy
======

Class is used for generation of Cauchy distributed real types random numbers.

.. _onemath_rng_cauchy_description:

.. rubric:: Description

The class object is used in the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` function to provide random numbers Cauchy distributed with displacement :math:`a`, and scale parameter :math:`(b, \beta)`, where :math:`a, \beta \in R; \beta > 0`.

The probability distribution is given by:

.. math::

    f_{a, \beta}(x) = \frac{1}{\pi\beta(1 + (\frac{x - a}{\beta})^2)}, x \in R.

The cumulative distribution function is as follows:

.. math::

    F_{a, \beta}(x) = \frac{1}{2} + \frac{1}{\pi} \arctan{(\frac{x - a}{\beta})}, x \in R.

.. _onemath_rng_cauchy_syntax:

class cauchy
------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    template<typename RealType = float, typename Method = cauchy_method::by_default>
    class cauchy {
    public:
        using method_type = Method;
        using result_type = RealType;
        cauchy();
        explicit cauchy(RealType a, RealType b);
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

        typename Method = oneapi::math::rng::cauchy_method::by_default
            Transformation method, which will be used for generation. Supported types:

                * ``oneapi::math::rng::cauchy_method::by_default``
                * ``oneapi::math::rng::cauchy_method::icdf``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_rng_distributions_template_parameter_onemath_rng_method_values>`.

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `cauchy()`_
          - Default constructor
        * - `explicit cauchy(RealType a, RealType b)`_
          - Constructor with parameters
        * - `RealType a() const`_
          - Method to obtain displacement value
        * - `RealType b() const`_
          - Method to obtain scalefactor value

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            cauchy::method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            cauchy::result_type = RealType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`cauchy()`:

        .. code-block:: cpp

            cauchy::cauchy()

        .. container:: section

            .. rubric:: Description

            Default constructor for distribution, parameters set as `a` = 0.0, and `b` = 1.0.

    .. container:: section

        .. _`explicit cauchy(RealType a, RealType b)`:

        .. code-block:: cpp

            explicit cauchy::cauchy(RealType a, RealType b)

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

            RealType cauchy::a() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `a` - displacement value.

    .. container:: section

        .. _`RealType b() const`:

        .. code-block:: cpp

            RealType cauchy::b() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `b` - scalefactor value.

**Parent topic:** :ref:`onemath_rng_distributions`
