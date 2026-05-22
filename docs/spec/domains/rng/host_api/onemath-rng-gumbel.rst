.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_gumbel:

gumbel
======

Class is used for generation of Gumbel distributed real types random numbers.

.. _onemath_rng_gumbel_description:

.. rubric:: Description

The class object is used in the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` function to provide random numbers Gumbel distributed with displacement :math:`a`, and scalefactor :math:`(b, \beta)`, where :math:`a, \beta \in R; \beta > 0`.

The probability distribution is given by:

.. math::

     f_{a, \beta}(x) = \frac{1}{\beta}exp(-\frac{x - a}{\beta})exp(-exp(\frac{x - a}{\beta})), x \in R.

The cumulative distribution function is as follows:

.. math::

     F_{a, \beta}(x) = 1 - exp(-exp(\frac{x - a}{\beta})), x \in R.

.. _onemath_rng_gumbel_syntax:

class gumbel
------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    template<typename RealType = float, typename Method = gumbel_method::by_default>
    class gumbel {
    public:
        using method_type = Method;
        using result_type = RealType;
        gumbel();
        explicit gumbel(RealType a, RealType b);
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

        typename Method = oneapi::math::rng::gumbel_method::by_default
            Transformation method, which will be used for generation. Supported types:

                * ``oneapi::math::rng::gumbel_method::by_default``
                * ``oneapi::math::rng::gumbel_method::icdf``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_rng_distributions_template_parameter_onemath_rng_method_values>`.

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `gumbel()`_
          - Default constructor
        * - `explicit gumbel(RealType a, RealType b)`_
          - Constructor with parameters
        * - `RealType a() const`_
          - Method to obtain displacement value
        * - `RealType b() const`_
          - Method to obtain scalefactor value

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            gumbel::method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            gumbel::result_type = RealType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`gumbel()`:

        .. code-block:: cpp

            gumbel::gumbel()

        .. container:: section

            .. rubric:: Description

            Default constructor for distribution, parameters set as `a` = 0.0, and `beta` = 1.0.

    .. container:: section

        .. _`explicit gumbel(RealType a, RealType b)`:

        .. code-block:: cpp

            explicit gumbel::gumbel(RealType a, RealType b)

        .. container:: section

            .. rubric:: Description

            Constructor with parameters. `a` is a displacement value, `beta` is a scalefactor value.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when :math:`b \leq` static_cast<RealType>(0.0)

.. container:: section

    .. rubric:: Characteristics

    .. container:: section

        .. _`RealType a() const`:

        .. code-block:: cpp

            RealType gumbel::a() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `a` - displacement value.

    .. container:: section

        .. _`RealType b() const`:

        .. code-block:: cpp

            RealType gumbel::b() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `b` - scalefactor value.

**Parent topic:** :ref:`onemath_rng_distributions`
