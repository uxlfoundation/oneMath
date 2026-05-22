.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_laplace:

laplace
=======

Class is used for generation of Laplace distributed real types random numbers.

.. _onemath_rng_laplace_description:

.. rubric:: Description

The class object is used in the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` function to provide random numbers Laplace distributed with mean value (or average) :math:`a`, and scalefactor :math:`(b, \beta)`, where :math:`a, \beta \in R; \beta > 0`. The scalefactor value determines the standard deviation as :math:`\sigma = \beta\sqrt{2}`.

The probability distribution is given by:

.. math::

     f_{a, \beta}(x) = \frac{1}{\sqrt{2\beta}}exp(-\frac{|x - a|}{\beta}), x \in R.

The cumulative distribution function is as follows:

.. math::

     F_{a, \beta}(x) = \left\{ \begin{array}{rcl} \frac{1}{2}exp(-\frac{|x - a|}{\beta}), x \ge a \\ 1 - \frac{1}{2}exp(-\frac{|x - a|}{\beta}), x < a \end{array}\right.

.. _onemath_rng_laplace_syntax:

class laplace
-------------

.. rubric:: Syntax

.. code-block:: cpp

    template<typename RealType = float, typename Method = laplace_method::by_default>
    class laplace {
    public:
        using method_type = Method;
        using result_type = RealType;
        laplace();
        explicit laplace(RealType a, RealType b);
        RealType a() const;
        RealType b() const;
    };

.. container:: section

    .. rubric:: Template parameters

    .. container:: section

        typename RealType
            Type of the produced values. Supported types:
                * ``float``
                * ``double``

    .. container:: section

        typename Method = oneapi::math::rng::laplace_method::by_default
            Transformation method, which will be used for generation. Supported types:

                * ``oneapi::math::rng::laplace_method::by_default``
                * ``oneapi::math::rng::laplace_method::icdf``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_rng_distributions_template_parameter_onemath_rng_method_values>`.

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `laplace()`_
          - Default constructor
        * - `explicit laplace(RealType a, RealType b)`_
          - Constructor with parameters
        * - `RealType a() const`_
          - Method to obtain mean value
        * - `RealType b() const`_
          - Method to obtain scalefactor value

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            laplace::method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            laplace::result_type = RealType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`laplace()`:

        .. code-block:: cpp

            laplace::laplace()

        .. container:: section

            .. rubric:: Description

            Default constructor for distribution, parameters set as `a` = 0.0, and `beta` = 1.0.

    .. container:: section

        .. _`explicit laplace(RealType a, RealType b)`:

        .. code-block:: cpp

            explicit laplace::laplace(RealType a, RealType b)

        .. container:: section

            .. rubric:: Description

            Constructor with parameters. `a` is a mean value, `beta` is a scalefactor value.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when :math:`b \leq` static_cast<RealType>(0.0)

.. container:: section

    .. rubric:: Characteristics

    .. container:: section

        .. _`RealType a() const`:

        .. code-block:: cpp

            RealType laplace::a() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `a` - mean value.

    .. container:: section

        .. _`RealType b() const`:

        .. code-block:: cpp

            RealType laplace::b() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `b` - scalefactor value.

**Parent topic:** :ref:`onemath_rng_distributions`
