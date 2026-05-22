.. SPDX-FileCopyrightText: 2024 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_device_rng_gamma:

gamma
=====


Generates gamma distributed random numbers.

.. rubric:: Description

The ``gamma`` class object is used in the ``generate`` function to provide
random numbers with gamma distribution that has shape :math:`\alpha`,
displacement :math:`a`, and scale parameter :math:`\beta`, where
:math:`a, \alpha, \beta \in R; \alpha > 0; \beta > 0`.

The probability distribution is given by:

.. math::

    f_{a, \alpha, \beta}(x) = \left\{ \begin{array}{rcl} \frac{1}{\Gamma(\alpha)\beta^{\alpha}}(x - a)^{\alpha - 1}e^{-(x - a) / \beta}, x \ge a \\ 0, x < a \end{array}\right.

The cumulative distribution function is as follows:

.. math::

    F_{a, \alpha, \beta}(x) = \left\{ \begin{array}{rcl} \int^x_a\frac{1}{\Gamma(\alpha)\beta^{\alpha}}(y - a)^{\alpha - 1}e^{-(y - a) / \beta}dy, x \ge a \\ 0, x < a \end{array}\right.


class gamma
-----------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::rng::device {
     template<typename RealType, typename Method>
     class gamma {
     public:
       using method_type = Method;
       using result_type = RealType;

       gamma();
       explicit gamma(RealType alpha, RealType a, RealType beta);

       RealType alpha() const;
       RealType a() const;
       RealType beta() const;
       std::size_t count_rejected_numbers() const;
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

        typename Method
            Generation method. The type is unspecified.


.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `gamma()`_
          - Default constructor
        * - `explicit gamma(RealType alpha, RealType a, RealType beta)`_
          - Constructor with parameters
        * - `RealType alpha() const`_
          - Method to obtain shape value
        * - `RealType a() const`_
          - Method to obtain displacement value
        * - `RealType beta() const`_
          - Method to obtain scale parameter
        * - `size_t count_rejected_numbers() const`_
          - Method to obtain amount of random numbers that were rejected during
            the last ``generate`` function call. If no ``generate`` calls, ``0`` is returned.

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            gamma::method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            gamma::result_type = RealType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`gamma()`:

        .. code-block:: cpp

            gamma::gamma()

        .. container:: section

            .. rubric:: Description

            Default constructor for distribution, parameters set as
            ``alpha`` = 1.0, ``a`` = 0.0, ``beta`` = 1.0.

    .. container:: section

        .. _`explicit gamma(RealType alpha, RealType a, RealType beta)`:

        .. code-block:: cpp

            explicit gamma::gamma(RealType alpha, RealType a, RealType beta)

        .. container:: section

            .. rubric:: Description

            Constructor with parameters. ``alpha`` is a shape, ``a`` is a displacement, ``beta`` is a scale parameter.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when :math:`alpha \leq 0` or :math:`beta \leq 0`

.. container:: section

    .. rubric:: Characteristics

    .. container:: section

        .. _`RealType alpha() const`:

        .. code-block:: cpp

            RealType gamma::alpha() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter ``alpha`` - shape.

    .. container:: section

        .. _`RealType a() const`:

        .. code-block:: cpp

            RealType gamma::a() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter ``a`` - displacement.

    .. container:: section

        .. _`RealType beta() const`:

        .. code-block:: cpp

            RealType gamma::beta() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter ``beta`` - scale parameter value.

    .. container:: section

        .. _`size_t count_rejected_numbers() const`:

        .. code-block:: cpp

            std::size_t gamma::count_rejected_numbers() const

        .. container:: section

            .. rubric:: Return Value

            Returns the amount of random numbers that were rejected during
            the last ``generate`` function call. If no ``generate`` calls, ``0`` is returned.

**Parent topic:** :ref:`onemath_device_rng_distributions`
