.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_gaussian:

gaussian
========

Class is used for generation of normally distributed real types random numbers.

.. _onemath_rng_gaussian_description:

.. rubric:: Description

The class object is used in the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` function to provide random numbers normally distributed with mean :math:`(mean, a)` and standard deviation :math:`(stddev, \sigma)`, where :math:`a, \sigma \in R; \sigma > 0`.

The probability distribution is given by:

.. math::

    f_{a, \sigma}(x) = \frac{1}{\sigma\sqrt{2\pi}}exp(-\frac{(x - a)^2}{2*\sigma^2}), x \in R.

The cumulative distribution function is as follows:

.. math::

    F_{a, \sigma}(x) = \int^x_{-\infty}\frac{1}{\sigma\sqrt{2\pi}}exp(-\frac{(y - a)^2}{2*\sigma^2})dy, x \in R.


.. _onemath_rng_gaussian_syntax:

class gaussian
--------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    template<typename RealType = float, typename Method = gaussian_method::by_default>
    class gaussian {
    public:
        using method_type = Method;
        using result_type = RealType;
        gaussian();
        explicit gaussian(RealType mean, RealType stddev);
        RealType mean() const;
        RealType stddev() const;
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

        typename Method = oneapi::math::rng::gaussian_method::by_default
            Transformation method, which will be used for generation. Supported types:

                * ``oneapi::math::rng::gaussian_method::by_default``
                * ``oneapi::math::rng::gaussian_method::box_muller``
                * ``oneapi::math::rng::gaussian_method::box_muller2``
                * ``oneapi::math::rng::gaussian_method::icdf``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_rng_distributions_template_parameter_onemath_rng_method_values>`

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `gaussian()`_
          - Default constructor
        * - `explicit gaussian(RealType mean, RealType stddev)`_
          - Constructor with parameters
        * - `RealType mean() const`_
          - Method to obtain mean value
        * - `RealType stddev() const`_
          - Method to obtain standard deviation value

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            gaussian::method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            gaussian::result_type = RealType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`gaussian()`:

        .. code-block:: cpp

            gaussian::gaussian()

        .. container:: section

            .. rubric:: Description

            Default constructor for distribution, parameters set as `mean` = 0.0, `stddev` = 1.0.

    .. container:: section

        .. _`explicit gaussian(RealType mean, RealType stddev)`:

        .. code-block:: cpp

            explicit gaussian::gaussian(RealType mean, RealType stddev)

        .. container:: section

            .. rubric:: Description

            Constructor with parameters. `mean` is a mean value, `stddev` is a standard deviation value.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when `stddev` :math:`\leq` `static_cast<RealType>(0.0)`

.. container:: section

    .. rubric:: Characteristics

    .. container:: section

        .. _`RealType mean() const`:

        .. code-block:: cpp

            RealType gaussian::mean() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `mean` - mean value.

    .. container:: section

        .. _`RealType stddev() const`:

        .. code-block:: cpp

            RealType gaussian::stddev() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `stddev` - standard deviation value.

**Parent topic:** :ref:`onemath_rng_distributions`
