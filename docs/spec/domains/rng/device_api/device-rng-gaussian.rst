.. SPDX-FileCopyrightText: 2023 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_device_rng_gaussian:

gaussian
========

Generates normally distributed random numbers.

.. rubric:: Description

The ``gaussian`` class object is used in the ``generate`` and function to provide 
random numbers with normal (Gaussian) distribution with mean (``a``) and standard deviation
(``stddev``, :math:`\sigma` ), where :math:`a, \sigma \in \mathbb{R}; \sigma > 0`

The probability density function is given by:

.. math::

   f_{a, \sigma} (x) =
   \frac{1}{\sigma \sqrt{2 \pi}}
   \exp
   \left(
      - \frac{(y-a)^2}{2\sigma^2}
   \right) dy,
   - \infty < x < + \infty

The cumulative distribution function is as follows:

.. math::

   F_{a, \sigma} (x) =
   \int_{-\infty}^{x}
   \frac{1}{\sigma \sqrt{2 \pi}}
   \exp
   \left(
      - \frac{(y-a)^2}{2\sigma^2}
   \right) dy,
   - \infty < x < + \infty


The cumulative distribution function :math:`F_{a, \sigma}(x)` can be expressed
in terms of standard normal distribution :math:`\phi(x)` as

.. math::

   F_{a,\sigma}(x) = \phi((x - a)/\sigma)


class gaussian
--------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::rng::device {
     template<typename RealType, typename Method>
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

        typename Method
            Generation method. The specific values are as follows:

                * ``oneapi::math::rng::device::gaussian_method::by_default``
                * ``oneapi::math::rng::device::gaussian_method::box_muller2``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_device_rng_distributions_method>`

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
          - Method to obtain left bound `a`
        * - `RealType stddev() const`_
          - Method to obtain right bound `b`

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

            explicit gaussian::gaussian(RealType a, RealType b)

        .. container:: section

            .. rubric:: Description

            Constructor with parameters. `mean` is a mean value, `stddev` is a standard deviation value.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when `stddev` :math:`\leq 0`

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

**Parent topic:** :ref:`onemath_device_rng_distributions`
