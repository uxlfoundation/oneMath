.. SPDX-FileCopyrightText: 2023 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_device_rng_poisson:

poisson
=======

Generates Poisson distributed random values.

.. rubric:: Description

The ``poisson`` class object is used in the ``generate`` and function
to provide Poisson distributed random numbers with distribution parameter :math:`\lambda`, where :math:`\lambda \in R; \lambda > 0`.


The probability distribution is given by:

.. math::

   P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}

:math:`k \in \{0, 1, 2, \ldots \}`.

The cumulative distribution function is as follows:

.. math::

   F_{\lambda}(x) =
   \begin{cases}
      \sum_{k=0}^{\lfloor x \rfloor} \frac{\lambda^k e^{-\lambda}}{k!}, & x \geq 0 \\
      0, & x < 0
   \end{cases},
   x \in R


class poisson
-------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::rng::device {
     template<typename IntType, typename Method>
     class poisson {
     public:
       using method_type = Method;
       using result_type = IntType;

       poisson();
       explicit poisson(double lambda);

       double lambda() const;
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

        typename Method = oneapi::math::rng::poisson_method::by_default
            Transformation method, which will be used for generation. Supported types:

                * ``oneapi::math::rng::device::poisson_method::by_default``
                * ``oneapi::math::rng::device::poisson_method::devroye``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_rng_distributions_template_parameter_onemath_rng_method_values>`.

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `poisson()`_
          - Default constructor
        * - `explicit poisson(double lambda)`_
          - Constructor with parameters
        * - `double lambda() const`_
          - Method to obtain distribution parameter

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            poisson::method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            poisson::result_type = IntType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`poisson()`:

        .. code-block:: cpp

            poisson::poisson()

        .. container:: section

            .. rubric:: Description

            Default constructor for distribution, parameters set as `lambda` = 0.5.

    .. container:: section

        .. _`explicit poisson(double lambda)`:

        .. code-block:: cpp

            explicit poisson::poisson(double lambda)

        .. container:: section

            .. rubric:: Description

            Constructor with parameters. `lambda` is a distribution parameter.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when :math:`lambda \leq 0`

.. container:: section

    .. rubric:: Characteristics

    .. container:: section

        .. _`double lambda() const`:

        .. code-block:: cpp

            double poisson::lambda() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `lambda`.

**Parent topic:** :ref:`onemath_device_rng_distributions`
