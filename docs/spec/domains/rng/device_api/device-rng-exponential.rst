.. SPDX-FileCopyrightText: 2023 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_device_rng_exponential:

exponential
===========


Generates exponentially distributed random numbers.

.. rubric:: Description

The ``exponential`` class object is used in the ``generate`` function to provide
random numbers with exponential distribution that has displacement :math:`a` and scalefactor :math:`\beta`,
where :math:`a, \beta \in R ; \beta > 0`.


The probability density function is given by:

.. math::

   f_{a, \beta}(x) =
   \begin{cases}
      \frac{1}{\beta} \exp (-\frac{(x-a)}{\beta}), & x \geq a \\
      0, & x < a
   \end{cases},
   - \infty < x < + \infty

The cumulative distribution function is as follows:

.. math::

   F_{a, \beta}(x) =
   \begin{cases}
      1 - \exp (-\frac{(x-a)}{\beta}), & x \geq a \\
      0, & x < a
   \end{cases},
   - \infty < x < + \infty


class exponential
-----------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::rng::device {
     template<typename RealType, typename Method>
     class exponential {
     public:
       using method_type = Method;
       using result_type = RealType;

       exponential();
       explicit exponential(RealType a, RealType beta);

       RealType a() const;
       RealType beta() const;
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

        typename Method = oneapi::math::rng::exponential_method::by_default
            Generation method. The specific values are as follows:

                * ``oneapi::math::rng::device::exponential_method::by_default``
                * ``oneapi::math::rng::device::exponential_method::icdf``
                * ``oneapi::math::rng::device::exponential_method::icdf_accurate``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_device_rng_distributions_method>`.


.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `exponential()`_
          - Default constructor
        * - `explicit exponential(RealType a, RealType beta)`_
          - Constructor with parameters
        * - `RealType a() const`_
          - Method to obtain displacement value
        * - `RealType beta() const`_
          - Method to obtain scalefactor

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            exponential::method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            exponential::result_type = RealType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`exponential()`:

        .. code-block:: cpp

            exponential::exponential()

        .. container:: section

            .. rubric:: Description

            Default constructor for distribution, parameters set as `a` = 0.0, `beta` = 1.0.

    .. container:: section

        .. _`explicit exponential(RealType a, RealType beta)`:

        .. code-block:: cpp

            explicit exponential::exponential(RealType a, RealType beta)

        .. container:: section

            .. rubric:: Description

            Constructor with parameters. `a` is a displacement, `beta` is a scalefactor.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when :math:`beta \leq 0`

.. container:: section

    .. rubric:: Characteristics

    .. container:: section

        .. _`RealType a() const`:

        .. code-block:: cpp

            RealType exponential::a() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `a` - displacement.

    .. container:: section

        .. _`RealType beta() const`:

        .. code-block:: cpp

            RealType exponential::beta() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `beta` - scalefactor value.

**Parent topic:** :ref:`onemath_device_rng_distributions`
