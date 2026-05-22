.. SPDX-FileCopyrightText: 2023 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_device_rng_uniform_continuous:

uniform (Continuous)
====================


Generates random numbers with uniform distribution.

.. rubric:: Description

The class object is used in ``generate`` function to provide random numbers uniformly 
distributed over the interval [``a``, ``b``), where ``a``, ``b`` are the left and right bounds of the interval, 
respectively, and :math:`a, b \in R ; a < b`
``a``, ``b∈R`` ; ``a`` < ``b``.


The probability density function is given by:

.. math::

   f_{a, b}(x) =
     \begin{cases}
       \frac{1}{b-a}, & x \in [a, b)\\
       1, & x \notin [a, b) \end{cases}, -\infty < x < +\infty

The cumulative distribution function is as follows:


.. math::

   F_{a, b}(x) =
   \begin{cases}
      0, & x < a \\
      \frac{x-a}{b-a}, & a \leq x < b \\
      1, & x \geq b
   \end{cases},
   -\infty < x < +\infty


class uniform
-------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::rng::device {
     template<typename Type = float, typename Method = uniform_method::by_default>
     class uniform {
     public:
       using method_type = Method;
       using result_type = Type;
  
       uniform();
       explicit uniform(Type a, Type b);
  
       Type a() const;
       Type b() const;
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

                * ``oneapi::math::rng::device::uniform_method::by_default``
                * ``oneapi::math::rng::device::uniform_method::standard``
                * ``oneapi::math::rng::device::uniform_method::accurate``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_device_rng_distributions_method>`


.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `uniform()`_
          - Default constructor
        * - `explicit uniform(RealType a, RealType b)`_
          - Constructor with parameters
        * - `RealType a() const`_
          - Method to obtain left bound `a`
        * - `RealType b() const`_
          - Method to obtain right bound `b`

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            uniform::method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            uniform::result_type = RealType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`uniform()`:

        .. code-block:: cpp

            uniform::uniform()

        .. container:: section

            .. rubric:: Description

            Default constructor for distribution, parameters set as `a` = 0.0, `b` = 1.0.

    .. container:: section

        .. _`explicit uniform(RealType a, RealType b)`:

        .. code-block:: cpp

            explicit uniform::uniform(RealType a, RealType b)

        .. container:: section

            .. rubric:: Description

            Constructor with parameters. `a` is a left bound, `b` is a right bound, assume :math:`a < b`.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when :math:`a \ge b`

.. container:: section

    .. rubric:: Characteristics

    .. container:: section

        .. _`RealType a() const`:

        .. code-block:: cpp

            RealType uniform::a() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `a` - left bound.

    .. container:: section

        .. _`RealType b() const`:

        .. code-block:: cpp

            RealType uniform::b() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `b` - right bound.

**Parent topic:** :ref:`onemath_device_rng_distributions`
