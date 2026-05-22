.. SPDX-FileCopyrightText: 2023 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_device_rng_uniform_discrete:

uniform (Discrete)
==================


Generates random numbers uniformly distributed over the interval ``[a, b)``.

.. rubric:: Description

The ``uniform`` class object is used in ``generate`` and function
to provide random numbers uniformly distributed over the interval ``[a, b)``,
where ``a, b`` are the left and right
bounds of the interval respectively, and :math:`a, b \in Z ; a < b`.


The probability distribution is given by:

.. math::

   P(X = k) = \frac{1}{b-a},
   k \in \{a, a + 1, \ldots, b-1\}

The cumulative distribution function is as follows:

.. math::

   F_{a, b}(x) =
   \begin{cases}
       0, & x < a \\
       \frac{x-a + 1}{b-a}, & a \leq x < b \\
       1, & x \geq b
   \end{cases},
   x \in R


class uniform
-------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::rng::device {
     template<typename Type, typename Method>
     class uniform<Type, Method> {
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

        typename Type
            Type of the produced values. Supported types:

                * ``std::int8_t``
                * ``std::uint8_t``
                * ``std::int16_t``
                * ``std::uint16_t``
                * ``std::int32_t``
                * ``std::uint32_t``
                * ``std::int64_t``
                * ``std::uint64_t``

    .. container:: section

        typename Method = oneapi::math::rng::uniform_method::by_default
            Transformation method, which will be used for generation. Supported types:

                * ``oneapi::math::rng::device::uniform_method::by_default``
                * ``oneapi::math::rng::device::uniform_method::standard``
                * ``oneapi::math::rng::device::uniform_method::accurate``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_device_rng_distributions_method>`.


.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `uniform()`_
          - Default constructor
        * - `explicit uniform(Type a, Type b)`_
          - Constructor with parameters
        * - `Type a() const`_
          - Method to obtain left bound `a`
        * - `Type b() const`_
          - Method to obtain right bound `b`

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`uniform()`:

        .. code-block:: cpp

            uniform::uniform()

        .. container:: section

            .. rubric:: Description

            Default constructor for distribution, parameters set as `a` = 0, `b` = (1 << 23) with ``uniform_method::standard``
            or std::numeric_limits<Type>::max() with ``uniform_method::accurate``.

    .. container:: section

        .. _`explicit uniform(Type a, Type b)`:

        .. code-block:: cpp

            explicit uniform::uniform(Type a, Type b)

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

        .. _`Type a() const`:

        .. code-block:: cpp

            uniform::a() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `a` - left bound.

    .. container:: section

        .. _`Type b() const`:

        .. code-block:: cpp

            uniform::b() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `b` - right bound.

**Parent topic:** :ref:`onemath_device_rng_distributions`
