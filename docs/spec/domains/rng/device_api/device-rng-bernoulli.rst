.. SPDX-FileCopyrightText: 2023 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_device_rng_bernoulli:

bernoulli
=========

Generates Bernoulli distributed random values.

.. rubric:: Description

The ``bernoulli`` class object is used in the ``generate`` and function
to provide Bernoulli distributed random numbers with probability ``p`` of a single trial success,
where :math:`p \in R; 0 \leq p \leq 1`.

The probability distribution is given by:

.. math::

    P(X = 1) = p

.. math::

    P(X = 0) = 1 - p

The cumulative distribution function is as follows:

.. math::

   F_p(x) =
   \begin{cases}
      0, & x < 0 \\
      1 - p, & 0 \leq x < 1, x \in R \\
      1, & x \geq 1
   \end{cases}


class bernoulli
---------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::rng::device {
     template<typename IntType, typename Method>
     class bernoulli {
     public:
       using method_type = Method;
       using result_type = IntType;

       bernoulli();
       explicit bernoulli(float p);
       
       float p() const;
     };
   }


.. container:: section

    .. rubric:: Template parameters

    .. container:: section

        typename IntType
            Type of the produced values. Supported types:
                * ``std::int8_t``
                * ``std::uint8_t``
                * ``std::int16_t``
                * ``std::uint16_t``
                * ``std::int32_t``
                * ``std::uint32_t``

    .. container:: section

        typename Method = oneapi::math::rng::bernoulli_method::by_default
            Transformation method, which will be used for generation. Supported types:

                * ``oneapi::math::rng::bernoulli_method::by_default``
                * ``oneapi::math::rng::bernoulli_method::icdf``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_rng_distributions_template_parameter_onemath_rng_method_values>`.

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `bernoulli()`_
          - Default constructor
        * - `explicit bernoulli(float p)`_
          - Constructor with parameters
        * - `float p() const`_
          - Method to obtain probability `p`

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            bernoulli::method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            bernoulli::result_type = IntType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`bernoulli()`:

        .. code-block:: cpp

            bernoulli::bernoulli()

        .. container:: section

            .. rubric:: Description

            Default constructor for distribution, parameters set as `p` = 0.5f.

    .. container:: section

        .. _`explicit bernoulli(float p)`:

        .. code-block:: cpp

            explicit bernoulli::bernoulli(float p)

        .. container:: section

            .. rubric:: Description

            Constructor with parameters. `p` is a probability.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when `p > 1`, or `p < 0`

.. container:: section

    .. rubric:: Characteristics

    .. container:: section

        .. _`float p() const`:

        .. code-block:: cpp

            float bernoulli::p() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `p` - probability.

**Parent topic:** :ref:`onemath_device_rng_distributions`

