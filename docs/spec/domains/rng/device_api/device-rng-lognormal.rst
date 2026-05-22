.. SPDX-FileCopyrightText: 2023 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_device_rng_lognormal:

lognormal
=========


Generates lognormally distributed random numbers.

.. rubric:: Description

The ``lognormal`` class object is used in the ``generate`` and function to provide 
random numbers with average of distribution (``m``, ``a``) and standard deviation (``s``, :math:`\sigma`) of
subject normal distribution, displacement (``displ``, ``b``), and scalefactor (``scale``, :math:`\beta`), where
:math:`a, \sigma, b, \beta \in \mathbb{R}; \sigma > 0, \beta > 0`.

The probability density function is given by:

.. math::

   f_{a, \sigma, b, \beta} (x) =
   \begin{cases}
      \frac{1}{\sigma (x - b) \sqrt {2\pi}}
      \exp \left(
         -\frac{\ln( \frac{x - b}{\beta}) - a)^2}{2\sigma^2}
      \right), & x > b \\
      0, & x \leq b
   \end{cases}


The cumulative distribution function is as follows:

.. math::

   F_{a, \sigma, b, \beta} (x) =
   \begin{cases}
      \Phi \left(\frac{\ln( \frac{x - b}{\beta}) - a}{\sigma}\right), & x > b \\
      0, & x \leq b
   \end{cases}


class lognormal
---------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::rng::device {
     template<typename RealType, typename Method>
     class lognormal {
     public:
        using method_type = Method;
        using result_type = RealType;
  
        lognormal();
        explicit lognormal(RealType m, RealType s, RealType displ = (RealType)0.0, RealType scale = (RealType)1.0);
  
        RealType m() const;
        RealType s() const;
        RealType displ() const;
        RealType scale() const;
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

        typename Method = oneapi::math::rng::lognormal_method::by_default
            Transformation method, which will be used for generation. Supported types:

                * ``oneapi::math::rng::device::lognormal_method::by_default``
                * ``oneapi::math::rng::device::lognormal_method::box_muller2``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_device_rng_distributions_method>`.


.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `lognormal()`_
          - Default constructor
        * - `explicit lognormal(RealType m, RealType s, RealType displ = (RealType)0.0, RealType scale = (RealType)1.0)`_
          - Constructor with parameters
        * - `RealType m() const`_
          - Method to obtain mean value
        * - `RealType s() const`_
          - Method to obtain standard deviation value
        * - `RealType displ() const`_
          - Method to obtain displacement value
        * - `RealType scale() const`_
          - Method to obtain scalefactor value


.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            lognormal::method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            lognormal::result_type = RealType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`lognormal()`:

        .. code-block:: cpp

            lognormal::lognormal()

        .. container:: section

            .. rubric:: Description

            Default constructor for distribution, parameters set as `m` = 0.0, `s` = 1.0, `displ` = 0.0, `scale` = 1.0.

    .. container:: section

        .. _`explicit lognormal(RealType m, RealType s, RealType displ = (RealType)0.0, RealType scale = (RealType)1.0)`:

        .. code-block:: cpp

            explicit lognormal::lognormal(RealType m, RealType s, RealType displ = (RealType)0.0, RealType scale = (RealType)1.0)

        .. container:: section

            .. rubric:: Description

            Constructor with parameters. `m` is a mean value, `s` is a standard deviation value, `displ` is a displacement value, `scale` is a scalefactor value.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when :math:`s \leq 0`, or :math:`scale \leq 0`

.. container:: section

    .. rubric:: Characteristics

    .. container:: section

        .. _`RealType m() const`:

        .. code-block:: cpp

            RealType lognormal::m() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `m` - mean value.

    .. container:: section

        .. _`RealType s() const`:

        .. code-block:: cpp

            RealType lognormal::s() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `s` - standard deviation value.

    .. container:: section

        .. _`RealType displ() const`:

        .. code-block:: cpp

            RealType lognormal::displ() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `displ` - displacement value.

    .. container:: section

        .. _`RealType scale() const`:

        .. code-block:: cpp

            RealType lognormal::scale() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `scale` - scalefactor value.

**Parent topic:** :ref:`onemath_device_rng_distributions`
