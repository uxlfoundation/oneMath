.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_gaussian_mv:

gaussian_mv
===========

Class is used for generation of multivariate normally distributed real types random numbers.

.. _onemath_rng_gaussian_mv_description:

.. rubric:: Description

The class object is used in the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` function to provide n random numbers :math:`d`-variate normally distributed, with mean :math:`a` and variance-covariance matrix :math:`C`, where :math:`a \in R^d;` :math:`C` is dxd symmetric positive matrix.

The probability density function is given by:

.. math::

    f_{a, C}(x) = \frac{1}{\sqrt{det(2\pi C)}}exp(-1 / 2(x - a)^T C^{-1}(x-a)).

.. _onemath_rng_gaussian_mv_syntax:

class gaussian_mv
-----------------

Let ``SequenceContainerOrView`` be a type that can be one of C++ Sequence containers or C++ Views (``span``, ``mdspan``).
It's implementation defined which type ``SequenceContainerOrView`` represents.

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    template<typename RealType = std::int32_t, layout Layout = layout::packed, typename Method = gaussian_mv_method::by_default>
    class gaussian_mv {
    public:
        using method_type = Method;
        using result_type = RealType;
        explicit gaussian_mv(std::uint32_t dimen, SequenceContainerOrView<RealType> mean, SequenceContainerOrView<RealType> matrix);
        std::int32_t dimen() const;
        SequenceContainerOrView<RealType> mean() const;
        SequenceContainerOrView<RealType> matrix() const;
    };
    }

.. container:: section

    .. rubric:: Template parameters

    .. container:: section

        typename RealType
            Type of the produced values. Supported types:
                * ``float``
                * ``double``

    .. rubric:: Template parameters

    .. container:: section

        oneapi::math::rng::layout Layout
            Matrix layout:
                * ``oneapi::math::rng::layout::full``
                * ``oneapi::math::rng::layout::packed``
                * ``oneapi::math::rng::layout::diagonal``

    .. container:: section

        typename Method = oneapi::math::rng::gaussian_mv_method::by_default
            Transformation method, which will be used for generation. Supported types:

                * ``oneapi::math::rng::gaussian_mv_method::by_default``
                * ``oneapi::math::rng::gaussian_mv_method::box_muller``
                * ``oneapi::math::rng::gaussian_mv_method::box_muller2``
                * ``oneapi::math::rng::gaussian_mv_method::icdf``

            See description of the methods in :ref:`Distributions methods template parameter<onemath_rng_distributions_template_parameter_onemath_rng_method_values>`.

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `explicit gaussian_mv(std::uint32_t dimen, SequenceContainerOrView<RealType> mean, SequenceContainerOrView<RealType> matrix)`_
          - Constructor with parameters
        * - `std::int32_t dimen() const`_
          - Method to obtain number of dimensions in output random vectors
        * - `SequenceContainerOrView<double> mean() const`_
          - Method to obtain mean vector `a` of dimension d.
        * - `SequenceContainerOrView<double> matrix() const`_
          - Method to obtain variance-covariance matrix `C`

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            gaussian_mv::method_type = Method

        .. container:: section

            .. rubric:: Description

            The type which defines transformation method for generation.

    .. container:: section

        .. code-block:: cpp

            gaussian_mv::result_type = RealType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`explicit gaussian_mv(std::uint32_t dimen, SequenceContainerOrView<RealType> mean, SequenceContainerOrView<RealType> matrix)`:

        .. code-block:: cpp

            explicit gaussian_mv::gaussian_mv(std::uint32_t dimen, SequenceContainerOrView<RealType> mean, SequenceContainerOrView<RealType> matrix)

        .. container:: section

            .. rubric:: Description

            Constructor with parameters. ``dimen`` is the number of dimensions, ``mean`` is a mean vector, ``matrix`` is a variance-covariance matrix.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when :math:`mean.size() \leq 0`, or :math:`matrix.size() \leq 0`

.. container:: section

    .. rubric:: Characteristics

    .. container:: section

        .. _`std::int32_t dimen() const`:

        .. code-block:: cpp

            std::int32_t gaussian_mv::dimen() const

        .. container:: section

            .. rubric:: Return Value

            Returns the distribution parameter `dimen`.

    .. container:: section

        .. _`SequenceContainerOrView<double> mean() const`:

        .. code-block:: cpp

            SequenceContainerOrView<double> gaussian_mv::mean() const

        .. container:: section

            .. rubric:: Return Value

            Returns the mean vector.

    .. container:: section

        .. _`SequenceContainerOrView<double> matrix() const`:

        .. code-block:: cpp

            SequenceContainerOrView<double> gaussian_mv::matrix() const

        .. container:: section

            .. rubric:: Return Value

            Returns the variance-covariance matrix.

**Parent topic:** :ref:`onemath_rng_distributions`
