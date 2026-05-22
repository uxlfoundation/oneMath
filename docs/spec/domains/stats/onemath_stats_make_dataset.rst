.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_stats_make_dataset:

make_dataset
============

Entry point to create a dataset from the provided parameters.

.. _onemath_stats_make_dataset_description:

.. rubric:: Description and Assumptions

The oneapi::math::stats::make_dataset function is used to create a dataset from the provided storage of the observations matrix, the number of dimensions and observations, and other parameters.

:ref:`onemath_stats_make_dataset` supports the following precisions for data:

    .. list-table::
        :header-rows: 1

        * - T
        * - ``float``
        * - ``double``


.. _onemath_stats_make_dataset_buffer:

make_dataset (buffer version)
-----------------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::stats {
    template<layout ObservationsLayout = layout::row_major, typename Type>
        dataset<sycl::buffer<Type, 1>, ObservationsLayout> make_dataset(
        std::int64_t n_dims,
        std::int64_t n_observations,
        sycl::buffer<Type, 1> observations,
        sycl::buffer<Type, 1> weights = {0},
        sycl::buffer<std::int64_t, 1> indices = {0});
    }

.. container:: section

    .. rubric:: Template Parameters

    ObservationsLayout
        Data layout. The specific values are described in :ref:`onemath_stats_dataset`.

    Type
        Data precision.

.. container:: section

    .. rubric:: Input Parameters

    n_dims
        The number of dimensions.

    n_observations
        The number of observations.

    observations
        Matrix of observations.

    weights
        Optional parameter. Array of weights of size n_observations. Elements of the array are non-negative members.  If the parameter is not specified, each observation has weight equal to 1.

    indices
        Optional parameter. Array of vector components that are processed. The size of the array is n_dims. If the parameter is not specified, all components are processed.

.. container:: section

    .. rubric:: Throws

    oneapi::math::invalid_argument
        Exception is thrown when :math: `n_dims \leq 0`, or :math: `n_observations \leq 0`, or observations.get_count() == 0

.. container:: section

    .. rubric:: Return Value

    Dataset holding specified parameters.


.. _onemath_stats_make_dataset_usm:

make_dataset (USM version)
--------------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::stats {
    template<layout ObservationsLayout = layout::row_major, typename Type>
        dataset<Type*, ObservationsLayout> make_dataset(std::nt64_t
        n_dims, std::int64_t n_observations,
        Type* observations, Type* weights = nullptr, std::int64_t* indices = nullptr);
    }

.. container:: section

    .. rubric:: Template Parameters

    ObservationsLayout
        Data layout. The specific values are described in :ref:`onemath_stats_dataset`.

    Type
        Data precision.

.. container:: section

    .. rubric:: Input Parameters

    n_dims
        The number of dimensions.

    n_observations
        The number of observations.

    observations
        Matrix of observations.

    weights
        Optional parameter. Array of weights of size n_observations. Elements of the array are non-negative members.  If the parameter is not specified, each observation has weight equal to 1.

    indices
        Optional parameter. Array of vector components that are processed. Size of array is n_dims. If the    parameter  is   not specified, all components are processed.

.. container:: section

    .. rubric:: Throws

    oneapi::math::invalid_argument
        Exception is thrown when :math:`n\_dims \leq 0`, or :math:`n\_observations \leq 0`, or observations == nullptr

.. container:: section

    .. rubric:: Return Value

    Dataset holding specified parameters.


**Parent topic:** :ref:`onemath_stats_service_routines`
