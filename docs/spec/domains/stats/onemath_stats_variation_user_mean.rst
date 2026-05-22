.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_stats_variation_user_mean:

variation with provided mean
============================

Entry point to compute variation with the provided mean values.

.. _onemath_stats_variation_user_mean_description:

.. rubric:: Description and Assumptions

The oneapi::math::stats::variation function is used to compute an array of variation (variation for each dataset's dimension) with the provided mean values.

:ref:`onemath_stats_variation_user_mean` supports the following precisions for data:

    .. list-table::
        :header-rows: 1

        * - T
        * - ``float``
        * - ``double``


.. rubric:: Mathematical Notation

Vector of variation coefficients:

:math:`VC(X) = \left( VC_1(X), \; \dots, VC_p(X) \right), \; VC_i(X) = \frac{V^{0.5}_i(X)}{M_i(X)}`

Where:

:math:`i = 1, \dots, p`


.. _onemath_stats_variation_user_mean_buffer:

oneapi::math::stats::variation (buffer version)
-----------------------------------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::stats {
    template<method Method = method::fast, typename Type, layout ObservationsLayout>
        void variation(sycl::queue& queue, sycl::buffer<Type, 1> mean,
        const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
        sycl::buffer<Type, 1> variation);
    }

.. container:: section

    .. rubric:: Template Parameters

    Method
        Method which is used for estimate computation. The specific values are as follows:

        *  ``oneapi::math::stats::method::fast``

    Type
        Data precision.

    ObservationsLayout
        Data layout. The specific values are described in :ref:`onemath_stats_dataset`.


.. container:: section

    .. rubric:: Input Parameters

    queue
        The queue where the routine should be executed.

    mean
        sycl::buffer to the array of provided mean values.

    data
        Dataset which is used for computation.

.. container:: section

    .. rubric:: Output Parameters

    variation
        sycl::buffer array of variation values.

.. container:: section

    .. rubric:: Throws

    oneapi::math::invalid_argument
        Exception is thrown when variation.get_count() == 0, or mean.get_count() == 0, or dataset object is invalid

.. _onemath_stats_variation_user_mean_usm:

variation with provided mean (USM version)
------------------------------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::stats {
    template<method Method = method::fast, typename Type, layout ObservationsLayout>
        sycl::event variation(sycl::queue& queue,
        Type* mean,
        const dataset<ObservationsLayout, Type*>& data,
        Type* variation,
        const std::vector<sycl::event> &dependencies = {});
    }

.. container:: section

    .. rubric:: Template Parameters

    Method
        Method which is used for estimate computation. The specific values are as follows:

        *  ``oneapi::math::stats::method::fast``

    Type
        Data precision.

    ObservationsLayout
        Data layout. The specific values are described in :ref:`onemath_stats_dataset`.

.. container:: section

    .. rubric:: Input Parameters

    queue
        The queue where the routine should be executed.

    mean
        Pointer to the array of provided mean values.

    data
        Dataset which is used for computation.

    dependencies
        Optional parameter. List of events to wait for before starting computation, if any.

.. container:: section

    .. rubric:: Output Parameters

    variation
        Pointer to the array of the variation values.

.. container:: section

    .. rubric:: Throws

    oneapi::math::invalid_argument
        Exception is thrown when variation == nullptr, or mean == nullptr, or dataset object is invalid

.. container:: section

    .. rubric:: Return Value

    Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemath_stats_routines`

