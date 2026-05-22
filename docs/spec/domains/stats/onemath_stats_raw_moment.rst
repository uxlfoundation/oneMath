.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_stats_raw_moment:

raw_moment
==========

Entry point to compute raw moments up to the 4th order.

.. _onemath_stats_raw_moment_description:

.. rubric:: Description and Assumptions

The oneapi::math::stats::raw_moment function is used to compute array of raw moments up to the 4th order (raw moments for each dataset's dimension).

:ref:`onemath_stats_raw_moment` supports the following precisions for data:

    .. list-table::
        :header-rows: 1

        * - T
        * - ``float``
        * - ``double``


.. rubric:: Mathematical Notation

Vector of raw moments values:

:math:`R^k(X) = \left( R^k_1(X), \; \dots, R^k_p(X) \right), \; R^k_i(X) = \frac{1}W \sum_{j=1}^n \omega_j x_{ij}^k`

Where:

:math:`W = \sum_{j=1}^n \omega_j, \; j = 1, \dots, p; k = 1, 2, 3, 4`


.. _onemath_stats_raw_moment_buffer:

oneapi::math::stats::raw_moment (buffer version)
------------------------------------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::stats {
    template<method Method = fast, typename Type, layout ObservationsLayout>
        void raw_moment(sycl::queue& queue,
        const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
        sycl::buffer<Type, 1> mean,
        sycl::buffer<Type, 1> raw_moment_2 = {0},
        sycl::buffer<Type, 1> raw_moment_3 = {0},
        sycl::buffer<Type, 1> raw_moment_4 = {0});
    }

.. container:: section

    .. rubric:: Template Parameters

    Method
        Method which is used for estimate computation. The specific values are as follows:

        *  ``oneapi::math::stats::method::fast``
        *  ``oneapi::math::stats::method::one_pass``

    Type
        Data precision.

    ObservationsLayout
        Data layout. The specific values are described in :ref:`onemath_stats_dataset`.

.. container:: section

    .. rubric:: Input Parameters

    queue
        The queue where the routine should be executed.

    data
        Dataset which is used for computation.

.. container:: section

    .. rubric:: Output Parameters

    mean
        sycl::buffer array of mean values.

    raw_moment_2
        Optional parameter. sycl::buffer array of 2nd order raw moment values.

    raw_moment_3
        Optional parameter. sycl::buffer array of 3rd order raw moment values.

    raw_moment_4
        Optional parameter. sycl::buffer array of 4th order raw moment values.

.. container:: section

    .. rubric:: Throws

    oneapi::math::invalid_argument
        Exception is thrown when mean.get_count() == 0 & raw_moment_2.get_count() == 0 & raw_moment_3.get_count() == 0 & raw_moment_4.get_count() == 0, or dataset object is invalid

.. _onemath_stats_raw_moment_usm:

raw_moment (USM version)
------------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::stats {
    template<method Method = method::fast, typename Type, layout ObservationsLayout>
        sycl::event raw_moment(
        sycl::queue& queue,
        const dataset<ObservationsLayout, Type*>& data,
        Type* mean,
        Type* raw_moment_2 = nullptr,
        Type* raw_moment_3 = nullptr,
        Type* raw_moment_4 = nullptr,
        const std::vector<sycl::event> &dependencies = {});
    }

.. container:: section

    .. rubric:: Template Parameters

    Method
        Method which is used for estimate computation. The specific values are as follows:

        *  ``oneapi::math::stats::method::fast``
        *  ``oneapi::math::stats::method::one_pass``

    Type
        Data precision.

    ObservationsLayout
        Data layout. The specific values are described in :ref:`onemath_stats_dataset`.

.. container:: section

    .. rubric:: Input Parameters

    queue
        The queue where the routine should be executed.

    data
        Dataset which is used for computation.

    dependencies
        Optional parameter. List of events to wait for before starting computation, if any.

.. container:: section

    .. rubric:: Output Parameters

    mean
        Pointer to the array of mean values.

    raw_moment_2
        Optional parameter. Pointer to the array of the 2nd order raw moment values.

    raw_moment_3
        Optional parameter. Pointer to the array of the 3rd order raw moment values.

    raw_moment_4
        Optional parameter. Pointer to the array of the 2nd order raw moment values.

.. container:: section

    .. rubric:: Throws

    oneapi::math::invalid_argument
        Exception is thrown when mean == nullptr & raw_moment_2 == nullptr & raw_moment_3 == nullptr & raw_moment_4 == nullptr, or dataset object is invalid

.. container:: section

    .. rubric:: Return Value

    Output event to wait on to ensure computation is complete.


**Parent topic:** :ref:`onemath_stats_routines`

