.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_stats_raw_sum:

raw_sum
=======

Entry point to compute raw sums up to the 4th order.

.. _onemath_stats_raw_sum_description:

.. rubric:: Description and Assumptions

The oneapi::math::stats::raw_sum function is used to compute an array of raw sums up to the 4th order (raw sums for each dataset's dimension).

:ref:`onemath_stats_raw_sum` supports the following precisions for data:

    .. list-table::
        :header-rows: 1

        * - T
        * - ``float``
        * - ``double``


.. rubric:: Mathematical Notation

Vector of raw sums:

:math:`S^k(X) = \left( S^k_1(X), \; \dots, S^k_p(X) \right), \; S_i^k(X) = \sum_{j=1}^n \omega_j x^k_{ij}`

Where:

:math:`i = 1, \dots, p, \; k = 1, 2, 3, 4`


.. _onemath_stats_raw_sum_buffer:

raw_sum (Buffer version)
------------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::stats {
    template<method Method = method::fast, typename Type, layout ObservationsLayout>
        void raw_sum(sycl::queue& queue,
        const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
        sycl::buffer<Type, 1> sum,
        sycl::buffer<Type, 1> raw_sum_2 = {0},
        sycl::buffer<Type, 1> raw_sum_3 = {0},
        sycl::buffer<Type, 1> raw_sum_4 = {0});
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

    sum
        sycl::buffer array of sum values.

    raw_sum_2
        Optional parameter. sycl::buffer array of 2nd order raw sum values.

    raw_sum_3
        Optional parameter. sycl::buffer array of 3rd order raw sum values.

    raw_sum_4
        Optional parameter. sycl::buffer array of 4th order raw sum values.

.. container:: section

    .. rubric:: Throws

    oneapi::math::invalid_argument
        Exception is thrown when sum.get_count() == 0 & raw_sum_2.get_count() == 0 & raw_sum_3.get_count() == 0 & raw_sum_4.get_count() == 0, or dataset object is invalid

.. _onemath_stats_raw_sum_usm:

raw_sum (USM version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::stats {
    template<method Method = method::fast, typename Type, layout ObservationsLayout>
        sycl::event raw_sum(sycl::queue& queue,
        const dataset<ObservationsLayout, Type*>& data,
        Type* sum,
        Type* raw_sum_2 = nullptr,
        Type* raw_sum_3 = nullptr,
        Type* raw_sum_4 = nullptr,
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

    sum
        Pointer to the array of sum values.

    raw_sum_2
        Optional parameter. Pointer to the array of the 2nd order raw sum values.

    raw_sum_3
        Optional parameter. Pointer to the array of the 3rd order raw sum values.

    raw_sum_4
        Optional parameter. Pointer to the array of the 2nd order raw sum values.

.. container:: section

    .. rubric:: Throws

    oneapi::math::invalid_argument
        Exception is thrown when sum == nullptr & raw_sum_2 == nullptr & raw_sum_3 == nullptr & raw_sum_4 == nullptr, or dataset object is invalid

.. container:: section

    .. rubric:: Return Value

    Output event to wait on to ensure computation is complete.


**Parent topic:** :ref:`onemath_stats_routines`

