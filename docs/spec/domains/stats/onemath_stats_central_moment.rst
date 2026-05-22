.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_stats_central_moment:

central_moment
==============

Entry point to compute central moments up to the 4th order.

.. _onemath_stats_central_moment_description:

.. rubric:: Description and Assumptions

The oneapi::math::stats::central_moment function is used to compute an array of central moments up to the 4th order (central moments for each dataset's dimension).

:ref:`onemath_stats_central_moment` supports the following precisions for data:

    .. list-table::
        :header-rows: 1

        * - T
        * - ``float``
        * - ``double``


.. rubric:: Mathematical Notation

Vector of central moments:

:math:`C^k(X) = \left( C^k_1(X), \; \dots, C^k_p(X) \right), \; C_i^k(X) =  \frac{1}{W} \sum_{j=1}^{n} \omega_j \left( x_{ij} - M_i(X) \right)^k`

Where:

:math:`W = \sum_{j=1}^n \omega_j, \, i = 1, \dots, p, \; k = 2, 3, 4`


.. _onemath_stats_central_moment_buffer:

central_moment (buffer version)
-------------------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::stats {
    template<method Method = oneapi::math::stats::method::fast, typename Type,
        layout ObservationsLayout>
        void central_moment(sycl::queue& queue,
        const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
        sycl::buffer<Type, 1> central_moment_2,
        sycl::buffer<Type, 1> central_moment_3 = {0},
        sycl::buffer<Type, 1> central_moment_4 = {0});
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

    central_moment_2
        sycl::buffer array of 2nd order central moment values.

    central_moment_3
        Optional parameter. sycl::buffer array of 3rd order central moment values.

    central_moment_4
        Optional parameter. sycl::buffer array of 4th order central moment values.

.. container:: section

    .. rubric:: Throws

    oneapi::math::invalid_argument
        Exception is thrown when central_moment_2.get_count() == 0 & central_moment_3.get_count() == 0 & central_moment_4.get_count() == 0, or dataset object is invalid

.. _onemath_stats_central_moment_usm:

central_moment (USM version)
----------------------------

.. rubric:: Syntax


.. code-block:: cpp

    namespace oneapi::math::stats {
    template<method Method = method::fast, typename Type, layout ObservationsLayout>
        sycl::event central_moment(sycl::queue& queue,
        const dataset<ObservationsLayout, Type*>& data, Type* central_moment_2,
        Type* central_moment_3 = nullptr, Type* central_moment_4 = nullptr,
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

    central_moment_2
        Pointer to the array of the 2nd order central moment values.

    central_moment_3
        Optional parameter. Pointer to the array of the 3rd order central moment values.

    central_moment_4
        Optional parameter. Pointer to the array of the 2nd order central moment values.

.. container:: section

    .. rubric:: Throws

    oneapi::math::invalid_argument
        Exception is thrown when central_moment_2 == nullptr & central_moment_3 == nullptr & central_moment_4 == nullptr, or dataset object is invalid

.. container:: section

    .. rubric:: Return Value

    Output event to wait on to ensure computation is complete.



**Parent topic:** :ref:`onemath_stats_routines`

