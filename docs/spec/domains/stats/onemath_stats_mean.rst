.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_stats_mean:

mean
====

Entry point to compute mean values.

.. _onemath_stats_mean_description:

.. rubric:: Description and Assumptions

The oneapi::math::stats::mean function is used to compute a mean array (mean value for each dataset's dimension).

:ref:`onemath_stats_mean` supports the following precisions for data:

    .. list-table::
        :header-rows: 1

        * - T
        * - ``float``
        * - ``double``


.. rubric:: Mathematical Notation

Vector of mean values:

:math:`M(X) = \left( M_1(X), \; \dots, M_p(X) \right), \; M_i(X) = \frac{1}{W} \sum_{j=1}^n \omega_j x_{ij}`

Where:

:math:`W = \sum_{j=1}^n \omega_j, \; i = 1, \dots, p`


.. _onemath_stats_mean_buffer:

mean (buffer version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::stats {
    template<method Method = method::fast, typename Type, layout ObservationsLayout>
        void mean(sycl::queue& queue,
        const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
        sycl::buffer<Type, 1> mean);
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

.. container:: section

    .. rubric:: Throws

    oneapi::math::invalid_argument
        Exception is thrown when mean.get_count() == 0, or dataset object is invalid

.. _onemath_stats_mean_usm:

mean (USM version)
------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::stats {
    template<method Method = method::fast, typename Type, layout ObservationsLayout>
        sycl::event mean(sycl::queue& queue,
        const dataset<ObservationsLayout, Type*>& data,
        Type* mean,
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

.. container:: section

    .. rubric:: Throws

    oneapi::math::invalid_argument
        Exception is thrown when mean == nullptr, or dataset object is invalid

.. container:: section

    .. rubric:: Return Value

    Output event to wait on to ensure computation is complete.


**Parent topic:** :ref:`onemath_stats_routines`

