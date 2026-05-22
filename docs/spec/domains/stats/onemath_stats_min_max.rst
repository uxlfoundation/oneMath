.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_stats_min_max:

min_max
=======

Entry point to compute min and max values.

.. _onemath_stats_min_max_description:

.. rubric:: Description and Assumptions

The oneapi::math::stats::min_max function is used to compute min and max arrays (min and max values for each dataset's dimension).

:ref:`onemath_stats_min_max` supports the following precisions for data:

    .. list-table::
        :header-rows: 1

        * - T
        * - ``float``
        * - ``double``


.. rubric:: Mathematical Notation

Vector of maximum values:

:math:`\max(X) = \left( {\max}_1(X), \; \dots, {\max}_p(X) \right)`

Vector of minimum values:

:math:`\min(X) = \left( {\min}_1(X), \; \dots, {\min}_p(X) \right)`

Where:

:math:`i = 1, \dots, p`


.. _onemath_stats_min_max_buffer:

min_max (buffer version)
------------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::stats {
    template<method Method = method::fast, typename Type, layout ObservationsLayout>
        void min_max(sycl::queue& queue,
        const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
        sycl::buffer<Type, 1> min,
        sycl::buffer<Type, 1> max);
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

    data
        Dataset which is used for computation.

.. container:: section

    .. rubric:: Output Parameters

    min
        sycl::buffer array of min values.

    max
        sycl::buffer array of max values.

.. container:: section

    .. rubric:: Throws

    oneapi::math::invalid_argument
        Exception is thrown when min.get_count() == 0, or max.get_count() == 0, or dataset object is invalid

.. _onemath_stats_min_max_usm:

min_max (USM version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::stats {
    template<method Method = method::fast, typename Type, layout ObservationsLayout>
        sycl::event min_max(sycl::queue& queue,
        const dataset<ObservationsLayout, Type*>& data,
        Type* min,
        Type* max,
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

    data
        Dataset which is used for computation.

    dependencies
        Optional parameter. List of events to wait for before starting computation, if any.

.. container:: section

    .. rubric:: Output Parameters

    min
        Pointer to the array of min values.

    max
        Pointer to the array of max values.

.. container:: section

    .. rubric:: Throws

    oneapi::math::invalid_argument
        Exception is thrown when min == nullptr, or max == nullptr, or dataset object is invalid

.. container:: section

    .. rubric:: Return Value

    Output event to wait on to ensure computation is complete.


**Parent topic:** :ref:`onemath_stats_routines`

