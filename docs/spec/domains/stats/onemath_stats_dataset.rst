.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_stats_dataset:

dataset
=======

The structure consolidates the information of a multi-dimensional dataset.

.. _onemath_stats_dataset_description:

.. rubric:: Description

The ``dataset`` struct object is used in :ref:`onemath_stats_routines` as a multi-dimensional data storage. ``dataset`` struct contains information about observations matrix and its size (dimensions x observations), observations weights and indices for dimensions (defines dimensions to be processed).

.. _onemath_stats_dataset_syntax_buffer:

structure dataset (Buffer version)
----------------------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::stats {
    template<layout ObservationsLayout, typename Type>
	struct dataset<ObservationsLayout, sycl::buffer<Type, 1>> {

    	explicit dataset(std::int64_t n_dims_, std::int64_t n_observations_,
                    sycl::buffer<Type, 1> observations_, sycl::buffer<Type, 1> weights_ = {0},
                    sycl::buffer<std::int64_t, 1> indices_ = {0});

    	std::int64_t n_dims;
    	std::int64_t n_observations;
    	sycl::buffer<Type, 1> observations;
    	sycl::buffer<Type, 1> weights = {0};
    	sycl::buffer<std::int64_t, 1> indices = {0};
        static constexpr layout layout = ObservationsLayout;
	};
    }


.. container:: section

    .. rubric:: Template parameters

    .. container:: section

        typename Type
            Type of the multi-dimensional data. Supported types:

                * ``float``
                * ``double``

    .. container:: section

        oneapi::math::stats::layout ObservationsLayout
            Type of the multi-dimensional data layout. Supported types:

                * ``oneapi::math::stats::layout::row_major``
                * ``oneapi::math::stats::layout::col_major``

.. container:: section

    .. rubric:: Struct Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `explicit dataset(std::int64_t n_dims_, std::int64_t n_observations_, sycl::buffer<Type, 1> observations_, sycl::buffer<Type, 1> weights_ = {0}, sycl::buffer<std::int64_t, 1> indices_ = {0})`_
          - Constructor


.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`explicit dataset(std::int64_t n_dims_, std::int64_t n_observations_, sycl::buffer<Type, 1> observations_, sycl::buffer<Type, 1> weights_ = {0}, sycl::buffer<std::int64_t, 1> indices_ = {0})`:

        .. code-block:: cpp

            explicit dataset::dataset(std::int64_t n_dims_, std::int64_t n_observations_,
                sycl::buffer<Type, 1> observations_,
                sycl::buffer<Type, 1> weights_ = {0},
                sycl::buffer<std::int64_t, 1> indices_ = {0});

        .. container:: section

            .. rubric:: Description

            Constructor with parameters.

            	* `n_dims_` is the number of dimensions
            	* `n_observations_` is the number of observations
            	* `observations_` is the matrix of observations
            	* `weights_` is an optional parameter, represents an array of weights for observations (of size `n_observations`). If the parameter is not specified, each observation is assigned a weight equal 1.
            	* `indices_` is an optional parameter, represents an array of dimensions that are processed (of size `n_dims`). If the parameter is not specified, all dimensions are processed.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when `n_dims_` :math:`\leq 0`, or `n_observations_` :math:`\leq 0`, or `observations_.get_count() == 0`

.. _onemath_stats_dataset_syntax_usm:

structure dataset (USM version)
-------------------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::stats {
    template<layout ObservationsLayout, typename Type>
	struct dataset<Type*, ObservationsLayout> {
    	explicit dataset(std::int64_t n_dims_, std::int64_t n_observations_, Type* observations_,
                    Type* weights_ = nullptr, std::int64_t* indices_ = nullptr);

    	std::int64_t n_dims;
    	std::int64_t n_observations;
    	Type* observations;
    	Type* weights = nullptr;
    	std::int64_t* indices = nullptr;
        static constexpr layout layout = ObservationsLayout;
	};
    }


.. container:: section

    .. rubric:: Template parameters

    .. container:: section

        typename Type
            Type of the multi-dimensional data. Supported types:

                * ``float``
                * ``double``

    .. container:: section

        oneapi::math::stats::layout ObservationsLayout
            Type of the multi-dimensional data layout. Supported types:

                * ``oneapi::math::stats::layout::row_major``
                * ``oneapi::math::stats::layout::col_major``

.. container:: section

    .. rubric:: Struct Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `explicit dataset(std::int64_t n_dims_, std::int64_t n_observations_, Type* observations_, Type* weights_ =  nullptr, std::int64_t* indices_ = nullptr)`_
          - Constructor


.. container:: section

    .. rubric:: Constructors

    .. container:: section

        .. _`explicit dataset(std::int64_t n_dims_, std::int64_t n_observations_, Type* observations_, Type* weights_ = nullptr, std::int64_t* indices_ = nullptr)`:

        .. code-block:: cpp

            explicit dataset::dataset(std::int64_t n_dims_, std::int64_t n_observations_,
                Type* observations_,
                Type* weights_ = nullptr,
                std::int64_t* indices_ = nullptr);

        .. container:: section

            .. rubric:: Description

            Constructor with parameters.

            	* `n_dims_` is the number of dimensions
            	* `n_observations_` is the number of observations
            	* `observations_` is the matrix of observations
            	* `weights_` is an optional parameter, represents an array of weights for observations (of size `n_observations`). If the parameter is not specified, each observation is assigned a weight equal 1.
            	* `indices_` is an optional parameter, represents an array of dimensions that are processed (of size `n_dims`). If the parameter is not specified, all dimensions are processed.

        .. container:: section

            .. rubric:: Throws

            oneapi::math::invalid_argument
                Exception is thrown when `n_dims_` :math:`\leq 0`, or `n_observations_` :math:`\leq 0`, or `observations_ == nullptr`

**Parent topic:** :ref:`onemath_stats`

