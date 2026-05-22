.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_dft_enums:

DFT-related scoped enumeration types
------------------------------------

The following scoped enumeration types, defined in the ``oneapi::math::dft``
namespace, are used for constructing and configuring ``descriptor`` objects
consistently with the DFT(s) they are meant to define. The usage of prepended
namespace specifiers ``oneapi::math::dft`` is omitted below for conciseness.

.. list-table::
      :header-rows: 1
      :widths: 24 73

      * -   Scoped enumeration type
        -   Description
      * -   ``precision``
        -   Type for the floating-point format to be used in the desired DFT
            calculations (for arithmetic operations and data representation). A
            template parameter ``prec`` of this type is used for the
            ``descriptor`` :ref:`class template<onemath_dft_descriptor>`.
      * -   ``domain``
        -   Type for the kind of forward domain for the desired DFT calculations.
            A template parameter ``dom`` of this type is used for the
            ``descriptor`` :ref:`class template<onemath_dft_descriptor>`.
      * -   ``config_param``
        -   Type for configuration parameters of ``descriptor`` objects.
      * -   ``config_value``
        -   Type for non-numeric configuration values corresponding to some of
            the :ref:`configuration parameters<onemath_dft_enum_config_param>`.

.. _onemath_dft_enum_precision:

``precision``
+++++++++++++

Values in this scoped enumeration type represent the floating-point format to be
used for the desired DFT(s). The same format is to be used for the user-provided
data, the computation being carried out by oneMath and the delivered results.

.. container:: section
 
   .. rubric:: Syntax

   .. code:: cpp

      namespace oneapi::math::dft {
        enum class precision {
            SINGLE,
            DOUBLE
        };
      }

   .. list-table::
      :header-rows: 1
      :widths: 24 73

      * -   ``precision`` enumerator
        -   Description
      * -   ``SINGLE``
        -   Single-precision floating-point format (FP32) is used for data
            representation and arithmetic operations.
      * -   ``DOUBLE``
        -   Double-precision floating-point format (FP64) is used for data
            representation and arithmetic operations.

.. _onemath_dft_enum_domain:

``domain``
++++++++++

Values in this scoped enumeration type represent the kind of forward domain for
the desired DFT(s). As explained in the :ref:`introduction<onemath_dft_definitions>`,
the backward domain type is always complex.

.. container:: section

   .. rubric:: Syntax

   .. code:: cpp

      namespace oneapi::math::dft {
        enum class domain {
            REAL,
            COMPLEX
        };
      }
   
   .. list-table::
      :header-rows: 1
      :widths: 24 73

      * -   ``domain`` enumerator
        -   Description
      * -   ``REAL``
        -   The forward domain is the set of real :math:`d`-dimensional sequences.
      * -   ``COMPLEX``
        -   The forward domain is the set of complex :math:`d`-dimensional sequences.


.. _onemath_dft_enum_config_param:

``config_param``
++++++++++++++++

Values in this scoped enumeration type represent configuration parameters for
``descriptor`` objects.

.. container:: section

   .. code:: cpp

      namespace oneapi::math::dft {
        enum class config_param {
            // read-only parameters:
            FORWARD_DOMAIN,
            DIMENSION,
            LENGTHS,
            PRECISION,
            COMMIT_STATUS,
            WORKSPACE_EXTERNAL_BYTES,
            // writable parameters:
            FORWARD_SCALE,
            BACKWARD_SCALE,

            NUMBER_OF_TRANSFORMS,

            COMPLEX_STORAGE,

            PLACEMENT,

            FWD_STRIDES,
            BWD_STRIDES,
            INPUT_STRIDES, // deprecated
            OUTPUT_STRIDES, // deprecated

            FWD_DISTANCE,
            BWD_DISTANCE,

            WORKSPACE_PLACEMENT
        };
      }

   Configuration parameters represented by ``config_param::FORWARD_DOMAIN`` and
   ``config_param::PRECISION`` are associated with configuration values of type
   ``domain`` and ``precision`` respectively. Other configuration parameters are
   associated with configuration values of type ``config_value`` or of a native
   type like ``std::int64_t``, ``std::vector<std::int64_t>``, ``float`` or
   ``double``. This is further specified in the following table.

   .. list-table::
      :header-rows: 1
      :widths: 10 50 40

      * -   ``config_param`` enumerator
        -   Represented configuration parameter
        -   | Type of associated configuration value
            | [default value]
      * -   ``FORWARD_DOMAIN``
        -   Kind of forward domain for the DFT. This parameter is read-only.
        -   | ``domain``
            | [specialization value used for template parameter ``dom`` of the
               ``descriptor`` :ref:`class template<onemath_dft_descriptor>`]
      * -   ``DIMENSION``
        -   Dimension :math:`d` of the DFT. This parameter is read-only.
        -   | ``std::int64_t``
            | [:math:`d`]
      * -   ``LENGTHS``
        -   Lengths :math:`\lbrace n_1, \ldots, n_d\rbrace` of the DFT. This
            parameter is read-only and can only be set via the
            :ref:`parameterized constructors<onemath_dft_descriptor_constructors>`.
        -   | ``std::vector<std::int64_t>`` of size :math:`d` or, if :math:`d = 1`, ``std::int64_t``
            | [``std::vector<int64_t>({n_1,...,n_d})``]
      * -   ``PRECISION``
        -   Floating-point format to use for the DFT data and computation. This
            parameter is read-only.
        -   | ``precision``
            | [specialization value used for template parameter ``prec`` of the
               ``descriptor`` :ref:`class template<onemath_dft_descriptor>`]
      * -   ``COMMIT_STATUS``
        -   Status flag indicating whether the object is ready for computations
            after a successful call to its ``commit``
            :ref:`member function<onemath_dft_descriptor_commit>`. This parameter
            is read-only.
        -   | ``config_value`` (possible values are ``config_value::COMMITTED`` or ``config_value::UNCOMMITTED``).
            | [``config_value::UNCOMMITTED``]
      * -   ``FORWARD_SCALE``
        -   :math:`\sigma_{\delta}` for the forward DFT (:math:`\delta = -1`).
        -   | ``float`` (resp. ``double``) for single-precision (resp. double-precision) descriptors
            | [:math:`1.0`]
      * -   ``BACKWARD_SCALE``
        -   :math:`\sigma_{\delta}` for the backward DFT (:math:`\delta = +1`).
        -   | ``float`` (resp. ``double``) for single-precision (resp. double-precision) descriptors
            | [:math:`1.0`]
      * -   ``NUMBER_OF_TRANSFORMS``
        -   Batch size :math:`M`. This is relevant (and *must* be set) for
            batched DFT(s), *i.e.*, if :math:`M > 1`.
        -   | ``std::int64_t``
            | [:math:`1`]
      * -   ``COMPLEX_STORAGE``
        -   :ref:`Data storage<onemath_dft_complex_storage>` type used (relevant
            for complex descriptors only).
        -   | ``config_value`` (possible values are ``config_value::COMPLEX_COMPLEX`` or ``config_value::REAL_REAL``)
            | [``config_value::COMPLEX_COMPLEX``]
      * -   ``PLACEMENT``
        -   | Parameter specifying whether the DFT calculations should be done in-place (results overwriting the input data) or out-of-place (input and output in separate data containers having no common elements).
            | Note: even for out-of-place configurations, some implementations may not preserve the original input data.
        -   | ``config_value`` (possible values are ``config_value::INPLACE`` or ``config_value::NOT_INPLACE``)
            | [``config_value::INPLACE``]
      * -   ``FWD_STRIDES``
        -   Offset and strides :ref:`defining the layout<onemath_dft_fwd_bwd_strides>`
            within a given data sequence in the forward domain.
        -   | ``std::vector<std::int64_t>`` of size :math:`(d+1)`
            | [see the :ref:`dedicated section<onemath_dft_fwd_bwd_strides>`]
      * -   ``BWD_STRIDES``
        -   Offset and strides :ref:`defining the layout<onemath_dft_fwd_bwd_strides>`
            within a given data sequence in the backward domain.
        -   | ``std::vector<std::int64_t>`` of size :math:`(d+1)`
            | [see the :ref:`dedicated section<onemath_dft_fwd_bwd_strides>`]
      * -   ``INPUT_STRIDES`` (deprecated)
        -   Offset and strides :ref:`defining the layout<onemath_dft_io_strides_deprecated>`
            within a given *input* data sequence.
        -   | ``std::vector<std::int64_t>`` of size :math:`(d+1)`
            | [``std::vector<std::int64_t>(d+1, 0)``]
      * -   ``OUTPUT_STRIDES`` (deprecated)
        -   Offset and strides :ref:`defining the layout<onemath_dft_io_strides_deprecated>`
            within a given *output* data sequence.
        -   | ``std::vector<std::int64_t>`` of size :math:`(d+1)`
            | [``std::vector<std::int64_t>(d+1, 0)``]
      * -   ``FWD_DISTANCE``
        -   :ref:`Distance<onemath_dft_num_dft_data_layouts_batched_dfts>` in
            number of elements of
            :ref:`implicitly-assumed data type<onemath_dft_config_data_implicitly_assumed_elementary_data_type>`
            between forward-domain entries
            :math:`\left(\cdot\right)^{m}_{k_1, k_2, \ldots, k_d}` and
            :math:`\left(\cdot\right)^{m + 1}_{k_1, k_2, \ldots, k_d}` for all
            :math:`0\leq m < M - 1` and
            :math:`\left(k_1, k_2, \ldots, k_d\right)` in
            :ref:`valid range<onemath_dft_elementary_range_of_indices>`. This is
            relevant (and *must* be set) for batched DFT(s), *i.e.*, if
            :math:`M > 1`.
        -   | ``std::int64_t``
            | [:math:`0`]
      * -   ``BWD_DISTANCE``
        -   :ref:`Distance<onemath_dft_num_dft_data_layouts_batched_dfts>` in
            number of elements of
            :ref:`implicitly-assumed data type<onemath_dft_config_data_implicitly_assumed_elementary_data_type>`
            between backward-domain entries
            :math:`\left(\cdot\right)^{m}_{k_1, k_2, \ldots, k_d}` and
            :math:`\left(\cdot\right)^{m + 1}_{k_1, k_2, \ldots, k_d}` for all
            :math:`0\leq m < M - 1` and
            :math:`\left(k_1, k_2, \ldots, k_d\right)` in
            :ref:`valid range<onemath_dft_elementary_range_of_indices>`. This is
            relevant (and *must* be set) for batched DFT(s), *i.e.*, if
            :math:`M > 1`.
        -   | ``std::int64_t``
            | [:math:`0`]
      * -   ``WORKSPACE_PLACEMENT``
        -   Some FFT algorithm computation steps require a
            :ref:`scratch space<onemath_dft_config_workspace_placement>` for
            permutations or other purposes. This parameter controls whether this
            scratch space is automatically allocated or provided by the user.
        -   | ``config_value`` (possible values are ``config_value::WORKSPACE_AUTOMATIC`` or ``config_value::WORKSPACE_EXTERNAL``).
            | [``config_value::WORKSPACE_AUTOMATIC``]
      * -   ``WORKSPACE_EXTERNAL_BYTES``
        -   The required minimum external workspace size for use by the
            ``set_workspace`` :ref:`member function<onemath_dft_descriptor_set_workspace>`.
            This parameter is read-only and can be queried only if the
            ``descriptor`` object is committed.
        -   | ``std::int64_t``
            | [N/A]

.. _onemath_dft_enum_config_value:

``config_value``
++++++++++++++++

Values in this scoped enumeration type represent possible non-numeric
configuration values associated with some
:ref:`configuration parameters<onemath_dft_enum_config_param>`.

.. container:: section

   .. code:: cpp

      namespace oneapi::math::dft {
        enum class config_value {
            // for config_param::COMMIT_STATUS
            COMMITTED,
            UNCOMMITTED,

            // for config_param::COMPLEX_STORAGE,
            COMPLEX_COMPLEX,
            REAL_REAL,

            // for config_param::PLACEMENT
            INPLACE,
            NOT_INPLACE

            // For config_param::WORKSPACE_PLACEMENT
            WORKSPACE_AUTOMATIC,
            WORKSPACE_EXTERNAL,
        };
      }


   .. list-table::
      :header-rows: 1
      :widths: 30 67

      * -   ``config_value`` enumerators
        -   Description
      * -   | ``COMMITTED``
            | ``UNCOMMITTED``
        -   Possible configuration values associated with the configuration
            parameter ``config_param::COMMIT_STATUS`` indicating whether a
            ``descriptor`` object was successfully commited
            (``config_value::COMMITTED``) or not (``config_value::UNCOMMITTED``).
      * -   | ``COMPLEX_COMPLEX``
            | ``REAL_REAL``
        -   Possible configuration values associated with the configuration
            parameter ``config_param::COMPLEX_STORAGE`` indicating whether real
            and imaginary parts of complex entries are stored separately
            (``config_value::REAL_REAL``) or not (``config_value::COMPLEX_COMPLEX``).
            More details are available in the page dedicated to
            :ref:`data storage<onemath_dft_data_storage>`.
      * -   | ``INPLACE``
            | ``NOT_INPLACE``
        -   Possible configuration values associated with the configuration
            parameter ``config_param::PLACEMENT`` indicating whether the DFT
            computation must be performed in-place (``config_value::INPLACE``)
            or not (``config_value::NOT_INPLACE``).
      * -   | ``WORKSPACE_AUTOMATIC``
            | ``WORKSPACE_EXTERNAL``
        -   Possible configuration values associated with the configuration
            parameter ``config_param::WORKSPACE_PLACEMENT``. More details are
            available in the page dedicated to
            :ref:`workspace placement<onemath_dft_config_workspace_placement>`.

**Parent topic:** :ref:`onemath_dft`

.. toctree::
   :hidden:

   config_params/data_layouts
   config_params/storage_formats
   config_params/workspace_placement
