.. SPDX-FileCopyrightText: 2023 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_dft_config_data_layouts:

Configuration of data layouts
-----------------------------

The usage of prepended namespace specifiers ``oneapi::math::dft`` is
omitted below for conciseness.

The DFT interface provides the configuration parameters
``config_param::FWD_STRIDES`` (resp. ``config_param::BWD_STRIDES``)
to define the data layout locating entries of relevant data sequences in the
forward (resp. backward) domain. In case of batched transforms, *i.e.*, if
:math:`M > 1` is configured by setting ``config_param::NUMBER_OF_TRANSFORMS``
accordingly, ``config_param::FWD_DISTANCE`` (resp.
``config_param::BWD_DISTANCE``) completes the description of the data layout by
specifying the distances between successive data sequences in the forward (resp.
backward) domain.

Using the notations from the :ref:`introduction<onemath_dft_definitions>` and the
superscript :math:`\text{fwd}` (resp. :math:`\text{bwd}`) for data sequences
belonging to forward (resp. backward) domain, for any :math:`m` and multi-index
:math:`\left(k_1, k_2, \ldots, k_d\right)` within :ref:`valid
range<onemath_dft_elementary_range_of_indices>`, the corresponding entry
:math:`\left(\cdot\right)^{m}_{k_{1}, k_{2}, \dots, k_d }` – or the real or
imaginary part thereof – of the relevant data sequence is located at index

.. math::
    s^{\text{xwd}}_0 + k_1\ s^{\text{xwd}}_1 + k_2\ s^{\text{xwd}}_2 + \dots + k_d\ s^{\text{xwd}}_d + m\ l^{\text{xwd}}
    :label: eq_idx_data_layout

of the corresponding data container (``sycl::buffer`` object or
device-accessible USM allocation) provided to the compute function, the base
data type of which is (possibly implicitly re-interpreted) as documented in the
:ref:`table<onemath_dft_config_data_implicitly_assumed_elementary_data_type>`
below. In the index expression :eq:`eq_idx_data_layout`,
:math:`\text{x} = \text{f}` (resp. :math:`\text{x} = \text{b}`) for entries of
forward-domain (resp. backward-domain) data sequences and

- :math:`s^{\text{xwd}}_j`, :math:`\forall j \in \lbrace 0, \ldots, d\rbrace`
  represents the :ref:`offset and generalized
  strides<onemath_dft_fwd_bwd_strides>` defining the locations of entries within
  each :math:`d`-dimensional data sequence in the forward (resp. backward)
  domain if :math:`\text{x} = \text{f}` (resp. if :math:`\text{x} = \text{b}`),
  counted in number of elements of the relevant :ref:`implicitly-assumed
  elementary data type<onemath_dft_config_data_implicitly_assumed_elementary_data_type>`;
- :math:`l^{\text{xwd}}` represents the
  :ref:`distance<onemath_dft_num_dft_data_layouts_batched_dfts>` between successive
  :math:`d`-dimensional data sequences in the forward (resp. backward) domain if
  :math:`\text{x} = \text{f}` (resp. if :math:`\text{x} = \text{b}`), counted
  in number of elements of the relevant :ref:`implicitly-assumed elementary data
  type<onemath_dft_config_data_implicitly_assumed_elementary_data_type>`.

.. note::
    All data sequences (or respective real and imaginary parts thereof if
    separately stored) must belong to the same block allocation, as a
    consequence of the generalized index :eq:`eq_idx_data_layout`.

.. _onemath_dft_config_data_implicitly_assumed_elementary_data_type:

.. rubric:: Implicitly-assumed elementary data type 

When reading or writing an element at index :eq:`eq_idx_data_layout` of any
user-provided data container used at compute time, a ``descriptor`` object may
re-interpret the base data type of that data container into an
implicitly-assumed elementary data type.
That implicitly-assumed data type depends on the object type, *i.e.*, on the
specialization values used for the template parameters when instantiating the
``descriptor`` :ref:`class template<onemath_dft_descriptor>`, and, in case of
complex descriptors, on the configuration value set for its configuration parameter
``config_param::COMPLEX_STORAGE``. The table below lists the implicitly-assumed
data type in either domain (last 2 columns) based on the object type and
its configuration value for ``config_param::COMPLEX_STORAGE`` (first 2 columns).

.. list-table::
    :header-rows: 1
    :class: longtable

    * -   Object type
      -   Configuration value for configuration parameter ``config_param::COMPLEX_STORAGE``
      -   Implicitly-assumed elementary data type in forward domain
      -   Implicitly-assumed elementary data type in backward domain
    * -   ``descriptor<precision::SINGLE, domain::COMPLEX>``
      -   ``config_value::COMPLEX_COMPLEX``
      -   ``std::complex<float>``
      -   ``std::complex<float>``
    * -   ``descriptor<precision::DOUBLE, domain::COMPLEX>``
      -   ``config_value::COMPLEX_COMPLEX``
      -   ``std::complex<double>``
      -   ``std::complex<double>``
    * -   ``descriptor<precision::SINGLE, domain::COMPLEX>``
      -   ``config_value::REAL_REAL``
      -   ``float``
      -   ``float``
    * -   ``descriptor<precision::DOUBLE, domain::COMPLEX>``
      -   ``config_value::REAL_REAL``
      -   ``double``
      -   ``double``
    * -   ``descriptor<precision::SINGLE, domain::REAL>``
      -   irrelevant
      -   ``float``
      -   ``std::complex<float>``
    * -   ``descriptor<precision::DOUBLE, domain::REAL>``
      -   irrelevant
      -   ``double``
      -   ``std::complex<double>``

.. _onemath_dft_num_dft_data_layouts_batched_dfts:

.. rubric:: Configuring data layouts for batched transforms

The value :math:`l^{\text{xwd}}` in :eq:`eq_idx_data_layout` above is
communicated as an ``std::int64_t`` configuration value, set for the
configuration parameter ``config_param::FWD_DISTANCE`` if :math:`\text{x} =
\text{f}` (resp. ``config_param::BWD_DISTANCE`` if :math:`\text{x} = \text{b}`).
This value is irrelevant for unbatched transforms, *i.e.*, for descriptors set
to handle a number of transforms :math:`M` equal to :math:`1` (default behavior).

In case of batched transforms, the number :math:`M > 1` of desired DFTs *must*
be set explicitly as an ``std::int64_t`` configuration value for the
configuration parameter ``config_param::NUMBER_OF_TRANSFORMS``. In that case,
the configuration parameters ``config_param::FWD_DISTANCE`` and
``config_param::BWD_DISTANCE`` *must also* be set explicitly since their default
configuration values of :math:`0` would break the :ref:`consistency
requirements<onemath_dft_data_layout_requirements>` for any :math:`M > 1`.

.. _onemath_dft_fwd_bwd_strides:

.. rubric:: Configuring strides in forward and backward domains

The values :math:`s^{\text{xwd}}_0, s^{\text{xwd}}_1, \dots, s^{\text{xwd}}_d`
in :eq:`eq_idx_data_layout` above are communicated as elements, in that order,
of a :math:`(d+1)`-long ``std::vector<std::int64_t>`` configuration value, set
for the configuration parameter ``config_param::FWD_STRIDES`` if
:math:`\text{x} = \text{f}` (resp. ``config_param::BWD_STRIDES`` if
:math:`\text{x} = \text{b}`). The element :math:`s^{\text{xwd}}_0` represents an
absolute offset (or "displacement") in the data sets while the subsequent
elements :math:`s^{\text{xwd}}_j\ (j > 0)` are generalized strides to be
considered along dimensions :math:`j \in \lbrace 1, \ldots, d\rbrace`.

The default values set for the forward and backward strides correspond to the
data layout configurations for unbatched, in-place transforms using unit stride
along the last dimension with no offset (and minimal padding in forward
domain in case of real descriptors, aligning with the :ref:`requirements for
in-place transforms<onemath_dft_data_layout_requirements>`). In other words, the
default values are :math:`s^{\text{fwd}}_0 = s^{\text{bwd}}_0 = 0`,
:math:`s^{\text{fwd}}_d = s^{\text{bwd}}_d = 1` and, for :math:`d`-dimensional
DFTs with :math:`d > 1`,

- :math:`s^{\text{fwd}}_{d-1} = s^{\text{bwd}}_{d-1} = n_{d}` for complex
  descriptors;
- :math:`s^{\text{bwd}}_{d-1} = \lfloor \frac{n_{d}}{2} \rfloor + 1`, and
  :math:`s^{\text{fwd}}_{d-1} = 2 s^{\text{bwd}}_{d-1}` for real descriptors;
- if :math:`d > 2`, :math:`s^{\text{xwd}}_k = n_{k+1} s^{\text{xwd}}_{k+1}`
  for :math:`k \in \lbrace 1, \ldots, d - 2\rbrace` (for
  :math:`\text{x} = \text{f}` and :math:`\text{x} = \text{b}`).

.. _onemath_dft_data_layout_requirements:

.. rubric:: General consistency requirements

In general, the distances and strides must be set so that every index value
:eq:`eq_idx_data_layout` corresponds to a *unique* entry of the data sequences
under consideration. In other words, there must not be one index value as
expressed in :eq:`eq_idx_data_layout` that corresponds to two different
:math:`(d+1)`-tuples :math:`(m, k_{1}, k_{2}, \dots, k_d)` that are both within
the :ref:`elementary range of indices considered by
oneMath<onemath_dft_elementary_range_of_indices>`.

Additionally, for in-place transforms (configuration value
``config_value::INPLACE`` associated with configuration parameter
``config_param::PLACEMENT``), the smallest stride value must be associated with
the same dimension in forward and backward domains and the data layouts must
abide by following "*consistency requirement*": the memory address(es) of
leading entry(ies) along the last dimension must be identical in forward and
backward domains. Specifically, considering any :math:`(d+1)`-tuple
:math:`(m, k_{1}, k_{2}, \dots, k_{d-1}, 0)` within :ref:`valid
range<onemath_dft_elementary_range_of_indices>`, the memory address of the
element of corresponding index value :eq:`eq_idx_data_layout` in forward domain
(considering the :ref:`implicitly assumed
type<onemath_dft_config_data_implicitly_assumed_elementary_data_type>` in forward
domain) must be identical to the memory address of the element of corresponding
index value :eq:`eq_idx_data_layout` in backward domain (considering the
:ref:`implicitly assumed
type<onemath_dft_config_data_implicitly_assumed_elementary_data_type>` in
backward domain). Equivalently,

- for complex descriptors, the offset, stride(s) (and distances, if relevant)
  must be equal in forward and backward domain;
- for real descriptors, offsets and strides must satisfy
  :math:`s^{\text{fwd}}_{j} = 2 s^{\text{bwd}}_{j}\ \forall j \in \lbrace 0,
  \ldots, d - 1\rbrace` (note that :math:`0 \leq j < d`) and distances, if
  relevant, must satisfy :math:`l^{\text{fwd}} = 2 l^{\text{bwd}}`. Note that
  this leads to some data padding being required in forward domain if unit
  strides are used along the last dimension in forward and backward domains.

.. _onemath_dft_io_strides_deprecated:

.. rubric:: Configuring strides for input and output data [deprecated, **not** recommended]

Instead of specifying strides by domain, one may choose to specify the strides
for input and output data sequences. Let
:math:`s^{\text{x}}_{j}, \ j \in \lbrace 0, 1, \ldots, d\rbrace` be the stride
values for input (resp. output) data sequences if :math:`\text{x} = \text{i}`
(resp. :math:`\text{x} = \text{o}`). Such
:math:`s^{\text{x}}_0, s^{\text{x}}_1, \dots, s^{\text{x}}_d` values may be
communicated as elements, in that order, of a :math:`(d+1)`-long
``std::vector<std::int64_t>`` configuration value, set for the (deprecated)
configuration parameter ``config_param::INPUT_STRIDES`` if
:math:`\text{x} = \text{i}` (resp. ``config_param::OUTPUT_STRIDES`` if
:math:`\text{x} = \text{o}`).

The values of :math:`s^{\text{i}}_{j}` and :math:`s^{\text{o}}_{j}` are to be
used and considered by oneMath if and only if
:math:`s^{\text{fwd}}_{j} = s^{\text{bwd}}_{j} = 0, \forall j \in \lbrace 0, 1, \ldots, d\rbrace`.
This will happen automatically if ``config_param::INPUT_STRIDES`` and
``config_param::OUTPUT_STRIDES`` are set and ``config_param::FWD_STRIDES`` and
``config_param::BWD_STRIDES`` are not (see note below).
In such a case, ``descriptor`` objects must consider the data layouts
corresponding to the two compute directions separately. As detailed above,
relevant data sequence entries are accessed as elements of data containers
(``sycl::buffer`` objects or device-accessible USM allocations) provided to the
compute function, the base data type of which is (possibly implicitly re-interpreted)
as documented in the above
:ref:`table<onemath_dft_config_data_implicitly_assumed_elementary_data_type>`. If
using input and output strides, for any :math:`m` and multi-index
:math:`\left(k_1, k_2, \ldots, k_d\right)` within :ref:`valid
range<onemath_dft_elementary_range_of_indices>`, the index to be used when
accessing a data sequence entry – or part thereof – in forward domain is

.. math::
    s^{\text{x}}_0 + k_1\ s^{\text{x}}_1 + k_2\ s^{\text{x}}_2 + \dots + k_d\ s^{\text{x}}_d + m\ l^{\text{fwd}}

where :math:`\text{x} = \text{i}` (resp. :math:`\text{x} = \text{o}`) for
forward (resp. backward) DFT(s). Similarly, the index to be used when accessing
a data sequence entry – or part thereof – in backward domain is

.. math::
    s^{\text{x}}_0 + k_1\ s^{\text{x}}_1 + k_2\ s^{\text{x}}_2 + \dots + k_d\ s^{\text{x}}_d + m\ l^{\text{bwd}}

where :math:`\text{x} = \text{o}` (resp. :math:`\text{x} = \text{i}`) for
forward (resp. backward) DFT(s).

As a consequence, configuring ``descriptor`` objects using these deprecated
configuration parameters makes their configuration direction-dependent when
different stride values are used in forward and backward domains. Since the
intended compute direction is unknown to the object when
:ref:`committing<onemath_dft_descriptor_commit>` it, every direction that results
in a :ref:`consistent data layout<onemath_dft_data_layout_requirements>` in
forward and backward domains must be supported by successfully-committed
``descriptor`` objects.

.. note::
    For ``descriptor`` objects with strides configured via these deprecated
    configuration parameters, the
    :ref:`consistency requirements<onemath_dft_data_layout_requirements>` may be
    satisfied for only one of the two compute directions, *i.e.*, for only one
    of the forward or backward DFT(s). Such a configuration should not cause an
    exception to be thrown by the descriptor's ``commit``
    :ref:`member function<onemath_dft_descriptor_commit>` but the behavior of
    oneMath is undefined if using that object for the compute direction that does
    not align with the :ref:`consistency requirements<onemath_dft_data_layout_requirements>`.

.. note::
    Setting either of ``config_param::INPUT_STRIDES`` or
    ``config_param::OUTPUT_STRIDES`` triggers any default or previously-set
    values for ``config_param::FWD_STRIDES`` and ``config_param::BWD_STRIDES``
    to reset to ``std::vector<std::int64_t>(d+1, 0)``, and vice versa.
    This default behavior prevents mix-and-matching usage of either of
    ``config_param::INPUT_STRIDES`` or ``config_param::OUTPUT_STRIDES`` with
    either of ``config_param::FWD_STRIDES`` or ``config_param::BWD_STRIDES``,
    which is **not** to be supported. If such a configuration is attempted, an
    exception is to be thrown at commit time due to invalid configuration, as
    the stride values that were implicitly reset surely invalidate the
    :ref:`consistency requirements<onemath_dft_data_layout_requirements>` for any
    non-trivial DFT.

If specifying the data layout strides using these deprecated configuration
parameters and if the strides differ in forward and backward domain, the
descriptor *must* be re-configured and re-committed for computing the DFT in
the reverse direction as shown below.

.. code-block:: cpp

   namespace dft = oneapi::math::dft;
   // ...
   desc.set_value(dft::config_param::INPUT_STRIDES,  fwd_domain_strides);
   desc.set_value(dft::config_param::OUTPUT_STRIDES, bwd_domain_strides);
   desc.commit(queue);
   compute_forward(desc, ...);
   // ...
   desc.set_value(dft::config_param::INPUT_STRIDES,  bwd_domain_strides);
   desc.set_value(dft::config_param::OUTPUT_STRIDES, fwd_domain_strides);
   desc.commit(queue);
   compute_backward(desc, ...);

The ``config_param::INPUT_STRIDES`` and ``config_param::OUTPUT_STRIDES``
parameters are deprecated. A warning message "{IN,OUT}PUT_STRIDES are deprecated:
please use {F,B}WD_STRIDES, instead." is to be reported to applications using
these configuration parameters.

**Parent topic** :ref:`onemath_dft_enums`
