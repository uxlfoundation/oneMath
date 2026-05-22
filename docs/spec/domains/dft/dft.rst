.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_dft:

Discrete Fourier Transform Functions
------------------------------------

oneMath provides a DPC++ interface to :math:`d`-dimensional :math:`\left(d \in
\mathbb{Z}_{>0}\right)` Discrete Fourier Transforms (DFTs).

.. _onemath_dft_definitions:

Definitions
+++++++++++

Let :math:`w` be a set of :math:`M` finite :math:`d`-dimensional discrete
sequences :math:`w^{m}` of length(s) :math:`n_1 \times n_2 \times \dots \times n_d`
(:math:`d \in \mathbb{Z}_{>0}`, :math:`M \in \mathbb{Z}_{>0}`,
:math:`n_{\ell} \in \mathbb{Z}_{>0}\ \forall \ell \in \lbrace 1, 2, \ldots, d \rbrace`,
and :math:`m \in \lbrace 0, 1, \ldots, M-1\rbrace`).
Let :math:`w^{m}_{k_1, k_2, \ldots, k_d}` be the entry of multi-index
:math:`\left(k_1, k_2, \ldots, k_d\right)` in :math:`w^{m}` wherein integer
indices :math:`k_{\ell}` are such that
:math:`0 \leq k_{\ell} < n_{\ell},\ \forall \ell \in \lbrace 1, 2, \ldots, d \rbrace`.

For every :math:`m \in \lbrace 0, 1, \ldots, M - 1 \rbrace`, the DFT of sequence
:math:`w^{m}` is the :math:`d`-dimensional discrete sequence :math:`z^{m}` of
length(s) :math:`n_1 \times n_2 \times \dots \times n_d` whose entries are defined as

.. _onemath_dft_formula:

.. math::
    z^{m}_{k_1, k_2,\ldots, k_d} = \sigma_{\delta} \displaystyle\sum_{j_d=0}^{n_d-1}\dots\displaystyle\sum_{j_2=0}^{n_2-1}\displaystyle\sum_{j_1=0}^{n_1-1} w^{m}_{j_1, j_2,\dots,j_d} \exp \left[ \delta 2\pi \imath \left( \sum_{\ell=1}^{d} \frac{j_{\ell}k_{\ell}}{n_{\ell}} \right) \right],
    :label: eq_dft_definition

where :math:`\imath^2 = -1`. In :eq:`eq_dft_definition`, :math:`\delta`
determines one of the two "directions" of the DFT: :math:`\delta=-1` defines the
"forward DFT" while :math:`\delta=+1` defines the "backward DFT".
:math:`\sigma_{\delta}` is a (real) scaling factor associated with either operation.

The domain of input (resp. output) discrete sequences for a forward (resp.
backward) DFT is referred to as "forward domain". Conversely, the domain of
output (resp. input) discrete sequences for forward (resp. backward) DFT is
referred to as "backward domain".

oneMath supports single-precision (fp32) and double-precision (fp64)
floating-point arithmetic for the calculation of DFTs, using two kinds of
forward domains:

- the set of complex :math:`d`-dimensional discrete sequences, referred to as
  "complex forward domain";
- the set of real :math:`d`-dimensional discrete sequences, referred to as
  "real forward domain".

Similarly, we refer to DFTs of complex (resp. real) forward domain as "complex
DFTs" (resp. "real DFTs"). Regardless of the kind of forward domain, the
backward domain's data sequences are always complex.

The calculation of the same DFT for several, *i.e.*, :math:`M > 1`, data sets of
the same kind of forward domain, using the same precision is referred to as a
"batched DFT".

.. _onemath_dft_elementary_range_of_indices:

Elementary range of indices
+++++++++++++++++++++++++++

In general, all entries of multi-indices :math:`\left(k_1, \ldots, k_d\right)`
such that :math:`0\leq k_{\ell} < n_{\ell}, \forall \ell \in \lbrace 1, \ldots, d \rbrace`
unambiguously determine any relevant :math:`d`-dimensional sequence
unambiguously (for any valid :math:`m`). In case of real DFTs, the data
sequences in backward domain can be fully determined from a smaller range of
indices. Indeed, if all entries of :math:`w` are real in
:eq:`eq_dft_definition`, then the entries of :math:`z` are complex and, for any
valid :math:`m`,
:math:`\left(z^{m}_{k_1, k_2, \dots, k_d}\right)^{*} = z^{m}_{j_{1}, j_{2}, \dots, j_{d}}` where
:math:`j_{\ell} = \left(n_{\ell} - k_{\ell}\right) \pmod {n_{\ell}}, \ \forall \ell \in \lbrace 1, \ldots, d \rbrace`
and :math:`\lambda^{*}` represents the conjugate of complex number
:math:`\lambda`. This conjugate symmetry relation makes roughly half the data
redundant in backward domain: in case of real DFTs, the data sequences in
backward domain can be fully determined even if one of the :math:`d` indices
:math:`k_{\ell}` is limited to the range
:math:`0\leq k_{\ell} \leq \lfloor \frac{n_{\ell}}{2}\rfloor`. In oneMath, the
index :math:`k_d`, *i.e.*, the last dimension's index, is restricted as such to
capture an elementary set of non-redundant entries for data sequences belonging
to the backward domain of real DFTs.

In other words, oneMath expects and produces a set of :math:`M`
:math:`d`-dimensional data sequences
:math:`\left(\cdot \right)^{m}_{k_1, k_2,\ldots, k_d}` with integer indices 
:math:`m` and
:math:`k_{\ell}\ \left(\ell \in \lbrace 1, \ldots, d \rbrace\right)` in the
elementary range

- :math:`0 \leq m < M`;
- :math:`0 \leq k_j < n_j,\ \forall j \in \lbrace1, \ldots, d - 1\rbrace`, if :math:`d > 1`;
- :math:`0 \leq k_d < n_d`, except for backward domain's data sequences of real DFTs;
- :math:`0 \leq k_d \leq \lfloor\frac{n_d}{2}\rfloor`, for backward domain's data sequences of real DFTs.

.. _onemath_dft_additional_constraints_in_bwd_domain_for_real_dft:

Additional constraints for data in backward domain of real DFTs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, note that the conjugate symmetry relation further constrains some of
the entries (or pairs thereof) in the backward domain's data sequences for real
DFTs. Specifically, for any of the :math:`M` sequences,

- the imaginary part must be :math:`0` for any entry of multi-index
  :math:`\left(k_1, k_2, \ldots, k_d\right)` such that
  :math:`k_{\ell} \equiv \left(n_{\ell} - k_{\ell}\right) \pmod {n_{\ell}}, \forall \ell \in \lbrace{1, \ldots, d\rbrace}`,
  *e.g.*, entry of multi-index :math:`\left(0, 0, \ldots, 0\right)`;
- pairs of entries of multi-indices :math:`\left(k_1, k_2, \ldots, k_d\right)`
  and :math:`\left(j_1, j_2, \ldots, j_d\right)` such that
  :math:`k_{\ell} \equiv \left(n_{\ell} - j_{\ell}\right) \pmod {n_{\ell}}, \forall \ell \in \lbrace{1, \ldots, d\rbrace}`
  must be complex conjugates of one another, *e.g.*, entries of multi-indices
  :math:`\left(1, 0, \ldots, 0\right)` and
  :math:`\left(n_1 - 1, 0, \ldots, 0\right)` must be complex conjugates (note
  that this case falls back to the above constraint if :math:`n_1 = 2`).

.. note::
  The behavior of oneMath is undefined for real backward DFT if the input data
  does not satisfy those constraints. oneMath considers it the user's
  responsibility to guarantee that these constraints are satisfied by the input
  data for real backward DFTs.

.. _onemath_dft_recommended_usage:

Recommended usage
+++++++++++++++++

The desired DFT to be computed is entirely defined by an object ``desc`` of a
specialization of the ``oneapi::math::dft::descriptor``
:ref:`class template<onemath_dft_descriptor>`.
The desired floating-point format and kind of forward domain are determined by
``desc``'s particular class, *i.e.*, by the specialization values of the
:ref:`template parameters<onemath_dft_descriptor_template_parameters>` ``prec``
and ``dom`` of the ``descriptor`` class template, respectively. Once ``desc`` is
created, the length(s) :math:`\lbrace n_1, n_2, \ldots, n_d\rbrace` (and the
dimension :math:`d`) cannot be changed, as they are read-only parameters. Other
configuration details for the DFT under consideration may be specified by
invoking the appropriate
:ref:`configuration-setting member function(s)<onemath_dft_descriptor_set_value>`
of ``desc`` for every relevant configuration parameter (*e.g.*, the number
:math:`M` of sequences to consider in case of a batched DFT). Once configured
as desired, ``desc`` must be initialized for computation by using its
:ref:`committing member function<onemath_dft_descriptor_commit>`, which requires
a ``sycl::queue`` object. The successful completion of that operation makes
``desc`` ready to compute the desired DFT *as configured*, for the particular
device and context encapsulated by the latter. ``desc`` may then be used with
user-provided, device-accessible data, in a
``oneapi::math::dft::compute_forward`` (resp.
``oneapi::math::dft::compute_backward``) function to enqueue operations relevant
to the desired forward (resp. backward) DFT calculations.

.. note::
  Objects of any ``oneapi::math::dft::descriptor`` class

  - must be successfully committed prior to providing them to any compute
    function;
  - must be re-committed to account for any change in configuration operated
    after they were already successfully committed;
  - deliver best performance for DFT calculations when created, configured and
    committed outside applications' hotpath(s) that use them multiple times for
    identically-configured DFTs. ``oneapi::math::dft::compute_forward`` and/or
    ``oneapi::math::dft::compute_backward`` should be the only oneMath DFT-related
    routines invoked in programs' hotpaths.

Summary table
~~~~~~~~~~~~~

The table below summarizes the identifiers of the ``oneapi::math::dft`` namespace
relevant to computing DFTs.

.. _onemath_dft_summary_table:
.. container::


   .. container:: tablenoborder

      .. 

      .. list-table:: 
         :header-rows: 1
         :widths: 33 64   

         * -     Identifier in ``oneapi::math::dft``
           -     Description
         * -     ``descriptor``
           -     A template for classes whose instances define a specific DFT to
                 be calculated and its configuration. Template
                 parameters are omitted in this table for conciseness (more
                 details are available in the page dedicated to
                 :ref:`the descriptor class template<onemath_dft_descriptor>`).
         * -     ``domain``, ``precision``, ``config_param`` and ``config_value``
           -     :ref:`Scoped enumerations<onemath_dft_enums>` pertaining to the
                 definition of any configuration parameter or its associated
                 value, for any instance of a ``descriptor`` class.
         * -     ``descriptor::set_value``
           -     Member functions to
                 :ref:`set (writable) configuration parameters<onemath_dft_descriptor_set_value>`
                 for any instance of a ``descriptor`` class.
         * -     ``descriptor::get_value``
           -     Member functions to
                 :ref:`query configuration parameters<onemath_dft_descriptor_get_value>`
                 from any instance of a ``descriptor`` class.
         * -     ``descriptor::commit``
           -     A member function to
                 :ref:`commit<onemath_dft_descriptor_commit>` any instance
                 of a ``descriptor`` class to the DFT calculations it defines,
                 on a given queue.
         * -     ``descriptor::set_workspace``
           -     A member function to equip any instance of a ``descriptor``
                 class with an
                 :ref:`externally-allocated workspace<onemath_dft_descriptor_set_workspace>`.
         * -     ``compute_forward``
           -     Function templates for computing a forward DFT, as defined by
                 a (successfully-committed) instance of a ``descriptor`` class
                 (required argument). Template parameters are omitted in this
                 table for conciseness (more details are available in the
                 :ref:`dedicated page<onemath_dft_compute_forward>`).
         * -     ``compute_backward``
           -     Function templates for computing a backward DFT, as defined by
                 a (successfully-committed) instance of a ``descriptor`` class
                 (required argument). Template parameters are omitted in this
                 table for conciseness (more details are available in the
                 :ref:`dedicated page<onemath_dft_compute_backward>`).


**Parent topic:** :ref:`onemath_domains`

.. toctree::
    :hidden:

    descriptor
    enums_and_config_params
    compute_forward
    compute_backward

