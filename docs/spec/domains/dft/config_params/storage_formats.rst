.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_dft_data_storage:

Data storage
============

The usage of prepended namespace specifiers ``oneapi::math::dft`` is
omitted below for conciseness.

The data storage convention observed by a ``descriptor`` object depends on
whether it is a real or complex descriptor and, in case of complex descriptors,
on the configuration value associated with configuration parameter
``config_param::COMPLEX_STORAGE``.

.. _onemath_dft_complex_storage:

Complex descriptors
-------------------

For a complex descriptor, the configuration parameter
``config_param::COMPLEX_STORAGE`` specifies how the entries of the complex data
sequences it consumes and produces are stored. If that configuration parameter is
associated with a configuration value ``config_value::COMPLEX_COMPLEX`` (default
behavior), those entries are accessed and stored as ``std::complex<float>``
(resp. ``std::complex<double>``) elements of a single data container
(device-accessible USM allocation or ``sycl::buffer`` object) if the
``descriptor`` object is a single-precision (resp. double-precision) descriptor.
If the configuration value ``config_value::REAL_REAL`` is used instead, the real
and imaginary parts of those entries are accessed and stored as ``float`` (resp.
``double``) elements of two separate, non-overlapping data containers
(device-accessible USM allocations or ``sycl::buffer`` objects) if the
``descriptor`` object is a single-precision (resp. double-precision) descriptor.

These two behaviors are further specified and illustrated below.

.. _onemath_dft_complex_storage_complex_complex:

.. rubric:: ``config_value::COMPLEX_COMPLEX`` for ``config_param::COMPLEX_STORAGE``

For complex descriptors with parameter ``config_param::COMPLEX_STORAGE`` set to
``config_value::COMPLEX_COMPLEX``, each of forward- and backward-domain data
sequences must belong to a single data container (device-accessible USM
allocation or ``sycl::buffer`` object). Any relevant entry
:math:`\left(\cdot\right)^{m}_{k_1, k_2,\dots ,k_d}` is accessed/stored from/in
a data container provided at compute time at the index value expressed in eq.
:eq:`eq_idx_data_layout` (see the page dedicated to the
:ref:`configuration of data layout<onemath_dft_config_data_layouts>`)
of that data container, whose elementary data type is (possibly implicitly
re-interpreted as) ``std::complex<float>`` (resp. ``std::complex<double>``) for
single-precision (resp. double-precision) descriptors.

The same unique data container is to be used for forward- and backward-domain
data sequences for in-place transforms (for ``descriptor`` objects with
configuration value ``config_value::INPLACE`` for configuration parameter
``config_param::PLACEMENT``). Two separate data containers sharing no common
elements are to be used for out-of-place transforms (for ``descriptor`` objects
with configuration value ``config_value::NOT_INPLACE`` for configuration
parameter ``config_param::PLACEMENT``).

The following snippet illustrates the usage of ``config_value::COMPLEX_COMPLEX``
for configuration parameter ``config_param::COMPLEX_STORAGE``, in the
context of in-place, single-precision (fp32) calculations of :math:`M`
three-dimensional :math:`n_1 \times n_2 \times n_3` complex transforms, using
identical (default) strides and distances in forward and backward domains, with
USM allocations.

.. code-block:: cpp

    namespace dft = oneapi::math::dft;
    dft::descriptor<dft::precision::SINGLE, dft::domain::COMPLEX> desc({n1, n2, n3});
    std::vector<std::int64_t> strides({0, n2*n3, n3, 1});
    std::int64_t dist = n1*n2*n3;
    std::complex<float> *Z = (std::complex<float> *) malloc_device(2*sizeof(float)*n1*n2*n3*M, queue);
    desc.set_value(dft::config_param::FWD_STRIDES, strides);
    desc.set_value(dft::config_param::BWD_STRIDES, strides);
    desc.set_value(dft::config_param::FWD_DISTANCE, dist);
    desc.set_value(dft::config_param::BWD_DISTANCE, dist);
    desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, M);
    desc.set_value(dft::config_param::COMPLEX_STORAGE, dft::config_value::COMPLEX_COMPLEX);
    desc.commit(queue);

    // initialize forward-domain data such that entry {m;k1,k2,k3}
    //   = Z[ strides[0] + k1*strides[1] + k2*strides[2] + k3*strides[3] + m*dist ]
    auto ev = compute_forward(desc, Z); // complex-to-complex in-place DFT
    // Upon completion of ev, in backward domain: entry {m;k1,k2,k3}
    //   = Z[ strides[0] + k1*strides[1] + k2*strides[2] + k3*strides[3] + m*dist ]

.. _onemath_dft_complex_storage_real_real:

.. rubric:: ``config_value::REAL_REAL`` for ``config_param::COMPLEX_STORAGE``

For complex descriptors with parameter ``config_param::COMPLEX_STORAGE`` set to
``config_value::REAL_REAL``, forward- and backward-domain data sequences are
read/stored from/in two different, non-overlapping data containers
(device-accessible USM allocations or ``sycl::buffer`` objects) encapsulating
the real and imaginary parts of the relevant entries separately. The real and
imaginary parts of any relevant complex entry
:math:`\left(\cdot\right)^{m}_{k_1, k_2,\dots ,k_d}` are both stored at the
index value expressed in eq. :eq:`eq_idx_data_layout` (see the page dedicated to
the :ref:`configuration of data layout<onemath_dft_config_data_layouts>`) of
their respective data containers, whose elementary data type is (possibly
implicitly re-interpreted as) ``float`` (resp. ``double``) for single-precision
(resp. double-precision) descriptors.

The same two data containers are to be used for real and imaginary parts of
forward- and backward-domain data sequences for in-place transforms (for
``descriptor`` objects with configuration value ``config_value::INPLACE`` for
configuration parameter ``config_param::PLACEMENT``). Four separate data
containers sharing no common elements are to be used for out-of-place transforms
(for ``descriptor`` objects with configuration value ``config_value::NOT_INPLACE``
for configuration parameter ``config_param::PLACEMENT``).

The following snippet illustrates the usage of ``config_value::REAL_REAL``
set for configuration parameter ``config_param::COMPLEX_STORAGE``, in the
context of in-place, single-precision (fp32) calculation of :math:`M`
three-dimensional :math:`n_1 \times n_2 \times n_3` complex transforms, using
identical (default) strides and distances in forward and backward domains, with
USM allocations.

.. code-block:: cpp

    namespace dft = oneapi::math::dft;
    dft::descriptor<dft::precision::SINGLE, dft::domain::COMPLEX> desc({n1, n2, n3});
    std::vector<std::int64_t> strides({0, n2*n3, n3, 1});
    std::int64_t dist = n1*n2*n3;
    float *ZR = (float *) malloc_device(sizeof(float)*n1*n2*n3*M, queue); // data container for real parts
    float *ZI = (float *) malloc_device(sizeof(float)*n1*n2*n3*M, queue); // data container for imaginary parts
    desc.set_value(dft::config_param::FWD_STRIDES, strides);
    desc.set_value(dft::config_param::BWD_STRIDES, strides);
    desc.set_value(dft::config_param::FWD_DISTANCE, dist);
    desc.set_value(dft::config_param::BWD_DISTANCE, dist);
    desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, M);
    desc.set_value(dft::config_param::COMPLEX_STORAGE, dft::config_value::REAL_REAL);
    desc.commit(queue);

    // initialize forward-domain data such that the real part of entry {m;k1,k2,k3}
    //   = ZR[ strides[0] + k1*strides[1] + k2*strides[2] + k3*strides[3] + m*dist ]
    // and the imaginary part of entry {m;k1,k2,k3}
    //   = ZI[ strides[0] + k1*strides[1] + k2*strides[2] + k3*strides[3] + m*dist ]
    auto ev = compute_forward<decltype(desc), float>(desc, ZR, ZI); // complex-to-complex in-place DFT
    // Upon completion of ev, in backward domain: the real part of entry {m;k1,k2,k3}
    //   = ZR[ strides[0] + k1*strides[1] + k2*strides[2] + k3*strides[3] + m*dist ]
    // and the imaginary part of entry {m;k1,k2,k3}
    //   = ZI[ strides[0] + k1*strides[1] + k2*strides[2] + k3*strides[3] + m*dist ]

.. _onemath_dft_real_storage:

Real descriptors
----------------

Real descriptors observe only one type of data storage. Any relevant (real)
entry :math:`\left(\cdot\right)^{m}_{k_1, k_2,\dots ,k_d}` of a data sequence
in forward domain is accessed and stored as a ``float`` (resp. ``double``)
element of a single data container (device-accessible USM allocation or
``sycl::buffer`` object) if the ``descriptor`` object is a single-precision
(resp. double-precision) descriptor. Any relevant (complex) entry
:math:`\left(\cdot\right)^{m}_{k_1, k_2,\dots ,k_d}` of a data sequence in
backward domain is accessed and stored as a ``std::complex<float>`` (resp.
``std::complex<double>``) element of a single data container (device-accessible
USM allocation or ``sycl::buffer`` object) if the
``descriptor`` object is a single-precision (resp. double-precision) descriptor.

The following snippet illustrates the usage of a real, single-precision
descriptor (and the corresponding data storage) for the in-place,
single-precision (fp32), calculation of :math:`M` three-dimensional
:math:`n_1 \times n_2 \times n_3` real transforms, using default strides in
forward and backward domains, with USM allocations.

.. code-block:: cpp

    namespace dft = oneapi::math::dft;
    dft::descriptor<dft::precision::SINGLE, dft::domain::REAL> desc({n1, n2, n3});
    // Note: integer divisions here below
    std::vector<std::int64_t> fwd_strides({0, 2*n2*(n3/2 + 1), 2*(n3/2 + 1), 1});
    std::vector<std::int64_t> bwd_strides({0,   n2*(n3/2 + 1),   (n3/2 + 1), 1});
    std::int64_t fwd_dist = 2*n1*n2*(n3/2 + 1);
    std::int64_t bwd_dist =   n1*n2*(n3/2 + 1);
    float *data = (float *) malloc_device(sizeof(float)*fwd_dist*M, queue); // data container
    desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
    desc.set_value(dft::config_param::BWD_STRIDES, bwd_strides);
    desc.set_value(dft::config_param::FWD_DISTANCE, fwd_dist);
    desc.set_value(dft::config_param::BWD_DISTANCE, bwd_dist);
    desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, M);
    desc.commit(queue);

    // initialize forward-domain data such that real entry {m;k1,k2,k3}
    //   = data[ fwd_strides[0] + k1*fwd_strides[1] + k2*fwd_strides[2] + k3*fwd_strides[3] + m*fwd_dist ]
    auto ev = compute_forward(desc, data); // real-to-complex in-place DFT
    // In backward domain, the implicitly-assumed type is complex so, consider
    //   std::complex<float>* complex_data = static_cast<std::complex<float>*>(data);
    // upon completion of ev, the backward-domain entry {m;k1,k2,k3} is
    //   = complex_data[ bwd_strides[0] + k1*bwd_strides[1] + k2*bwd_strides[2] + k3*bwd_strides[3] + m*bwd_dist ]
    //   for 0 <= k3 <= n3/2.
    //   Note: if n3/2 < k3 < n3, entry {m;k1,k2,k3} is not stored explicitly
    //   since it is equal to std::conj(entry {m;n1-k1,n2-k2,n3-k3})

**Parent topic** :ref:`onemath_dft_enums`
