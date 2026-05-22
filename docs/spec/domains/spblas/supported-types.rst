.. SPDX-FileCopyrightText: 2024 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_sparse_supported_types:

Supported template types
========================

.. container:: tablenoborder

   .. list-table::
      :header-rows: 1

      * -  Data Types ``<dataType>``
        -  Integer Types ``<indexType>``
      * -  ``float``
        -  ``std::int32_t``
      * -  ``double``
        -  ``std::int64_t``
      * -  ``std::complex<float>``
        -   
      * -  ``std::complex<double>``
        -   

``dataType`` is used to describe the precision (i.e. number of bits) and domain
(i.e. real or complex) of the :ref:`data handles <onemath_sparse_data_handles>`
and the operations using them.

``indexType`` is used to describe the range of integer types such as indices,
offsets or sizes of the :ref:`data handles <onemath_sparse_data_handles>` and the
operations using them.

**Parent topic:** :ref:`onemath_spblas`
