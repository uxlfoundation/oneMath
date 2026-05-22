.. SPDX-FileCopyrightText: 2023 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_device_rng_bits:

bits
====

Generates bits of underlying engine (BRNG) integer sequence.

.. rubric:: Description

The ``bits`` class object is used in ``generate`` and function to provide integer 
random values. Each integer can be treated as a vector of several bits. In pseudorandom generators 
this randomness can be violated. See :ref:`VS Notes<onemath_rng_bibliography>` for details.


class bits
----------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::rng::device {
     template<typename UIntType = std::uint32_t>
     class bits {
       using result_type = UIntType;
     };
   }


.. container:: section

    .. rubric:: Template parameters

    .. container:: section

        typename UIntType
            Type of the produced values. Supported types:
                * ``std::uint32_t`` for philox4x32x10, mrg32k3a and mcg31m1 engines.
                * ``std::uint64_t`` for mcg59.

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            bits::result_type = UIntType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

**Parent topic:** :ref:`onemath_device_rng_distributions`
