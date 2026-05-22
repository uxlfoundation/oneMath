.. SPDX-FileCopyrightText: 2023 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_device_rng_uniform_bits:

uniform_bits
============

Generates uniformly distributed bits in 32/64-bit chunks.

.. rubric:: Description

The ``uniform_bits`` class object is used in ``generate`` and function to generate uniformly distributed bits 
in 32/64-bit chunks. It is designed to ensure each bit in the 32/64-bit chunk is uniformly distributed. This distribution 
is supported for philox4x32x10 and mcg59 engines. When generating 64-bit chunks, twice as much engine offset needs to 
be provided.

``UIntType`` denotes the chunk size and can be ``std::uint32_t``, ``std::uint64_t``. See :ref:`VS Notes<onemath_rng_bibliography>` for details.


class uniform_bits
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::rng::device {
     template<typename UIntType = std::uint32_t>
     class uniform_bits {
       using result_type = UIntType;
     };
   }


.. container:: section

    .. rubric:: Template parameters

    .. container:: section

        typename UIntType
            Type of the produced values. Supported types:
                * ``std::uint32_t``
                * ``std::uint64_t``

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            uniform_bits::result_type = UIntType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

**Parent topic:** :ref:`onemath_device_rng_distributions`
