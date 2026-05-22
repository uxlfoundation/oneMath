.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_uniform_bits:

uniform_bits
============

Class is used for generation of uniformly distributed bits in 32/64-bit chunks.

.. _onemath_rng_uniform_bits_description:

.. rubric:: Description

The class object is used in the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` function to provide uniformly distributed bits in 32/64-bit chunks. It is designed to ensure each bit in the 32/64-bit chunk is uniformly distributed. Can be not supported by the specific engine.

class uniform_bits
------------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    template<typename UIntType = std::uint32_t>
    class uniform_bits {
    public:
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

**Parent topic:** :ref:`onemath_rng_distributions`
