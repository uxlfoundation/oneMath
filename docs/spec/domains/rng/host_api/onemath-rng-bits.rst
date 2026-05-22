.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_bits:

bits
====

Class is used for generation of underlying engine integer recurrence.

.. _onemath_rng_bits_description:

.. rubric:: Description

The class object is used in the :ref:`oneapi::math::rng::generate()<onemath_rng_generate>` function to provide integer random numbers. Each integer can be treated as a vector of several bits. In a truly random generator, these bits are random, while in pseudorandom generators this randomness can be violated.

class bits
----------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng {
    template<typename UIntType = std::uint32_t>
    class bits {
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

.. container:: section

    .. rubric:: Member types

    .. container:: section

        .. code-block:: cpp

            bits::result_type = UIntType

        .. container:: section

            .. rubric:: Description

            The type which defines type of generated random numbers.

**Parent topic:** :ref:`onemath_rng_distributions`
