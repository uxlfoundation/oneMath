.. SPDX-FileCopyrightText: 2023 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_device_rng_mcg31m1:

mcg31m1
=======

The 31-bit multiplicative congruential pseudorandom number generator MCG(:math:`1132489760, 2^{32}-1`) :ref:`[L'Ecuyer99a] <onemath_rng_bibliography>`.

.. rubric:: Description

The mcg31m1 engine is a 31-bit multiplicative congruential generator :ref:`[L'Ecuyer99] <onemath_rng_bibliography>`. 
The mcg31m1 generator belongs to linear congruential generators with the period length of approximately :math:`2^{31}`. 
Such generators are still used as default random number generators in various software systems, mainly due to the 
simplicity of the portable versions implementation, speed, and compatibility with the earlier systems versions. 
However, their period length does not meet the requirements for modern basic generators. Still, the mcg31m1 generator 
possesses good statistic properties and you may successfully use it to generate random numbers of different 
distributions for small samplings.

.. container:: section

    .. rubric:: Generation algorithm

    :math:`x_n=ax_{n-1}(mod \ m)`

    :math:`u_n = x_n / m`

    :math:`a = 1132489760, m=2^{31} - 1`

class mcg31m1
-------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::rng::device {
     template<std::int32_t VecSize = 1>
     class mcg31m1 {
     public:
       static constexpr std::uint32_t default_seed = 1;
       static constexpr std::int32_t vec_size = VecSize;
       
       mcg31m1();
       mcg31m1(std::uint32_t seed, std::uint64_t offset = 0);
     };
   }

.. container:: section

    .. rubric:: Class Template Parameters

    VecSize
        Describes the size of vector which will be produced by generate function by this engine. VecSize values 
        may be 1, 2, 3, 4, 8, 16 as ``sycl::vec`` class size. By default VecSize = 1, for this case, a single 
        random number is returned by the ``generate`` function.

.. container:: section

    .. rubric:: Class Members

    .. list-table::
        :header-rows: 1

        * - Routine
          - Description
        * - `mcg31m1()`_
          - Default constructor
        * - `mcg31m1(std::uint32_t seed, std::uint64_t offset = 0)`_
          - Constructor for common seed initialization of the engine and common number of skipped elements

.. container:: section

    .. rubric:: Constructors

    .. _`mcg31m1()`:

    .. code-block:: cpp
    
        mcg31m1::mcg31m1()

    .. _`mcg31m1(std::uint32_t seed, std::uint64_t offset = 0)`:

    .. code-block:: cpp
    
        mcg31m1::mcg31m1(std::uint32_t seed, std::uint64_t offset = 0)

    .. container:: section

        .. rubric:: Input Parameters

        seed
            The initial conditions of the generator state, assume :math:`x_0 = seed \ mod \ 0x7FFFFFFF`, 
            if :math:`x_0 = 0`, assume :math:`x_0 = 1`.
        
        offset
            Number of skipped elements.

**Parent topic:** :ref:`onemath_device_rng_engines`
