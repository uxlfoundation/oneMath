.. SPDX-FileCopyrightText: 2023 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_device_rng_mcg59:

mcg59
=====

The 59-bit multiplicative congruential pseudorandom number generator MCG(:math:`13^{13}, 2^{59}`)
from NAG Numerical Libraries.

.. rubric:: Description

The mcg59 engine is a 59-bit multiplicative congruential generator from NAG Numerical Libraries :ref:`NAG <onemath_rng_bibliography>`. 
The mcg59 generator belongs to linear congruential generators with the period length of approximately :math:`2^{57}`.

.. container:: section

    .. rubric:: Generation algorithm

    :math:`x_n=ax_{n-1}(mod \ m)`

    :math:`u_n = x_n / m`

    :math:`a = 13^{13}, m=2^{59}`

class mcg59
-----------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::rng::device {
     template<std::int32_t VecSize = 1>
     class mcg59 {
     public:
       static constexpr std::uint32_t default_seed = 1;
       static constexpr std::int32_t vec_size = VecSize;
       
       mcg59();
       mcg59(std::uint64_t seed, std::uint64_t offset = 0);
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
        * - `mcg59()`_
          - Default constructor
        * - `mcg59(std::uint64_t seed, std::uint64_t offset = 0)`_
          - Constructor for common seed initialization of the engine and common number of skipped elements

.. container:: section

    .. rubric:: Constructors

    .. _`mcg59()`:

    .. code-block:: cpp
    
        mcg59::mcg59()

    .. _`mcg59(std::uint64_t seed, std::uint64_t offset = 0)`:

    .. code-block:: cpp
    
        mcg59::mcg59(std::uint64_t seed, std::uint64_t offset = 0)

    .. container:: section

        .. rubric:: Input Parameters

        seed
            The initial conditions of the generator state, assume :math:`x_0 = seed \ mod \ 2^{59}`, 
            if :math:`x_0 = 0`, assume :math:`x_0 = 1`.
        
        offset
            Number of skipped elements.

**Parent topic:** :ref:`onemath_device_rng_engines`
