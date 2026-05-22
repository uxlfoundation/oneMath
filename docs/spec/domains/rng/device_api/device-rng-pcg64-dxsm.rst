.. SPDX-FileCopyrightText: 2025 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_device_rng_pcg64_dxsm:

pcg64_dxsm
==========

A permuted congruential pseudorandom number generator PCG64 DXSM with a period of :math:`2^{128}` :ref:`[pcg2014] <onemath_rng_bibliography>`.

.. rubric:: Description

The PCG is a family of random number generators based on linear congruential generator (LCG). The algorithm applies permutation function to improve statistical properties of LCG.
PCG64 DXSM uses `double xorshift multiply` output function and has a period of :math:`2^{128}`.

.. container:: section

    .. rubric:: Generation algorithm

    :math:`x_n = x_{n-1} \cdot a + b`

    :math:`hi_n = x_n \gg 64`

    :math:`lo_n = x_n \land \left(2^{64} - 1\right) \lor 1`
    
    :math:`hi_n = hi_n \oplus (hi_n \gg 32)`
    
    :math:`hi_n = hi_n \cdot a`
    
    :math:`hi_n = hi_n \oplus (hi_n \gg 48)`
    
    :math:`u_n = hi_n \cdot lo_n`

    :math:`a = \text{0xDA942042E4DD58B5}, b = \text{0x5851F42D4C957F2D14057B7EF767814F}`

class pcg64_dxsm
----------------

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::math::rng::device {
        template<std::int32_t VecSize = 1>
        class pcg64_dxsm {
        public:
            static constexpr std::uint64_t default_seed = 1;
            static constexpr std::int32_t vec_size = VecSize;

            pcg64_dxsm();
            pcg64_dxsm(std::uint64_t seed, std::uint64_t offset = 0);
            pcg64_dxsm(std::initializer_list<std::uint64_t> seed, std::uint64_t offset = 0);
            pcg64_dxsm(std::uint64_t seed, std::initializer_list<std::uint64_t> offset);
            pcg64_dxsm(std::initializer_list<std::uint64_t> seed, std::initializer_list<std::uint64_t> offset);
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
        * - `pcg64_dxsm()`_
          - Default constructor
        * - `pcg64_dxsm(std::uint64_t seed, std::uint64_t offset = 0)`_
          - Constructor for common seed initialization of the engine and common number of skipped elements
        * - `pcg64_dxsm(std::initializer_list<std::uint64_t> seed, std::uint64_t offset = 0)`_
          - Constructor for extended seed initialization of the engine and common number of skipped elements
        * - `pcg64_dxsm(std::uint64_t seed, std::initializer_list<std::uint64_t> offset)`_
          - Constructor for common seed initialization of the engine and extended number of skipped elements
        * - `pcg64_dxsm(std::initializer_list<std::uint64_t> seed, std::initializer_list<std::uint64_t> offset)`_
          - Constructor for extended seed initialization of the engine and extended number of skipped elements

.. container:: section

    .. rubric:: Constructors

    .. _`pcg64_dxsm()`:

    .. code-block:: cpp
    
        pcg64_dxsm::pcg64_dxsm()

    .. _`pcg64_dxsm(std::uint64_t seed, std::uint64_t offset = 0)`:

    .. code-block:: cpp
    
        pcg64_dxsm::pcg64_dxsm(std::uint64_t seed, std::uint64_t offset = 0)

    .. container:: section

        .. rubric:: Input Parameters

        seed
            The initial conditions of the generator state.
        
        offset
            Number of skipped elements.
            
    .. _`pcg64_dxsm(std::initializer_list<std::uint64_t> seed, std::uint64_t offset = 0)`:

    .. code-block:: cpp
    
        pcg64_dxsm::pcg64_dxsm(std::initializer_list<std::uint64_t> seed, std::uint64_t offset = 0)

    .. container:: section

        .. rubric:: Input Parameters

        seed
            The initial conditions of the generator state.
        
        offset
            Number of skipped elements.

    .. _`pcg64_dxsm(std::uint64_t seed, std::initializer_list<std::uint64_t> offset)`:

    .. code-block:: cpp
    
        pcg64_dxsm::pcg64_dxsm(std::uint64_t seed, std::initializer_list<std::uint64_t> offset)

    .. container:: section

        .. rubric:: Input Parameters

        seed
            The initial conditions of the generator state.
        
        offset
            Number of skipped elements. Offset is calculated as: ``num_to_skip`` [0]+ ``num_to_skip`` [1]*2\ :sup:`64`.
            
    .. _`pcg64_dxsm(std::initializer_list<std::uint64_t> seed, std::initializer_list<std::uint64_t> offset)`:

    .. code-block:: cpp
    
        pcg64_dxsm::pcg64_dxsm(std::initializer_list<std::uint64_t> seed, std::initializer_list<std::uint64_t> offset)

    .. container:: section

        .. rubric:: Input Parameters

        seed
            The initial conditions of the generator state.
        
        offset
            Number of skipped elements. Offset is calculated as: ``num_to_skip`` [0]+ ``num_to_skip`` [1]*2\ :sup:`64`.

**Parent topic:** :ref:`onemath_device_rng_engines`
