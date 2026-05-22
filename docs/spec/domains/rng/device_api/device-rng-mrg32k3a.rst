.. SPDX-FileCopyrightText: 2023 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_device_rng_mrg32k3a:

mrg32k3a
========

The combined multiple recursive pseudorandom number generator MRG32k3a.

.. rubric:: Description

MRG32k3a engine is a 32-bit combined multiple recursive generator with two components of order 3
[:ref:`L'Ecuyer99a<onemath_rng_bibliography>`]. MRG32k3a combined generator meets the requirements for 
modern RNGs, such as good multidimensional uniformity, or a long period (:math:`p \approx 2^{191}`).


.. container:: section

    .. rubric:: Generation algorithm


    :math:`x_n=a_{11} x_{n-1} + a_{12} x_{n-2} + a_{13} x_{n-3}(mod \ m_{1})`

    :math:`y_n = a_{21} y_{n-1} + a_{22} y_{n-2} + a_{23} (mod \ m_2)`

    :math:`z_n = x_n - y_n (mod \ m_{1})`

    :math:`u_n = z_n / m_1`

    :math:`a_{11} = 0, a_{12} = 1403580, a_{13} = -810728, m_1 = 2^{32} - 209`

    :math:`a_{21} = 527612, a_{22} = 0, a_{23} = -1370589, m_2 = 2^{32} - 22853`


class mrg32k3a
--------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::rng::device {
     template<std::int32_t VecSize = 1>
     class mrg32k3a {
     public:
       static constexpr std::uint32_t default_seed = 1;
       static constexpr std::int32_t vec_size = VecSize;
   
       mrg32k3a();
       mrg32k3a(std::uint32_t seed, std::uint64_t offset = 0);
       mrg32k3a(std::initializer_list<std::uint32_t> seed, std::uint64_t offset = 0);
       mrg32k3a(std::uint32_t seed, std::initializer_list<std::uint64_t> offset);
       mrg32k3a(std::initializer_list<std::uint32_t> seed, std::initializer_list<std::uint64_t> offset);
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
        * - `mrg32k3a()`_
          - Default constructor
        * - `mrg32k3a(std::uint32_t seed, std::uint64_t offset = 0)`_
          - Constructor for common seed initialization of the engine and common number of skipped elements
        * - `mrg32k3a(std::initializer_list<std::uint32_t> seed, std::uint64_t offset = 0)`_
          - Constructor for extended seed initialization of the engine and common number of skipped elements
        * - `mrg32k3a(std::uint32_t seed, std::initializer_list<std::uint64_t> offset)`_
          - Constructor for common seed initialization of the engine and extended number of skipped elements
        * - `mrg32k3a(std::initializer_list<std::uint32_t> seed, std::initializer_list<std::uint64_t> offset)`_
          - Constructor for extended seed initialization of the engine and extended number of skipped elements

.. container:: section

    .. rubric:: Constructors

    .. _`mrg32k3a()`:

    .. code-block:: cpp
    
        mrg32k3a::mrg32k3a()

    .. _`mrg32k3a(std::uint32_t seed, std::uint64_t offset = 0)`:

    .. code-block:: cpp
    
        mrg32k3a::mrg32k3a(std::uint32_t seed, std::uint64_t offset = 0)

    .. container:: section

        .. rubric:: Input Parameters

        seed
            The initial conditions of the generator state, assume
            if :math:`n = 0: x_{-3} = x_{-2} = x_{-1} = y_{-3} = y_{-2} = y_{-1} = 1`

            if :math:`n = 1: x_{-3} = seed[0] \ mod \ m_1, x_{-2} = x_{-1} = y_{-3} = y_{-2} = y_{-1} = 1`

            if :math:`n = 2: x_{-3} = seed[0] \ mod \ m_1, x_{-2} = seed[1] \ mod \ m_1, x_{-1} = y_{-3} = y_{-2} = y_{-1} = 1`

            if :math:`n = 3: x_{-3} = seed[0] \ mod \ m_1, x_{-2} = seed[1] \ mod \ m_1, x_{-1} = seed[2] \ mod \ m_1`

                :math:`y_{-3} = y_{-2} = y_{-1} = 1`

            if :math:`n = 4: x_{-3} = seed[0] \ mod \ m_1, x_{-2} = seed[1] \ mod \ m_1, x_{-1} = seed[2] \ mod \ m_1`
            
                :math:`y_{-3} = seed[3] \ mod \ m_2, y_{-2} = y_{-1} = 1`

            if :math:`n = 5: x_{-3} = seed[0] \ mod \ m_1, x_{-2} = seed[1] \ mod \ m_1, x_{-1} = seed[2] \ mod \ m_1`

                :math:`y_{-3} = seed[3] \ mod \ m_2, y_{-2} = seed[4] \ mod \ m_2, y_{-1} = 1`

            if :math:`n \geqslant 6: x_{-3} = seed[0] \ mod \ m_1, x_{-2} = seed[1] \ mod \ m_1, x_{-1} = seed[2] \ mod \ m_1`

                :math:`y_{-3} = seed[3] \ mod \ m_2, y_{-2} = seed[4] \ mod \ m_2, y_{-1} = seed[5] \ mod \ m_2`

            if the values prove to be :math:`x_{-3} = x_{-2} = x_{-1} = 0`, assume :math:`x_{-3} = 1`

            if the values prove to be :math:`y_{-3} = y_{-2} = y_{-1} = 0`, assume :math:`y_{-3} = 1`.
        
        offset
            Number of skipped elements.

    .. _`mrg32k3a(std::initializer_list<std::uint32_t> seed, std::uint64_t offset = 0)`:

    .. code-block:: cpp
    
        mrg32k3a::mrg32k3a(std::initializer_list<std::uint32_t> seed, std::uint64_t offset = 0)

    .. container:: section

        .. rubric:: Input Parameters

        seed
            Initial conditions of the engine state.

        offset
            Number of skipped elements.

    .. _`mrg32k3a(std::uint32_t seed, std::initializer_list<std::uint64_t> offset)`:

    .. code-block:: cpp

        mrg32k3a::mrg32k3a(std::uint32_t seed, std::initializer_list<std::uint64_t> offset)

    .. container:: section

        .. rubric:: Input Parameters

        seed
            The initial conditions of the generator state, assume
            if :math:`n = 0: x_{-3} = x_{-2} = x_{-1} = y_{-3} = y_{-2} = y_{-1} = 1`

            if :math:`n = 1: x_{-3} = seed[0] \ mod \ m_1, x_{-2} = x_{-1} = y_{-3} = y_{-2} = y_{-1} = 1`

            if :math:`n = 2: x_{-3} = seed[0] \ mod \ m_1, x_{-2} = seed[1] \ mod \ m_1, x_{-1} = y_{-3} = y_{-2} = y_{-1} = 1`

            if :math:`n = 3: x_{-3} = seed[0] \ mod \ m_1, x_{-2} = seed[1] \ mod \ m_1, x_{-1} = seed[2] \ mod \ m_1`

                :math:`y_{-3} = y_{-2} = y_{-1} = 1`

            if :math:`n = 4: x_{-3} = seed[0] \ mod \ m_1, x_{-2} = seed[1] \ mod \ m_1, x_{-1} = seed[2] \ mod \ m_1`
            
                :math:`y_{-3} = seed[3] \ mod \ m_2, y_{-2} = y_{-1} = 1`

            if :math:`n = 5: x_{-3} = seed[0] \ mod \ m_1, x_{-2} = seed[1] \ mod \ m_1, x_{-1} = seed[2] \ mod \ m_1`

                :math:`y_{-3} = seed[3] \ mod \ m_2, y_{-2} = seed[4] \ mod \ m_2, y_{-1} = 1`

            if :math:`n \geqslant 6: x_{-3} = seed[0] \ mod \ m_1, x_{-2} = seed[1] \ mod \ m_1, x_{-1} = seed[2] \ mod \ m_1`

                :math:`y_{-3} = seed[3] \ mod \ m_2, y_{-2} = seed[4] \ mod \ m_2, y_{-1} = seed[5] \ mod \ m_2`

            if the values prove to be :math:`x_{-3} = x_{-2} = x_{-1} = 0`, assume :math:`x_{-3} = 1`

            if the values prove to be :math:`y_{-3} = y_{-2} = y_{-1} = 0`, assume :math:`y_{-3} = 1`.

        offset
            Number of skipped elements. Offset is calculated as: ``num_to_skip`` [0]+ ``num_to_skip`` [1]*2\ :sup:`64` + ``num_to_skip`` [2]\* 2\ :sup:`128` + … + ``num_to_skip`` [``n``-1]\*2\ :sup:`64` \*(``n``-1).

    .. _`mrg32k3a(std::initializer_list<std::uint32_t> seed, std::initializer_list<std::uint64_t> offset)`:

    .. code-block:: cpp

        mrg32k3a::mrg32k3a(std::initializer_list<std::uint32_t> seed, std::initializer_list<std::uint64_t> offset)

    .. container:: section

        .. rubric:: Input Parameters

        seed
            Initial conditions of the engine state.

        offset
            Number of skipped elements. Offset is calculated as: ``num_to_skip`` [0]+ ``num_to_skip`` [1]*2\ :sup:`64` + ``num_to_skip`` [2]\* 2\ :sup:`128` + … + ``num_to_skip`` [``n``-1]\*2\ :sup:`64` \*(``n``-1).

**Parent topic:** :ref:`onemath_device_rng_engines`
