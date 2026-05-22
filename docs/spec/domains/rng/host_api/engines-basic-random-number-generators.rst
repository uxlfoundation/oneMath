.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_engines_basic_random_number_generators:

Host Engines (Basic Random Number Generators)
=============================================


.. container::


   oneMath RNG provides pseudorandom, quasi-random, and non-deterministic
   random number generators for Data Parallel C++:


   .. container:: tablenoborder


      .. list-table::
         :header-rows: 1

         * -  Routine
           -  Description
         * -                      \ :ref:`onemath_rng_default_engine`\
           -     The default random engine
         * -                      \ :ref:`onemath_rng_mrg32k3a`\
           -     The combined multiple recursive pseudorandom number       generator      ``MRG32k3a``\ :ref:`[L'Ecuyer99a] <onemath_rng_bibliography>`\
         * -                      \ :ref:`onemath_rng_philox4x32x10`\
           -     Philox4x32-10 counter-based pseudorandom number       generator with a period of      2\ :sup:`128`\ ``PHILOX4X32X10``\ :ref:`[Salmon11] <onemath_rng_bibliography>`\
         * -                      \ :ref:`onemath_rng_mcg31m1`\
           -     The 31-bit multiplicative congruential pseudorandom       number generator MCG(1132489760, 231 -1)      :ref:`[L'Ecuyer99] <onemath_rng_bibliography>`\
         * -                      \ :ref:`onemath_rng_r250`\
           -     The 32-bit generalized feedback shift register       pseudorandom number generator      ``GFSR(250,103)``\ :ref:`[Kirkpatrick81] <onemath_rng_bibliography>`\
         * -                      \ :ref:`onemath_rng_mcg59`\
           -     The 59-bit multiplicative congruential pseudorandom       number generator ``MCG(13``\ :sup:`13`\ ``, 2``\ :sup:`59`) from      NAG Numerical Libraries      :ref:`[NAG] <onemath_rng_bibliography>`\
         * -                      \ :ref:`onemath_rng_wichmann_hill`\
           -     Wichmann-Hill pseudorandom number generator (a set of       273 basic generators) from NAG Numerical Libraries      :ref:`[NAG] <onemath_rng_bibliography>`\
         * -                      \ :ref:`onemath_rng_mt19937`\
           -     Mersenne Twister pseudorandom number generator       ``MT19937``\ :ref:`[Matsumoto98] <onemath_rng_bibliography>`      with period length 2\ :sup:`19937`-1 of the produced sequence
         * -                      \ :ref:`onemath_rng_mt2203`\
           -     Set of Mersenne Twister pseudorandom number       generators      ``MT2203``\ :ref:`[Matsumoto98] <onemath_rng_bibliography>`,      :ref:`[Matsumoto00] <onemath_rng_bibliography>`.      Each of them generates a sequence of period length equal to      2\ :sup:`2203`-1. Parameters of the generators provide mutual      independence of the corresponding sequences.
         * -                      \ :ref:`onemath_rng_sfmt19937`\
           -     SIMD-oriented Fast Mersenne Twister pseudorandom       number generator      ``SFMT19937``\ :ref:`[Saito08] <onemath_rng_bibliography>`      with a period length equal to 2\ :sup:`19937`-1 of the produced      sequence.
         * -                      \ :ref:`onemath_rng_sobol`\
           -     Sobol quasi-random number generator       :ref:`[Sobol76] <onemath_rng_bibliography>`,      :ref:`[Bratley88] <onemath_rng_bibliography>`,      which works in arbitrary dimension.
         * -                      \ :ref:`onemath_rng_niederreiter`\
           -     Niederreiter quasi-random number generator       :ref:`[Bratley92] <onemath_rng_bibliography>`,      which works in arbitrary dimension.
         * -                      \ :ref:`onemath_rng_ars5`\
           -     ARS-5 counter-based pseudorandom number generator with       a period of 2\ :sup:`128`, which uses instructions from the AES-NI      set      ``ARS5``\ :ref:`[Salmon11] <onemath_rng_bibliography>`.
         * -                      \ :ref:`onemath_rng_nondeterministic`\
           -     Non-deterministic random number generator \




   For some basic generators, oneMath RNG provides two methods of
   creating independent states in multiprocessor computations, which are
   the leapfrog method and the block-splitting method. These sequence
   splitting methods are also useful in sequential Monte Carlo. The
   description of these functions can be found in the :ref:`onemath_rng_service_routines` section.


   In addition, the MT2203 pseudorandom number generator is a set of
   generators designed to create independent random
   sequences, which might be used in parallel Monte Carlo simulations.
   Another generator that has the same feature is Wichmann-Hill. It
   allows creating up to 273 independent random streams. The properties
   of the generators designed for parallel computations are discussed in
   detail in
   [:ref:`Coddington94 <onemath_rng_bibliography>`].


   **Parent topic:** :ref:`onemath_rng_manual_offload_routines`


.. container::


   -  :ref:`onemath_rng_default_engine`
      The default random engine (implementation defined)
   -  :ref:`onemath_rng_mrg32k3a`
      The combined multiple recursive pseudorandom number generator
      MRG32k3a [ L'Ecuyer99a]
   -  :ref:`onemath_rng_philox4x32x10`
      A Philox4x32-10 counter-based pseudorandom number generator.
      [Salmon11].
   -  :ref:`onemath_rng_mcg31m1`
      The 31-bit multiplicative congruential pseudorandom number
      generator MCG(1132489760, 231 -1) [L'Ecuyer99]
   -  :ref:`onemath_rng_mcg59`
      The 59-bit multiplicative congruential pseudorandom number
      generator MCG(1313, 259) from NAG Numerical Libraries [NAG].
   -  :ref:`onemath_rng_r250`
      The 32-bit generalized feedback shift register pseudorandom number
      generator GFSR(250,103)[Kirkpatrick81].
   -  :ref:`onemath_rng_wichmann_hill`
      Wichmann-Hill pseudorandom number generator (a set of 273 basic
      generators) from NAG Numerical Libraries [NAG].
   -  :ref:`onemath_rng_mt19937`
      Mersenne Twister pseudorandom number generator MT19937
      [Matsumoto98] with period length 2\ :sup:`19937`-1 of the produced
      sequence.
   -  :ref:`onemath_rng_sfmt19937`
      SIMD-oriented Fast Mersenne Twister pseudorandom number generator
      SFMT19937 [Saito08] with a period length equal to
      2\ :sup:`19937`-1 of the produced sequence.
   -  :ref:`onemath_rng_mt2203`
      Set of Mersenne Twister pseudorandom number generators MT2203
      [Matsumoto98], [Matsumoto00]. Each of them generates a sequence of
      period length equal to 2\ :sup:`2203`-1. Parameters of the
      generators provide mutual independence of the corresponding
      sequences..
   -  :ref:`onemath_rng_ars5`
      ARS-5 counter-based pseudorandom number generator with a period of
      2\ :sup:`128`, which uses instructions from the AES-NI set
      ARS5[Salmon11].
   -  :ref:`onemath_rng_sobol`
      Sobol quasi-random number generator [Sobol76], [Bratley88], which
      works in arbitrary dimension.
   -  :ref:`onemath_rng_niederreiter`
      Niederreiter quasi-random number generator [Bratley92], which
      works in arbitrary dimension.
   -  :ref:`onemath_rng_nondeterministic`
      Non-deterministic random number generator.

.. toctree::
    :hidden:

    onemath-rng-default_engine
    onemath-rng-mrg32k3a
    onemath-rng-philox4x32x10
    onemath-rng-mcg31m1
    onemath-rng-mcg59
    onemath-rng-r250
    onemath-rng-wichmann_hill
    onemath-rng-mt19937
    onemath-rng-sfmt19937
    onemath-rng-mt2203
    onemath-rng-ars5
    onemath-rng-sobol
    onemath-rng-niederreiter
    onemath-rng-nondeterministic
