.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_distributions:

Host Distributions
==================


.. container::


   oneMath RNG routines are used to generate random
   numbers with different types of distribution. Each function group is
   introduced below by the type of underlying distribution and contains
   a short description of its functionality, as well as specifications
   of the call sequence and the explanation of input and output
   parameters. `Table Continuous Distribution Generators`_
   and `Table Discrete Distribution Generators`_ list the
   random number generator routines with data types and output
   distributions, and sets correspondence between data types of the
   generator routines and the basic random number generators.


   _`Table Continuous Distribution Generators`

   .. container:: tablenoborder


      .. list-table::
         :header-rows: 1

         * -     Routine
           -     Description
         * -           \ :ref:`onemath_rng_uniform_continuous`\
           -     Uniform continuous distribution on the interval       [``a,b``)
         * -           \ :ref:`onemath_rng_gaussian`\
           -     Normal (Gaussian) distribution
         * -           \ :ref:`onemath_rng_exponential`\
           -     Exponential distribution
         * -           \ :ref:`onemath_rng_laplace`\
           -     Laplace distribution (double exponential distribution)
         * -           \ :ref:`onemath_rng_weibull`\
           -     Weibull distribution
         * -           \ :ref:`onemath_rng_cauchy`\
           -     Cauchy distribution
         * -           \ :ref:`onemath_rng_rayleigh`\
           -     Rayleigh distribution
         * -           \ :ref:`onemath_rng_lognormal`\
           -     Lognormal distribution
         * -           \ :ref:`onemath_rng_gumbel`\
           -     Gumbel (extreme value) distribution
         * -           \ :ref:`onemath_rng_gamma`\
           -     Gamma distribution
         * -           \ :ref:`onemath_rng_beta`\
           -     Beta distribution
         * -           \ :ref:`onemath_rng_chi_square`\
           -     Chi-Square distribution
         * -           \ :ref:`onemath_rng_gaussian_mv`\
           -     Normal Multivariate (Gaussian Multivariate) distribution


   _`Table Discrete Distribution Generators`

   .. container:: tablenoborder


      .. list-table::
         :header-rows: 1

         * -     Type of Distribution
           -     Description
         * -           \ :ref:`onemath_rng_uniform_discrete`\
           -     Uniform discrete distribution on the interval       [``a,b``)
         * -           \ :ref:`onemath_rng_uniform_bits`\
           -     Uniformly distributed bits in 32/64-bit chunks
         * -           \ :ref:`onemath_rng_bits`\
           -     Bits of underlying BRNG integer recurrence
         * -           \ :ref:`onemath_rng_bernoulli`\
           -     Bernoulli distribution
         * -           \ :ref:`onemath_rng_geometric`\
           -     Geometric distribution
         * -           \ :ref:`onemath_rng_binomial`\
           -     Binomial distribution
         * -           \ :ref:`onemath_rng_hypergeometric`\
           -     Hypergeometric distribution
         * -           \ :ref:`onemath_rng_poisson`\
           -     Poisson distribution
         * -           \ :ref:`onemath_rng_poisson_v`\
           -     Poisson distribution with varying mean
         * -           \ :ref:`onemath_rng_negative_binomial`\
           -     Negative binomial distribution, or Pascal distribution
         * -           \ :ref:`onemath_rng_multinomial`\
           -     Multinomial distribution




   .. container:: section

      .. rubric:: Modes of random number generation

      The library provides two modes of random number generation,
      accurate and fast. Accurate generation mode is intended for
      applications that are highly demanding to accuracy of
      calculations. When used in this mode, the generators produce
      random numbers lying completely within the definitional domain for all
      values of the distribution parameters. For example, random numbers
      obtained from the generator of continuous distribution that is
      uniform on interval [``a``,\ ``b``] belong to this interval
      irrespective of what ``a`` and ``b`` values may be. Fast mode
      provides high performance generation and also guarantees that
      generated random numbers belong to the definitional domain except
      for some specific values of distribution parameters. The
      generation mode is set by specifying the relevant value of the method
      parameter in generator routines. The list of distributions that
      support accurate mode of generation is given in the table below.


    _`Table Distribution Generators with Accurate Method`
   .. container:: tablenoborder


      .. list-table::
         :header-rows: 1

         * -     Distribution
           -     Method
         * -           \ :ref:`onemath_rng_uniform_continuous`\
           -     `oneapi::math::rng::unform_method::accurate`
         * -           \ :ref:`onemath_rng_exponential`\
           -     `oneapi::math::rng::exponential_method::icdf_accurate`
         * -           \ :ref:`onemath_rng_weibull`\
           -     `oneapi::math::rng::weibull_method::icdf_accurate`
         * -           \ :ref:`onemath_rng_rayleigh`\
           -     `oneapi::math::rng::rayleigh_method::icdf_accurate`
         * -           \ :ref:`onemath_rng_lognormal`\
           -     `oneapi::math::rng::lognormal_method::box_muller2_accurate`, `oneapi::math::rng::lognormal_method::icdf_accurate`
         * -           \ :ref:`onemath_rng_gamma`\
           -     `oneapi::math::rng::gamma_method::marsaglia_accurate`
         * -           \ :ref:`onemath_rng_beta`\
           -     `oneapi::math::rng::beta_method::cja_accurate`
               
   **Parent topic:** :ref:`onemath_rng_manual_offload_routines`



.. toctree::
    :hidden:

    distributions-template-parameter-onemath-rng-method-values
    onemath-rng-uniform-continuous
    onemath-rng-gaussian
    onemath-rng-exponential
    onemath-rng-laplace
    onemath-rng-weibull
    onemath-rng-cauchy
    onemath-rng-rayleigh
    onemath-rng-lognormal
    onemath-rng-gumbel
    onemath-rng-gamma
    onemath-rng-beta
    onemath-rng-chi_square
    onemath-rng-gaussian_mv
    onemath-rng-uniform-discrete
    onemath-rng-uniform_bits
    onemath-rng-bits
    onemath-rng-bernoulli
    onemath-rng-geometric
    onemath-rng-binomial
    onemath-rng-hypergeometric
    onemath-rng-poisson
    onemath-rng-poisson_v
    onemath-rng-negbinomial
    onemath-rng-multinomial
