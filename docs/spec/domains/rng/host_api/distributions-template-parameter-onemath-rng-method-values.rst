.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_distributions_template_parameter_onemath_rng_method_values:

Distributions Template Parameter Method
=======================================


.. container:: section

    .. list-table::
        :header-rows: 1

        * -  Method
          -  Distributions
          -  Math Description
        * -  ``uniform_method::standard``\         \ ``uniform_method::accurate``\
          -  \ ``uniform(s,d)``\       \ ``uniform(i)``\
          -     Standard method. ``uniform_method::standard_accurate`` supported for ``uniform(s, d)`` only.
        * -  ``gaussian_method::box_muller``
          -  \ ``gaussian``\
          -     Generates normally distributed random number x thru the pair of uniformly distributed numbers :math:`u_1` and :math:`u_2` according to the formula: :math:`x = \sqrt{-2lnu_1}\sin(2 \pi u_2)`
        * -  ``gaussian_method::box_muller2``
          -  \ ``gaussian``\       \ ``lognormal``\
          -     Generates normally distributed random numbers :math:`x_1` and :math:`x_2` thru the pair of uniformly distributed numbers :math:`u_1` and :math:`u_2` according to the formulas: \ :math:`x_1 = \sqrt{-2lnu_1}\sin{2\pi u_2}`\       \ :math:`x_2 = \sqrt{-2lnu_1}\cos{2\pi u_2}`\
        * -  ``gaussian_method::icdf``
          -  \ ``gaussian``\
          -     Inverse cumulative distribution function (ICDF) method.
        * -  \ ``exponential_method::icdf``\         \ ``exponential_method::icdf_accurate``\
          -  \ ``exponential``\
          -     Inverse cumulative distribution function (ICDF) method.
        * -  \ ``weibull_method::icdf``\         \ ``weibull_method::icdf_accurate``\
          -  \ ``weibull``\
          -     Inverse cumulative distribution function (ICDF) method.
        * -  \ ``cauchy_method::icdf``\
          -  \ ``cauchy``\
          -     Inverse cumulative distribution function (ICDF) method.
        * -  \ ``rayleigh_method::icdf``\         \ ``rayleigh_method::icdf_accurate``\
          -  \ ``rayleigh``\
          -     Inverse cumulative distribution function (ICDF) method.
        * -  \ ``bernoulli_method::icdf``\
          -  \ ``bernoulli``\
          -     Inverse cumulative distribution function (ICDF) method.
        * -  \ ``geometric_method::icdf``\
          -  \ ``geometric``\
          -     Inverse cumulative distribution function (ICDF) method.
        * -  \ ``gumbel_method::icdf``\
          -  \ ``gumbel``\
          -     Inverse cumulative distribution function (ICDF) method.
        * -  \ ``lognormal_method::icdf``\         \ ``lognormal_method::icdf_accurate``\
          -  \ ``lognormal``\
          -     Inverse cumulative distribution function (ICDF) method.
        * -  \ ``lognormal_method::box_muller2``\         \ ``lognormal_method::box_muller2_accurate``\
          -  \ ``lognormal``\
          -     Generated normally distributed random numbers :math:`x_1` and :math:`x_2` by box_muller2 method are converted to lognormal distribution.
        * -  \ ``gamma_method::marsaglia``\         \ ``gamma_method::marsaglia_accurate``\
          -     \ ``gamma``\
          -     For :math:`\alpha > 1`, a gamma distributed random number is generated as a cube of properly scaled normal random number; for :math:`0.6 \leq \alpha < 1`, a gamma distributed random number is generated using rejection from Weibull distribution; for :math:`\alpha < 0.6`, a gamma distributed random number is obtained using transformation of exponential power distribution; for :math:`\alpha = 1`, gamma distribution is reduced to exponential distribution.
        * -  \ ``beta_method::cja``\         \ ``beta_method::cja_accurate``\
          -     \ ``beta``\
          -     Cheng-Jonhnk-Atkinson method. For :math:`min(p, q) > 1`, Cheng method is used; for :math:`min(p, q) < 1`, Johnk method is used, if :math:`q + K*p2 + C \leq 0 (K = 0.852..., C=-0.956...)` otherwise, Atkinson switching algorithm is used; for :math:`max(p, q) < 1`, method of Johnk is used; for :math:`min(p, q) < 1, max(p, q)> 1`, Atkinson switching algorithm is used (CJA stands for Cheng, Johnk, Atkinson); for :math:`p = 1` or :math:`q = 1`, inverse cumulative distribution function method is used; for :math:`p = 1` and :math:`q = 1`, beta distribution is reduced to uniform distribution.
        * -  ``chi_square_method::gamma_based``
          -     \ ``chi_square``\
          -     (most common):       If :math:`\nu \ge 17` or :math:`\nu` is odd and :math:`5 \leq \nu \leq 15`, a chi-square distribution is reduced to a Gamma distribution with these parameters: Shape :math:`\alpha = \nu / 2`\ Offset :math:`a = 0`\       \Scale factor :math:`\beta = 2`\       The random numbers of the Gamma distribution are generated.
        * -  ``binomial_method::btpe``
          -     \ ``binomial``\
          -     Acceptance/rejection method for :math:`ntrial * min(p, 1 - p) \ge 30` with decomposition into four regions: two parallelograms, triangle, left exponential tail, right exponential tail.
        * -  ``poisson_method::ptpe``
          -     \ ``poisson``\
          -     Acceptance/rejection method for :math:`\lambda \ge 27` with decomposition into four regions: two parallelograms, triangle, left exponential tail, right exponential tail.
        * -  ``poisson_method::gaussian_icdf_based``
          -           \ ``poisson``\
          -     for :math:`\lambda \ge 1`, method based on Poisson inverse CDF approximation by Gaussian inverse CDF; for :math:`\lambda < 1`, table lookup method is used.
        * -  ``poisson_v_method::gaussian_icdf_based``
          -           \ ``poisson_v``\
          -     for :math:`\lambda \ge 1`, method based on Poisson inverse CDF approximation by Gaussian inverse CDF; for :math:`\lambda < 1`, table lookup method is used.
        * -  ``hypergeometric_method::h2pe``
          -     \ ``hypergeometric``\
          -     Acceptance/rejection method for large mode of distribution with decomposition into three regions: rectangular, left exponential tail, right exponential tail.
        * -  ``negative_binomial_method::nbar``
          -     \ ``negative_binomial``\
          -     Acceptance/rejection method for: :math:`\frac{(a - 1)(1 - p)}{p} \ge 100` with decomposition      into five regions: rectangular, 2 trapezoid, left exponential tail, right exponential tail.
        * -  ``multinomial_method::poisson_icdf_based``
          -     \ ``multinomial``\
          -     Multinomial distribution with parameters :math:`m, k`, and a probability vector :math:`p`. Random numbers of the multinomial distribution are generated by Poisson Approximation method.
        * -  ``gaussian_mv_method::box_muller``
          -     \ ``gaussian_mv``\
          -     BoxMuller method for gaussian_mv method.
        * -  ``gaussian_mv_method::box_muller2``
          -     \ ``gaussian_mv``\
          -     BoxMuller2 method for gaussian_mv method.
        * -  ``gaussian_mv_method::icdf``
          -     \ ``gaussian_mv``\
          -     Inverse cumulative distribution function (ICDF) method.

**Parent
topic:** :ref:`onemath_rng_distributions`
