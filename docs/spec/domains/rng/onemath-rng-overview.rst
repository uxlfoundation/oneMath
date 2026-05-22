.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_overview:

Random Number Generators (RNG)
==============================

.. rubric:: Definitions


The pseudo-random number generator is defined by a structure(:math:`S`, :math:`\mu`, :math:`f`, :math:`U`, :math:`g`), where:

   -  :math:`S` is a finite set of states (the state space)
   -  :math:`\mu` is a probability distribution on :math:`S` for the initial state
      (or seed) :math:`s_0`
   -  :math:`f:S \rightarrow S` is the transition function
   -  :math:`U` – a finite set of output symbols
   -  :math:`g:S \rightarrow U` an output function


The generation of random numbers is as follows:

   #. Generate the initial state (called the seed) :math:`s_0` according to :math:`\mu` and compute :math:`u_0 = g(s_0)`.
   #. Iterate for :math:`i = 1 , ... , ` :math:`s_i = f(s_{i-1})` and :math:`u_i = g(s_i)`. Output values :math:`u_i` are the so-called random numbers produced by the PRNG.


In computational statistics, random variate generation is usually made in two steps:

   #. Generating imitations of independent and identically distributed
      (i.i.d.) random variables having the uniform distribution over the
      interval :math:`(0, 1)`
   #. Applying transformations to these i.i.d. :math:`U(0, 1)` random variates
      in order to generate (or imitate) random variates and random
      vectors from arbitrary distributions.


.. rubric:: Execution Models

RNG domain supports two execution models:

   #. :ref:`Host API<onemath_rng_manual_offload_routines>`, which is aligned with the rest of oneMath domains 
      :ref:`oneMath domains<onemath_member_functions>`.
   #. :ref:`Device API<onemath_device_rng_routines>`, which is specific for RNG domain. These APIs 
      are designed to be callable from the User's kernels as well as Host code.


.. toctree::
   :hidden:

   host_api/rng-host-routines.rst
   device_api/device-routines.rst

**Parent topic:** :ref:`onemath_rng`
