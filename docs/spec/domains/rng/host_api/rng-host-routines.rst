.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_rng_manual_offload_routines:

Random Number Generators Host Routines
======================================

.. rubric:: Structure

RNG domain contains two classes types:

   -  Engines (basic random number generators) classes, which holds
      the state of generator and is a source of independent and identically distributed random variables.
      Refer to :ref:`onemath_rng_engines_basic_random_number_generators`
      for a detailed description.
   -  Distribution classes templates (transformation classes) for different types of statistical
      distributions, for example, uniform, normal (Gaussian), binomial,
      etc. These classes contain all of the distribution’s parameters
      (including generation method). Refer to :ref:`onemath_rng_distributions` for
      a detailed description of the distributions.

The RNG domain also contains two types of free functions:

   -  Generation routines. The current routines are used to obtain random
      numbers from a given engine with proper statistics defined by a
      given distribution. Refer to the :ref:`onemath_rng_generate_routine`
      section for a detailed description.
   -  Service routines. The routines are used to modify the engine state. Refer to :ref:`onemath_rng_service_routines` for a
      description of these routines.


Engine classes work with both generation and service routines. Distribution classes are used in
generation routines only. Refer to the :ref:`onemath_rng_usage_model`
section for the description of typical RNG scenario.

.. toctree::
   :hidden:

   onemath-rng-usage-model.rst
   generate-routine.rst
   engines-basic-random-number-generators.rst
   service-routines.rst
   distributions.rst
   ../bibliography.rst

**Parent topic:** :ref:`onemath_rng_overview`