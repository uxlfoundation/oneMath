.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_stats_overview:

Summary Statistics
==================

.. rubric:: Definitions

The oneMath Summary Statistics domains consists of:

   * Dataset structure. The structure consolidates the information of a multi-dimensional dataset (see detailed description in :ref:`onemath_stats_dataset`).
   * Computation routines. The routines are represented as free functions (see detailed description for each routine in :ref:`onemath_stats_routines`):

      *  Raw and central sums / moments up to the fourth order
      *  Variation coefficient
      *  Skewness and excess kurtosis (further referred as kurtosis)
      *  Minimum and maximum

Refer to :ref:`onemath_stats_usage_model`.

.. toctree::
	:hidden:

	onemath_stats_usage_model.rst
	onemath_stats_dataset.rst
	onemath_stats_routines.rst
	onemath_stats_service_routines.rst

**Parent topic:** :ref:`onemath_stats`