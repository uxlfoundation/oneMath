.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _oneMath-section:

=======
oneMath
=======

The |onemath_full_name| (oneMath) defines a set of fundamental mathematical routines for use in high-performance computing and other applications. As part of oneAPI, oneMath is designed to allow execution on a wide variety of computational devices: CPUs, GPUs, FPGAs, and other accelerators. The functionality is subdivided into several domains: dense linear algebra, sparse linear algebra, discrete Fourier transforms, random number generators and vector math.


The general assumptions, design features and requirements for the oneMath library and host-to-device computational routines will be described in :ref:`onemath_architecture`. 
The individual domains and their APIs are described in :ref:`onemath_domains`.  
Other design considerations that are not necessarily part of the oneMath specification but that are worth mentioning will be discussed in :ref:`onemath_appendix`.


.. toctree::
   :maxdepth: 2

   architecture/architecture.rst
   domains/domains.rst
   appendix/appendix.rst
