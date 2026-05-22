.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_arch_other:

Other Features
----------------
This section covers all other features in the design of oneMath architecture.


.. _onemath_spec_current_version:

Specification Version and Compliance
+++++++++++++++++++++++++++++++++++++++++++++

Each oneMath domain must define a preprocessor macro to represent the version of the specification that the implementation is compliant with.

The macros for each domain are listed as follows:

  | ONEMATH_BLAS_SPEC_VERSION
  | ONEMATH_LAPACK_SPEC_VERSION
  | ONEMATH_SPBLAS_SPEC_VERSION
  | ONEMATH_DFT_SPEC_VERSION
  | ONEMATH_RNG_SPEC_VERSION
  | ONEMATH_STATS_SPEC_VERSION
  | ONEMATH_VM_SPEC_VERSION

The specification version can be created by appending all digits of the specification version in the format of <MAJOR><MINOR>. MINOR version always uses two digits. This version can be used to check the compatibility of the implementation with the specification version. Note that the revision is not included here because it reflects changes only for the specification document without affecting the implementation. If the implementation is not compliant with any release of the specification, then the macro must have a numerical value of `001`. Versions between `001` and `100` are reserved for alpha/beta support of the specification in oneMKL and should only be used for special cases. A preprocessor may evaluate an undefined macro to `0`, so `000` should not be used.

Version Example

  | oneAPI 1.1 rev 1 will be represented as a numerical value of 101
  | oneAPI 1.2 rev 1 will be represented as a numerical value of 102
  | oneAPI 1.2 rev 2 will be represented as a numerical value of 102

Macro Example

.. code-block:: c

  // For oneAPI 1.2 rev 1
  #define ONEMATH_BLAS_SPEC_VERSION 102

  // For oneAPI 1.2 rev 2
  #define ONEMATH_DFT_SPEC_VERSION 102

  // For oneAPI 1.3 rev 1
  #define ONEMATH_VM_SPEC_VERSION 103

Versioning details are defined here: https://github.com/uxlfoundation/oneAPI-spec/blob/main/doc/versioning.rst#provisional-versions

.. _onemath_pre_post_conditions:

Pre/Post Condition Checking
+++++++++++++++++++++++++++++++++++++++

The individual oneMath computational routines will define any preconditions and postconditions and will define in this specification any specific checks or verifications that should be enabled for all implementations.


