.. _building_the_project_with_openblas:

Building the Project with OpenBLAS
==================================

This page describes building oneMath with the OpenBLAS backend using
different SYCL implementations:

- DPC++ (Intel oneAPI)
- AdaptiveCpp

Environment Setup
#################

#. 
   Install and build OpenBLAS. The installation prefix will be referred to as
   ``<path to openblas>``.

#. 
   Clone this project. The root directory will be referred to as
   ``<path to onemath>``.

#. 
   (Optional) Install a reference BLAS/LAPACK implementation for functional
   testing. The installation prefix will be referred to as
   ``<path to reference lapack install>``.

#. 
   Ensure required shared libraries are visible at runtime:

.. code-block:: bash

  export LD_LIBRARY_PATH=<path to system libraries>:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=<path to openblas>/lib:$LD_LIBRARY_PATH

  # Optional (only if functional tests are enabled)
  export LD_LIBRARY_PATH=<path to reference lapack install>/lib:$LD_LIBRARY_PATH


Building with DPC++
###################

If using Intel oneAPI DPC++, load the compiler environment:

.. code-block:: bash

  source <path to dpcpp environment script>

Build commands:

.. code-block:: bash

  cd <path to onemath>
  mkdir build && cd build

  cmake ..                                                        \
        -DCMAKE_BUILD_TYPE=Release                                \
        -DCMAKE_C_COMPILER=<path to icx>                          \
        -DCMAKE_CXX_COMPILER=<path to icpx>                       \
        -DONEMATH_SYCL_IMPLEMENTATION=dpc++                       \
        -DENABLE_MKLCPU_BACKEND=False                             \
        -DENABLE_MKLGPU_BACKEND=False                             \
        -DENABLE_NETLIB_BACKEND=False                             \
        -DENABLE_OPENBLAS_BACKEND=True                            \
        -DOPENBLAS_DIR=<path to openblas>                         \
        -DBUILD_FUNCTIONAL_TESTS=True                             \
        -DBUILD_EXAMPLES=True                                     \
        -DCMAKE_INSTALL_PREFIX=<path to install directory>        \
        -DREF_BLAS_ROOT=<path to reference lapack install>        \
        -DREF_LAPACK_ROOT=<path to reference lapack install>

  make -j && make install 


Building with AdaptiveCpp
#########################

If using AdaptiveCpp, load the compiler environment:

.. code-block:: bash

  source <path to adaptivecpp environment script>

Build commands:

.. code-block:: bash

  cd <path to onemath>
  mkdir build && cd build

  cmake ..                                                        \
        -DCMAKE_BUILD_TYPE=Release                                \
        -DCMAKE_C_COMPILER=<path to clang>                        \
        -DCMAKE_CXX_COMPILER=<path to acpp>                       \
        -DONEMATH_SYCL_IMPLEMENTATION=adaptivecpp                 \
        -DACPP_TARGETS=omp                                        \
        -DENABLE_MKLCPU_BACKEND=False                             \
        -DENABLE_MKLGPU_BACKEND=False                             \
        -DENABLE_NETLIB_BACKEND=False                             \
        -DENABLE_OPENBLAS_BACKEND=True                            \
        -DOPENBLAS_DIR=<path to openblas>                         \
        -DBUILD_FUNCTIONAL_TESTS=True|False                       \
        -DBUILD_EXAMPLES=True                                     \
        -DCMAKE_INSTALL_PREFIX=<path to install directory>

 make -j && make install 


Common Build Options
####################

.. list-table::
   :header-rows: 1

   * - CMake Option
     - Supported Values
     - Description
   * - ONEMATH_SYCL_IMPLEMENTATION
     - dpc++, adaptivecpp
     - Selects the SYCL implementation
   * - ENABLE_OPENBLAS_BACKEND
     - True, False
     - Enables the OpenBLAS backend
   * - OPENBLAS_DIR
     - Path
     - Path to the OpenBLAS installation
   * - ENABLE_MKLCPU_BACKEND
     - True, False
     - Enables/disables MKL CPU backend
   * - ENABLE_MKLGPU_BACKEND
     - True, False
     - Enables/disables MKL GPU backend
   * - ENABLE_NETLIB_BACKEND
     - True, False
     - Enables/disables Netlib backend
   * - BUILD_FUNCTIONAL_TESTS
     - True, False
     - Enables functional tests
   * - BUILD_EXAMPLES
     - True, False
     - Builds example programs
   * - CMAKE_INSTALL_PREFIX
     - Path
     - Installation directory


Running Tests
#############

Running Test Binaries Directly
-----------------------------

The BLAS test executables are located in the ``bin`` directory inside the
build folder.

.. code-block:: bash

  cd <path to onemath>/build

  # Run BLAS runtime tests
  ./bin/test_main_blas_rt

  # Run BLAS compile-time tests
  ./bin/test_main_blas_ct


Notes
#####

* OpenBLAS is used as the CPU BLAS backend in this configuration.
* Ensure that OpenBLAS shared libraries are available via
  ``LD_LIBRARY_PATH`` or system linker configuration.
* AdaptiveCpp requires proper target configuration via ``ACPP_TARGETS``.
* Functional tests require a reference BLAS/LAPACK installation.
* Test binaries are generated only if functional tests are enabled.
