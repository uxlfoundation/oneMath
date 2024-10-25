.. _building_and_running_tests:

Building and Running Tests
==========================

The functional tests are enabled by default, and can be enabled/disabled
with the CMake build parameter ``-DBUILD_FUNCTIONAL_TESTS=True/False``. Only
the tests relevant to the enabled backends and target domains will be built.

Building tests for BLAS and LAPACK domains requires additional libraries for
reference.

* BLAS: Requires a reference BLAS library.
* LAPACK: Requires a reference LAPACK library.

For both BLAS and LAPACK, shared libraries supporting both 32 and 64 bit
indexing are required.

A reference LAPACK implementation (including BLAS) can be built as the
following:

.. code-block:: bash

  git clone https://github.com/Reference-LAPACK/lapack.git 
  cd lapack; mkdir -p build; cd build 
  cmake -DCMAKE_INSTALL_PREFIX=~/lapack -DCBLAS=True -DLAPACK=True -DLAPACKE=True -DBUILD_INDEX64=True -DBUILD_SHARED_LIBS=True ..
  cmake --build . -j --target install 
  cmake -DCMAKE_INSTALL_PREFIX=~/lapack -DCBLAS=True -DLAPACK=True -DLAPACKE=True -DBUILD_INDEX64=False -DBUILD_SHARED_LIBS=True ..
  cmake --build . -j --target install

and then used in oneMKL by setting ``-REF_BLAS_ROOT=/path/to/lapack/install``
and ``-DREF_LAPACK_ROOT=/path/to/lapack/install``.

You can re-run tests without re-building the entire project.

To run the tests, either run test binaries individually, or use ``ctest`` CMake test driver program.

.. code-block:: bash

  # Run all tests
  ctest
  # Run only Gpu specific tests
  ctest -R Gpu
  # Exclude Cpu tests
  ctest -E Cpu

For more ``ctest`` options, refer to `ctest manual page <https://cmake.org/cmake/help/v3.13/manual/ctest.1.html>`_.

