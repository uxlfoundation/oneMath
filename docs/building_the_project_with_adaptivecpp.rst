.. _building_the_project_with_adaptivecpp:

Building the Project with AdaptiveCpp
=====================================

.. _build_setup_with_adaptivecpp:

Environment Setup
#################

#. 
   Build and install AdaptiveCpp. For a detailed description of available
   AdaptiveCpp backends, their dependencies, and installation, see the
   `AdaptiveCpp installation readme
   <https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md#compilation-flows>`_.

#. 
   Clone this project. The root directory of the cloned repository will be
   referred to as ``<path to onemath>``.

#. 
   Download and install the `required dependencies
   <https://github.com/uxlfoundation/oneMath?tab=readme-ov-file#software-requirements>`_
   manually.

Build Commands
###############

In most cases, building oneMath is as simple as setting the compiler and
selecting the desired backends to build with.

On Linux (other OSes are not supported with the AdaptiveCpp compiler):

.. code-block:: bash

  # Inside <path to onemath>
  mkdir build && cd build
  cmake .. -DONEMATH_SYCL_IMPLEMENTATION=hipsycl   \ # Indicate that AdaptiveCpp is being used.
          -DENABLE_MKLGPU_BACKEND=False            \ # MKLGPU backend is not supported by AdaptiveCpp
          -DENABLE_<BACKEND_NAME>_BACKEND=True     \ # Enable backend(s) (optional)
          -DENABLE_<BACKEND_NAME_2>_BACKEND=True   \ # Multiple backends can be enabled at once.
          -DHIPSYCL_TARGETS=omp/;hip:gfx90a,gfx906 \ # Set target architectures depending on supported devices.
          -DBUILD_FUNCTIONAL_TESTS=False           \ # See section *Building the tests* for more on building tests. True by default.
          -DBUILD_EXAMPLES=False                   # Optional: True by default.
  cmake --build .
  cmake --install . --prefix <path_to_install_dir> # required to have full package structure

Backends should be enabled by setting ``-DENABLE_<BACKEND_NAME>_BACKEND=True`` for
each desired backend. By default, the ``MKLGPU`` and ``MKLCPU`` backends are
enabled, but ``MKLGPU`` must be disabled with AdaptiveCpp. The supported
backends for the compilers are given in the table at `oneMath supported
configurations table
<https://github.com/uxlfoundation/oneMath?tab=readme-ov-file#supported-configurations>`_,
and the CMake option names are given in the table below. Some backends may
require additional parameters to be set. See the relevant section below for
additional guidance. The target architectures must be specified with
``HIP_TARGETS``. See the `AdaptiveCpp documentation
<https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/using-hipsycl.md#adaptivecpp-targets-specification>`_.

If a backend library supports multiple domains (i.e. BLAS, RNG), it may be
desirable to only enable selected domains. For this, the ``TARGET_DOMAINS``
variable should be set. For further details, see :ref:`_build_target_domains`.

By default, the library also additionally builds examples and tests. These can
be disabled by setting the parameters ``BUILD_FUNCTIONAL_TESTS`` and
``BUILD_EXAMPLES`` to False. Building the functional tests may require additional
external libraries. See the section :ref:`building_and_running_tests` for more
information.

The most important supported build options are:

.. list-table::
   :header-rows: 1

   * - CMake Option
     - Supported Values
     - Default Value 
   * - ENABLE_MKLCPU_BACKEND
     - True, False
     - True      
   * - ENABLE_CUBLAS_BACKEND
     - True, False
     - False     
   * - ENABLE_CURAND_BACKEND
     - True, False
     - False     
   * - ENABLE_NETLIB_BACKEND
     - True, False
     - False     
   * - ENABLE_ROCBLAS_BACKEND
     - True, False
     - False     
   * - ENABLE_ROCRAND_BACKEND
     - True, False
     - False     
   * - ENABLE_MKLCPU_THREAD_TBB
     - True, False
     - True      
   * - BUILD_FUNCTIONAL_TESTS
     - True, False
     - True      
   * - BUILD_EXAMPLES
     - True, False
     - True      
   * - TARGET_DOMAINS (list)
     - blas, rng
     - All supported domains

Some additional build options are given in
:ref:`build_additional_options_dpcpp`.

Backends
########

.. _build_for_cuda_adaptivecpp:

Building for CUDA
~~~~~~~~~~~~~~~~~

The CUDA backends can be enabled with ``ENABLE_CUBLAS_BACKEND`` and
``ENABLE_CURAND_BACKEND``.

The target architecture must be set using the ``HIPSYCL_TARGETS`` parameter. For
example, to target a Nvidia A100 (Ampere architecture), set
``-DHIPSYCL_TARGETS=cuda:sm_80``, where the figure ``80`` corresponds to a CUDA
compute capability of 8.0. The correspondence between compute capabilities and
Nvidia GPU products is given on the `Nvidia website
<https://developer.nvidia.com/cuda-gpus>`_. Multiple architectures can be
enabled using a comma separated list. See the `AdaptiveCpp documentation
<https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/using-hipsycl.md#adaptivecpp-targets-specification>`_.

No additional parameters are required for using CUDA libraries. In most cases,
the CUDA libraries should be found automatically by CMake.

.. _build_for_rocm_adaptivecpp:

Building for ROCm
~~~~~~~~~~~~~~~~~

The ROCm backends can be enabled with ``ENABLE_ROCBLAS_BACKEND`` and
``ENABLE_ROCRAND_BACKEND``.

The target architecture must be set using the ``HIPSYCL_TARGETS`` parameter. See
the `AdaptiveCpp documentation
<https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/using-hipsycl.md#adaptivecpp-targets-specification>`_.
For example, to target the MI200 series, set ``-DHIPSYCL_TARGETS=hip:gfx90a``.
Multiple architectures can be enabled using a comma separated list. For example,
``-DHIPSYCL_TARGETS=hip:gfx906,gfx90a``, and multiple APIs with a semicolon
(``-DHIPSYCL_TARGETS=omp\;hip:gfx906,gfx90a``).

For common AMD GPU architectures, see the :ref:`build_for_ROCM_dpcpp` in the
DPC++ build guide.

.. _project_cleanup:

Project Cleanup
###############

Most use-cases involve building the project without the need to clean up the
build directory. However, if you wish to clean up the build directory, you can
delete the ``build`` folder and create a new one. If you wish to clean up the
build files but retain the build configuration, following commands will help you
do so.

.. code-block:: sh

  # If you use "GNU/Unix Makefiles" for building,
  make clean

  # If you use "Ninja" for building
  ninja -t clean
