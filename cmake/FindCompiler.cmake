#===============================================================================
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
#
# SPDX-License-Identifier: Apache-2.0
#===============================================================================

include_guard()
include(CheckCXXCompilerFlag)
include(FindPackageHandleStandardArgs)
check_cxx_compiler_flag("-fsycl" is_dpcpp)

if(is_dpcpp)
  # Workaround for internal compiler error during linking if -fsycl is used
  get_filename_component(SYCL_BINARY_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
  find_library(SYCL_LIBRARY NAMES sycl
    PATHS
      "${SYCL_BINARY_DIR}/lib"
      "${SYCL_BINARY_DIR}/../lib"
      "${SYCL_BINARY_DIR}/../../lib"
      ENV LIBRARY_PATH
      ENV PATH
  )
  if(NOT SYCL_LIBRARY)
    message(FATAL_ERROR "SYCL library is not found in ${SYCL_BINARY_DIR}/../lib, PATH, and LIBRARY_PATH")
  endif()

  add_library(ONEMATH::SYCL::SYCL INTERFACE IMPORTED)
  if(UNIX)
    set(UNIX_INTERFACE_COMPILE_OPTIONS -fsycl)
    set(UNIX_INTERFACE_LINK_OPTIONS -fsycl)
    # Check if the Nvidia target is supported. PortFFT uses this for choosing default configuration.
    check_cxx_compiler_flag("-fsycl -fsycl-targets=nvptx64-nvidia-cuda" dpcpp_supports_nvptx64)

    # Assemble the SYCL offload target triples for every enabled GPU backend.
    # DPC++ only honors the *last* -fsycl-targets flag on the command line, so
    # all target triples must be passed together as a single comma-separated
    # list. Previously each vendor backend appended its own -fsycl-targets flag
    # (here and in the individual backend CMake files); in a multi-vendor build
    # all but the last triple were then silently dropped, so the SYCL device
    # kernels of the other vendor(s) were missing at runtime (see issue #708).
    set(ONEMATH_SYCL_TARGET_TRIPLES "")
    set(ONEMATH_SYCL_TARGET_ARCH_OPTIONS "")

    if(ENABLE_CUBLAS_BACKEND OR ENABLE_CURAND_BACKEND OR ENABLE_CUSOLVER_BACKEND
        OR ENABLE_CUFFT_BACKEND OR ENABLE_CUSPARSE_BACKEND)
      list(APPEND ONEMATH_SYCL_TARGET_TRIPLES "nvptx64-nvidia-cuda")
      if(DEFINED CUDA_TARGETS AND NOT "${CUDA_TARGETS}" STREQUAL "")
        list(APPEND ONEMATH_SYCL_TARGET_ARCH_OPTIONS
          -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=${CUDA_TARGETS})
      endif()
    endif()

    if(ENABLE_ROCBLAS_BACKEND OR ENABLE_ROCRAND_BACKEND OR ENABLE_ROCSOLVER_BACKEND
        OR ENABLE_ROCFFT_BACKEND OR ENABLE_ROCSPARSE_BACKEND)
      list(APPEND ONEMATH_SYCL_TARGET_TRIPLES "amdgcn-amd-amdhsa")
      list(APPEND ONEMATH_SYCL_TARGET_ARCH_OPTIONS
        -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${HIP_TARGETS})
    endif()

    if(ONEMATH_SYCL_TARGET_TRIPLES)
      # Preserve the Intel GPU (spir64) device image when an Intel oneMKL GPU
      # backend is enabled alongside a CUDA/HIP backend; otherwise it would be
      # dropped from the combined -fsycl-targets list.
      if(ENABLE_MKLGPU_BACKEND)
        list(APPEND ONEMATH_SYCL_TARGET_TRIPLES "spir64")
      endif()
      list(JOIN ONEMATH_SYCL_TARGET_TRIPLES "," ONEMATH_SYCL_TARGETS_ARG)
      list(APPEND UNIX_INTERFACE_COMPILE_OPTIONS
        -fsycl-targets=${ONEMATH_SYCL_TARGETS_ARG} -fsycl-unnamed-lambda
        ${ONEMATH_SYCL_TARGET_ARCH_OPTIONS})
      list(APPEND UNIX_INTERFACE_LINK_OPTIONS
        -fsycl-targets=${ONEMATH_SYCL_TARGETS_ARG}
        ${ONEMATH_SYCL_TARGET_ARCH_OPTIONS})
    endif()

    set_target_properties(ONEMATH::SYCL::SYCL PROPERTIES
      INTERFACE_COMPILE_OPTIONS "${UNIX_INTERFACE_COMPILE_OPTIONS}"
      INTERFACE_LINK_OPTIONS "${UNIX_INTERFACE_LINK_OPTIONS}"
      INTERFACE_LINK_LIBRARIES ${SYCL_LIBRARY})
  else()
    set_target_properties(ONEMATH::SYCL::SYCL PROPERTIES
      INTERFACE_COMPILE_OPTIONS "-fsycl"
      INTERFACE_LINK_LIBRARIES ${SYCL_LIBRARY})
  endif()

  if(ENABLE_ROCBLAS_BACKEND OR ENABLE_ROCRAND_BACKEND OR ENABLE_ROCSOLVER_BACKEND OR ENABLE_ROCSPARSE_BACKEND)
    # Allow find_package(HIP) to find the correct path to libclang_rt.builtins.a
    # HIP's CMake uses the command `${HIP_CXX_COMPILER} -print-libgcc-file-name --rtlib=compiler-rt` to find this path.
    # This can print a non-existing file if the compiler used is icpx.
    if(NOT HIP_CXX_COMPILER)
      find_path(HIP_CXX_COMPILER clang++
        HINTS ENV HIPROOT ENV ROCM_PATH
      )
    endif()
  endif()

endif(is_dpcpp)
