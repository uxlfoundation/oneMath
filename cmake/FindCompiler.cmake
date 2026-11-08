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

    if(ENABLE_CURAND_BACKEND OR ENABLE_CUSOLVER_BACKEND OR ENABLE_CUSPARSE_BACKEND)
      list(APPEND UNIX_INTERFACE_COMPILE_OPTIONS
        -fsycl-targets=nvptx64-nvidia-cuda -fsycl-unnamed-lambda)
      list(APPEND UNIX_INTERFACE_LINK_OPTIONS
        -fsycl-targets=nvptx64-nvidia-cuda)
    elseif(ENABLE_ROCBLAS_BACKEND OR ENABLE_ROCRAND_BACKEND
                OR ENABLE_ROCSOLVER_BACKEND OR ENABLE_ROCSPARSE_BACKEND)
      list(APPEND UNIX_INTERFACE_COMPILE_OPTIONS
        -fsycl-targets=amdgcn-amd-amdhsa -fsycl-unnamed-lambda 
	-Xsycl-target-backend --offload-arch=${HIP_TARGETS})
      list(APPEND UNIX_INTERFACE_LINK_OPTIONS
        -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend 
	--offload-arch=${HIP_TARGETS})
      # Recent DPC++/LLVM defaults to AMD code object ABI version 6, which is
      # only supported by ROCm 6.3+. When building against an older ROCm, fall
      # back to code object version 5 so the ROCm device library can be found.
      # See https://github.com/uxlfoundation/oneMath/issues/687
      if(NOT CMAKE_CXX_FLAGS MATCHES "-mcode-object-version")
        set(_onemath_rocm_version_file "")
        foreach(_onemath_rocm_root "$ENV{ROCM_PATH}" "$ENV{HIPROOT}" "/opt/rocm")
          if(_onemath_rocm_root AND EXISTS "${_onemath_rocm_root}/.info/version")
            set(_onemath_rocm_version_file "${_onemath_rocm_root}/.info/version")
            break()
          endif()
        endforeach()
        if(_onemath_rocm_version_file)
          file(READ "${_onemath_rocm_version_file}" _onemath_rocm_version)
          string(STRIP "${_onemath_rocm_version}" _onemath_rocm_version)
          if(_onemath_rocm_version MATCHES "^([0-9]+)\\.([0-9]+)")
            set(_onemath_rocm_ver "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
            if(_onemath_rocm_ver VERSION_LESS "6.3")
              message(STATUS "Detected ROCm ${_onemath_rocm_ver} (< 6.3); "
                "adding -mcode-object-version=5 for AMD targets")
              list(APPEND UNIX_INTERFACE_COMPILE_OPTIONS -mcode-object-version=5)
              list(APPEND UNIX_INTERFACE_LINK_OPTIONS -mcode-object-version=5)
            endif()
          endif()
        endif()
      endif()
    endif()
    if(ENABLE_CURAND_BACKEND OR ENABLE_CUSOLVER_BACKEND OR ENABLE_CUSPARSE_BACKEND OR ENABLE_ROCBLAS_BACKEND
	    OR ENABLE_ROCRAND_BACKEND OR ENABLE_ROCSOLVER_BACKEND OR ENABLE_ROCSPARSE_BACKEND)
      set_target_properties(ONEMATH::SYCL::SYCL PROPERTIES
        INTERFACE_COMPILE_OPTIONS "${UNIX_INTERFACE_COMPILE_OPTIONS}"
        INTERFACE_LINK_OPTIONS "${UNIX_INTERFACE_LINK_OPTIONS}"
        INTERFACE_LINK_LIBRARIES ${SYCL_LIBRARY})
    else()
      set_target_properties(ONEMATH::SYCL::SYCL PROPERTIES
        INTERFACE_COMPILE_OPTIONS "-fsycl"
        INTERFACE_LINK_OPTIONS "-fsycl"
        INTERFACE_LINK_LIBRARIES ${SYCL_LIBRARY})
    endif()
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
