#===============================================================================
# Copyright 2020-2022 Intel Corporation
# Copyright (C) 2022 Heidelberg University, Engineering Mathematics and Computing Lab (EMCL) and Computing Centre (URZ)
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

cmake_minimum_required (VERSION 3.13)

# Define build type
set(DEFAULT_BUILD_TYPE "Release")

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE: None, set to ${DEFAULT_BUILD_TYPE} by default")
    set(CMAKE_BUILD_TYPE ${DEFAULT_BUILD_TYPE} CACHE STRING
            "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel RelWithAssert" FORCE)
else()
    message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
endif()

# Build options
option(BUILD_SHARED_LIBS "Build dynamic libraries" ON)

## Backends
option(ENABLE_MKLCPU_BACKEND "Enable the Intel oneMKL CPU backend for supported interfaces" ON)
option(ENABLE_MKLGPU_BACKEND "Enable the Intel oneMKL GPU backend for supported interfaces" ON)
if(ENABLE_MKLCPU_BACKEND)
  option(ENABLE_MKLCPU_THREAD_TBB "Enable the use of Intel TBB with the oneMath CPU backend" ON)
endif()

option(ENABLE_ARMPL_BACKEND "Enable the ArmPl backend for BLAS/LAPACK interface" OFF)
if(ENABLE_ARMPL_BACKEND)
  option(ENABLE_ARMPL_OMP "Enable OpenMP for the ArmPl backend" ON)
endif()

# blas
option(ENABLE_CUBLAS_BACKEND "Enable the cuBLAS backend for the BLAS interface" OFF)
option(ENABLE_ROCBLAS_BACKEND "Enable the rocBLAS backend for the BLAS interface" OFF)
option(ENABLE_NETLIB_BACKEND "Enable the Netlib backend for the BLAS interface" OFF)
option(ENABLE_GENERIC_BLAS_BACKEND "Enable the generic BLAS backend for the BLAS interface. Cannot be used with other BLAS backends." OFF)

# rand
option(ENABLE_CURAND_BACKEND "Enable the cuRAND backend for the RNG interface" OFF)
option(ENABLE_ROCRAND_BACKEND "Enable the rocRAND backend for the RNG interface" OFF)

# lapack
option(ENABLE_CUSOLVER_BACKEND "Enable the cuSOLVER backend for the LAPACK interface" OFF)
option(ENABLE_ROCSOLVER_BACKEND "Enable the rocSOLVER backend for the LAPACK interface" OFF)

# dft
option(ENABLE_CUFFT_BACKEND "Enable the cuFFT backend for the DFT interface" OFF)
option(ENABLE_ROCFFT_BACKEND "Enable the rocFFT backend for the DFT interface" OFF)
option(ENABLE_PORTFFT_BACKEND "Enable the portFFT DFT backend for the DFT interface. Cannot be used with other DFT backends." OFF)

# sparse
option(ENABLE_CUSPARSE_BACKEND "Enable the cuSPARSE backend for the SPARSE_BLAS interface" OFF)
option(ENABLE_ROCSPARSE_BACKEND "Enable the rocSPARSE backend for the SPARSE_BLAS interface" OFF)

set(ONEMATH_SYCL_IMPLEMENTATION "dpc++" CACHE STRING "Name of the SYCL compiler")
set(HIP_TARGETS "" CACHE STRING "Target HIP architectures")

## Testing
option(BUILD_FUNCTIONAL_TESTS "" ON)

## Examples
option(BUILD_EXAMPLES "" ON)

## Documentation
option(BUILD_DOC "" OFF)

if(DEFINED ENABLE_PORTBLAS_BACKEND)
  message(WARNING "PORTBLAS_BACKEND is deprecated, please use ENABLE_GENERIC_BLAS_BACKEND.")
  set(ENABLE_GENERIC_BLAS_BACKEND ${ENABLE_PORTBLAS_BACKEND})
endif()

## Supported domains
set(DOMAINS_LIST "")
if(ENABLE_MKLCPU_BACKEND
        OR ENABLE_MKLGPU_BACKEND
        OR ENABLE_CUBLAS_BACKEND
        OR ENABLE_ROCBLAS_BACKEND
        OR ENABLE_NETLIB_BACKEND
        OR ENABLE_GENERIC_BLAS_BACKEND
        OR ENABLE_ARMPL_BACKEND)
  list(APPEND DOMAINS_LIST "blas")
endif()
if(ENABLE_MKLCPU_BACKEND
        OR ENABLE_MKLGPU_BACKEND
        OR ENABLE_CUSOLVER_BACKEND
        OR ENABLE_ROCSOLVER_BACKEND
        OR ENABLE_ARMPL_BACKEND)
  list(APPEND DOMAINS_LIST "lapack")
endif()
if(ENABLE_MKLCPU_BACKEND
        OR ENABLE_MKLGPU_BACKEND
        OR ENABLE_CURAND_BACKEND
        OR ENABLE_ROCRAND_BACKEND
        OR ENABLE_ARMPL_BACKEND)
  list(APPEND DOMAINS_LIST "rng")
endif()
if(ENABLE_MKLGPU_BACKEND
        OR ENABLE_MKLCPU_BACKEND
        OR ENABLE_CUFFT_BACKEND
        OR ENABLE_ROCFFT_BACKEND
        OR ENABLE_PORTFFT_BACKEND
        OR ENABLE_ARMPL_BACKEND)
  list(APPEND DOMAINS_LIST "dft")
endif()
if(ENABLE_MKLCPU_BACKEND
        OR ENABLE_MKLGPU_BACKEND
        OR ENABLE_CUSPARSE_BACKEND
        OR ENABLE_ROCSPARSE_BACKEND)
  list(APPEND DOMAINS_LIST "sparse_blas")
endif()

if(ENABLE_GENERIC_BLAS_BACKEND
	AND (ENABLE_MKLCPU_BACKEND
		OR ENABLE_MKLGPU_BACKEND
		OR ENABLE_CUBLAS_BACKEND
		OR ENABLE_ROCBLAS_BACKEND
		OR ENABLE_NETLIB_BACKEND))
	message(FATAL_ERROR "ENABLE_GENERIC_BLAS_BACKEND cannot be enabled at the same time as other BLAS backends.")
endif()

if (ENABLE_PORTFFT_BACKEND
	AND (ENABLE_MKLCPU_BACKEND
		OR ENABLE_MKLGPU_BACKEND
		OR ENABLE_ROCFFT_BACKEND
		OR ENABLE_CUFFT_BACKEND))
	message(FATAL_ERROR "ENABLE_PORTFFT_BACKEND cannot be enabled at the same time as other DFT backends.")
endif()

# Define required CXX compilers before project
if(CMAKE_CXX_COMPILER OR NOT ONEMATH_SYCL_IMPLEMENTATION STREQUAL "dpc++")
  if(WIN32)
    string(REPLACE "\\" "/" CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER})
  endif()
else()
  if(ENABLE_CUBLAS_BACKEND OR ENABLE_CURAND_BACKEND OR ENABLE_CUSOLVER_BACKEND OR ENABLE_CUFFT_BACKEND OR ENABLE_CUSPARSE_BACKEND
    OR ENABLE_ROCBLAS_BACKEND OR ENABLE_ROCRAND_BACKEND OR ENABLE_ROCSOLVER_BACKEND OR ENABLE_ROCFFT_BACKEND OR ENABLE_ROCSPARSE_BACKEND)
    set(CMAKE_CXX_COMPILER "clang++")
  elseif(ENABLE_MKLGPU_BACKEND)
    if(UNIX)
      set(CMAKE_CXX_COMPILER "icpx")
    else()
      set(CMAKE_CXX_COMPILER "icx")
    endif()
  else()
    if(UNIX)
      find_program(ICPX_ICX_PATH icpx)
    else()
      find_program(ICPX_ICX_PATH icx)
    endif()
    if(ICPX_ICX_PATH)
      if(UNIX)
        message(STATUS "CXX compiler: icpx was found in PATH, using icpx")
        set(CMAKE_CXX_COMPILER "icpx")
      else()
        message(STATUS "CXX compiler: icx was found in PATH, using icx")
        set(CMAKE_CXX_COMPILER "icx")
      endif()
    else()
      if(WIN32)
        message(STATUS "CXX compiler: icx was not found in PATH, using clang-cl instead")
        set(CMAKE_CXX_COMPILER "clang-cl")
      else()
        message(STATUS "CXX compiler: icpx was not found in PATH, using clang++ instead")
        set(CMAKE_CXX_COMPILER "clang++")
      endif()
    endif()
  endif()
endif()

# Define required C compilers before project
if(CMAKE_C_COMPILER OR NOT ONEMATH_SYCL_IMPLEMENTATION STREQUAL "dpc++")
  if(WIN32)
    string(REPLACE "\\" "/" CMAKE_C_COMPILER ${CMAKE_C_COMPILER})
  endif()
else()
  find_program(ICX_PATH icx)
  if(ICX_PATH)
    message(STATUS "C compiler: icx was found in PATH, using icx")
    set(CMAKE_C_COMPILER "icx")
  else()
    if(WIN32)
      message(STATUS "C compiler: icx was not found in PATH, using clang-cl instead")
      set(CMAKE_C_COMPILER "clang-cl")
    else()
      message(STATUS "C compiler: icx was not found in PATH, using clang instead")
      set(CMAKE_C_COMPILER "clang")
    endif()
  endif()
endif()

project(oneMath VERSION 0.5.0 LANGUAGES CXX)

# Override default CXX compile/link lines for Windows after project
if(WIN32 AND ONEMATH_SYCL_IMPLEMENTATION STREQUAL "dpc++")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function -w")
  foreach (flag_var
           CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
           CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    string(REPLACE "/MD" "" ${flag_var} "${${flag_var}}")
  endforeach()
  set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> -fsycl /nologo <DEFINES> <INCLUDES> /EHsc <FLAGS> /Fo<OBJECT> -c <SOURCE>")
  set(CMAKE_CXX_CREATE_STATIC_LIBRARY "lib /nologo <OBJECTS> /out:<TARGET>")
  if(CMAKE_VERSION VERSION_LESS "3.25.2")
    set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_CXX_COMPILER> -fsycl -fsycl-device-code-split=per_kernel /nologo <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
    set(CMAKE_CXX_CREATE_SHARED_LIBRARY "<CMAKE_CXX_COMPILER> -fsycl -fsycl-device-code-split=per_kernel /nologo <OBJECTS> <LINK_LIBRARIES> /link /out:<TARGET> /implib:<TARGET_IMPLIB> /pdb:<TARGET_PDB> /dll /version:<TARGET_VERSION_MAJOR>.<TARGET_VERSION_MINOR>")
  endif()
endif()

# Temporary disable sycl 2020 deprecations warnings for cuSOLVER and rocSOLVER
if(ONEMATH_SYCL_IMPLEMENTATION STREQUAL "dpc++" AND (ENABLE_ROCSOLVER_BACKEND))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSYCL2020_DISABLE_DEPRECATION_WARNINGS")
endif()

# Target domains
if(NOT TARGET_DOMAINS OR TARGET_DOMAINS STREQUAL "None")
  # Set to all by default
  set(TARGET_DOMAINS ${DOMAINS_LIST})
else()
  # Make sure the input was converted to list
  string(REPLACE " " ";" TARGET_DOMAINS ${TARGET_DOMAINS})
  set(NOT_FOUND 0)
  foreach(domain ${TARGET_DOMAINS})
    if(NOT ${domain} IN_LIST DOMAINS_LIST)
      set(NOT_FOUND 1)
      break()
    endif()
  endforeach()
  if(NOT_FOUND)
    message(STATUS "TARGET_DOMAINS contains unsupported options, reset to all")
    set(TARGET_DOMAINS ${DOMAINS_LIST})
  endif()
endif()
message(STATUS "TARGET_DOMAINS: ${TARGET_DOMAINS}")

# Include Intel(R) oneAPI Math Kernel Library (oneMKL)
if(ENABLE_MKLGPU_BACKEND OR ENABLE_MKLCPU_BACKEND)
  set(MKL_ARCH intel64)
  set(MKL_INTERFACE ilp64)
  if(ENABLE_MKLCPU_THREAD_TBB)
    set(MKL_THREADING tbb_thread)
  else()
    set(MKL_THREADING sequential)
  endif()
  if(BUILD_SHARED_LIBS AND NOT WIN32)
    set(MKL_LINK dynamic)
  else()
    set(MKL_LINK static)
  endif()
  # Enable SYCL API
  set(DPCPP_COMPILER ON)
  set(SYCL_COMPILER ON)
  # In case Intel oneMKL package doesn't include MKLConfig,
  # use MKLConfig from the repo
  find_package(MKL REQUIRED
          HINTS ${MKL_ROOT}/lib/cmake
                ${MKL_ROOT}/lib/cmake/mkl
                $ENV{MKLROOT}
                ${PROJECT_SOURCE_DIR}/cmake/mkl)
endif()

# Set output directories for the project
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add CMake Finders
add_subdirectory(cmake)

# Include general cmake config files
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Add DPC++ options for Linux
if(WIN32)
  add_library(ONEMATH::SYCL::SYCL INTERFACE IMPORTED)
else()
  # Find necessary packages
  if(ONEMATH_SYCL_IMPLEMENTATION)
    string( TOLOWER "${ONEMATH_SYCL_IMPLEMENTATION}" ONEMATH_SYCL_IMPLEMENTATION)
    if (ONEMATH_SYCL_IMPLEMENTATION STREQUAL "adaptivecpp")
      message(STATUS "Looking for AdaptiveCpp")
      find_package(AdaptiveCpp CONFIG REQUIRED)
      if (${AdaptiveCpp_VERSION} VERSION_LESS "24.6")
        message(FATAL_ERROR "AdaptiveCpp version 24.6 or higher is required but ${AdaptiveCpp_VERSION} was found")
      endif()
      set(USE_ADD_SYCL_TO_TARGET_INTEGRATION true)
      set (CMAKE_CXX_STANDARD 17)
      add_library(ONEMATH::SYCL::SYCL INTERFACE IMPORTED)
    elseif(ONEMATH_SYCL_IMPLEMENTATION STREQUAL "dpc++")
      message(STATUS "Looking for dpc++")
      set(USE_ADD_SYCL_TO_TARGET_INTEGRATION false)
      find_package(Compiler REQUIRED)
    else()
      message(FATAL_ERROR "SYCL implementation ${ONEMATH_SYCL_IMPLEMENTATION} is not known")
    endif()
  else()
    message(STATUS "Looking for dpc++")
    set(USE_ADD_SYCL_TO_TARGET_INTEGRATION false)
    find_package(Compiler REQUIRED)
  endif()
endif()

if(DEFINED REF_BLAS_ROOT)
  find_file(ONEMATH_REF_BLAS_LIBNAME NAMES blas.dll libblas.so HINTS ${REF_BLAS_ROOT} PATH_SUFFIXES lib lib64)
  find_file(ONEMATH_REF_CBLAS_LIBNAME NAMES cblas.dll libcblas.so HINTS ${REF_BLAS_ROOT} PATH_SUFFIXES lib lib64)
endif()

# Add source directory and output to bin/
add_subdirectory(src bin)

# Functional Tests
if(BUILD_FUNCTIONAL_TESTS OR BUILD_EXAMPLES)
  enable_testing()
endif()

if(BUILD_FUNCTIONAL_TESTS)
  add_subdirectory(tests)
endif()

# Examples
if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(BUILD_DOC)
  add_subdirectory(docs)
endif()

install(DIRECTORY include/
  DESTINATION include
  COMPONENT Devel
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/oneMathConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

export(EXPORT oneMathTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/oneMathTargets.cmake"
  NAMESPACE ONEMATH::
)
configure_file("${PROJECT_SOURCE_DIR}/cmake/oneMathConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/oneMathConfig.cmake"
  COPYONLY
)

set(config_package_location "lib/cmake/${PROJECT_NAME}")
install(EXPORT oneMathTargets
  FILE oneMathTargets.cmake
  NAMESPACE ONEMATH::
  DESTINATION ${config_package_location}
)
install(
  FILES
  "${PROJECT_SOURCE_DIR}/cmake/oneMathConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/oneMathConfigVersion.cmake"
  DESTINATION ${config_package_location}
  COMPONENT Devel
)
