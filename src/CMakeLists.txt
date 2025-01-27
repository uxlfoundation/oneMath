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

# Define common build flags for oneMath libraries
set(ONEMATH_BUILD_COPT "")
if(WIN32 AND BUILD_SHARED_LIBS)
  list(APPEND ONEMATH_BUILD_COPT "-Donemath_EXPORTS")
endif()

# store path to CMAKE_CURRENT_BINARY_DIR to use it later (makes FetchContent_Declare workable)
set(ONEMATH_GENERATED_INCLUDE_PATH ${CMAKE_CURRENT_BINARY_DIR})


set(ONEMATH_INCLUDE_DIRS
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
)

# Create a CMake library as a deprecated alias of target_name
# The deprecated target name is based on the target_name with "onemath" replaced to "onemkl"
# Deprecated targets are planned to be removed late 2025
function(add_deprecated_library target_name)
  string(REPLACE "onemath" "onemkl" deprecated_name "${target_name}")
  if("${target_name}" EQUAL "${deprecated_name}")
    message(FATAL_ERROR "Internal error: add_deprecated_library was not able to generate a deprecated target name")
  endif()
  # The INTERFACE IMPORTED target works like an alias which can have different properties
  add_library(deprecated_name INTERFACE IMPORTED)
  target_link_libraries(deprecated_name INTERFACE target_name)
  set_target_properties(deprecated_name PROPERTIES DEPRECATION "${deprecated_name} target is deprecated, please use ${target_name} instead")
endfunction()


# Due to using the same directory structure and file name for different headers
# in this library and in the Intel(R) oneAPI Math Kernel Library, the compiler
# may not include the expected headers.
# Intel oneMKL include path is set as system include meaning it is always
# searched last no matter the order the order of the include flag in the command
# line argument.
# Using the -iquote flag is not supported on Windows.
# To avoid confusion the include paths are set up with a different "root" folder
# i.e.:
#   * the oneMath include path is `${PROJECT_SOURCE_DIR}/include` and its
#     deprecated headers can be included using `#include "oneapi/mkl/mkl.hpp"`
#     for instance.
#   * the Intel oneMKL include path is `${MKL_INCLUDE}/include/oneapi` and its
#     headers can be included using `#include "mkl/mkl.hpp"` for instance.
function(target_add_intel_onemkl_include target_name)
  target_include_directories(${target_name}
    PRIVATE ${MKL_INCLUDE}/oneapi
  )
endfunction()

# Build loader and backends for each domain
add_custom_target(onemath_backend_libs)
foreach(domain ${TARGET_DOMAINS})
  add_subdirectory(${domain})
endforeach()

# Generate header with enabled backends for testing

function(generate_header_file)
  # Following if-conditions allow to decouple cmake configuration variables with
  # the corresponding generated macro. This is done to be conformant with ES.33
  # C++ Core Guidelines
  set(ONEMATH_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  set(ONEMATH_ENABLE_MKLCPU_BACKEND ${ENABLE_MKLCPU_BACKEND})
  set(ONEMATH_ENABLE_MKLGPU_BACKEND ${ENABLE_MKLGPU_BACKEND})
  set(ONEMATH_ENABLE_CUBLAS_BACKEND ${ENABLE_CUBLAS_BACKEND})
  set(ONEMATH_ENABLE_ROCBLAS_BACKEND ${ENABLE_ROCBLAS_BACKEND})
  set(ONEMATH_ENABLE_NETLIB_BACKEND ${ENABLE_NETLIB_BACKEND})
  set(ONEMATH_ENABLE_ARMPL_BACKEND ${ENABLE_ARMPL_BACKEND})
  set(ONEMATH_ENABLE_GENERIC_BLAS_BACKEND ${ENABLE_GENERIC_BLAS_BACKEND})
  set(ONEMATH_ENABLE_CURAND_BACKEND ${ENABLE_CURAND_BACKEND})
  set(ONEMATH_ENABLE_ROCRAND_BACKEND ${ENABLE_ROCRAND_BACKEND})
  set(ONEMATH_ENABLE_CUSOLVER_BACKEND ${ENABLE_CUSOLVER_BACKEND})
  set(ONEMATH_ENABLE_ROCSOLVER_BACKEND ${ENABLE_ROCSOLVER_BACKEND})
  set(ONEMATH_ENABLE_CUFFT_BACKEND ${ENABLE_CUFFT_BACKEND})
  set(ONEMATH_ENABLE_ROCFFT_BACKEND ${ENABLE_ROCFFT_BACKEND})
  set(ONEMATH_ENABLE_PORTFFT_BACKEND ${ENABLE_PORTFFT_BACKEND})
  set(ONEMATH_ENABLE_CUSPARSE_BACKEND ${ENABLE_CUSPARSE_BACKEND})
  set(ONEMATH_ENABLE_ROCSPARSE_BACKEND ${ENABLE_ROCSPARSE_BACKEND})

  configure_file(config.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/oneapi/math/config.hpp.configured")
  file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/oneapi/math/detail/config.hpp"
    INPUT "${CMAKE_CURRENT_BINARY_DIR}/oneapi/math/config.hpp.configured"
  )
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/oneapi/math/detail/config.hpp"
    DESTINATION include/oneapi/math/detail
    COMPONENT Devel
  )
endfunction()

generate_header_file()

# Add recipe for onemath loader library
if(BUILD_SHARED_LIBS)
  add_library(onemath SHARED)

  # The loader library depends on all the backend libraries as it uses
  # dlopen to load them at runtime.
  # Use add_dependencies to ensure that all the backend libraries are
  # (re-)built when compiling the loader or runtime binaries.
  add_dependencies(onemath onemath_backend_libs)

  target_include_directories(onemath
    PUBLIC ${ONEMATH_INCLUDE_DIRS}
  )
  set_target_properties(onemath PROPERTIES
    SOVERSION ${PROJECT_VERSION_MAJOR}
  )
  # w/a for setting oneMath installed headers as -I instead of -isystem for cmake >= 3.25 for workable find_package(MKL) combination
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.25.0")
    set_target_properties(onemath PROPERTIES EXPORT_NO_SYSTEM true)
  endif()

  # Build dispatcher library
  set (ONEMATH_LIBS ${TARGET_DOMAINS})
  list(TRANSFORM ONEMATH_LIBS PREPEND onemath_)
  target_link_libraries(onemath PUBLIC ${ONEMATH_LIBS})
  set_target_properties(onemath PROPERTIES
      INSTALL_RPATH "\$ORIGIN"
      BUILD_WITH_INSTALL_RPATH TRUE
  )

  # Add the library to install package
  foreach(domain_lib ${ONEMATH_LIBS})
    install(TARGETS ${domain_lib} EXPORT oneMathTargets)
  endforeach()
  install(TARGETS onemath EXPORT oneMathTargets
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
  )

  add_deprecated_library(onemkl)
endif()
