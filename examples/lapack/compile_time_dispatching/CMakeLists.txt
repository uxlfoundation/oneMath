#===============================================================================
# Copyright 2022 Intel Corporation
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

# The example is written for the MKLCPU and CUSOLVER backends
if(NOT (ENABLE_MKLCPU_BACKEND AND ENABLE_CUSOLVER_BACKEND))
  return()
endif()

set(EXAMPLE_TARGET example_lapack_getrs_usm_mklcpu_cusolver)

# External applications should use find_package or FetchContent to include oneMath first.
# See https://github.com/uxlfoundation/oneMath/blob/develop/docs/using_onemath_with_cmake.rst

# Create a CMake target with one source file
add_executable(${EXAMPLE_TARGET} getrs_usm_mklcpu_cusolver.cpp)

# May be needed to compile source files with SYCL kernels
if(USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET ${EXAMPLE_TARGET} SOURCES getrs_usm_mklcpu_cusolver.cpp)
endif()

# Linking against onemath_lapack_mklcpu and onemath_lapack_cusolver in CMake will add the required include directories and dependencies.
# One can also link against `onemath_lapack` to link against all the lapack backends built.
# These targets should only be used for compile-time dispatching.
target_link_libraries(${EXAMPLE_TARGET} PUBLIC
  onemath_lapack_mklcpu
  onemath_lapack_cusolver
)

# Include directories specific to the examples
target_include_directories(${EXAMPLE_TARGET} PUBLIC
  ${PROJECT_SOURCE_DIR}/examples/include
)

# Register example as ctest
add_test(NAME lapack/EXAMPLE/CT/getrs_usm_mklcpu_cusolver COMMAND ${EXAMPLE_TARGET})
