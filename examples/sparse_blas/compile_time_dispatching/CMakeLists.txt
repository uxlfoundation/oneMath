#===============================================================================
# Copyright 2023 Intel Corporation
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

# The example is written for the MKLCPU and CUSPARSE backends
if(NOT (ENABLE_MKLCPU_BACKEND AND ENABLE_CUSPARSE_BACKEND))
  return()
endif()

set(EXAMPLE_TARGET example_sparse_blas_spmv_usm_mklcpu_cusparse)

# External applications should use find_package or FetchContent to include oneMath first.
# See https://github.com/uxlfoundation/oneMath/blob/develop/docs/using_onemath_with_cmake.rst

# Create a CMake target with one source file
add_executable(${EXAMPLE_TARGET} spmv_usm_mklcpu_cusparse.cpp)

# Linking against onemath_sparse_blas_mklcpu and onemath_sparse_blas_cusparse in CMake will add the required include directories and dependencies.
# One can also link against `onemath_sparse_blas` to link against all the sparse_blas backends built.
# These targets should only be used for compile-time dispatching.
target_link_libraries(${EXAMPLE_TARGET} PUBLIC
  onemath_sparse_blas_mklcpu
  onemath_sparse_blas_cusparse
)

# Include directories specific to the examples
target_include_directories(${EXAMPLE_TARGET} PUBLIC
  ${PROJECT_SOURCE_DIR}/examples/include
)

# Enable warnings
include(WarningsUtils)
target_link_libraries(${EXAMPLE_TARGET} PRIVATE onemath_warnings)

# Register example as ctest
add_test(NAME sparse_blas/EXAMPLE/CT/spmv_usm_mklcpu_cusparse COMMAND ${EXAMPLE_TARGET})
