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

# The example is written for the MKLCPU and CURAND backends
if(NOT (ENABLE_MKLCPU_BACKEND AND ENABLE_CURAND_BACKEND))
  return()
endif()

set(EXAMPLE_TARGET example_rng_uniform_usm_mklcpu_curand)

# External applications should use find_package or FetchContent to include oneMath first.
# See https://github.com/uxlfoundation/oneMath/blob/develop/docs/using_onemath_with_cmake.rst

# Create a CMake target with one source file
add_executable(${EXAMPLE_TARGET} uniform_usm_mklcpu_curand.cpp)

# Linking against onemath_rng_mklcpu and onemath_rng_curand in CMake will add the required include directories and dependencies.
# One can also link against `onemath_rng` to link against all the rng backends built.
# These targets should only be used for compile-time dispatching.
target_link_libraries(${EXAMPLE_TARGET} PUBLIC
  onemath_rng_mklcpu
  onemath_rng_curand
)

# Include directories specific to the examples
target_include_directories(${EXAMPLE_TARGET} PUBLIC
  ${PROJECT_SOURCE_DIR}/examples/include
)

# Register example as ctest
add_test(NAME rng/EXAMPLE/CT/uniform_usm_mklcpu_curand COMMAND ${EXAMPLE_TARGET})
