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

# NOTE: user needs to set env var ONEAPI_DEVICE_SELECTOR to use runtime example (no need to specify backend when building with CMake)

# Set up for the right backend for run-time dispatching examples
# If users build more than one backend (i.e. mklcpu and mklgpu, or mklcpu and CUDA), they may need to
# overwrite ONEAPI_DEVICE_SELECTOR in their environment to run on the desired backend
set(DEVICE_FILTERS "")
if(ENABLE_MKLCPU_BACKEND)
  list(APPEND DEVICE_FILTERS "opencl:cpu")
endif()
# RNG only supports mklcpu backend on Windows
if(ENABLE_MKLGPU_BACKEND)
  list(APPEND DEVICE_FILTERS "level_zero:gpu")
endif()
if(ENABLE_CURAND_BACKEND)
  list(APPEND DEVICE_FILTERS "cuda:gpu")
endif()
if(ENABLE_ROCRAND_BACKEND)
  list(APPEND DEVICE_FILTERS "hip:gpu")
endif()

message(STATUS "ONEAPI_DEVICE_SELECTOR will be set to the following value(s): [${DEVICE_FILTERS}] for run-time dispatching examples")

set(EXAMPLE_TARGET example_rng_uniform)

# External applications should use find_package or FetchContent to include oneMath first.
# See https://github.com/uxlfoundation/oneMath/blob/develop/docs/using_onemath_with_cmake.rst

# Create a CMake target with one source file
add_executable(${EXAMPLE_TARGET} uniform.cpp)

# Linking against onemath in CMake will add the required include directories and dependencies.
# This target should only be used for runtime dispatching.
target_link_libraries(${EXAMPLE_TARGET} PUBLIC onemath)

# Include directories specific to the examples
target_include_directories(${EXAMPLE_TARGET} PUBLIC
  ${PROJECT_SOURCE_DIR}/examples/rng/device/include
  ${PROJECT_SOURCE_DIR}/examples/include
)

# May be needed to compile source files with SYCL kernels
if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET ${EXAMPLE_TARGET} SOURCES uniform.cpp)
endif()

if(NOT ${ONEMATH_SYCL_IMPLEMENTATION} STREQUAL "hipsycl")
  # set strict warning flags for rng device API examples
  target_compile_options(${EXAMPLE_TARGET} PUBLIC -Wall -Werror -Wextra -Wpedantic -Wunreachable-code -Wfloat-conversion -Wextra-semi -Wshadow)
  target_link_options(${EXAMPLE_TARGET} PUBLIC -fsycl -fsycl-device-code-split=per_kernel)
endif()

# Register example as ctest
foreach(device_filter ${DEVICE_FILTERS})
  add_test(NAME rng/EXAMPLE/DEVICE/uniform/${device_filter} COMMAND ${EXAMPLE_TARGET})
  # Set ONEAPI_DEVICE_SELECTOR environment variable to select a device at runtime
  set_property(TEST rng/EXAMPLE/DEVICE/uniform/${device_filter} PROPERTY
    ENVIRONMENT ONEAPI_DEVICE_SELECTOR=${device_filter})
endforeach(device_filter)
