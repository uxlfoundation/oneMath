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

# Build object from all test sources
set(SERVICE_DEVICE_TESTS_SOURCES "skip_ahead.cpp")

add_library(rng_device_service_ct OBJECT ${SERVICE_DEVICE_TESTS_SOURCES})
target_compile_options(rng_device_service_ct PRIVATE -DNOMINMAX)
target_include_directories(rng_device_service_ct
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${PROJECT_SOURCE_DIR}/deps/googletest/include
    PUBLIC ${CMAKE_BINARY_DIR}/bin
)
if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET rng_device_service_ct SOURCES ${SERVICE_DEVICE_TESTS_SOURCES})
else()
  target_link_libraries(rng_device_service_ct PUBLIC ONEMATH::SYCL::SYCL)
endif()

if(NOT ${ONEMATH_SYCL_IMPLEMENTATION} STREQUAL "hipsycl")
  target_link_options(rng_device_service_ct PUBLIC -fsycl -fsycl-device-code-split=per_kernel)
endif()
