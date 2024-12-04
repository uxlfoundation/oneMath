#===============================================================================
# Copyright 2021 Intel Corporation
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

# Build backends
add_subdirectory(backends)

# Recipe for LAPACK loader object
if(BUILD_SHARED_LIBS)
add_library(onemath_lapack OBJECT)
add_deprecated_library(onemath_lapack)
target_sources(onemath_lapack PRIVATE lapack_loader.cpp)
target_include_directories(onemath_lapack
  PRIVATE ${PROJECT_SOURCE_DIR}/include
          ${PROJECT_SOURCE_DIR}/src
          ${PROJECT_SOURCE_DIR}/src/include
          ${CMAKE_BINARY_DIR}/bin
          ${ONEMATH_GENERATED_INCLUDE_PATH}
          $<TARGET_FILE_DIR:onemath>
)

target_compile_options(onemath_lapack PRIVATE ${ONEMATH_BUILD_COPT})

set_target_properties(onemath_lapack PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(onemath_lapack PUBLIC ONEMATH::SYCL::SYCL)
endif()

