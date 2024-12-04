#===============================================================================
# Copyright 2024 Intel Corporation
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

add_subdirectory(backends)

if(BUILD_SHARED_LIBS)
  add_library(onemath_sparse_blas OBJECT)
  add_deprecated_library(onemath_sparse_blas)
  target_sources(onemath_sparse_blas PRIVATE sparse_blas_loader.cpp)
  target_include_directories(onemath_sparse_blas
    PRIVATE ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/src/include
            ${CMAKE_BINARY_DIR}/bin
            ${ONEMATH_GENERATED_INCLUDE_PATH}
            $<TARGET_FILE_DIR:onemath>
  )

  target_compile_options(onemath_sparse_blas PRIVATE ${ONEMATH_BUILD_COPT})

  set_target_properties(onemath_sparse_blas PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )
  if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
    add_sycl_to_target(TARGET onemath_sparse_blas SOURCES sparse_blas_loader.cpp)
  else()
    target_link_libraries(onemath_sparse_blas PUBLIC ONEMATH::SYCL::SYCL)
  endif()

  include(WarningsUtils)
  target_link_libraries(onemath_sparse_blas PRIVATE onemath_warnings)

endif()
