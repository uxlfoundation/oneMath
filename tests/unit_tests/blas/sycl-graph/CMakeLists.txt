#===============================================================================
# Copyright 2025 Intel Corporation
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
set(SYCL_GRAPH_SOURCES)

set(SYCL_GRAPH_SOURCES_W_CBLAS "gemm_usm.cpp" "gemm_batch_usm.cpp")

if(CBLAS_FOUND)
    list(APPEND SYCL_GRAPH_SOURCES ${SYCL_GRAPH_SOURCES_W_CBLAS})
endif()

if(BUILD_SHARED_LIBS)
  add_library(blas_sycl_graph_rt OBJECT ${SYCL_GRAPH_SOURCES})
  target_compile_options(blas_sycl_graph_rt PRIVATE -DCALL_RT_API -DNOMINMAX)
  target_include_directories(blas_sycl_graph_rt
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      PUBLIC ${PROJECT_SOURCE_DIR}/include
      PUBLIC ${PROJECT_SOURCE_DIR}/deps/googletest/include
      PUBLIC ${CMAKE_BINARY_DIR}/bin
      $<$<BOOL:${CBLAS_FOUND}>:${CBLAS_INCLUDE}>
  )
  if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
    add_sycl_to_target(TARGET blas_sycl_graph_rt SOURCES ${SYCL_GRAPH_SOURCES})
  else()
    target_link_libraries(blas_sycl_graph_rt PUBLIC ONEMATH::SYCL::SYCL)
  endif()
endif()

add_library(blas_sycl_graph_ct OBJECT ${SYCL_GRAPH_SOURCES})
target_compile_options(blas_sycl_graph_ct PRIVATE  -DNOMINMAX)
target_include_directories(blas_sycl_graph_ct
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${PROJECT_SOURCE_DIR}/deps/googletest/include
    PUBLIC ${CMAKE_BINARY_DIR}/bin
    $<$<BOOL:${CBLAS_FOUND}>:${CBLAS_INCLUDE}>
)
if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
    add_sycl_to_target(TARGET blas_sycl_graph_ct SOURCES ${SYCL_GRAPH_SOURCES})
else()
  target_link_libraries(blas_sycl_graph_ct PUBLIC ONEMATH::SYCL::SYCL)
endif()
