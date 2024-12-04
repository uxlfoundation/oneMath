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

# Build object from all test sources
# TODO: add list of tests without Netlib dependency
set(EXTENSIONS_SOURCES)

set(EXTENSIONS_SOURCES_W_CBLAS "gemm_bias.cpp" "gemmt.cpp" "gemm_bias_usm.cpp" "gemmt_usm.cpp" "omatcopy.cpp" "omatcopy_usm.cpp" "imatcopy.cpp" "imatcopy_usm.cpp" "omatadd.cpp" "omatadd_usm.cpp" "omatcopy2.cpp" "omatcopy2_usm.cpp")

if(CBLAS_FOUND)
  list(APPEND EXTENSIONS_SOURCES ${EXTENSIONS_SOURCES_W_CBLAS})
endif()


if(BUILD_SHARED_LIBS)
  add_library(blas_extensions_rt OBJECT ${EXTENSIONS_SOURCES})
  target_compile_options(blas_extensions_rt PRIVATE -DCALL_RT_API -DNOMINMAX)
  target_include_directories(blas_extensions_rt
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      PUBLIC ${PROJECT_SOURCE_DIR}/include
      PUBLIC ${PROJECT_SOURCE_DIR}/deps/googletest/include
      PUBLIC ${CMAKE_BINARY_DIR}/bin
      $<$<BOOL:${CBLAS_FOUND}>:${CBLAS_INCLUDE}>
  )
  if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
    add_sycl_to_target(TARGET blas_extensions_rt SOURCES ${EXTENSIONS_SOURCES})
  else()
    target_link_libraries(blas_extensions_rt PUBLIC ONEMATH::SYCL::SYCL)
  endif()
endif()

add_library(blas_extensions_ct OBJECT ${EXTENSIONS_SOURCES})
target_compile_options(blas_extensions_ct PRIVATE  -DNOMINMAX)
target_include_directories(blas_extensions_ct
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${PROJECT_SOURCE_DIR}/deps/googletest/include
    PUBLIC ${CMAKE_BINARY_DIR}/bin
    $<$<BOOL:${CBLAS_FOUND}>:${CBLAS_INCLUDE}>
)
if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET blas_extensions_ct  SOURCES ${EXTENSIONS_SOURCES})
else()
  target_link_libraries(blas_extensions_ct PUBLIC ONEMATH::SYCL::SYCL)
endif()
