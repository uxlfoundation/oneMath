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
set(L3_SOURCES)

set(L3_SOURCES_W_CBLAS "gemm.cpp" "symm.cpp" "syrk.cpp" "hemm.cpp" "herk.cpp" "syr2k.cpp" "her2k.cpp" "trmm.cpp" "trsm.cpp" "gemm_usm.cpp" "symm_usm.cpp" "syrk_usm.cpp" "hemm_usm.cpp" "herk_usm.cpp" "syr2k_usm.cpp" "her2k_usm.cpp" "trmm_usm.cpp" "trsm_usm.cpp")

if(CBLAS_FOUND)
  list(APPEND L3_SOURCES ${L3_SOURCES_W_CBLAS})
endif()

if(BUILD_SHARED_LIBS)
  add_library(blas_level3_rt OBJECT ${L3_SOURCES})
  target_compile_options(blas_level3_rt PRIVATE -DCALL_RT_API -DNOMINMAX)
  target_include_directories(blas_level3_rt
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      PUBLIC ${PROJECT_SOURCE_DIR}/include
      PUBLIC ${PROJECT_SOURCE_DIR}/deps/googletest/include
      PUBLIC ${CMAKE_BINARY_DIR}/bin
      $<$<BOOL:${CBLAS_FOUND}>:${CBLAS_INCLUDE}>
  )
  if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
    add_sycl_to_target(TARGET blas_level3_rt SOURCES ${L3_SOURCES})
  else()
    target_link_libraries(blas_level3_rt PUBLIC ONEMKL::SYCL::SYCL)
  endif()
endif()

add_library(blas_level3_ct OBJECT ${L3_SOURCES})
target_compile_options(blas_level3_ct PRIVATE  -DNOMINMAX)
target_include_directories(blas_level3_ct
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${PROJECT_SOURCE_DIR}/deps/googletest/include
    PUBLIC ${CMAKE_BINARY_DIR}/bin
    $<$<BOOL:${CBLAS_FOUND}>:${CBLAS_INCLUDE}>
)
if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET blas_level3_ct SOURCES ${L3_SOURCES})
else()
  target_link_libraries(blas_level3_ct PUBLIC ONEMKL::SYCL::SYCL)
endif()


