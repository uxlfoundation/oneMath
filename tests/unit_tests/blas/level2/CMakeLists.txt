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
set(L2_SOURCES)

set(L2_SOURCES_W_CBLAS "hpr2.cpp" "hpmv.cpp" "her.cpp" "her2.cpp" "hemv.cpp" "hbmv.cpp" "geru.cpp" "ger.cpp" "gerc.cpp" "gemv.cpp" "gbmv.cpp" "trsv.cpp" "trmv.cpp" "tpsv.cpp" "tpmv.cpp" "tbsv.cpp" "tbmv.cpp" "syr.cpp" "syr2.cpp" "symv.cpp" "spr.cpp" "spr2.cpp" "spmv.cpp" "sbmv.cpp" "hpr.cpp" "hpr2_usm.cpp" "hpmv_usm.cpp" "her_usm.cpp" "her2_usm.cpp" "hemv_usm.cpp" "hbmv_usm.cpp" "geru_usm.cpp" "ger_usm.cpp" "gerc_usm.cpp" "gemv_usm.cpp" "gbmv_usm.cpp" "trsv_usm.cpp" "trmv_usm.cpp" "tpsv_usm.cpp" "tpmv_usm.cpp" "tbsv_usm.cpp" "tbmv_usm.cpp" "syr_usm.cpp" "syr2_usm.cpp" "symv_usm.cpp" "spr_usm.cpp" "spr2_usm.cpp" "spmv_usm.cpp" "sbmv_usm.cpp" "hpr_usm.cpp")

if(CBLAS_FOUND)
  list(APPEND L2_SOURCES ${L2_SOURCES_W_CBLAS})
endif()

if(BUILD_SHARED_LIBS)
  add_library(blas_level2_rt OBJECT ${L2_SOURCES})
  target_compile_options(blas_level2_rt PRIVATE -DCALL_RT_API -DNOMINMAX)
  target_include_directories(blas_level2_rt
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      PUBLIC ${PROJECT_SOURCE_DIR}/include
      PUBLIC ${PROJECT_SOURCE_DIR}/deps/googletest/include
      PUBLIC ${CMAKE_BINARY_DIR}/bin
      $<$<BOOL:${CBLAS_FOUND}>:${CBLAS_INCLUDE}>
  )
  if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
    add_sycl_to_target(TARGET blas_level2_rt SOURCES ${L2_SOURCES})
  else()
    target_link_libraries(blas_level2_rt PUBLIC ONEMATH::SYCL::SYCL)
  endif()
endif()

add_library(blas_level2_ct OBJECT ${L2_SOURCES})
target_compile_options(blas_level2_ct PRIVATE -DNOMINMAX)
target_include_directories(blas_level2_ct
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${PROJECT_SOURCE_DIR}/deps/googletest/include
    PUBLIC ${CMAKE_BINARY_DIR}/bin
    $<$<BOOL:${CBLAS_FOUND}>:${CBLAS_INCLUDE}>
)
if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET blas_level2_ct SOURCES ${L2_SOURCES})
else()
  target_link_libraries(blas_level2_ct PUBLIC ONEMATH::SYCL::SYCL)
endif()

