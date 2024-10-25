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
set(BATCH_SOURCES)

set(BATCH_SOURCES_W_CBLAS "copy_batch_stride.cpp" "axpy_batch_stride.cpp" "dgmm_batch_stride.cpp" "gemm_batch_stride.cpp" "gemv_batch_stride.cpp" "trsm_batch_stride.cpp" "syrk_batch_stride.cpp" "copy_batch_usm.cpp" "copy_batch_stride_usm.cpp" "axpy_batch_usm.cpp" "axpy_batch_stride_usm.cpp" "dgmm_batch_usm.cpp" "dgmm_batch_stride_usm.cpp" "gemm_batch_usm.cpp" "gemm_batch_stride_usm.cpp" "gemv_batch_usm.cpp" "gemv_batch_stride_usm.cpp" "trsm_batch_usm.cpp" "trsm_batch_stride_usm.cpp" "syrk_batch_usm.cpp" "syrk_batch_stride_usm.cpp" "omatcopy_batch_stride.cpp" "omatcopy_batch_stride_usm.cpp" "imatcopy_batch_stride.cpp" "imatcopy_batch_stride_usm.cpp" "omatadd_batch_stride.cpp" "omatadd_batch_stride_usm.cpp" "omatcopy_batch_usm.cpp" "imatcopy_batch_usm.cpp")

if(CBLAS_FOUND)
  list(APPEND BATCH_SOURCES ${BATCH_SOURCES_W_CBLAS})
endif()

if(BUILD_SHARED_LIBS)
  add_library(blas_batch_rt OBJECT ${BATCH_SOURCES})
  target_compile_options(blas_batch_rt PRIVATE -DCALL_RT_API -DNOMINMAX)
  target_include_directories(blas_batch_rt
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      PUBLIC ${PROJECT_SOURCE_DIR}/include
      PUBLIC ${PROJECT_SOURCE_DIR}/deps/googletest/include
      PUBLIC ${CMAKE_BINARY_DIR}/bin
      $<$<BOOL:${CBLAS_FOUND}>:${CBLAS_INCLUDE}>
  )
  if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
    add_sycl_to_target(TARGET blas_batch_rt SOURCES ${BATCH_SOURCES})
  else()
    target_link_libraries(blas_batch_rt PUBLIC ONEMKL::SYCL::SYCL)
  endif()
endif()

add_library(blas_batch_ct OBJECT ${BATCH_SOURCES})
target_compile_options(blas_batch_ct PRIVATE  -DNOMINMAX)
target_include_directories(blas_batch_ct
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${PROJECT_SOURCE_DIR}/deps/googletest/include
    PUBLIC ${CMAKE_BINARY_DIR}/bin
    $<$<BOOL:${CBLAS_FOUND}>:${CBLAS_INCLUDE}>
)
if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET blas_batch_ct SOURCES ${BATCH_SOURCES})
else()
  target_link_libraries(blas_batch_ct PUBLIC ONEMKL::SYCL::SYCL)
endif()
