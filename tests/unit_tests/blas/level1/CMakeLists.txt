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
set(L1_SOURCES)

set(L1_SOURCES_W_CBLAS "nrm2.cpp" "iamin.cpp" "iamax.cpp" "dotu.cpp" "dot.cpp" "dotc.cpp" "copy.cpp" "axpy.cpp" "axpby.cpp" "asum.cpp" "swap.cpp" "sdsdot.cpp" "scal.cpp" "rotmg.cpp" "rotm.cpp" "rotg.cpp" "rot.cpp" "nrm2_usm.cpp" "iamin_usm.cpp" "iamax_usm.cpp" "dotu_usm.cpp" "dot_usm.cpp" "dotc_usm.cpp" "copy_usm.cpp" "axpy_usm.cpp" "axpby_usm.cpp" "asum_usm.cpp" "swap_usm.cpp" "sdsdot_usm.cpp" "scal_usm.cpp" "rotmg_usm.cpp" "rotm_usm.cpp" "rotg_usm.cpp" "rot_usm.cpp")

if(CBLAS_FOUND)
  list(APPEND L1_SOURCES ${L1_SOURCES_W_CBLAS})
endif()

if(BUILD_SHARED_LIBS)
  add_library(blas_level1_rt OBJECT ${L1_SOURCES})
  target_compile_options(blas_level1_rt PRIVATE -DCALL_RT_API -DNOMINMAX)
  target_include_directories(blas_level1_rt
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      PUBLIC ${PROJECT_SOURCE_DIR}/include
      PUBLIC ${PROJECT_SOURCE_DIR}/deps/googletest/include
      PUBLIC ${CMAKE_BINARY_DIR}/bin
      $<$<BOOL:${CBLAS_FOUND}>:${CBLAS_INCLUDE}>
  )
  if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
    add_sycl_to_target(TARGET blas_level1_rt SOURCES ${L1_SOURCES})
  else()
    target_link_libraries(blas_level1_rt PUBLIC ONEMKL::SYCL::SYCL)
  endif()
endif()

add_library(blas_level1_ct OBJECT ${L1_SOURCES})
target_compile_options(blas_level1_ct PRIVATE -DNOMINMAX)
target_include_directories(blas_level1_ct
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${PROJECT_SOURCE_DIR}/deps/googletest/include
    PUBLIC ${CMAKE_BINARY_DIR}/bin
    $<$<BOOL:${CBLAS_FOUND}>:${CBLAS_INCLUDE}>
)
if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET blas_level1_ct SOURCES ${L1_SOURCES})
else()
  target_link_libraries(blas_level1_ct PUBLIC ONEMKL::SYCL::SYCL)
endif()
