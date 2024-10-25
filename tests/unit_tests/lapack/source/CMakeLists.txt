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

#Build object from all test sources
# TODO: add list of tests without Netlib dependency
set(LAPACK_SOURCES)

set(LAPACK_SOURCES_W_LAPACKE
    "gebrd.cpp"
    "geqrf.cpp"
    "geqrf_batch_group.cpp"
    "geqrf_batch_stride.cpp"
    "gerqf.cpp"
    "gesvd.cpp"
    "getrf.cpp"
    "getrf_batch_group.cpp"
    "getrf_batch_stride.cpp"
    "getri.cpp"
    "getri_batch_group.cpp"
    "getri_batch_stride.cpp"
    "getrs.cpp"
    "getrs_batch_group.cpp"
    "getrs_batch_stride.cpp"
    "heevd.cpp"
    "hegvd.cpp"
    "hetrd.cpp"
    "hetrf.cpp"
    "orgbr.cpp"
    "orgqr.cpp"
    "orgqr_batch_group.cpp"
    "orgqr_batch_stride.cpp"
    "orgtr.cpp"
    "ormqr.cpp"
    "ormrq.cpp"
    "ormtr.cpp"
    "potrf.cpp"
    "potrf_batch_group.cpp"
    "potrf_batch_stride.cpp"
    "potri.cpp"
    "potrs.cpp"
    "potrs_batch_group.cpp"
    "potrs_batch_stride.cpp"
    "syevd.cpp"
    "sygvd.cpp"
    "sytrd.cpp"
    "sytrf.cpp"
    "trtrs.cpp"
    "ungbr.cpp"
    "ungqr.cpp"
    "ungqr_batch_group.cpp"
    "ungqr_batch_stride.cpp"
    "ungtr.cpp"
    "unmqr.cpp"
    "unmrq.cpp"
    "unmtr.cpp"
)

if(LAPACKE_FOUND)
  list(APPEND LAPACK_SOURCES ${LAPACK_SOURCES_W_LAPACKE})
endif()

if(BUILD_SHARED_LIBS)
  add_library(lapack_source_rt OBJECT ${LAPACK_SOURCES})
  target_compile_options(lapack_source_rt PRIVATE -DCALL_RT_API -DNOMINMAX)
  target_include_directories(lapack_source_rt
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      PUBLIC ${PROJECT_SOURCE_DIR}/include
      PUBLIC ${PROJECT_SOURCE_DIR}/deps/googletest/include
      PUBLIC ${CMAKE_BINARY_DIR}/bin
      $<$<BOOL:${LAPACKE_FOUND}>:${LAPACKE_INCLUDE}>
  )
  target_link_libraries(lapack_source_rt PUBLIC ONEMKL::SYCL::SYCL)
endif()

add_library(lapack_source_ct OBJECT ${LAPACK_SOURCES})
target_compile_options(lapack_source_ct PRIVATE -DNOMINMAX)
target_include_directories(lapack_source_ct
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${PROJECT_SOURCE_DIR}/deps/googletest/include
    PUBLIC ${CMAKE_BINARY_DIR}/bin
    $<$<BOOL:${LAPACKE_FOUND}>:${LAPACKE_INCLUDE}>
)
target_link_libraries(lapack_source_ct PUBLIC ONEMKL::SYCL::SYCL)
