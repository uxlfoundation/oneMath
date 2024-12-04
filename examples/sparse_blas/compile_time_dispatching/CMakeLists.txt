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

#Build object from all sources
set(SPARSE_CT_SOURCES "")
if(ENABLE_MKLCPU_BACKEND AND ENABLE_CUSPARSE_BACKEND)
  list(APPEND SPARSE_CT_SOURCES "sparse_blas_spmv_usm_mklcpu_cusparse")
endif()

include(WarningsUtils)

foreach(sparse_ct_source ${SPARSE_CT_SOURCES})
  add_executable(${sparse_ct_source} ${sparse_ct_source}.cpp)
  target_include_directories(${sparse_ct_source}
      PUBLIC ${PROJECT_SOURCE_DIR}/examples/include
      PUBLIC ${PROJECT_SOURCE_DIR}/include
      PUBLIC ${CMAKE_BINARY_DIR}/bin
  )

  target_link_libraries(${sparse_ct_source} PRIVATE ONEMATH::SYCL::SYCL onemath_sparse_blas_mklcpu onemath_sparse_blas_cusparse)

  # Register example as ctest
  add_test(NAME sparse_blas/EXAMPLE/CT/${sparse_ct_source} COMMAND ${sparse_ct_source})
endforeach(sparse_ct_source)

