#===============================================================================
# Copyright 2022 Intel Corporation
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
set(BLAS_CT_SOURCES "")
if(ENABLE_MKLCPU_BACKEND AND ENABLE_CUBLAS_BACKEND)
  list(APPEND BLAS_CT_SOURCES "gemm_usm_mklcpu_cublas")
endif()

foreach(blas_ct_source ${BLAS_CT_SOURCES})
  add_executable(example_${domain}_${blas_ct_source} ${blas_ct_source}.cpp)
  target_include_directories(example_${domain}_${blas_ct_source}
      PUBLIC ${PROJECT_SOURCE_DIR}/examples/include
      PUBLIC ${PROJECT_SOURCE_DIR}/include
      PUBLIC ${CMAKE_BINARY_DIR}/bin
  )

  if(domain STREQUAL "blas" AND ENABLE_MKLCPU_BACKEND AND ENABLE_CUBLAS_BACKEND)
    add_dependencies(example_${domain}_${blas_ct_source} onemath_${domain}_mklcpu onemath_${domain}_cublas)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_${domain}_mklcpu onemath_${domain}_cublas)
  endif()

  target_link_libraries(example_${domain}_${blas_ct_source} PUBLIC
      ${ONEMATH_LIBRARIES_${domain}}
      ONEMATH::SYCL::SYCL
  )

  # Register example as ctest
  add_test(NAME ${domain}/EXAMPLE/CT/${blas_ct_source} COMMAND example_${domain}_${blas_ct_source})

endforeach(blas_ct_source)
