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
set(DFT_CT_SOURCES "")
if (ENABLE_MKLCPU_BACKEND AND ENABLE_CUFFT_BACKEND)
  list(APPEND DFT_CT_SOURCES "complex_fwd_usm_mklcpu_cufft")
endif()

include(WarningsUtils)

foreach(dft_ct_source ${DFT_CT_SOURCES})
  set(EXAMPLE_NAME example_${domain}_${dft_ct_source})
  add_executable(${EXAMPLE_NAME} ${dft_ct_source}.cpp)
  target_include_directories(${EXAMPLE_NAME}
      PUBLIC ${PROJECT_SOURCE_DIR}/examples/include
      PUBLIC ${CMAKE_BINARY_DIR}/bin
  )

  if(domain STREQUAL "dft" AND ENABLE_MKLCPU_BACKEND AND ENABLE_CUFFT_BACKEND)
    add_dependencies(${EXAMPLE_NAME} onemath_${domain}_mklcpu onemath_${domain}_cufft)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_${domain}_mklcpu onemath_${domain}_cufft)
  endif()

  target_link_libraries(${EXAMPLE_NAME} PUBLIC
    ${ONEMATH_LIBRARIES_${domain}}
    onemath_warnings
  )

  # Register example as ctest
  add_test(NAME dft/EXAMPLE/CT/${dft_ct_source} COMMAND ${EXAMPLE_NAME})

endforeach(dft_ct_source)

