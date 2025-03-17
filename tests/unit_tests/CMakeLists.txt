#===============================================================================
# Copyright 2020-2023 Intel Corporation
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

set(TEST_TARGET_DOMAINS ${TARGET_DOMAINS})

if("blas" IN_LIST TEST_TARGET_DOMAINS)
  find_package(CBLAS)
  if(NOT CBLAS_FOUND)
    # TODO: add list of tests without Netlib dependency
    message(WARNING "Netlib CBLAS headers or libraries are not found, BLAS unit tests will be skipped")
    list(REMOVE_ITEM TEST_TARGET_DOMAINS "blas")
  endif()
endif()

if("lapack" IN_LIST TEST_TARGET_DOMAINS)
  find_package(LAPACKE)
  if(NOT LAPACKE_FOUND)
    # TODO: add list of tests without Netlib dependency
    message(WARNING "Netlib LAPACKE headers or libraries are not found, LAPACK unit tests will be skipped")
    list(REMOVE_ITEM TEST_TARGET_DOMAINS "lapack")
  endif()
endif()

foreach(domain ${TEST_TARGET_DOMAINS})
  # Build tests first
  add_subdirectory(${domain})
endforeach()

include(GoogleTest)

get_target_property(GTEST_INCLUDE_DIR gtest INTERFACE_INCLUDE_DIRECTORIES)

# Build final test binaries: test_main_rt is for testing RunTime API (RT), test_main_ct is for testing CompileTime API (CT)

# BLAS config
set(blas_TEST_LIST
        blas_level1
        blas_level2
        blas_level3
        blas_batch
        blas_extensions)

if(${ONEMATH_SYCL_IMPLEMENTATION} STREQUAL "dpc++")
  list(APPEND blas_TEST_LIST blas_sycl_graph)
endif()

set(blas_TEST_LINK "")

# LAPACK config
set(lapack_TEST_LIST
        lapack_source)

if(LAPACKE_FOUND)
  set(lapack_TEST_LINK ${LAPACKE_LINK})
endif()

# RNG config
set(rng_TEST_LIST
      rng_statistics
      rng_service)
set(rng_DEVICE_TEST_LIST
      rng_device_moments
      rng_device_service
)

set(rng_TEST_LINK "")

# DFT config
set(dft_TEST_LIST
      dft_source)

set(dft_TEST_LINK "")

# Sparse BLAS config
set(sparse_blas_TEST_LIST
      spblas_source)

set(sparse_blas_TEST_LINK "")

foreach(domain ${TEST_TARGET_DOMAINS})
  # Generate RT and CT test lists
  set(${domain}_TEST_LIST_RT ${${domain}_TEST_LIST})
  set(${domain}_TEST_LIST_CT ${${domain}_TEST_LIST})
  set(${domain}_DEVICE_TEST_LIST_CT ${${domain}_DEVICE_TEST_LIST})
  list(TRANSFORM ${domain}_TEST_LIST_RT APPEND _rt)
  list(TRANSFORM ${domain}_TEST_LIST_CT APPEND _ct)
  list(TRANSFORM ${domain}_DEVICE_TEST_LIST_CT APPEND _ct)

  add_executable(test_main_${domain}_ct main_test.cpp)
  target_include_directories(test_main_${domain}_ct PUBLIC ${GTEST_INCLUDE_DIR})
  
  if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
    add_sycl_to_target(TARGET test_main_${domain}_ct SOURCES main_test.cpp)
  else()
    target_compile_options(test_main_${domain}_ct PRIVATE -fsycl)
  endif()

  if(BUILD_SHARED_LIBS)
    add_executable(test_main_${domain}_rt main_test.cpp)
    target_include_directories(test_main_${domain}_rt PUBLIC ${GTEST_INCLUDE_DIR})
    if(NOT ${ONEMATH_SYCL_IMPLEMENTATION} STREQUAL "adaptivecpp")
      target_compile_options(test_main_${domain}_rt PRIVATE -fsycl)
    endif()
    target_link_libraries(test_main_${domain}_rt PUBLIC
      gtest
      gtest_main
      ${CMAKE_DL_LIBS}
      ${${domain}_TEST_LINK}
      ONEMATH::SYCL::SYCL
      onemath
      ${${domain}_TEST_LIST_RT}
    )
    if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
      add_sycl_to_target(TARGET test_main_${domain}_rt SOURCES main_test.cpp)
    endif()
  endif()

  if(ENABLE_MKLCPU_BACKEND)
    add_dependencies(test_main_${domain}_ct onemath_${domain}_mklcpu)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_${domain}_mklcpu)
  endif()

  if(ENABLE_MKLGPU_BACKEND)
    add_dependencies(test_main_${domain}_ct onemath_${domain}_mklgpu)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_${domain}_mklgpu)
  endif()

  if(domain STREQUAL "blas" AND ENABLE_CUBLAS_BACKEND)
    add_dependencies(test_main_${domain}_ct onemath_${domain}_cublas)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_${domain}_cublas)
  endif()

  if(domain STREQUAL "blas" AND ENABLE_ROCBLAS_BACKEND)
    add_dependencies(test_main_${domain}_ct onemath_${domain}_rocblas)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_${domain}_rocblas)
  endif()

  if(domain STREQUAL "blas" AND ENABLE_NETLIB_BACKEND)
    add_dependencies(test_main_${domain}_ct onemath_${domain}_netlib)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_${domain}_netlib)
  endif()

  if(domain STREQUAL "blas" AND ENABLE_ARMPL_BACKEND)
    add_dependencies(test_main_${domain}_ct onemath_${domain}_armpl)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_${domain}_armpl)
  endif()

  if(domain STREQUAL "blas" AND ENABLE_GENERIC_BLAS_BACKEND)
    add_dependencies(test_main_${domain}_ct onemath_${domain}_generic)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_${domain}_generic)
  endif()

  if(domain STREQUAL "lapack" AND ENABLE_CUSOLVER_BACKEND)
    add_dependencies(test_main_${domain}_ct onemath_${domain}_cusolver)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_${domain}_cusolver)
  endif()

  if(domain STREQUAL "lapack" AND ENABLE_ROCSOLVER_BACKEND)
    add_dependencies(test_main_${domain}_ct onemath_${domain}_rocsolver)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_${domain}_rocsolver)
  endif()

  if(domain STREQUAL "lapack" AND ENABLE_ARMPL_BACKEND)
    add_dependencies(test_main_${domain}_ct onemath_${domain}_armpl)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_${domain}_armpl)
  endif()

  if(domain STREQUAL "rng" AND ENABLE_CURAND_BACKEND)
    add_dependencies(test_main_${domain}_ct onemath_${domain}_curand)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_${domain}_curand)
  endif()

  if(domain STREQUAL "rng" AND ENABLE_ROCRAND_BACKEND)
    add_dependencies(test_main_${domain}_ct onemath_${domain}_rocrand)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_${domain}_rocrand)
  endif()

  if(domain STREQUAL "rng" AND ENABLE_ARMPL_BACKEND)
    add_dependencies(test_main_${domain}_ct onemath_${domain}_armpl)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_${domain}_armpl)
  endif()

  if(domain STREQUAL "dft" AND ENABLE_CUFFT_BACKEND)
    add_dependencies(test_main_${domain}_ct onemath_${domain}_cufft)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_${domain}_cufft)
  endif()

  if(domain STREQUAL "dft" AND ENABLE_ROCFFT_BACKEND)
    add_dependencies(test_main_${domain}_ct onemath_dft_rocfft)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_dft_rocfft)
  endif()

  if(domain STREQUAL "dft" AND ENABLE_PORTFFT_BACKEND)
    add_dependencies(test_main_${domain}_ct onemath_dft_portfft)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_dft_portfft)
  endif()

  if(domain STREQUAL "dft" AND ENABLE_ARMPL_BACKEND)
    add_dependencies(test_main_${domain}_ct onemath_${domain}_armpl)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_${domain}_armpl)
  endif()

  if(domain STREQUAL "sparse_blas" AND ENABLE_CUSPARSE_BACKEND)
    add_dependencies(test_main_${domain}_ct onemath_${domain}_cusparse)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_${domain}_cusparse)
  endif()

  if(domain STREQUAL "sparse_blas" AND ENABLE_ROCSPARSE_BACKEND)
    add_dependencies(test_main_${domain}_ct onemath_${domain}_rocsparse)
    list(APPEND ONEMATH_LIBRARIES_${domain} onemath_${domain}_rocsparse)
  endif()

  target_link_libraries(test_main_${domain}_ct PUBLIC
      gtest
      gtest_main
      ${CMAKE_DL_LIBS}
      ${${domain}_TEST_LINK}
      ${ONEMATH_LIBRARIES_${domain}}
      ONEMATH::SYCL::SYCL
      ${${domain}_TEST_LIST_CT}
      ${${domain}_DEVICE_TEST_LIST_CT}
  )
  
  if(NOT ${ONEMATH_SYCL_IMPLEMENTATION} STREQUAL "adaptivecpp")
    target_link_options(test_main_${domain}_ct PUBLIC -fsycl-device-code-split=per_kernel)
  endif()

  string(TOUPPER ${domain} DOMAIN_PREFIX)

  if(domain STREQUAL "blas" AND CBLAS_FOUND)
    set(TEST_LD_LIBRARY_PATH ${CMAKE_BINARY_DIR}/lib:${CBLAS_LIB_DIR}:$ENV{LD_LIBRARY_PATH})
  else()
    set(TEST_LD_LIBRARY_PATH ${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH})
  endif()

  if(BUILD_SHARED_LIBS)
    set_target_properties(test_main_${domain}_rt
      PROPERTIES BUILD_RPATH $<TARGET_FILE_DIR:onemath>)
  # Find individual tests within executable
    gtest_discover_tests(test_main_${domain}_rt
      PROPERTIES BUILD_RPATH ${CMAKE_BINARY_DIR}/lib
      PROPERTIES ENVIRONMENT LD_LIBRARY_PATH=${TEST_LD_LIBRARY_PATH}
      PROPERTIES TEST_PREFIX ${DOMAIN_PREFIX}/RT/
      DISCOVERY_TIMEOUT 30
    )
  endif()

  gtest_discover_tests(test_main_${domain}_ct
    PROPERTIES BUILD_RPATH ${CMAKE_BINARY_DIR}/lib
    PROPERTIES ENVIRONMENT LD_LIBRARY_PATH=${TEST_LD_LIBRARY_PATH}
    PROPERTIES TEST_PREFIX ${DOMAIN_PREFIX}/CT/
    DISCOVERY_TIMEOUT 30
  )

  if(BUILD_SHARED_LIBS)
    if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
      add_sycl_to_target(TARGET test_main_${domain}_rt)
    endif()
  endif()
endforeach()
