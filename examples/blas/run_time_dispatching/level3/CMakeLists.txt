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

# NOTE: user needs to set env var ONEAPI_DEVICE_SELECTOR to use runtime example without specifying backend in CMake
# $ENV{ONEAPI_DEVICE_SELECTOR}


# Build object from all example sources
set(BLAS_RT_SOURCES "gemm_usm")

# Set up for the right backend for run-time dispatching examples
# If users build more than one backend (i.e. mklcpu and mklgpu, or mklcpu and CUDA), they may need to
# overwrite ONEAPI_DEVICE_SELECTOR in their environment to run on the desired backend
set(DEVICE_FILTERS "")
if(ENABLE_MKLCPU_BACKEND)
  list(APPEND DEVICE_FILTERS "opencl:cpu")
endif()
if(ENABLE_MKLGPU_BACKEND)
  list(APPEND DEVICE_FILTERS "level_zero:gpu")
endif()
if(ENABLE_CUBLAS_BACKEND)
  list(APPEND DEVICE_FILTERS "cuda:gpu")
endif()
if(ENABLE_ROCBLAS_BACKEND)
  list(APPEND DEVICE_FILTERS "hip:gpu")
endif()
if(ENABLE_PORTBLAS_BACKEND)
  if(PORTBLAS_TUNING_TARGET)
    if(PORTBLAS_TUNING_TARGET MATCHES "INTEL_CPU")
      list(APPEND DEVICE_FILTERS "opencl:cpu")
    elseif(PORTBLAS_TUNING_TARGET MATCHES "_GPU")
      list(APPEND DEVICE_FILTERS "*:gpu")
    endif()
  else()
    # portBLAS default sycl-target is spir64, testing runtime on both supported
    # devices.
    list(APPEND DEVICE_FILTERS "opencl:cpu;level_zero:gpu")
  endif()
endif()

message(STATUS "ONEAPI_DEVICE_SELECTOR will be set to the following value(s): [${DEVICE_FILTERS}] for run-time dispatching examples")

foreach(blas_rt_source ${BLAS_RT_SOURCES})
  add_executable(example_${domain}_${blas_rt_source} ${blas_rt_source}.cpp)
  target_include_directories(example_${domain}_${blas_rt_source}
      PUBLIC ${PROJECT_SOURCE_DIR}/examples/include
      PUBLIC ${PROJECT_SOURCE_DIR}/include
      PUBLIC ${CMAKE_BINARY_DIR}/bin
  )

  add_dependencies(example_${domain}_${blas_rt_source} onemath)

  if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
    add_sycl_to_target(TARGET example_${domain}_${blas_rt_source} SOURCES ${BLAS_RT_SOURCES})
  endif()

  target_link_libraries(example_${domain}_${blas_rt_source} PUBLIC
      onemath
      ONEMATH::SYCL::SYCL
      ${CMAKE_DL_LIBS}
  )

  # Register example as ctest
  foreach(device_filter ${DEVICE_FILTERS})
    add_test(NAME ${domain}/EXAMPLE/RT/${blas_rt_source}/${device_filter} COMMAND example_${domain}_${blas_rt_source})
    set_property(TEST ${domain}/EXAMPLE/RT/${blas_rt_source}/${device_filter} PROPERTY
      ENVIRONMENT LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}
      ENVIRONMENT ONEAPI_DEVICE_SELECTOR=${device_filter})
  endforeach(device_filter)

endforeach(blas_rt_source)
