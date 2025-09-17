#==========================================================================
#  Copyright (C) Codeplay Software Limited
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  For your convenience, a copy of the License has been included in this
#  repository.
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#=========================================================================

set(LIB_NAME onemath_blas_generic)
set(LIB_OBJ ${LIB_NAME}_obj)

if(NOT DEFINED GENERIC_BLAS_TUNING_TARGET)
  option(GENERIC_BLAS_TUNING_TARGET "Set a TUNING_TARGET for generic backend" "")
endif()

if(DEFINED PORTBLAS_TUNING_TARGET)
  message(WARNING "PORTBLAS_TUNING_TARGET is deprecated, please use GENERIC_BLAS_TUNING_TARGET.")
  set(GENERIC_BLAS_TUNING_TARGET ${PORTBLAS_TUNING_TARGET})
endif()
# Parse compiler flags and return a list of SYCL targets
# The list is empty if no targets are set
function(get_sycl_targets FLAGS)
  string(REGEX MATCH "-fsycl-targets=[^ ]*" SYCL_TARGETS_FLAG "${FLAGS}")
  string(REPLACE "-fsycl-targets=" "" SYCL_TARGETS "${SYCL_TARGETS_FLAG}")
  string(REPLACE "," ";" SYCL_TARGETS "${SYCL_TARGETS}")
  set(SYCL_TARGETS ${SYCL_TARGETS} PARENT_SCOPE)
endfunction(get_sycl_targets)

# Generic blas backend supports tuning for some device types, but can only be
# compiled for one at a time currently. Work out which device to tune for based
# on the DPC++ target triple specified via -fsycl-targets
if(TARGET ONEMATH::SYCL::SYCL)
  get_target_property(ONEMATH_COMPILE_OPTIONS ONEMATH::SYCL::SYCL INTERFACE_COMPILE_OPTIONS)
endif()
get_sycl_targets("${ONEMATH_COMPILE_OPTIONS}")
list(LENGTH SYCL_TARGETS NUM_TARGETS)
if(NUM_TARGETS EQUAL 0)
  get_sycl_targets("${CMAKE_CXX_FLAGS}")
  list(LENGTH SYCL_TARGETS NUM_TARGETS)
endif()

if(GENERIC_BLAS_TUNING_TARGET)
  # Allow the user to manually enable a specific device type
  # for tuned generic backend configurations and sets sycl-target.
  if(GENERIC_BLAS_TUNING_TARGET STREQUAL "INTEL_CPU")
    set(ONEMATH_ENABLE_GENERIC_BLAS_BACKEND_INTEL_CPU "ON" CACHE INTERNAL "")
    set(GENERIC_BLAS_TUNING_TARGET "")
    target_compile_options(ONEMATH::SYCL::SYCL INTERFACE
      -fsycl-targets=spir64_x86_64 -fsycl-unnamed-lambda)
    target_link_options(ONEMATH::SYCL::SYCL INTERFACE
      -fsycl-targets=spir64_x86_64)
  elseif(GENERIC_BLAS_TUNING_TARGET STREQUAL "INTEL_GPU")
    set(ONEMATH_ENABLE_GENERIC_BLAS_BACKEND_INTEL_GPU "ON" CACHE INTERNAL "")
  elseif(GENERIC_BLAS_TUNING_TARGET STREQUAL "AMD_GPU")
    set(ONEMATH_ENABLE_GENERIC_BLAS_BACKEND_AMD_GPU "ON" CACHE INTERNAL "")
    if (is_dpcpp)
      target_compile_options(ONEMATH::SYCL::SYCL INTERFACE
        -fsycl-targets=amdgcn-amd-amdhsa -fsycl-unnamed-lambda
        -Xsycl-target-backend --offload-arch=${HIP_TARGETS})
      target_link_options(ONEMATH::SYCL::SYCL INTERFACE
        -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${HIP_TARGETS})
    else()
      message(WARNING "Compiler is not supported."
      " Unable to automatically set the required flags for the target '${GENERIC_BLAS_TUNING_TARGET}'."
      " Compilation may fail.")
    endif()
  elseif(GENERIC_BLAS_TUNING_TARGET STREQUAL "NVIDIA_GPU")
    set(ONEMATH_ENABLE_GENERIC_BLAS_BACKEND_NVIDIA_GPU "ON" CACHE INTERNAL "")
    if (is_dpcpp)
      target_compile_options(ONEMATH::SYCL::SYCL INTERFACE
        -fsycl-targets=nvptx64-nvidia-cuda -fsycl-unnamed-lambda)
      target_link_options(ONEMATH::SYCL::SYCL INTERFACE
        -fsycl-targets=nvptx64-nvidia-cuda)
      if(DEFINED CUDA_TARGETS)
        target_compile_options(ONEMATH::SYCL::SYCL INTERFACE
          -Xsycl-target-backend --cuda-gpu-arch=${CUDA_TARGETS})
        target_link_options(ONEMATH::SYCL::SYCL INTERFACE
          -Xsycl-target-backend --cuda-gpu-arch=${CUDA_TARGETS})
      endif()
    else()
      message(WARNING "Compiler is not supported."
      " Unable to automatically set the required flags for the target '${GENERIC_BLAS_TUNING_TARGET}'."
      " Compilation may fail.")
    endif()
  else()
    message(FATAL_ERROR "Unsupported GENERIC_BLAS_TUNING_TARGET: '${GENERIC_BLAS_TUNING_TARGET}'")
  endif()
elseif(NUM_TARGETS EQUAL 0)
  # Enable generic backend for all devices types
  set(ONEMATH_ENABLE_GENERIC_BLAS_BACKEND_INTEL_CPU "ON" CACHE INTERNAL "")
  set(ONEMATH_ENABLE_GENERIC_BLAS_BACKEND_INTEL_GPU "ON" CACHE INTERNAL "")
  set(ONEMATH_ENABLE_GENERIC_BLAS_BACKEND_AMD_GPU "ON" CACHE INTERNAL "")
  set(ONEMATH_ENABLE_GENERIC_BLAS_BACKEND_NVIDIA_GPU "ON" CACHE INTERNAL "")
else()
  # Try to automatically detect the GENERIC_BLAS_TUNING_TARGET
  foreach(SYCL_TARGET IN LISTS SYCL_TARGETS)
    if(SYCL_TARGETS MATCHES "^intel_gpu" OR SYCL_TARGETS MATCHES "^spir64_gen")
      set(ONEMATH_ENABLE_GENERIC_BLAS_BACKEND_INTEL_GPU "ON" CACHE INTERNAL "")
      set(GENERIC_BLAS_TUNING_TARGET "INTEL_GPU")
    elseif(SYCL_TARGETS MATCHES "^spir64_x86_64")
      set(ONEMATH_ENABLE_GENERIC_BLAS_BACKEND_INTEL_CPU "ON" CACHE INTERNAL "")
    elseif(SYCL_TARGETS MATCHES "^spir64")
      set(ONEMATH_ENABLE_GENERIC_BLAS_BACKEND_INTEL_CPU "ON" CACHE INTERNAL "")
      set(ONEMATH_ENABLE_GENERIC_BLAS_BACKEND_INTEL_GPU "ON" CACHE INTERNAL "")
      set(GENERIC_BLAS_TUNING_TARGET "INTEL_GPU")
    elseif(SYCL_TARGETS MATCHES "^amd_gpu" OR SYCL_TARGETS MATCHES "-amd-")
      set(ONEMATH_ENABLE_GENERIC_BLAS_BACKEND_AMD_GPU "ON" CACHE INTERNAL "")
      set(GENERIC_BLAS_TUNING_TARGET "AMD_GPU")
    elseif(SYCL_TARGETS MATCHES "^nvidia_gpu" OR SYCL_TARGETS MATCHES "-nvidia-")
      set(ONEMATH_ENABLE_GENERIC_BLAS_BACKEND_NVIDIA_GPU "ON" CACHE INTERNAL "")
      set(GENERIC_BLAS_TUNING_TARGET "NVIDIA_GPU")
    endif()
  endforeach()
  # Currently generic can only be tuned for one type of device.
  if(NUM_TARGETS GREATER 1)
    set(GENERIC_BLAS_TUNING_TARGET "")
  endif()
endif()

if(GENERIC_BLAS_TUNING_TARGET STREQUAL "INTEL_GPU")
  message(STATUS "Tuning generic BLAS for Intel GPU devices")
elseif(GENERIC_BLAS_TUNING_TARGET STREQUAL "AMD_GPU")
  message(STATUS "Tuning generic BLAS for AMD GPU devices")
elseif(GENERIC_BLAS_TUNING_TARGET STREQUAL "NVIDIA_GPU")
  message(STATUS "Tuning generic BLAS for Nvidia GPU devices")
else()
  message(STATUS "generic BLAS is not tuned for any device which can impact performance")
endif()

# If find_package doesn't work, download onemath_sycl_blas from Github. This is
# intended to make oneMath easier to use.
message(STATUS "Looking for oneMATH blas kernels")
find_package(ONEMATH_SYCL_BLAS 0.2.0 QUIET)
if (NOT ONEMATH_SYCL_BLAS_FOUND)
  message(STATUS "Looking for onemath_sycl_blas for generic backend - could not find onemath_sycl_blas with ONEMATH_SYCL_BLAS_DIR")
  include(FetchContent)
  if(NOT GENERIC_BLAS_TUNING_TARGET)
    set(GENERIC_BLAS_TUNING_TARGET "DEFAULT")
  endif()
  # Following variable TUNING_TARGET will be used in generic blas internal configuration
  set(TUNING_TARGET ${GENERIC_BLAS_TUNING_TARGET})
  # Set the policy to forward variables to generic blas configure step
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/deps")
  FetchContent_Declare(
    onemath_sycl_blas
    GIT_REPOSITORY https://github.com/uxlfoundation/generic-sycl-components
    GIT_TAG        main
    SOURCE_SUBDIR onemath/sycl/blas
  )
  FetchContent_MakeAvailable(onemath_sycl_blas)
  install(
    TARGETS onemath_sycl_blas
    EXPORT oneMathTargets
  )
  message(STATUS "Looking for onemath_sycl_blas - downloaded")

else()
  message(STATUS "Looking for oneMath blas kernels - found")
  add_library(onemath_sycl_blas ALIAS ONEMATH_SYCL_BLAS::onemath_sycl_blas)
endif()

# Read cmake options exported by the onemath_sycl_blas project into oneMath variables
set(ONEMATH_GENERIC_BLAS_ENABLE_COMPLEX ${BLAS_ENABLE_COMPLEX} CACHE INTERNAL "Enable complex data type support")
set(ONEMATH_GENERIC_BLAS_ENABLE_USM ${BLAS_ENABLE_USM} CACHE INTERNAL "Enable USM API support")

set(SOURCES
  generic_level1_double.cpp generic_level1_float.cpp
  generic_level2_double.cpp generic_level2_float.cpp
  generic_level3_double.cpp generic_level3_float.cpp
  generic_level3_half.cpp generic_level3_bfloat16.cpp
  generic_batch.cpp
  $<$<BOOL:${BUILD_SHARED_LIBS}>: generic_wrappers.cpp>)
add_library(${LIB_NAME})
add_deprecated_library(${LIB_NAME})
add_library(${LIB_OBJ} OBJECT ${SOURCES})
add_dependencies(onemath_backend_libs_blas ${LIB_NAME})

if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET ${LIB_OBJ} SOURCES ${SOURCES})
endif()

target_include_directories(${LIB_OBJ}
  PUBLIC  ${ONEMATH_INCLUDE_DIRS}
  PRIVATE ${PROJECT_SOURCE_DIR}/src/include
          ${PROJECT_SOURCE_DIR}/src
          ${CMAKE_BINARY_DIR}/bin
          ${ONEMATH_GENERATED_INCLUDE_PATH}
)

target_compile_options(${LIB_OBJ} PRIVATE ${ONEMATH_BUILD_COPT})
target_link_libraries(${LIB_OBJ} PUBLIC ONEMATH::SYCL::SYCL onemath_sycl_blas)

set_target_properties(${LIB_OBJ} PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${LIB_NAME} PRIVATE ${LIB_OBJ})
target_include_directories(${LIB_NAME} PUBLIC ${ONEMATH_INCLUDE_DIRS})

if(BUILD_SHARED_LIBS)
  set_target_properties(${LIB_NAME} PROPERTIES
    INTERFACE_LINK_LIBRARIES ONEMATH::SYCL::SYCL
  )
endif()

# Add major version to the library
set_target_properties(${LIB_NAME} PROPERTIES
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Add dependencies rpath to the library
list(APPEND CMAKE_BUILD_RPATH $<TARGET_FILE_DIR:${LIB_NAME}>)

# Add the library to install package
install(TARGETS ${LIB_OBJ} EXPORT oneMathTargets)
install(TARGETS ${LIB_NAME} EXPORT oneMathTargets
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
