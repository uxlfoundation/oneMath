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

set(LIB_NAME onemkl_blas_portblas)
set(LIB_OBJ ${LIB_NAME}_obj)

if(NOT DEFINED PORTBLAS_TUNING_TARGET)
  option(PORTBLAS_TUNING_TARGET "Set a TUNING_TARGET for portBLAS" "")
endif()

# Parse compiler flags and return a list of SYCL targets
# The list is empty if no targets are set
function(get_sycl_targets FLAGS)
  string(REGEX MATCH "-fsycl-targets=[^ ]*" SYCL_TARGETS_FLAG "${FLAGS}")
  string(REPLACE "-fsycl-targets=" "" SYCL_TARGETS "${SYCL_TARGETS_FLAG}")
  string(REPLACE "," ";" SYCL_TARGETS "${SYCL_TARGETS}")
  set(SYCL_TARGETS ${SYCL_TARGETS} PARENT_SCOPE)
endfunction(get_sycl_targets)

# portBLAS supports tuning for some device types, but can only be compiled
# for one at a time currently. Work out which device to tune for based on the
# DPC++ target triple specified via -fsycl-targets
if(TARGET ONEMKL::SYCL::SYCL)
  get_target_property(ONEMKL_COMPILE_OPTIONS ONEMKL::SYCL::SYCL INTERFACE_COMPILE_OPTIONS)
endif()
get_sycl_targets("${ONEMKL_COMPILE_OPTIONS}")
list(LENGTH SYCL_TARGETS NUM_TARGETS)
if(NUM_TARGETS EQUAL 0)
  get_sycl_targets("${CMAKE_CXX_FLAGS}")
  list(LENGTH SYCL_TARGETS NUM_TARGETS)
endif()

if(PORTBLAS_TUNING_TARGET)
  # Allow the user to manually enable a specific device type 
  # for tuned portBLAS configurations and sets sycl-target.
  if(PORTBLAS_TUNING_TARGET STREQUAL "INTEL_CPU")
    set(ONEMKL_ENABLE_PORTBLAS_BACKEND_INTEL_CPU "ON" CACHE INTERNAL "")
    set(PORTBLAS_TUNING_TARGET "")
    target_compile_options(ONEMKL::SYCL::SYCL INTERFACE
      -fsycl-targets=spir64_x86_64 -fsycl-unnamed-lambda)
    target_link_options(ONEMKL::SYCL::SYCL INTERFACE
      -fsycl-targets=spir64_x86_64)
  elseif(PORTBLAS_TUNING_TARGET STREQUAL "INTEL_GPU")
    set(ONEMKL_ENABLE_PORTBLAS_BACKEND_INTEL_GPU "ON" CACHE INTERNAL "")
  elseif(PORTBLAS_TUNING_TARGET STREQUAL "AMD_GPU")
    set(ONEMKL_ENABLE_PORTBLAS_BACKEND_AMD_GPU "ON" CACHE INTERNAL "")
    if (is_dpcpp)
      target_compile_options(ONEMKL::SYCL::SYCL INTERFACE
        -fsycl-targets=amdgcn-amd-amdhsa -fsycl-unnamed-lambda
        -Xsycl-target-backend --offload-arch=${HIP_TARGETS})
      target_link_options(ONEMKL::SYCL::SYCL INTERFACE
        -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${HIP_TARGETS})
    else()
      message(WARNING "Compiler is not supported."
      " Unable to automatically set the required flags for the target '${PORTBLAS_TUNING_TARGET}'."
      " Compilation may fail.")
    endif()
  elseif(PORTBLAS_TUNING_TARGET STREQUAL "NVIDIA_GPU")
    set(ONEMKL_ENABLE_PORTBLAS_BACKEND_NVIDIA_GPU "ON" CACHE INTERNAL "")
    if (is_dpcpp)
      target_compile_options(ONEMKL::SYCL::SYCL INTERFACE
        -fsycl-targets=nvptx64-nvidia-cuda -fsycl-unnamed-lambda)
      target_link_options(ONEMKL::SYCL::SYCL INTERFACE
        -fsycl-targets=nvptx64-nvidia-cuda)
      if(DEFINED CUDA_TARGET)
        target_compile_options(ONEMKL::SYCL::SYCL INTERFACE
          -Xsycl-target-backend --cuda-gpu-arch=${CUDA_TARGET})
        target_link_options(ONEMKL::SYCL::SYCL INTERFACE
          -Xsycl-target-backend --cuda-gpu-arch=${CUDA_TARGET})
      endif()
    else()
      message(WARNING "Compiler is not supported."
      " Unable to automatically set the required flags for the target '${PORTBLAS_TUNING_TARGET}'."
      " Compilation may fail.")
    endif()
  else()
    message(FATAL_ERROR "Unsupported PORTBLAS_TUNING_TARGET: '${PORTBLAS_TUNING_TARGET}'")
  endif()
elseif(NUM_TARGETS EQUAL 0)
  # Enable portBLAS backend for all devices types
  set(ONEMKL_ENABLE_PORTBLAS_BACKEND_INTEL_CPU "ON" CACHE INTERNAL "")
  set(ONEMKL_ENABLE_PORTBLAS_BACKEND_INTEL_GPU "ON" CACHE INTERNAL "")
  set(ONEMKL_ENABLE_PORTBLAS_BACKEND_AMD_GPU "ON" CACHE INTERNAL "")
  set(ONEMKL_ENABLE_PORTBLAS_BACKEND_NVIDIA_GPU "ON" CACHE INTERNAL "")
else()
  # Try to automatically detect the PORTBLAS_TUNING_TARGET
  foreach(SYCL_TARGET IN LISTS SYCL_TARGETS)
    if(SYCL_TARGETS MATCHES "^intel_gpu" OR SYCL_TARGETS MATCHES "^spir64_gen")
      set(ONEMKL_ENABLE_PORTBLAS_BACKEND_INTEL_GPU "ON" CACHE INTERNAL "")
      set(PORTBLAS_TUNING_TARGET "INTEL_GPU")
    elseif(SYCL_TARGETS MATCHES "^spir64_x86_64")
      set(ONEMKL_ENABLE_PORTBLAS_BACKEND_INTEL_CPU "ON" CACHE INTERNAL "")
    elseif(SYCL_TARGETS MATCHES "^spir64")
      set(ONEMKL_ENABLE_PORTBLAS_BACKEND_INTEL_CPU "ON" CACHE INTERNAL "")
      set(ONEMKL_ENABLE_PORTBLAS_BACKEND_INTEL_GPU "ON" CACHE INTERNAL "")
      set(PORTBLAS_TUNING_TARGET "INTEL_GPU")
    elseif(SYCL_TARGETS MATCHES "^amd_gpu" OR SYCL_TARGETS MATCHES "-amd-")
      set(ONEMKL_ENABLE_PORTBLAS_BACKEND_AMD_GPU "ON" CACHE INTERNAL "")
      set(PORTBLAS_TUNING_TARGET "AMD_GPU")
    elseif(SYCL_TARGETS MATCHES "^nvidia_gpu" OR SYCL_TARGETS MATCHES "-nvidia-")
      set(ONEMKL_ENABLE_PORTBLAS_BACKEND_NVIDIA_GPU "ON" CACHE INTERNAL "")
      set(PORTBLAS_TUNING_TARGET "NVIDIA_GPU")
    endif()
  endforeach()
  # Currently portBLAS can only be tuned for one type of device.
  if(NUM_TARGETS GREATER 1)
    set(PORTBLAS_TUNING_TARGET "")
  endif()
endif()

if(PORTBLAS_TUNING_TARGET STREQUAL "INTEL_GPU")
  message(STATUS "Tuning portBLAS for Intel GPU devices")
elseif(PORTBLAS_TUNING_TARGET STREQUAL "AMD_GPU")
  message(STATUS "Tuning portBLAS for AMD GPU devices")
elseif(PORTBLAS_TUNING_TARGET STREQUAL "NVIDIA_GPU")
  message(STATUS "Tuning portBLAS for Nvidia GPU devices")
else()
  message(STATUS "portBLAS is not tuned for any device which can impact performance")
endif()

# If find_package doesn't work, download portBLAS from Github. This is
# intended to make OneMKL easier to use.
message(STATUS "Looking for portBLAS")
find_package(PORTBLAS QUIET)
if (NOT PORTBLAS_FOUND)
  message(STATUS "Looking for portBLAS - could not find portBLAS with PORTBLAS_DIR")
  include(FetchContent)
  set(INSTALL_HEADER_ONLY ON)
  set(BLAS_BUILD_SAMPLES OFF)
  set(BLAS_ENABLE_BENCHMARK OFF)
  set(BLAS_ENABLE_TESTING OFF)
  set(ENABLE_EXPRESSION_TESTS OFF)
  if(NOT PORTBLAS_TUNING_TARGET)
    set(PORTBLAS_TUNING_TARGET "DEFAULT")
  endif()
  # Following variable TUNING_TARGET will be used in portBLAS internal configuration
  set(TUNING_TARGET ${PORTBLAS_TUNING_TARGET})
  set(BLAS_ENABLE_COMPLEX ON)
  # Set the policy to forward variables to portBLAS configure step
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/deps")
  FetchContent_Declare(
    portBLAS
    GIT_REPOSITORY https://github.com/codeplaysoftware/portBLAS
    GIT_TAG        main
  )
  FetchContent_MakeAvailable(portblas)
  message(STATUS "Looking for portBLAS - downloaded")

else()
  message(STATUS "Looking for portBLAS - found")
  add_library(portblas ALIAS PORTBLAS::portblas)
endif()

set(SOURCES
  portblas_level1_double.cpp portblas_level1_float.cpp
  portblas_level2_double.cpp portblas_level2_float.cpp
  portblas_level3_double.cpp portblas_level3_float.cpp
  portblas_level3_half.cpp portblas_level3_bfloat16.cpp
  portblas_batch.cpp
  $<$<BOOL:${BUILD_SHARED_LIBS}>: portblas_wrappers.cpp>)
add_library(${LIB_NAME})
add_library(${LIB_OBJ} OBJECT ${SOURCES})
add_dependencies(onemkl_backend_libs_blas ${LIB_NAME})

if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET ${LIB_OBJ} SOURCES ${SOURCES})
endif()

target_include_directories(${LIB_OBJ}
  PRIVATE ${PROJECT_SOURCE_DIR}/include
          ${PROJECT_SOURCE_DIR}/src/include
          ${PROJECT_SOURCE_DIR}/src
          ${CMAKE_BINARY_DIR}/bin
          ${ONEMKL_GENERATED_INCLUDE_PATH}
)

target_compile_options(${LIB_OBJ} PRIVATE ${ONEMKL_BUILD_COPT})
target_link_libraries(${LIB_OBJ} PUBLIC ONEMKL::SYCL::SYCL portblas)

set_target_properties(${LIB_OBJ} PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${LIB_NAME} PUBLIC ${LIB_OBJ})

if(BUILD_SHARED_LIBS)
  set_target_properties(${LIB_NAME} PROPERTIES
    INTERFACE_LINK_LIBRARIES ONEMKL::SYCL::SYCL
  )
endif()

# Add major version to the library
set_target_properties(${LIB_NAME} PROPERTIES
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Add dependencies rpath to the library
list(APPEND CMAKE_BUILD_RPATH $<TARGET_FILE_DIR:${LIB_NAME}>)

# Add the library to install package
install(TARGETS ${LIB_OBJ} EXPORT oneMKLTargets)
install(TARGETS ${LIB_NAME} EXPORT oneMKLTargets
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
