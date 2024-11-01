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

# Define common build flags for oneMKL libraries
set(ONEMKL_BUILD_COPT "")
if(WIN32 AND BUILD_SHARED_LIBS)
  list(APPEND ONEMKL_BUILD_COPT "-Donemkl_EXPORTS")
endif()

# store path to CMAKE_CURRENT_BINARY_DIR to use it later (makes FetchContent_Declare workable)
set(ONEMKL_GENERATED_INCLUDE_PATH ${CMAKE_CURRENT_BINARY_DIR})


set(ONEMKL_INTERFACE_INCLUDE_DIRS
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
)

# Build loader and backends for each domain
add_custom_target(onemkl_backend_libs)
foreach(domain ${TARGET_DOMAINS})
  add_subdirectory(${domain})
endforeach()

# Generate header with enabled backends for testing

function(generate_header_file)
  # Following if-conditions allow to decouple cmake configuration variables with
  # the corresponding generated macro. This is done to be conformant with ES.33
  # C++ Core Guidelines
  set(ONEMKL_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  set(ONEMKL_ENABLE_MKLCPU_BACKEND ${ENABLE_MKLCPU_BACKEND})
  set(ONEMKL_ENABLE_MKLGPU_BACKEND ${ENABLE_MKLGPU_BACKEND})
  set(ONEMKL_ENABLE_CUBLAS_BACKEND ${ENABLE_CUBLAS_BACKEND})
  set(ONEMKL_ENABLE_ROCBLAS_BACKEND ${ENABLE_ROCBLAS_BACKEND})
  set(ONEMKL_ENABLE_NETLIB_BACKEND ${ENABLE_NETLIB_BACKEND})
  set(ONEMKL_ENABLE_PORTBLAS_BACKEND ${ENABLE_PORTBLAS_BACKEND})
  set(ONEMKL_ENABLE_CURAND_BACKEND ${ENABLE_CURAND_BACKEND})
  set(ONEMKL_ENABLE_ROCRAND_BACKEND ${ENABLE_ROCRAND_BACKEND})
  set(ONEMKL_ENABLE_CUSOLVER_BACKEND ${ENABLE_CUSOLVER_BACKEND})
  set(ONEMKL_ENABLE_ROCSOLVER_BACKEND ${ENABLE_ROCSOLVER_BACKEND})
  set(ONEMKL_ENABLE_CUFFT_BACKEND ${ENABLE_CUFFT_BACKEND})
  set(ONEMKL_ENABLE_ROCFFT_BACKEND ${ENABLE_ROCFFT_BACKEND})
  set(ONEMKL_ENABLE_PORTFFT_BACKEND ${ENABLE_PORTFFT_BACKEND})
  set(ONEMKL_ENABLE_CUSPARSE_BACKEND ${ENABLE_CUSPARSE_BACKEND})
  set(ONEMKL_ENABLE_ROCSPARSE_BACKEND ${ENABLE_ROCSPARSE_BACKEND})

  configure_file(config.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/oneapi/mkl/config.hpp.configured")
  file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/oneapi/mkl/detail/config.hpp"
    INPUT "${CMAKE_CURRENT_BINARY_DIR}/oneapi/mkl/config.hpp.configured"
  )
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/oneapi/mkl/detail/config.hpp"
    DESTINATION include/oneapi/mkl/detail
    COMPONENT Devel
  )
endfunction()

generate_header_file()

# Add recipe for onemkl loader library
if(BUILD_SHARED_LIBS)
  add_library(onemkl SHARED)

  # The loader library depends on all the backend libraries as it uses
  # dlopen to load them at runtime.
  # Use add_dependencies to ensure that all the backend libraries are
  # (re-)built when compiling the loader or runtime binaries.
  add_dependencies(onemkl onemkl_backend_libs)

  target_include_directories(onemkl
    PUBLIC ${ONEMKL_INTERFACE_INCLUDE_DIRS}
  )
  set_target_properties(onemkl PROPERTIES
    SOVERSION ${PROJECT_VERSION_MAJOR}
  )
  # w/a for setting oneMKL Interfaces installed headers as -I instead of -isystem for cmake >= 3.25 for workable find_package(MKL) combination
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.25.0")
    set_target_properties(onemkl PROPERTIES EXPORT_NO_SYSTEM true)
  endif()

  # Build dispatcher library
  set (ONEMKL_LIBS ${TARGET_DOMAINS})
  list(TRANSFORM ONEMKL_LIBS PREPEND onemkl_)
  target_link_libraries(onemkl PUBLIC ${ONEMKL_LIBS})
  set_target_properties(onemkl PROPERTIES
      INSTALL_RPATH "\$ORIGIN"
      BUILD_WITH_INSTALL_RPATH TRUE
  )

  # Add the library to install package
  foreach(domain_lib ${ONEMKL_LIBS})
    install(TARGETS ${domain_lib} EXPORT oneMKLTargets)
  endforeach()
  install(TARGETS onemkl EXPORT oneMKLTargets
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
  )
endif()
