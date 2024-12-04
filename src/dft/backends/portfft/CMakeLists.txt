#===============================================================================
# Copyright Codeplay Software Ltd
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


check_cxx_compiler_flag("-fsycl" IS_DPCPP)

set(FOUND_TARGETS 0)

if (NOT (CMAKE_CXX_FLAGS STREQUAL ""))
  string(FIND ${CMAKE_CXX_FLAGS} "fsycl-targets" TARGETS_IDX)
  if (TARGETS_IDX GREATER_EQUAL 0)
    set(FOUND_TARGETS 1)
    message(STATUS "fsycl-targets flag found, not setting targets")
  endif()
endif()

if (IS_DPCPP AND UNIX AND NOT FOUND_TARGETS)
  message(WARNING "fsycl-targets flag not found, enabling all backends")
  set(TARGETS_COMPILE_OPTIONS -fsycl-unnamed-lambda)
  set(TARGETS_LINK_OPTIONS -fsycl-unnamed-lambda)

  # spir64 must be last in the list due to a bug in dpcpp 2024.0.0
  set(TARGETS_TRIPLES "spir64")
  if(dpcpp_supports_nvptx64)
    set(TARGETS_TRIPLES nvptx64-nvidia-cuda,${TARGETS_TRIPLES})
  endif()

  if (NOT (HIP_TARGETS STREQUAL ""))
    set(TARGETS_TRIPLES amdgcn-amd-amdhsa,${TARGETS_TRIPLES})
    list(APPEND TARGETS_COMPILE_OPTIONS -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${HIP_TARGETS})
    list(APPEND TARGETS_LINK_OPTIONS -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${HIP_TARGETS})
  else()
    message(WARNING "Can't enable hip backend, HIP_TARGETS has not been set.")
  endif()
  
  message(STATUS "portFFT target triple set to ${TARGETS_TRIPLES}")

  list(APPEND TARGETS_COMPILE_OPTIONS -fsycl-targets=${TARGETS_TRIPLES})
  list(APPEND TARGETS_LINK_OPTIONS -fsycl-targets=${TARGETS_TRIPLES})

  target_compile_options(ONEMATH::SYCL::SYCL INTERFACE ${TARGETS_COMPILE_OPTIONS})
  target_link_options(ONEMATH::SYCL::SYCL INTERFACE ${TARGETS_LINK_OPTIONS})
endif()

set(LIB_NAME onemath_dft_portfft)
set(LIB_OBJ ${LIB_NAME}_obj)

add_library(${LIB_NAME})
add_deprecated_library(${LIB_NAME})
add_library(${LIB_OBJ} OBJECT
  descriptor.cpp
  commit.cpp
  $<$<BOOL:${BUILD_SHARED_LIBS}>: portfft_wrappers.cpp>
)
add_dependencies(onemath_backend_libs_dft ${LIB_NAME})

find_package(portfft QUIET)
if (NOT portfft_FOUND)
	message(STATUS "portFFT - not found locally, downloading")

	include(FetchContent)
	set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/deps")
	FetchContent_Declare(
		portfft
		GIT_REPOSITORY https://github.com/codeplaysoftware/portFFT.git
		GIT_TAG        f29d8e7c103f8b981f719e1194199ca4d5216895
	)
	FetchContent_MakeAvailable(portfft)
	message(STATUS "portFFT - downloaded")
	target_link_libraries(${LIB_OBJ} PRIVATE portfft)
else()
	message(STATUS "portFFT - found")
	target_link_libraries(${LIB_OBJ} PRIVATE portfft::portfft)
endif()

target_link_libraries(${LIB_OBJ} PRIVATE onemath_warnings)

target_include_directories(${LIB_OBJ}
  PUBLIC ${ONEMATH_INCLUDE_DIRS}
)
target_include_directories(${LIB_NAME}
  PUBLIC ${ONEMATH_INCLUDE_DIRS}
)
target_include_directories(${LIB_OBJ}
  PRIVATE ${PROJECT_SOURCE_DIR}/src
          ${CMAKE_BINARY_DIR}/bin
)

target_compile_options(${LIB_OBJ} PRIVATE ${ONEMATH_BUILD_COPT})

target_link_libraries(${LIB_OBJ} PUBLIC ONEMATH::SYCL::SYCL)

set_target_properties(${LIB_OBJ} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(${LIB_NAME} PUBLIC ${LIB_OBJ})

#Set oneMath libraries as not transitive for dynamic
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
