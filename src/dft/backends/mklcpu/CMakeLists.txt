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

set(LIB_NAME onemath_dft_mklcpu)
set(LIB_OBJ ${LIB_NAME}_obj)

include(WarningsUtils)

add_library(${LIB_NAME})
add_deprecated_library(${LIB_NAME})
add_library(${LIB_OBJ} OBJECT
  commit.cpp
  descriptor.cpp
  forward.cpp
  backward.cpp
  $<$<BOOL:${BUILD_SHARED_LIBS}>: mkl_dft_cpu_wrappers.cpp>
)
add_dependencies(onemath_backend_libs_dft ${LIB_NAME})

target_include_directories(${LIB_OBJ}
  PUBLIC ${ONEMATH_INCLUDE_DIRS}
)
target_include_directories(${LIB_NAME}
  PUBLIC ${ONEMATH_INCLUDE_DIRS}
)

target_include_directories(${LIB_OBJ}
  PRIVATE ${PROJECT_SOURCE_DIR}/src
          ${PROJECT_SOURCE_DIR}/src/include
          ${CMAKE_BINARY_DIR}/bin
          ${ONEMATH_GENERATED_INCLUDE_PATH}
)

target_compile_options(${LIB_OBJ} PRIVATE ${ONEMATH_BUILD_COPT})
if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET ${LIB_OBJ} SOURCES ${SOURCES})
endif()

if(TARGET MKL::MKL_SYCL::DFT)
  target_link_libraries(${LIB_OBJ}
    PUBLIC ONEMATH::SYCL::SYCL
    PUBLIC MKL::MKL_SYCL::DFT
    PRIVATE onemath_warnings
  )
else()
  target_link_libraries(${LIB_OBJ}
    PUBLIC ONEMATH::SYCL::SYCL
    PUBLIC MKL::MKL_DPCPP
    PRIVATE onemath_warnings
  )
endif()

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
