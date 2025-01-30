#===============================================================================
# Copyright 2024 Intel Corporation
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

set(LIB_NAME onemath_sparse_blas_rocsparse)
set(LIB_OBJ ${LIB_NAME}_obj)

include(WarningsUtils)

add_library(${LIB_NAME})
add_library(${LIB_OBJ} OBJECT
  rocsparse_handles.cpp
  rocsparse_scope_handle.cpp
  operations/rocsparse_spmm.cpp
  operations/rocsparse_spmv.cpp
  operations/rocsparse_spsv.cpp
  $<$<BOOL:${BUILD_SHARED_LIBS}>: rocsparse_wrappers.cpp>
)
add_dependencies(onemath_backend_libs_sparse_blas ${LIB_NAME})

target_include_directories(${LIB_OBJ}
  PUBLIC  ${ONEMATH_INCLUDE_DIRS}
  PRIVATE ${PROJECT_SOURCE_DIR}/src/include
          ${PROJECT_SOURCE_DIR}/src
          ${CMAKE_BINARY_DIR}/bin
          ${ONEMATH_GENERATED_INCLUDE_PATH}
)

target_compile_options(${LIB_OBJ} PRIVATE ${ONEMATH_BUILD_COPT})

find_package(HIP REQUIRED)
find_package(rocsparse 3.1.2 REQUIRED)  # ROCm 6.1.0 or above

target_link_libraries(${LIB_OBJ} PRIVATE hip::host roc::rocsparse)

target_link_libraries(${LIB_OBJ}
  PUBLIC ONEMATH::SYCL::SYCL
  PRIVATE onemath_warnings
)

set_target_properties(${LIB_OBJ} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(${LIB_NAME} PRIVATE ${LIB_OBJ})
target_include_directories(${LIB_NAME} PUBLIC ${ONEMATH_INCLUDE_DIRS})

#Set oneMATH libraries as not transitive for dynamic
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
