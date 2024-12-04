#==========================================================================
#  Copyright (C) Codeplay Software Limited
#  Copyright 2022 Intel Corporation
#
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

set(LIB_NAME onemath_lapack_rocsolver)
set(LIB_OBJ ${LIB_NAME}_obj)
find_package(hip REQUIRED)
find_package(rocsolver REQUIRED)
find_package(Threads REQUIRED)

set(SOURCES rocsolver_lapack.cpp
		rocsolver_batch.cpp
	        $<$<STREQUAL:${ONEMATH_SYCL_IMPLEMENTATION},dpc++>:rocsolver_scope_handle.cpp>
          $<$<BOOL:${BUILD_SHARED_LIBS}>: rocsolver_wrappers.cpp>)
add_library(${LIB_NAME})
add_deprecated_library(${LIB_NAME})
add_library(${LIB_OBJ} OBJECT ${SOURCES})
add_dependencies(onemath_backend_libs_lapack ${LIB_NAME})

target_include_directories(${LIB_OBJ}
  PRIVATE ${PROJECT_SOURCE_DIR}/include
          ${PROJECT_SOURCE_DIR}/src/include
          ${PROJECT_SOURCE_DIR}/src
          ${CMAKE_BINARY_DIR}/bin
          ${ONEMATH_GENERATED_INCLUDE_PATH}
)
target_compile_options(${LIB_OBJ} PRIVATE ${ONEMATH_BUILD_COPT})
target_link_libraries(${LIB_OBJ} PRIVATE roc::rocsolver hip::host Threads::Threads)
target_link_libraries(${LIB_OBJ} PUBLIC ONEMATH::SYCL::SYCL)
target_compile_features(${LIB_OBJ} PUBLIC cxx_std_17)
set_target_properties(${LIB_OBJ} PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${LIB_NAME} PUBLIC ${LIB_OBJ})
if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET ${LIB_OBJ} SOURCES ${SOURCES})
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

