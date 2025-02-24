#===============================================================================
# Copyright 2025 SiPearl
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

set(LIB_NAME onemath_lapack_armpl)
set(LIB_OBJ ${LIB_NAME}_obj)

# Add third-party library
find_package(ARMPL REQUIRED)

add_library(${LIB_NAME})
add_library(${LIB_OBJ} OBJECT
  armpl_wrappers.cpp
  armpl_batch.cpp
  $<$<BOOL:${BUILD_SHARED_LIBS}>: armpl_wrappers_table_dyn.cpp>

)

target_include_directories(${LIB_OBJ}
  PUBLIC ${ONEMATH_INCLUDE_DIRS}
  PRIVATE ${PROJECT_SOURCE_DIR}/src/include
          ${PROJECT_SOURCE_DIR}/src
          ${ARMPL_INCLUDE}
)

target_link_libraries(${LIB_OBJ}
	PUBLIC ONEMATH::SYCL::SYCL ${ARMPL_LINK}
)

target_compile_features(${LIB_OBJ} PUBLIC cxx_std_14)
set_target_properties(${LIB_OBJ} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(${LIB_NAME} PRIVATE ${LIB_OBJ})
target_include_directories(${LIB_NAME} PUBLIC ${ONEMATH_INCLUDE_DIRS})

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
