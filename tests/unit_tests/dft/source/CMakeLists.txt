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

set(DFT_SOURCES "compute_tests.cpp" "descriptor_tests.cpp" "workspace_external_tests.cpp")

include(WarningsUtils)

if (BUILD_SHARED_LIBS)
    add_library(dft_source_rt OBJECT ${DFT_SOURCES})
    target_compile_options(dft_source_rt PRIVATE -DCALL_RT_API -DNOMINMAX)
    target_include_directories(dft_source_rt
            PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
            PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
            PUBLIC ${PROJECT_SOURCE_DIR}/include
            PUBLIC ${PROJECT_SOURCE_DIR}/deps/googletest/include
            PUBLIC ${CMAKE_BINARY_DIR}/bin
            )
    if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
        add_sycl_to_target(TARGET dft_source_rt SOURCES ${DFT_SOURCES})
    else ()
        target_link_libraries(dft_source_rt PUBLIC ONEMATH::SYCL::SYCL)
    endif ()
    target_link_libraries(dft_source_rt PRIVATE onemath_warnings)
endif ()

add_library(dft_source_ct OBJECT ${DFT_SOURCES})
target_compile_options(dft_source_ct PRIVATE -DNOMINMAX)
target_include_directories(dft_source_ct
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        PUBLIC ${PROJECT_SOURCE_DIR}/include
        PUBLIC ${PROJECT_SOURCE_DIR}/deps/googletest/include
        PUBLIC ${CMAKE_BINARY_DIR}/bin
        )
if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
    add_sycl_to_target(TARGET dft_source_ct SOURCES ${DFT_SOURCES})
else ()
    target_link_libraries(dft_source_ct PUBLIC ONEMATH::SYCL::SYCL)
endif ()
target_link_libraries(dft_source_ct PRIVATE onemath_warnings)

