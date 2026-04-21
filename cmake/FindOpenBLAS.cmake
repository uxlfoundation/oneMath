#===============================================================================
# Copyright 2020-2023 Intel Corporation
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

include_guard()
include(FindPackageHandleStandardArgs)

if(DEFINED OPENBLAS_DIR)
    set(_OPENBLAS_HINTS ${OPENBLAS_DIR})
elseif(DEFINED ENV{OPENBLAS_DIR})
    set(_OPENBLAS_HINTS $ENV{OPENBLAS_DIR})
elseif(CMAKE_PREFIX_PATH)
    set(_OPENBLAS_HINTS ${CMAKE_PREFIX_PATH})
endif()

find_library(OPENBLAS_LIBRARY
    NAMES openblas libopenblas
    HINTS ${_OPENBLAS_HINTS}
    PATH_SUFFIXES lib lib64
)

find_path(OPENBLAS_INCLUDE
    NAMES cblas.h
    HINTS ${_OPENBLAS_HINTS}
    PATH_SUFFIXES include include/openblas
)

find_package_handle_standard_args(OpenBLAS
    REQUIRED_VARS OPENBLAS_LIBRARY OPENBLAS_INCLUDE
)

if(OpenBLAS_FOUND)

    get_filename_component(OPENBLAS_LIB_DIR
        ${OPENBLAS_LIBRARY}
        DIRECTORY
    )

    add_library(ONEMATH::OPENBLAS::OPENBLAS UNKNOWN IMPORTED)

    set_target_properties(ONEMATH::OPENBLAS::OPENBLAS PROPERTIES
        IMPORTED_LOCATION ${OPENBLAS_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${OPENBLAS_INCLUDE}
    )

    if(UNIX AND OPENBLAS_LIB_DIR)
        set_target_properties(ONEMATH::OPENBLAS::OPENBLAS PROPERTIES
            INTERFACE_LINK_OPTIONS "-Wl,-rpath,${OPENBLAS_LIB_DIR}"
        )
    endif()

endif()

mark_as_advanced(
    OPENBLAS_LIBRARY
    OPENBLAS_INCLUDE
)
