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

include_guard()

find_library(CBLAS_file NAMES cblas.dll.lib cblas.lib cblas HINTS ${REF_BLAS_ROOT} ${REF_LAPACK_ROOT} PATH_SUFFIXES lib lib64)
find_package_handle_standard_args(CBLAS REQUIRED_VARS CBLAS_file)
find_library(BLAS_file NAMES blas.dll.lib blas.lib blas HINTS ${REF_BLAS_ROOT} ${REF_LAPACK_ROOT} PATH_SUFFIXES lib lib64)
find_package_handle_standard_args(CBLAS REQUIRED_VARS BLAS_file)

get_filename_component(CBLAS_LIB_DIR ${CBLAS_file} DIRECTORY)
find_path(CBLAS_INCLUDE cblas.h HINTS ${REF_BLAS_ROOT} ${REF_LAPACK_ROOT} PATH_SUFFIXES include)

if(UNIX)
  list(APPEND CBLAS_LINK "-Wl,-rpath,${CBLAS_LIB_DIR}")
endif()
list(APPEND CBLAS_LINK ${CBLAS_file})
list(APPEND CBLAS_LINK ${BLAS_file})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CBLAS REQUIRED_VARS CBLAS_INCLUDE CBLAS_LINK)
