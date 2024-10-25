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

# Add compile definitions for the reference libraries
add_compile_definitions(
  ONEMKL_REF_BLAS_LIBNAME="${ONEMKL_REF_BLAS_LIBNAME}"
  ONEMKL_REF_CBLAS_LIBNAME="${ONEMKL_REF_CBLAS_LIBNAME}")

add_subdirectory(level1)
add_subdirectory(level2)
add_subdirectory(level3)
add_subdirectory(batch)
add_subdirectory(extensions)
