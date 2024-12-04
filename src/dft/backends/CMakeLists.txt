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

add_custom_target(onemath_backend_libs_dft)
add_dependencies(onemath_backend_libs onemath_backend_libs_dft)

if(ENABLE_MKLGPU_BACKEND)
  add_subdirectory(mklgpu)
endif()

if(ENABLE_MKLCPU_BACKEND)
  add_subdirectory(mklcpu)
endif()

if(ENABLE_CUFFT_BACKEND)
  add_subdirectory(cufft)
endif()

if(ENABLE_ROCFFT_BACKEND)
  add_subdirectory(rocfft)
endif()

if(ENABLE_PORTFFT_BACKEND)
  add_subdirectory(portfft)
endif()
