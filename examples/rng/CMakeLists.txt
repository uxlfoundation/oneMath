#===============================================================================
# Copyright 2022 Intel Corporation
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

# Note: compile-time example uses both MKLCPU and CURAND backends, therefore
# cmake in the sub-directory will only build it if CURAND backend is enabled
add_subdirectory(compile_time_dispatching)
add_subdirectory(device)

# runtime compilation is only possible with dynamic libraries
if (BUILD_SHARED_LIBS)
  add_subdirectory(run_time_dispatching)
endif()
