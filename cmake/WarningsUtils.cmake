#===============================================================================
# Copyright Codeplay Software Ltd.
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

include_guard(GLOBAL)

add_library(onemath_warnings INTERFACE)

set(ONEMATH_WARNINGS "")

include(CheckCXXCompilerFlag)
macro(add_warning flag)
  check_cxx_compiler_flag(${flag} IS_SUPPORTED)
  if(${IS_SUPPORTED})
    list(APPEND ONEMATH_WARNINGS ${flag})
  else()
    message(WARNING "Compiler does not support ${flag}")
  endif()
endmacro()

add_warning("-Wall")
add_warning("-Wextra")
add_warning("-Wshadow")
add_warning("-Wconversion")
add_warning("-Wpedantic")

message(VERBOSE "Domains with warnings enabled use: ${ONEMATH_WARNINGS}")

# The onemath_warnings target can be linked to any other target to enable warnings.
target_compile_options(onemath_warnings INTERFACE ${ONEMATH_WARNINGS})

# Add the library to install package
install(TARGETS onemath_warnings EXPORT oneMathTargets)
