/*******************************************************************************
* Copyright Codeplay Software Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#include "oneapi/math/dft/detail/rocfft/onemath_dft_rocfft.hpp"
#include "dft/function_table.hpp"

#define WRAPPER_VERSION 1
#define BACKEND         rocfft

extern "C" dft_function_table_t onemath_dft_table = {
    WRAPPER_VERSION,
#include "dft/backends/backend_wrappers.cxx"
};

#undef WRAPPER_VERSION
#undef BACKEND
