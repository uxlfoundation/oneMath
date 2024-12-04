/*******************************************************************************
* Copyright 2023 Codeplay Software Ltd.
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

/*
This file lists functions matching those required by dft_function_table_t in 
src/dft/function_table.hpp.

To use this:

#define WRAPPER_VERSION <Wrapper version number>
#define BACKEND         <Backend name eg. mklgpu>

extern "C" dft_function_table_t onemath_dft_table = {
    WRAPPER_VERSION,
#include "dft/backends/backend_wrappers.cxx"
};

Changes to this file should be matched to changes in function_table.hpp. The required 
function template instantiations must be added to backend_backward_instantiations.cxx 
and backend_forward_instantiations.cxx.
*/

// clang-format off
oneapi::math::dft::BACKEND::create_commit,
oneapi::math::dft::BACKEND::create_commit,
oneapi::math::dft::BACKEND::create_commit,
oneapi::math::dft::BACKEND::create_commit,
// clang-format on

#undef ONEAPI_MKL_DFT_BACKEND_SIGNATURES
