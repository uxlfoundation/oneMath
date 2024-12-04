/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "lapack/function_table.hpp"
#include "oneapi/math/lapack/detail/mklcpu/onemath_lapack_mklcpu.hpp"

#define WRAPPER_VERSION 1

extern "C" ONEMATH_EXPORT lapack_function_table_t onemath_lapack_table = {
    WRAPPER_VERSION,
#define LAPACK_BACKEND mklcpu
#include "../mkl_common/lapack_wrappers.cxx"
#undef LAPACK_BACKEND
};
