/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#pragma once

#include "oneapi/math/detail/config.hpp"

#ifdef ONEMATH_ENABLE_MKLCPU_BACKEND
#include "oneapi/math/lapack/detail/mklcpu/lapack_ct.hpp"
#endif
#ifdef ONEMATH_ENABLE_MKLGPU_BACKEND
#include "oneapi/math/lapack/detail/mklgpu/lapack_ct.hpp"
#endif
#ifdef ONEMATH_ENABLE_CUSOLVER_BACKEND
#include "oneapi/math/lapack/detail/cusolver/lapack_ct.hpp"
#endif
#ifdef ONEMATH_ENABLE_ROCSOLVER_BACKEND
#include "oneapi/math/lapack/detail/rocsolver/lapack_ct.hpp"
#endif

#include "oneapi/math/lapack/detail/lapack_rt.hpp"
