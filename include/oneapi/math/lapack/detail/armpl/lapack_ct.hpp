/*******************************************************************************
* Copyright 2025 SiPearl
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

#pragma once

#include <complex>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "oneapi/math/types.hpp"
#include "oneapi/math/lapack/types.hpp"
#include "oneapi/math/detail/backend_selector.hpp"
#include "oneapi/math/lapack/detail/armpl/onemath_lapack_armpl.hpp"

namespace oneapi {
namespace math {
namespace lapack {

#define LAPACK_BACKEND armpl
#include "oneapi/math/lapack/detail/armpl/lapack_ct.hxx"
#undef LAPACK_BACKEND

} //namespace lapack
} //namespace math
} //namespace oneapi
