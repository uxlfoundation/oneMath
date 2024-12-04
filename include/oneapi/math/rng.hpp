/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef _ONEMATH_RNG_HPP_
#define _ONEMATH_RNG_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <complex>
#include <cstdint>

#include "oneapi/math/detail/config.hpp"
#include "oneapi/math/detail/get_device_id.hpp"

#include "oneapi/math/rng/predicates.hpp"
#include "oneapi/math/rng/detail/rng_loader.hpp"

#include "oneapi/math/rng/functions.hpp"
#include "oneapi/math/rng/distributions.hpp"
#include "oneapi/math/rng/engines.hpp"

#endif // _ONEMATH_RNG_HPP_
