/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef ONEMATH_RNG_DEVICE_ENGINE_BASE_HPP_
#define ONEMATH_RNG_DEVICE_ENGINE_BASE_HPP_

#include <cstdint>

#include <sycl/sycl.hpp>

namespace oneapi::math::rng::device::detail {

// internal structure to specify state of engine
template <typename EngineType>
struct engine_state {};

template <typename EngineType>
class engine_base {};

} // namespace oneapi::math::rng::device::detail

#include "oneapi/math/rng/device/detail/philox4x32x10_impl.hpp"
#include "oneapi/math/rng/device/detail/mrg32k3a_impl.hpp"
#include "oneapi/math/rng/device/detail/mcg31m1_impl.hpp"
#include "oneapi/math/rng/device/detail/mcg59_impl.hpp"

#endif // ONEMATH_RNG_DEVICE_ENGINE_BASE_HPP_
