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

#ifndef _ONEMATH_RNG_LOADER_HPP_
#define _ONEMATH_RNG_LOADER_HPP_

#include <cstdint>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/detail/export.hpp"
#include "oneapi/math/detail/get_device_id.hpp"

#include "oneapi/math/rng/detail/engine_impl.hpp"

namespace oneapi {
namespace math {
namespace rng {
namespace detail {

ONEMATH_EXPORT engine_impl* create_philox4x32x10(oneapi::math::device libkey, sycl::queue queue,
                                                 std::uint64_t seed);

ONEMATH_EXPORT engine_impl* create_philox4x32x10(oneapi::math::device libkey, sycl::queue queue,
                                                 std::initializer_list<std::uint64_t> seed);

ONEMATH_EXPORT engine_impl* create_mrg32k3a(oneapi::math::device libkey, sycl::queue queue,
                                            std::uint32_t seed);

ONEMATH_EXPORT engine_impl* create_mrg32k3a(oneapi::math::device libkey, sycl::queue queue,
                                            std::initializer_list<std::uint32_t> seed);

} // namespace detail
} // namespace rng
} // namespace math
} // namespace oneapi

#endif //_ONEMATH_RNG_LOADER_HPP_
