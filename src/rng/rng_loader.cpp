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

#include "oneapi/math/rng/detail/rng_loader.hpp"

#include "function_table_initializer.hpp"
#include "rng/function_table.hpp"

namespace oneapi {
namespace math {
namespace rng {
namespace detail {

static oneapi::math::detail::table_initializer<domain::rng, rng_function_table_t> function_tables;

engine_impl* create_philox4x32x10(oneapi::math::device libkey, sycl::queue queue,
                                  std::uint64_t seed) {
    return function_tables[{ libkey, queue }].create_philox4x32x10_sycl(queue, seed);
}

engine_impl* create_philox4x32x10(oneapi::math::device libkey, sycl::queue queue,
                                  std::initializer_list<std::uint64_t> seed) {
    return function_tables[{ libkey, queue }].create_philox4x32x10_ex_sycl(queue, seed);
}

engine_impl* create_mrg32k3a(oneapi::math::device libkey, sycl::queue queue, std::uint32_t seed) {
    return function_tables[{ libkey, queue }].create_mrg32k3a_sycl(queue, seed);
}

engine_impl* create_mrg32k3a(oneapi::math::device libkey, sycl::queue queue,
                             std::initializer_list<std::uint32_t> seed) {
    return function_tables[{ libkey, queue }].create_mrg32k3a_ex_sycl(queue, seed);
}

} // namespace detail
} // namespace rng
} // namespace math
} // namespace oneapi
