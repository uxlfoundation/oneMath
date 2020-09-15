/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/mkl/rng/detail/rng_loader.hpp"

#include "function_table_initializer.hpp"
#include "rng/function_table.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace detail {

static oneapi::mkl::detail::table_initializer<domain::rng, rng_function_table_t> function_tables;

engine_impl* create_philox4x32x10(oneapi::mkl::device libkey, cl::sycl::queue queue,
                                  std::uint64_t seed) {
    return function_tables[libkey].create_philox4x32x10_sycl(queue, seed);
}

engine_impl* create_philox4x32x10(oneapi::mkl::device libkey, cl::sycl::queue queue,
                                  std::initializer_list<std::uint64_t> seed) {
    return function_tables[libkey].create_philox4x32x10_ex_sycl(queue, seed);
}

} // namespace detail
} // namespace rng
} // namespace mkl
} // namespace oneapi
