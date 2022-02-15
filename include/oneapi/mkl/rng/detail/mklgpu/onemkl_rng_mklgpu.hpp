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

#ifndef _ONEMKL_RNG_MKLGPU_HPP_
#define _ONEMKL_RNG_MKLGPU_HPP_

#include <cstdint>
#include <CL/sycl.hpp>

#include "oneapi/mkl/detail/export.hpp"
#include "oneapi/mkl/rng/detail/engine_impl.hpp"

namespace oneapi::mkl::rng::mklgpu {

ONEMKL_EXPORT oneapi::mkl::rng::detail::engine_impl* create_philox4x32x10(cl::sycl::queue queue,
                                                                          std::uint64_t seed);

ONEMKL_EXPORT oneapi::mkl::rng::detail::engine_impl* create_philox4x32x10(
    cl::sycl::queue queue, std::initializer_list<std::uint64_t> seed);

ONEMKL_EXPORT oneapi::mkl::rng::detail::engine_impl* create_mrg32k3a(cl::sycl::queue queue,
                                                                     std::uint32_t seed);

ONEMKL_EXPORT oneapi::mkl::rng::detail::engine_impl* create_mrg32k3a(
    cl::sycl::queue queue, std::initializer_list<std::uint32_t> seed);

ONEMKL_EXPORT oneapi::mkl::rng::detail::engine_impl* create_mcg59(cl::sycl::queue queue,
                                                                  std::uint64_t seed);

ONEMKL_EXPORT oneapi::mkl::rng::detail::engine_impl* create_mcg59(
    cl::sycl::queue queue, std::initializer_list<std::uint64_t> seed);

} // namespace oneapi::mkl::rng::mklgpu

#endif //_ONEMKL_RNG_MKLGPU_HPP_
