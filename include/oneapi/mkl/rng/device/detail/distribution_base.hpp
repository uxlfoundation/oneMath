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

#ifndef _MKL_RNG_DISTRIBUTION_BASE_HPP_
#define _MKL_RNG_DISTRIBUTION_BASE_HPP_

#include <sycl/sycl.hpp>

#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/rng/device/types.hpp"

namespace oneapi::mkl::rng::device {

namespace detail {

template <typename DistrType>
class distribution_base {};

} // namespace detail

// declarations of distribution classes
template <typename Type = float, typename Method = uniform_method::by_default>
class uniform;

template <typename RealType = float, typename Method = gaussian_method::by_default>
class gaussian;

template <typename RealType = float, typename Method = lognormal_method::by_default>
class lognormal;

template <typename UIntType = std::uint32_t>
class uniform_bits;

template <typename UIntType = std::uint32_t>
class bits;

template <typename RealType = float, typename Method = exponential_method::by_default>
class exponential;

template <typename RealType = float, typename Method = beta_method::by_default>
class beta;

template <typename RealType = float, typename Method = gamma_method::by_default>
class gamma;

template <typename IntType = std::int32_t, typename Method = poisson_method::by_default>
class poisson;

template <typename IntType = std::uint32_t, typename Method = bernoulli_method::by_default>
class bernoulli;

} // namespace oneapi::mkl::rng::device

#include "oneapi/mkl/rng/device/detail/uniform_impl.hpp"
#include "oneapi/mkl/rng/device/detail/gaussian_impl.hpp"
#include "oneapi/mkl/rng/device/detail/lognormal_impl.hpp"
#include "oneapi/mkl/rng/device/detail/bits_impl.hpp"
#include "oneapi/mkl/rng/device/detail/uniform_bits_impl.hpp"
#include "oneapi/mkl/rng/device/detail/exponential_impl.hpp"
#include "oneapi/mkl/rng/device/detail/poisson_impl.hpp"
#include "oneapi/mkl/rng/device/detail/bernoulli_impl.hpp"
#include "oneapi/mkl/rng/device/detail/beta_impl.hpp"
#include "oneapi/mkl/rng/device/detail/gamma_impl.hpp"

#endif // _MKL_RNG_DISTRIBUTION_BASE_HPP_
