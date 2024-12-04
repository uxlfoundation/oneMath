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

#ifndef ONEMATH_RNG_DEVICE_ENGINES_HPP_
#define ONEMATH_RNG_DEVICE_ENGINES_HPP_

#include <limits>

#include "oneapi/math/rng/device/types.hpp"
#include "oneapi/math/rng/device/functions.hpp"
#include "oneapi/math/rng/device/detail/engine_base.hpp"

namespace oneapi::math::rng::device {

// PSEUDO-RANDOM NUMBER DEVICE-SIDE ENGINES

// Class template oneapi::math::rng::device::philox4x32x10
//
// Represents Philox4x32-10 counter-based pseudorandom number generator
//
// Supported parallelization methods:
//      skip_ahead
//
template <std::int32_t VecSize>
class philox4x32x10 : detail::engine_base<philox4x32x10<VecSize>> {
public:
    static constexpr std::uint64_t default_seed = 0;

    static constexpr std::int32_t vec_size = VecSize;

    philox4x32x10() : detail::engine_base<philox4x32x10<VecSize>>(default_seed) {}

    philox4x32x10(std::uint64_t seed, std::uint64_t offset = 0)
            : detail::engine_base<philox4x32x10<VecSize>>(seed, offset) {}

    philox4x32x10(std::initializer_list<std::uint64_t> seed, std::uint64_t offset = 0)
            : detail::engine_base<philox4x32x10<VecSize>>(seed.size(), seed.begin(), offset) {}

    philox4x32x10(std::uint64_t seed, std::initializer_list<std::uint64_t> offset)
            : detail::engine_base<philox4x32x10<VecSize>>(seed, offset.size(), offset.begin()) {}

    philox4x32x10(std::initializer_list<std::uint64_t> seed,
                  std::initializer_list<std::uint64_t> offset)
            : detail::engine_base<philox4x32x10<VecSize>>(seed.size(), seed.begin(), offset.size(),
                                                          offset.begin()) {}

private:
    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::uint64_t num_to_skip);

    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::initializer_list<std::uint64_t> num_to_skip);

    template <typename DistrType>
    friend class detail::distribution_base;
};

// Class oneapi::math::rng::device::mrg32k3a
//
// Represents the combined recurcive pseudorandom number generator
//
// Supported parallelization methods:
//      skip_ahead
//
template <std::int32_t VecSize>
class mrg32k3a : detail::engine_base<mrg32k3a<VecSize>> {
public:
    static constexpr std::uint32_t default_seed = 1;

    static constexpr std::int32_t vec_size = VecSize;

    mrg32k3a() : detail::engine_base<mrg32k3a<VecSize>>(default_seed) {}

    mrg32k3a(std::uint32_t seed, std::uint64_t offset = 0)
            : detail::engine_base<mrg32k3a<VecSize>>(seed, offset) {}

    mrg32k3a(std::initializer_list<std::uint32_t> seed, std::uint64_t offset = 0)
            : detail::engine_base<mrg32k3a<VecSize>>(seed.size(), seed.begin(), offset) {}

    mrg32k3a(std::uint32_t seed, std::initializer_list<std::uint64_t> offset)
            : detail::engine_base<mrg32k3a<VecSize>>(seed, offset.size(), offset.begin()) {}

    mrg32k3a(std::initializer_list<std::uint32_t> seed, std::initializer_list<std::uint64_t> offset)
            : detail::engine_base<mrg32k3a<VecSize>>(seed.size(), seed.begin(), offset.size(),
                                                     offset.begin()) {}

private:
    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::uint64_t num_to_skip);

    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::initializer_list<std::uint64_t> num_to_skip);

    template <typename DistrType>
    friend class detail::distribution_base;
};

// Class oneapi::math::rng::device::mcg31m1
//
//
//
// Supported parallelization methods:
//      skip_ahead
//
template <std::int32_t VecSize>
class mcg31m1 : detail::engine_base<mcg31m1<VecSize>> {
public:
    static constexpr std::uint32_t default_seed = 1;

    static constexpr std::int32_t vec_size = VecSize;

    mcg31m1() : detail::engine_base<mcg31m1<VecSize>>(default_seed) {}

    mcg31m1(std::uint32_t seed, std::uint64_t offset = 0)
            : detail::engine_base<mcg31m1<VecSize>>(seed, offset) {}

private:
    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::uint64_t num_to_skip);

    template <typename DistrType>
    friend class detail::distribution_base;
};

// Class oneapi::math::rng::device::mcg59
//
//
//
// Supported parallelization methods:
//      skip_ahead
//
template <std::int32_t VecSize>
class mcg59 : detail::engine_base<mcg59<VecSize>> {
public:
    static constexpr std::uint32_t default_seed = 1;

    static constexpr std::int32_t vec_size = VecSize;

    mcg59() : detail::engine_base<mcg59<VecSize>>(default_seed) {}

    mcg59(std::uint64_t seed, std::uint64_t offset = 0)
            : detail::engine_base<mcg59<VecSize>>(seed, offset) {}

private:
    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::uint64_t num_to_skip);

    template <typename DistrType>
    friend class detail::distribution_base;
};

} // namespace oneapi::math::rng::device

#endif // ONEMATH_RNG_DEVICE_ENGINES_HPP_
