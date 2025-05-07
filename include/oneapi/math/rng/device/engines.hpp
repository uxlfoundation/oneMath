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
class philox4x32x10 : public detail::engine_base<philox4x32x10<VecSize>> {
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
class mrg32k3a : public detail::engine_base<mrg32k3a<VecSize>> {
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
class mcg31m1 : public detail::engine_base<mcg31m1<VecSize>> {
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
class mcg59 : public detail::engine_base<mcg59<VecSize>> {
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

// Class template oneapi::math::rng::device::pcg64_dxsm
//
// Represents PCG64-DXSM pseudorandom number generator
//
// Supported parallelization methods:
//      skip_ahead
//
template <std::int32_t VecSize>
class pcg64_dxsm : public detail::engine_base<pcg64_dxsm<VecSize>> {
public:
    static constexpr std::uint64_t default_seed = 0;

    static constexpr std::int32_t vec_size = VecSize;

    pcg64_dxsm() : detail::engine_base<pcg64_dxsm<VecSize>>(default_seed) {}

    pcg64_dxsm(std::uint64_t seed, std::uint64_t offset = 0)
            : detail::engine_base<pcg64_dxsm<VecSize>>(seed, offset) {}

    pcg64_dxsm(std::initializer_list<std::uint64_t> seed, std::uint64_t offset = 0)
            : detail::engine_base<pcg64_dxsm<VecSize>>(seed.size(), seed.begin(), offset) {}

    pcg64_dxsm(std::uint64_t seed, std::initializer_list<std::uint64_t> offset)
            : detail::engine_base<pcg64_dxsm<VecSize>>(seed, offset.size(), offset.begin()) {}

    pcg64_dxsm(std::initializer_list<std::uint64_t> seed,
               std::initializer_list<std::uint64_t> offset)
            : detail::engine_base<pcg64_dxsm<VecSize>>(seed.size(), seed.begin(), offset.size(),
                                                       offset.begin()) {}

private:
    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::uint64_t num_to_skip);

    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::initializer_list<std::uint64_t> num_to_skip);

    template <typename DistrType>
    friend class detail::distribution_base;
};

// ENGINE ADAPTORS

// Class oneapi::math::rng::device::count_engine_adaptor
template <typename Engine>
class count_engine_adaptor {
public:
    static constexpr std::int32_t vec_size = Engine::vec_size;

    // ctors
    template <typename... Params>
    count_engine_adaptor(Params... params) : engine_(params...) {}

    count_engine_adaptor(const Engine& engine) : engine_(engine) {}
    count_engine_adaptor(Engine&& engine) : engine_(std::move(engine)) {}

    // methods
    template <typename RealType>
    auto generate(RealType a, RealType b) {
        counted_ += Engine::vec_size;
        return engine_.generate(a, b);
    }

    auto generate() {
        counted_ += Engine::vec_size;
        return engine_.generate();
    }

    template <typename RealType>
    RealType generate_single(RealType a, RealType b) {
        counted_++;
        return engine_.generate_single(a, b);
    }

    template <typename UIntType>
    auto generate_uniform_bits() {
        if constexpr (std::is_same<UIntType, std::uint32_t>::value) {
            counted_ += Engine::vec_size;
        }
        else {
            counted_ += 2 * Engine::vec_size;
        }
        return engine_.template generate_uniform_bits<UIntType>();
    }

    template <typename UIntType>
    auto generate_single_uniform_bits() {
        if constexpr (std::is_same<UIntType, std::uint32_t>::value) {
            counted_ += 1;
        }
        else {
            counted_ += 2;
        }
        return engine_.template generate_single_uniform_bits<UIntType>();
    }

    auto generate_bits() {
        counted_ += Engine::vec_size;
        return engine_.generate_bits();
    }

    // getters
    std::int64_t get_count() const {
        return counted_;
    }

    const Engine& base() const {
        return engine_;
    }

private:
    Engine engine_;
    std::int64_t counted_ = 0;

    template <typename DistrType>
    friend class detail::distribution_base;
};

} // namespace oneapi::math::rng::device

#endif // ONEMATH_RNG_DEVICE_ENGINES_HPP_
