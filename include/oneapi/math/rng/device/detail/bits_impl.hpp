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

#ifndef ONEMATH_RNG_DEVICE_BITS_IMPL_HPP_
#define ONEMATH_RNG_DEVICE_BITS_IMPL_HPP_

#include "engine_base.hpp"

namespace oneapi::math::rng::device::detail {

template <typename UIntType>
class distribution_base<oneapi::math::rng::device::bits<UIntType>> {
protected:
    template <typename EngineType>
    auto generate(EngineType& engine) -> typename std::enable_if<
        !std::is_same<EngineType, mcg59<EngineType::vec_size>>::value,
        typename std::conditional<EngineType::vec_size == 1, UIntType,
                                  sycl::vec<UIntType, EngineType::vec_size>>::type>::type {
        static_assert(std::is_same<UIntType, uint32_t>::value,
                      "oneMath: bits works only with std::uint32_t");
        return engine.generate();
    }

    template <typename EngineType>
    auto generate(EngineType& engine) -> typename std::enable_if<
        std::is_same<EngineType, mcg59<EngineType::vec_size>>::value,
        typename std::conditional<EngineType::vec_size == 1, UIntType,
                                  sycl::vec<UIntType, EngineType::vec_size>>::type>::type {
        static_assert(std::is_same<UIntType, uint64_t>::value,
                      "oneMath: bits for mcg59 works only with std::uint64_t");
        return engine.generate_bits();
    }

    template <typename EngineType>
    typename std::enable_if<!std::is_same<EngineType, mcg59<EngineType::vec_size>>::value,
                            UIntType>::type
    generate_single(EngineType& engine) {
        static_assert(std::is_same<UIntType, uint32_t>::value,
                      "oneMath: bits works only with std::uint32_t");
        return engine.generate_single();
    }

    template <typename EngineType>
    typename std::enable_if<std::is_same<EngineType, mcg59<EngineType::vec_size>>::value,
                            UIntType>::type
    generate_single(EngineType& engine) {
        static_assert(std::is_same<UIntType, uint64_t>::value,
                      "oneMath: bits for mcg59 works only with std::uint64_t");
        return engine.generate_single();
    }
};

} // namespace oneapi::math::rng::device::detail

#endif // ONEMATH_RNG_DEVICE_BITS_IMPL_HPP_
