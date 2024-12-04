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

#ifndef ONEMATH_RNG_DEVICE_UNIFORM_BITS_IMPL_HPP_
#define ONEMATH_RNG_DEVICE_UNIFORM_BITS_IMPL_HPP_

#include "engine_base.hpp"

namespace oneapi::math::rng::device::detail {

template <typename UIntType>
class distribution_base<oneapi::math::rng::device::uniform_bits<UIntType>> {
protected:
    template <typename EngineType>
    auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, UIntType,
                                  sycl::vec<UIntType, EngineType::vec_size>>::type {
        static_assert(std::is_same<EngineType, philox4x32x10<EngineType::vec_size>>::value ||
                          std::is_same<EngineType, mcg59<EngineType::vec_size>>::value,
                      "oneMath: uniform_bits works only with philox4x32x10/mcg59 engines");
        return engine.template generate_uniform_bits<UIntType>();
    }

    template <typename EngineType>
    UIntType generate_single(EngineType& engine) {
        static_assert(std::is_same<EngineType, philox4x32x10<EngineType::vec_size>>::value ||
                          std::is_same<EngineType, mcg59<EngineType::vec_size>>::value,
                      "oneMath: uniform_bits works only with philox4x32x10/mcg59 engines");
        return engine.template generate_single_uniform_bits<UIntType>();
    }
};

} // namespace oneapi::math::rng::device::detail

#endif // ONEMATH_RNG_DEVICE_UNIFORM_BITS_IMPL_HPP_
