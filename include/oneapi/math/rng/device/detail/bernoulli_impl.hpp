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

#ifndef ONEMATH_RNG_DEVICE_BERNOULLI_IMPL_HPP_
#define ONEMATH_RNG_DEVICE_BERNOULLI_IMPL_HPP_

namespace oneapi::math::rng::device::detail {

template <typename IntType, typename Method>
class distribution_base<oneapi::math::rng::device::bernoulli<IntType, Method>> {
public:
    struct param_type {
        param_type(float p) : p_(p) {}
        float p_;
    };

    distribution_base(float p) : p_(p) {
#ifndef __SYCL_DEVICE_ONLY__
        if ((p > 1.0f) || (p < 0.0f)) {
            throw oneapi::math::invalid_argument("rng", "bernoulli", "p < 0 || p > 1");
        }
#endif
    }

    float p() const {
        return p_;
    }

    param_type param() const {
        return param_type(p_);
    }

    void param(const param_type& pt) {
#ifndef __SYCL_DEVICE_ONLY__
        if ((pt.p_ > 1.0f) || (pt.p_ < 0.0f)) {
            throw oneapi::math::invalid_argument("rng", "bernoulli", "p < 0 || p > 1");
        }
#endif
        p_ = pt.p_;
    }

protected:
    template <typename EngineType>
    auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, IntType,
                                  sycl::vec<IntType, EngineType::vec_size>>::type {
        auto uni_res = engine.generate(0.0f, 1.0f);
        if constexpr (EngineType::vec_size == 1) {
            return IntType{ uni_res < p_ };
        }
        else {
            sycl::vec<IntType, EngineType::vec_size> vec_out(IntType{ 0 });
            for (int i = 0; i < EngineType::vec_size; ++i) {
                if (uni_res[i] < p_) {
                    vec_out[i] = IntType{ 1 };
                }
            }
            return vec_out;
        }
    }

    template <typename EngineType>
    IntType generate_single(EngineType& engine) {
        auto uni_res = engine.generate_single(0.0f, 1.0f);
        return IntType{ uni_res < p_ };
    }

    float p_;
};

} // namespace oneapi::math::rng::device::detail

#endif // ONEMATH_RNG_DEVICE_BERNOULLI_IMPL_HPP_
