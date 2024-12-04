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

#ifndef ONEMATH_RNG_DEVICE_LOGNORMAL_IMPL_HPP_
#define ONEMATH_RNG_DEVICE_LOGNORMAL_IMPL_HPP_

namespace oneapi::math::rng::device::detail {

template <typename RealType, typename Method>
class distribution_base<oneapi::math::rng::device::lognormal<RealType, Method>> {
public:
    struct param_type {
        param_type(RealType m, RealType s, RealType displ, RealType scale)
                : m_(m),
                  s_(s),
                  displ_(displ),
                  scale_(scale) {}
        RealType m_;
        RealType s_;
        RealType displ_;
        RealType scale_;
    };

    distribution_base(RealType m, RealType s, RealType displ, RealType scale)
            : gaussian_(m, s),
              displ_(displ),
              scale_(scale) {
#ifndef __SYCL_DEVICE_ONLY__
        if (scale <= static_cast<RealType>(0.0)) {
            throw oneapi::math::invalid_argument("rng", "lognormal", "scale <= 0");
        }
#endif
    }

    RealType m() const {
        return gaussian_.mean();
    }

    RealType s() const {
        return gaussian_.stddev();
    }

    RealType displ() const {
        return displ_;
    }

    RealType scale() const {
        return scale_;
    }

    param_type param() const {
        return param_type(gaussian_.mean(), gaussian_.stddev(), displ_, scale_);
    }

    void param(const param_type& pt) {
#ifndef __SYCL_DEVICE_ONLY__
        if (pt.scale_ <= static_cast<RealType>(0.0)) {
            throw oneapi::math::invalid_argument("rng", "lognormal", "scale <= 0");
        }
#endif
        gaussian_.param({ pt.m_, pt.s_ });
        displ_ = pt.displ_;
        scale_ = pt.scale_;
    }

protected:
    template <typename EngineType>
    auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, RealType,
                                  sycl::vec<RealType, EngineType::vec_size>>::type {
        auto res = gaussian_.generate(engine);
        return sycl::exp(res) * scale_ + displ_;
    }

    template <typename EngineType>
    RealType generate_single(EngineType& engine) {
        RealType res = gaussian_.generate_single(engine);
        return sycl::exp(res) * scale_ + displ_;
    }

    distribution_base<oneapi::math::rng::device::gaussian<RealType, gaussian_method::box_muller2>>
        gaussian_;
    RealType displ_;
    RealType scale_;
};

} // namespace oneapi::math::rng::device::detail

#endif // ONEMATH_RNG_DEVICE_LOGNORMAL_IMPL_HPP_
