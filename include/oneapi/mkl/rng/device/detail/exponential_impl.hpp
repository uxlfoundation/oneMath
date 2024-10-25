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

#ifndef _MKL_RNG_DEVICE_EXPONENTIAL_IMPL_HPP_
#define _MKL_RNG_DEVICE_EXPONENTIAL_IMPL_HPP_

#include "vm_wrappers.hpp"

namespace oneapi::mkl::rng::device::detail {

template <typename RealType, typename Method>
class distribution_base<oneapi::mkl::rng::device::exponential<RealType, Method>> {
public:
    struct param_type {
        param_type(RealType a, RealType beta) : a_(a), beta_(beta) {}
        RealType a_;
        RealType beta_;
    };

    distribution_base(RealType a, RealType beta) : a_(a), beta_(beta) {
#ifndef __SYCL_DEVICE_ONLY__
        if (beta <= static_cast<RealType>(0.0)) {
            throw oneapi::mkl::invalid_argument("rng", "exponential", "beta <= 0");
        }
#endif
    }

    RealType a() const {
        return a_;
    }

    RealType beta() const {
        return beta_;
    }

    param_type param() const {
        return param_type(a_, beta_);
    }

    void param(const param_type& pt) {
#ifndef __SYCL_DEVICE_ONLY__
        if (pt.beta_ <= static_cast<RealType>(0.0)) {
            throw oneapi::mkl::invalid_argument("rng", "exponential", "beta <= 0");
        }
#endif
        a_ = pt.a_;
        beta_ = pt.beta_;
    }

protected:
    template <typename EngineType>
    auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, RealType,
                                  sycl::vec<RealType, EngineType::vec_size>>::type {
        auto res = engine.generate(RealType(0), RealType(1));
        if constexpr (EngineType::vec_size == 1) {
            res = ln_wrapper(res);
        }
        else {
            for (int i = 0; i < EngineType::vec_size; ++i) {
                res[i] = ln_wrapper(res[i]);
            }
        }
        res = a_ - res * beta_;
        if constexpr (std::is_same<Method, exponential_method::icdf_accurate>::value) {
#ifndef __HIPSYCL__
            res = sycl::fmax(res, a_);
#else
            // a workaround for hipSYCL (AdaptiveCpp)
            if constexpr (EngineType::vec_size == 1) {
                res = std::fmax(res, a_);
            }
            else {
                for (int i = 0; i < EngineType::vec_size; i++) {
                    res[i] = std::fmax(res[i], a_);
                }
            }
#endif
        }
        return res;
    }

    template <typename EngineType>
    RealType generate_single(EngineType& engine) {
        RealType res = engine.generate_single(RealType(0), RealType(1));
        res = ln_wrapper(res);
        res = a_ - res * beta_;
        if constexpr (std::is_same<Method, exponential_method::icdf_accurate>::value) {
#ifndef __HIPSYCL__
            res = sycl::fmax(res, a_);
#else
            // a workaround for hipSYCL (AdaptiveCpp)
            if constexpr (EngineType::vec_size == 1) {
                res = std::fmax(res, a_);
            }
            else {
                for (int i = 0; i < EngineType::vec_size; i++) {
                    res[i] = std::fmax(res[i], a_);
                }
            }
#endif
        }
        return res;
    }

    RealType a_;
    RealType beta_;

    friend class distribution_base<
        oneapi::mkl::rng::device::poisson<std::int32_t, poisson_method::devroye>>;
    friend class distribution_base<
        oneapi::mkl::rng::device::poisson<std::uint32_t, poisson_method::devroye>>;
    friend class distribution_base<oneapi::mkl::rng::device::gamma<float, gamma_method::marsaglia>>;
    friend class distribution_base<
        oneapi::mkl::rng::device::gamma<double, gamma_method::marsaglia>>;
    friend class distribution_base<
        oneapi::mkl::rng::device::gamma<float, gamma_method::marsaglia_accurate>>;
    friend class distribution_base<
        oneapi::mkl::rng::device::gamma<double, gamma_method::marsaglia_accurate>>;
};

} // namespace oneapi::mkl::rng::device::detail

#endif // _MKL_RNG_DEVICE_EXPONENTIAL_IMPL_HPP_
