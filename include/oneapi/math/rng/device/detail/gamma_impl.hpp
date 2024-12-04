/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef ONEMATH_RNG_DEVICE_GAMMA_IMPL_HPP_
#define ONEMATH_RNG_DEVICE_GAMMA_IMPL_HPP_

#include "vm_wrappers.hpp"

namespace oneapi::math::rng::device::detail {

enum class gamma_algorithm { Exponential = 0, Vaduva, EPD_Transform, Marsaglia };

// 1/3
template <typename DataType>
inline DataType gamma_c1() {
    if constexpr (std::is_same_v<DataType, double>)
        return 0x1.5555555555555p-2;
    else
        return 0x1.555556p-2f;
}

// 0.0331
template <typename DataType>
inline DataType gamma_c2() {
    if constexpr (std::is_same_v<DataType, double>)
        return 0x1.0f27bb2fec56dp-5;
    else
        return 0x1.0f27bcp-5f;
}

// 0.6
template <typename DataType>
inline DataType gamma_c06() {
    if constexpr (std::is_same_v<DataType, double>)
        return 0x1.3333333333333p-1;
    else
        return 0x1.333334p-1f;
}

template <typename RealType, typename Method>
class distribution_base<oneapi::math::rng::device::gamma<RealType, Method>> {
public:
    struct param_type {
        param_type(RealType alpha, RealType a, RealType beta) : alpha_(alpha), a_(a), beta_(beta) {}
        RealType alpha_;
        RealType a_;
        RealType beta_;
    };

    distribution_base(RealType alpha, RealType a, RealType beta)
            : alpha_(alpha),
              a_(a),
              beta_(beta),
              count_(0) {
        set_algorithm();
#ifndef __SYCL_DEVICE_ONLY__
        if (alpha <= RealType(0.0)) {
            throw oneapi::math::invalid_argument("rng", "gamma", "alpha <= 0");
        }
        else if (beta <= RealType(0.0)) {
            throw oneapi::math::invalid_argument("rng", "gamma", "beta <= 0");
        }
#endif
    }

    RealType alpha() const {
        return alpha_;
    }

    RealType a() const {
        return a_;
    }

    RealType beta() const {
        return beta_;
    }

    std::size_t count_rejected_numbers() const {
        return count_;
    }

    param_type param() const {
        return param_type(alpha_, a_, beta_);
    }

    void param(const param_type& pt) {
#ifndef __SYCL_DEVICE_ONLY__
        if (pt.alpha_ <= RealType(0.0)) {
            throw oneapi::math::invalid_argument("rng", "gamma", "alpha <= 0");
        }
        else if (pt.beta_ <= RealType(0.0)) {
            throw oneapi::math::invalid_argument("rng", "gamma", "beta <= 0");
        }
#endif
        alpha_ = pt.alpha_;
        a_ = pt.a_;
        beta_ = pt.beta_;
        set_algorithm();
    }

protected:
    void set_algorithm() {
        if (alpha_ <= RealType(1.0)) {
            if (alpha_ == RealType(1.0)) {
                algorithm_ = gamma_algorithm::Exponential;
            }
            else if (alpha_ > gamma_c06<RealType>()) {
                algorithm_ = gamma_algorithm::Vaduva;
            }
            else {
                algorithm_ = gamma_algorithm::EPD_Transform;
            }
        }
        else {
            algorithm_ = gamma_algorithm::Marsaglia;
        }
    }

    template <typename T, int vecSize>
    inline std::pair<T, T> gauss_BM2_for_Marsaglia(const sycl::vec<T, vecSize>& vec) {
        T tmp, sin, cos, gauss_1, gauss_2;
        tmp = ln_wrapper(vec[0]);
        tmp = sqrt_wrapper(T(-2.0) * tmp);
        sin = sincospi_wrapper(T(2) * vec[2], cos);
        gauss_1 = (tmp * sin);
        gauss_2 = (tmp * cos);
        return { gauss_1, gauss_2 };
    }

    template <std::int32_t n, typename T, typename EngineType>
    T acc_rej_kernel(T& z, EngineType& engine) {
        RealType flC, flD;
        if (algorithm_ == gamma_algorithm::Vaduva) {
            flC = RealType(1.0) / alpha_;
            flD = (RealType(1.0) - alpha_) *
                  exp_wrapper(ln_wrapper(alpha_) * alpha_ / (RealType(1.0) - alpha_));
        }
        else if (algorithm_ == gamma_algorithm::EPD_Transform) {
            flC = RealType(1.0) / alpha_;
            flD = (RealType(1.0) - alpha_);
        }
        else if (algorithm_ == gamma_algorithm::Marsaglia) {
            flD = alpha_ - gamma_c1<RealType>();
            flC = sqrt_wrapper(RealType(1.0) / (RealType(9.0) * alpha_ - RealType(3.0)));
        }

        count_ = 0;
        RealType z1, z2, z3, z4;
        for (int i = 0; i < n; i++) {
            while (1) { // looping until satisfied
                if (!flag_) {
                    z1 = engine.generate_single(RealType(0), RealType(1));
                    z2 = engine.generate_single(RealType(0), RealType(1));
                }

                if (algorithm_ == gamma_algorithm::Vaduva) {
                    z1 = -ln_wrapper(z1);
                    z2 = -ln_wrapper(z2);
                    z[i] = powr_wrapper(z1, flC);
                    if (z1 + z2 >= z[i] + flD) {
                        break;
                    }
                }
                if (algorithm_ == gamma_algorithm::EPD_Transform) {
                    z2 = -ln_wrapper(z2);
                    if (z1 <= flD) {
                        z[i] = powr_wrapper(z1, flC);
                        if (z[i] <= z2) {
                            break;
                        }
                    }
                    else {
                        z1 = -ln_wrapper((RealType(1.0) - z1) * flC);
                        z[i] = powr_wrapper(flD + alpha_ * z1, flC);
                        if (z[i] <= z2 + z1) {
                            break;
                        }
                    }
                }
                if (algorithm_ == gamma_algorithm::Marsaglia) {
                    RealType local_uniform_2, local_gauss;
                    if (!flag_) {
                        z3 = engine.generate_single(RealType(0), RealType(1));
                        z4 = engine.generate_single(RealType(0), RealType(1));
                        auto gauss =
                            gauss_BM2_for_Marsaglia(sycl::vec<RealType, 4>{ z1, z2, z3, z4 });
                        local_uniform_2 = z2;
                        local_gauss = gauss.first;

                        saved_uniform_2_ = z4;
                        saved_gauss_ = gauss.second;
                    }
                    else {
                        local_uniform_2 = saved_uniform_2_;
                        local_gauss = saved_gauss_;
                    }
                    flag_ = !flag_;
                    z[i] = RealType(1.0) + flC * local_gauss;
                    if (z[i] > RealType(0.0)) {
                        z[i] = z[i] * z[i] * z[i];
                        local_gauss = local_gauss * local_gauss;
                        if (local_uniform_2 <
                            RealType(1.0) - gamma_c2<RealType>() * local_gauss * local_gauss) {
                            z[i] = flD * z[i];
                            break;
                        }
                        else {
                            RealType local_uniform_1 = ln_wrapper(z[i]);
                            local_uniform_2 = ln_wrapper(local_uniform_2);
                            if (local_uniform_2 <
                                RealType(0.5) * local_gauss +
                                    flD * (RealType(1.0) - z[i] + local_uniform_1)) {
                                z[i] = flD * z[i];
                                break;
                            }
                        }
                    }
                }
                ++count_;
            }
        }
        auto res = a_ + beta_ * z;
        if constexpr (std::is_same_v<Method, gamma_method::marsaglia_accurate>) {
            for (std::int32_t i = 0; i < EngineType::vec_size; i++) {
                if (res[i] < a_)
                    res[i] = a_;
            }
        }
        return res;
    }

    template <typename EngineType>
    auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, RealType,
                                  sycl::vec<RealType, EngineType::vec_size>>::type {
        if (algorithm_ == gamma_algorithm::Exponential) {
            distribution_base<oneapi::math::rng::device::exponential<RealType>> distr_exp(a_,
                                                                                          beta_);
            return distr_exp.generate(engine);
        }
        sycl::vec<RealType, EngineType::vec_size> res{};
        res = acc_rej_kernel<EngineType::vec_size>(res, engine);

        return res;
    }

    template <typename EngineType>
    RealType generate_single(EngineType& engine) {
        if (algorithm_ == gamma_algorithm::Exponential) {
            distribution_base<oneapi::math::rng::device::exponential<RealType>> distr_exp(a_,
                                                                                          beta_);
            RealType z = distr_exp.generate_single(engine);
            return z;
        }
        sycl::vec<RealType, 1> res{};
        res = acc_rej_kernel<1>(res, engine);

        return res[0];
    }

    RealType alpha_;
    RealType a_;
    RealType beta_;
    RealType saved_gauss_;
    RealType saved_uniform_2_;
    bool flag_ = false;
    std::size_t count_;
    gamma_algorithm algorithm_;
};

} // namespace oneapi::math::rng::device::detail

#endif // ONEMATH_RNG_DEVICE_GAMMA_IMPL_HPP_
