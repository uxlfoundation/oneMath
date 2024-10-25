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

#ifndef _MKL_RNG_DEVICE_BETA_IMPL_HPP_
#define _MKL_RNG_DEVICE_BETA_IMPL_HPP_

#include "vm_wrappers.hpp"

namespace oneapi::mkl::rng::device::detail {

enum class beta_algorithm { Johnk = 0, Atkinson1, Atkinson2, Atkinson3, Cheng, p1, q1, p1q1 };

// log(4)=1.3862944..
template <typename DataType>
inline DataType log4() {
    if constexpr (std::is_same_v<DataType, double>)
        return 0x1.62e42fefa39efp+0;
    else
        return 0x1.62e43p+0f;
}

// K=0.85225521765372429631847
template <typename DataType>
inline DataType beta_k() {
    if constexpr (std::is_same_v<DataType, double>)
        return 0x1.b45acbbf56123p-1;
    else
        return 0x1.b45accp-1f;
}

// C=-0.956240971340815081432202
template <typename DataType>
inline DataType beta_c() {
    if constexpr (std::is_same_v<DataType, double>)
        return -0x1.e9986aa60216p-1;
    else
        return -0x1.e9986ap-1f;
}

template <typename RealType, typename Method>
class distribution_base<oneapi::mkl::rng::device::beta<RealType, Method>> {
public:
    struct param_type {
        param_type(RealType p, RealType q, RealType a, RealType b) : p_(p), q_(q), a_(a), b_(b) {}
        RealType p_;
        RealType q_;
        RealType a_;
        RealType b_;
    };

    distribution_base(RealType p, RealType q, RealType a, RealType b)
            : p_(p),
              q_(q),
              a_(a),
              b_(b),
              count_(0) {
        set_algorithm();
#ifndef __SYCL_DEVICE_ONLY__
        if (p <= RealType(0.0)) {
            throw oneapi::mkl::invalid_argument("rng", "beta", "p <= 0");
        }
        else if (q <= RealType(0.0)) {
            throw oneapi::mkl::invalid_argument("rng", "beta", "q <= 0");
        }
        else if (b <= RealType(0.0)) {
            throw oneapi::mkl::invalid_argument("rng", "beta", "b <= 0");
        }
#endif
    }

    RealType p() const {
        return p_;
    }

    RealType q() const {
        return q_;
    }

    RealType a() const {
        return a_;
    }

    RealType b() const {
        return b_;
    }

    std::size_t count_rejected_numbers() const {
        return count_;
    }

    param_type param() const {
        return param_type(p_, q_, a_, b_);
    }

    void param(const param_type& pt) {
#ifndef __SYCL_DEVICE_ONLY__
        if (pt.p_ <= RealType(0.0)) {
            throw oneapi::mkl::invalid_argument("rng", "beta", "p <= 0");
        }
        else if (pt.q_ <= RealType(0.0)) {
            throw oneapi::mkl::invalid_argument("rng", "beta", "q <= 0");
        }
        else if (pt.b_ <= RealType(0.0)) {
            throw oneapi::mkl::invalid_argument("rng", "beta", "b <= 0");
        }
#endif
        p_ = pt.p_;
        q_ = pt.q_;
        a_ = pt.a_;
        b_ = pt.b_;
        set_algorithm();
    }

protected:
    template <std::int32_t n, typename T>
    T pq_kernel(T& z) {
        for (std::int32_t i = 0; i < n; i++) {
            if (p_ == RealType(1.0)) {
                z[i] = pow_wrapper(z[i], RealType(1) / q_);
                z[i] = RealType(1.0) - z[i];
            }
            if (q_ == RealType(1.0)) {
                z[i] = pow_wrapper(z[i], RealType(1) / p_);
            }
        }
        count_ = 0;

        // p1q1
        return a_ + b_ * z;
    }

    template <std::int32_t n, typename T, typename EngineType>
    T acc_rej_kernel(T& z, EngineType& engine) {
        RealType s, t;

        RealType flKoef1, flKoef2, flKoef3, flKoef4, flKoef5, flKoef6;
        RealType flDeg[2];

        if (algorithm_ == beta_algorithm::Atkinson1) {
            RealType flInv_s[2], flTmp[2];
            flTmp[0] = p_ * (RealType(1.0) - p_);
            flTmp[1] = q_ * (RealType(1.0) - q_);

            flTmp[0] = sqrt_wrapper(flTmp[0]);
            flTmp[1] = sqrt_wrapper(flTmp[1]);

            t = flTmp[0] / (flTmp[0] + flTmp[1]);

            s = q_ * t;
            s = s / (s + p_ * (RealType(1.0) - t));

            flInv_s[0] = RealType(1.0) / s;
            flInv_s[1] = RealType(1.0) / (RealType(1.0) - s);
            flDeg[0] = RealType(1.0) / p_;
            flDeg[1] = RealType(1.0) / q_;

            flInv_s[0] = pow_wrapper(flInv_s[0], flDeg[0]);
            flInv_s[1] = pow_wrapper(flInv_s[1], flDeg[1]);

            flKoef1 = t * flInv_s[0];
            flKoef2 = (RealType(1.0) - t) * flInv_s[1];
            flKoef3 = RealType(1.0) - q_;
            flKoef4 = RealType(1.0) - p_;
            flKoef5 = RealType(1.0) / (RealType(1.0) - t);
            flKoef6 = RealType(1.0) / t;
        }
        else if (algorithm_ == beta_algorithm::Atkinson2) {
            RealType flInv_s[2], flTmp;

            t = RealType(1.0) - p_;
            t /= (t + q_);

            flTmp = RealType(1.0) - t;
            flTmp = pow_wrapper(flTmp, q_);
            s = q_ * t;
            s /= (s + p_ * flTmp);

            flInv_s[0] = RealType(1.0) / s;
            flInv_s[1] = RealType(1.0) / (RealType(1.0) - s);
            flDeg[0] = RealType(1.0) / p_;
            flDeg[1] = RealType(1.0) / q_;

            flInv_s[0] = pow_wrapper(flInv_s[0], flDeg[0]);
            flInv_s[1] = pow_wrapper(flInv_s[1], flDeg[1]);

            flKoef1 = t * flInv_s[0];
            flKoef2 = (RealType(1.0) - t) * flInv_s[1];
            flKoef3 = RealType(1.0) - q_;
            flKoef4 = RealType(1.0) - p_;
        }
        else if (algorithm_ == beta_algorithm::Atkinson3) {
            RealType flInv_s[2], flTmp;

            t = RealType(1.0) - q_;
            t /= (t + p_);

            flTmp = RealType(1.0) - t;
            flTmp = pow_wrapper(flTmp, p_);
            s = p_ * t;
            s /= (s + q_ * flTmp);

            flInv_s[0] = RealType(1.0) / s;
            flInv_s[1] = RealType(1.0) / (RealType(1.0) - s);
            flDeg[0] = RealType(1.0) / q_;
            flDeg[1] = RealType(1.0) / p_;

            flInv_s[0] = pow_wrapper(flInv_s[0], flDeg[0]);
            flInv_s[1] = pow_wrapper(flInv_s[1], flDeg[1]);

            flKoef1 = t * flInv_s[0];
            flKoef2 = (RealType(1.0) - t) * flInv_s[1];
            flKoef3 = RealType(1.0) - p_;
            flKoef4 = RealType(1.0) - q_;
        }
        else if (algorithm_ == beta_algorithm::Cheng) {
            flKoef1 = p_ + q_;
            flKoef2 = (flKoef1 - RealType(2.0)) / (RealType(2.0) * p_ * q_ - flKoef1);
            flKoef2 = sqrt_wrapper(flKoef2);
            flKoef3 = p_ + RealType(1.0) / flKoef2;
        }

        RealType z1, z2;

        count_ = 0;
        for (int i = 0; i < n; i++) {
            while (1) { // looping until satisfied
                z1 = engine.generate_single(RealType(0), RealType(1));
                z2 = engine.generate_single(RealType(0), RealType(1));

                if (algorithm_ == beta_algorithm::Johnk) {
                    RealType flU1, flU2, flSum;
                    z1 = ln_wrapper(z1) / p_;
                    z2 = ln_wrapper(z2) / q_;

                    z1 = exp_wrapper(z1);
                    z2 = exp_wrapper(z2);

                    flU1 = z1;
                    flU2 = z2;
                    flSum = flU1 + flU2;
                    if (flSum > RealType(0.0) && flSum <= RealType(1.0)) {
                        z[i] = flU1 / flSum;
                        break;
                    }
                }
                if (algorithm_ == beta_algorithm::Atkinson1) {
                    RealType flU, flExp, flX, flLn;
                    z2 = ln_wrapper(z2);

                    flU = z1;
                    flExp = z2;
                    if (flU <= s) {
                        flU = pow_wrapper(flU, flDeg[0]);
                        flX = flKoef1 * flU;
                        flLn = (RealType(1.0) - flX) * flKoef5;
                        flLn = ln_wrapper(flLn);
                        if (flKoef3 * flLn + flExp <= RealType(0.0)) {
                            z[i] = flX;
                            break;
                        }
                    }
                    else {
                        flU = RealType(1.0) - flU;
                        flU = pow_wrapper(flU, flDeg[1]);
                        flX = RealType(1.0) - flKoef2 * flU;

                        flLn = flX * flKoef6;
                        flLn = ln_wrapper(flLn);
                        if (flKoef4 * flLn + flExp <= RealType(0.0)) {
                            z[i] = flX;
                            break;
                        }
                    }
                }
                if (algorithm_ == beta_algorithm::Atkinson2) {
                    RealType flU, flExp, flX, flLn;
                    z2 = ln_wrapper(z2);

                    flU = z1;
                    flExp = z2;
                    if (flU <= s) {
                        flU = pow_wrapper(flU, flDeg[0]);
                        flX = flKoef1 * flU;
                        flLn = (RealType(1.0) - flX);
                        flLn = ln_wrapper(flLn);
                        if (flKoef3 * flLn + flExp <= RealType(0.0)) {
                            z[i] = flX;
                            break;
                        }
                    }
                    else {
                        flU = RealType(1.0) - flU;
                        flU = pow_wrapper(flU, flDeg[1]);
                        flX = RealType(1.0) - flKoef2 * flU;

                        flLn = flX / t;
                        flLn = ln_wrapper(flLn);
                        if (flKoef4 * flLn + flExp <= RealType(0.0)) {
                            z[i] = flX;
                            break;
                        }
                    }
                }
                if (algorithm_ == beta_algorithm::Atkinson3) {
                    RealType flU, flExp, flX, flLn;
                    z2 = ln_wrapper(z2);

                    flU = z1;
                    flExp = z2;
                    if (flU <= s) {
                        flU = pow_wrapper(flU, flDeg[0]);
                        flX = flKoef1 * flU;
                        flLn = (RealType(1.0) - flX);
                        flLn = ln_wrapper(flLn);
                        if (flKoef3 * flLn + flExp <= RealType(0.0)) {
                            z[i] = RealType(1.0) - flX;
                            break;
                        }
                    }
                    else {
                        flU = RealType(1.0) - flU;
                        flU = pow_wrapper(flU, flDeg[1]);
                        flX = RealType(1.0) - flKoef2 * flU;

                        flLn = flX / t;
                        flLn = ln_wrapper(flLn);
                        if (flKoef4 * flLn + flExp <= RealType(0.0)) {
                            z[i] = RealType(1.0) - flX;
                            break;
                        }
                    }
                }
                if (algorithm_ == beta_algorithm::Cheng) {
                    RealType flU1, flU2, flV, flW, flInv;
                    RealType flTmp[2];
                    flU1 = z1;
                    flU2 = z2;

                    flV = flU1 / (RealType(1.0) - flU1);

                    flV = ln_wrapper(flV);

                    flV = flKoef2 * flV;

                    flW = flV;
                    flW = exp_wrapper(flW);
                    flW = p_ * flW;
                    flInv = RealType(1.0) / (q_ + flW);
                    flTmp[0] = flKoef1 * flInv;
                    flTmp[1] = flU1 * flU1 * flU2;
                    for (int i = 0; i < 2; i++) {
                        flTmp[i] = ln_wrapper(flTmp[i]);
                    }

                    if (flKoef1 * flTmp[0] + flKoef3 * flV - log4<RealType>() >= flTmp[1]) {
                        z[i] = flW * flInv;
                        break;
                    }
                }
                ++count_;
            }
        }
        return a_ + b_ * z;
    }

    template <typename EngineType>
    auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, RealType,
                                  sycl::vec<RealType, EngineType::vec_size>>::type {
        sycl::vec<RealType, EngineType::vec_size> res{};
        if (algorithm_ == beta_algorithm::p1 || algorithm_ == beta_algorithm::q1 ||
            algorithm_ == beta_algorithm::p1q1) {
            res = engine.generate(RealType(0), RealType(1));
            res = pq_kernel<EngineType::vec_size>(res);
        }
        else {
            res = acc_rej_kernel<EngineType::vec_size>(res, engine);
        }
        if constexpr (std::is_same_v<Method, beta_method::cja_accurate>) {
            for (std::int32_t i = 0; i < EngineType::vec_size; i++) {
                if (res[i] < a_)
                    res[i] = a_;
                if (res[i] > a_ + b_)
                    res[i] = a_ + b_;
            }
        }
        return res;
    }

    template <typename EngineType>
    RealType generate_single(EngineType& engine) {
        RealType res{};
        sycl::vec<RealType, 1> z{ res };
        if (algorithm_ == beta_algorithm::p1 || algorithm_ == beta_algorithm::q1 ||
            algorithm_ == beta_algorithm::p1q1) {
            z[0] = engine.generate_single(RealType(0), RealType(1));
            res = pq_kernel<1>(z);
        }
        else {
            res = acc_rej_kernel<1>(z, engine);
        }
        if constexpr (std::is_same_v<Method, beta_method::cja_accurate>) {
            for (std::int32_t i = 0; i < EngineType::vec_size; i++) {
                if (res[i] < a_)
                    res[i] = a_;
                if (res[i] > a_ + b_)
                    res[i] = a_ + b_;
            }
        }
        return res;
    }

    void set_algorithm() {
        if (p_ < RealType(1.0) && q_ < RealType(1.0)) {
            if (q_ + beta_k<RealType>() * p_ * p_ + beta_c<RealType>() <= RealType(0.0)) {
                algorithm_ = beta_algorithm::Johnk;
            }
            else {
                algorithm_ = beta_algorithm::Atkinson1;
            }
        }
        else if (p_ < RealType(1.0) && q_ > RealType(1.0)) {
            algorithm_ = beta_algorithm::Atkinson2;
        }
        else if (p_ > RealType(1.0) && q_ < RealType(1.0)) {
            algorithm_ = beta_algorithm::Atkinson3;
        }
        else if (p_ > RealType(1.0) && q_ > RealType(1.0)) {
            algorithm_ = beta_algorithm::Cheng;
        }
        else if (p_ == RealType(1.0) && q_ != RealType(1.0)) {
            algorithm_ = beta_algorithm::p1;
        }
        else if (q_ == RealType(1.0) && p_ != RealType(1.0)) {
            algorithm_ = beta_algorithm::q1;
        }
        else {
            algorithm_ = beta_algorithm::p1q1;
        }
    }

    RealType p_;
    RealType q_;
    RealType a_;
    RealType b_;
    std::size_t count_;
    beta_algorithm algorithm_;
};

} // namespace oneapi::mkl::rng::device::detail

#endif // _MKL_RNG_DEVICE_BETA_IMPL_HPP_
