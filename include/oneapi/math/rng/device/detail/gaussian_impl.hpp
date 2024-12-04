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

#ifndef ONEMATH_RNG_DEVICE_GAUSSIAN_IMPL_HPP_
#define ONEMATH_RNG_DEVICE_GAUSSIAN_IMPL_HPP_

#include "vm_wrappers.hpp"

namespace oneapi::math::rng::device::detail {

// sqrt(2)
template <typename RealType = float>
constexpr inline RealType sqrt2() {
    return 0x1.6A09E6P+0f; // 1.414213562
}

template <>
constexpr inline double sqrt2<double>() {
    return 0x1.6A09E667F3BCDP+0; // 1.414213562
}

template <typename RealType>
class distribution_base<
    oneapi::math::rng::device::gaussian<RealType, gaussian_method::box_muller2>> {
public:
    struct param_type {
        param_type(RealType mean, RealType stddev) : mean_(mean), stddev_(stddev) {}
        RealType mean_;
        RealType stddev_;
    };

    distribution_base(RealType mean, RealType stddev) : mean_(mean), stddev_(stddev) {
        flag_ = false;
#ifndef __SYCL_DEVICE_ONLY__
        if (stddev <= RealType(0)) {
            throw oneapi::math::invalid_argument("rng", "gaussian", "stddev <= 0");
        }
#endif
    }

    RealType mean() const {
        return mean_;
    }

    RealType stddev() const {
        return stddev_;
    }

    param_type param() const {
        return param_type(mean_, stddev_);
    }

    void param(const param_type& pt) {
#ifndef __SYCL_DEVICE_ONLY__
        if (pt.stddev_ <= RealType(0)) {
            throw oneapi::math::invalid_argument("rng", "gaussian", "stddev <= 0");
        }
#endif
        mean_ = pt.mean_;
        stddev_ = pt.stddev_;
    }

protected:
    template <typename EngineType>
    __attribute__((always_inline)) inline auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, RealType,
                                  sycl::vec<RealType, EngineType::vec_size>>::type {
        RealType u1, u2, u1_transformed;

        if constexpr (EngineType::vec_size == 1) {
            RealType res;
            if (!flag_) {
                u1 = engine.generate(RealType(0), RealType(1));
                u2 = engine.generate(RealType(0), RealType(1));
                u1_transformed = ln_wrapper(u1);
                u1_transformed = sqrt_wrapper(static_cast<RealType>(-2.0) * u1_transformed);
                res = u1_transformed * sinpi_wrapper(RealType(2) * u2) * stddev_ + mean_;
                u1_transformed_ = u1_transformed;
                u2_ = u2;
                flag_ = true;
                return res;
            }
            res = u1_transformed_ * cospi_wrapper(RealType(2) * u2_) * stddev_ + mean_;
            flag_ = false;
            return res;
        }
        else {
            RealType sin, cos;
            sycl::vec<RealType, EngineType::vec_size> res;
            if (!flag_) {
                constexpr std::int32_t tail = EngineType::vec_size % 2;
                auto uniform_res = engine.generate(RealType(0), RealType(1));
#pragma unroll
                for (std::int32_t i = 0; i < EngineType::vec_size - tail; i += 2) {
                    u1 = uniform_res[i];
                    u2 = uniform_res[i + 1];
                    u1_transformed = ln_wrapper(u1);
                    u1_transformed = sqrt_wrapper(static_cast<RealType>(-2.0) * u1_transformed);
                    sin = sincospi_wrapper(RealType(2.0) * u2, cos);
                    res[i] = (u1_transformed * sin) * stddev_ + mean_;
                    res[i + 1] = (u1_transformed * cos) * stddev_ + mean_;
                }
                if constexpr (tail) {
                    u1 = uniform_res[EngineType::vec_size - 1];
                    u2 = engine.generate_single(RealType(0), RealType(1));
                    u1_transformed = ln_wrapper(u1);
                    u1_transformed = sqrt_wrapper(static_cast<RealType>(-2.0) * u1_transformed);
                    res[EngineType::vec_size - 1] =
                        u1_transformed * sinpi_wrapper(RealType(2) * u2) * stddev_ + mean_;
                    u1_transformed_ = u1_transformed;
                    u2_ = u2;
                    flag_ = true;
                }
                return res;
            }

            res[0] = u1_transformed_ * cospi_wrapper(RealType(2) * u2_) * stddev_ + mean_;
            flag_ = false;
            constexpr std::int32_t tail = (EngineType::vec_size - 1) % 2;
#pragma unroll
            for (std::int32_t i = 1; i < EngineType::vec_size - tail; i += 2) {
                u1 = engine.generate_single(RealType(0), RealType(1));
                u2 = engine.generate_single(RealType(0), RealType(1));
                u1_transformed = ln_wrapper(u1);
                u1_transformed = sqrt_wrapper(static_cast<RealType>(-2.0) * u1_transformed);
                sin = sincospi_wrapper(RealType(2.0) * u2, cos);
                res[i] = (u1_transformed * sin) * stddev_ + mean_;
                res[i + 1] = (u1_transformed * cos) * stddev_ + mean_;
            }
            if constexpr (tail) {
                u1 = engine.generate_single(RealType(0), RealType(1));
                u2 = engine.generate_single(RealType(0), RealType(1));
                u1_transformed = ln_wrapper(u1);
                u1_transformed = sqrt_wrapper(static_cast<RealType>(-2.0) * u1_transformed);
                res[EngineType::vec_size - 1] =
                    u1_transformed * sinpi_wrapper(RealType(2) * u2) * stddev_ + mean_;
                u1_transformed_ = u1_transformed;
                u2_ = u2;
                flag_ = true;
            }
            return res;
        }
    }

    template <typename EngineType>
    __attribute__((always_inline)) inline RealType generate_single(EngineType& engine) {
        RealType u1, u2, u1_transformed;
        RealType res;
        if (!flag_) {
            u1 = engine.generate_single(RealType(0), RealType(1));
            u2 = engine.generate_single(RealType(0), RealType(1));
            u1_transformed = ln_wrapper(u1);
            u1_transformed = sqrt_wrapper(static_cast<RealType>(-2.0) * u1_transformed);
            res = u1_transformed * sinpi_wrapper(RealType(2) * u2) * stddev_ + mean_;
            u1_transformed_ = u1_transformed;
            u2_ = u2;
            flag_ = true;
            return res;
        }
        res = u1_transformed_ * cospi_wrapper(RealType(2) * u2_) * stddev_ + mean_;
        flag_ = false;
        return res;
    }

    RealType mean_;
    RealType stddev_;
    bool flag_ = false;
    RealType u1_transformed_;
    RealType u2_;

    friend class distribution_base<
        oneapi::math::rng::device::lognormal<RealType, lognormal_method::box_muller2>>;
    friend class distribution_base<
        oneapi::math::rng::device::poisson<std::int32_t, poisson_method::devroye>>;
    friend class distribution_base<
        oneapi::math::rng::device::poisson<std::uint32_t, poisson_method::devroye>>;
};

#if MKL_RNG_USE_BINARY_CODE

template <typename RealType>
class distribution_base<oneapi::math::rng::device::gaussian<RealType, gaussian_method::icdf>> {
public:
    struct param_type {
        param_type(RealType mean, RealType stddev) : mean_(mean), stddev_(stddev) {}
        RealType mean_;
        RealType stddev_;
    };

    distribution_base(RealType mean, RealType stddev) : mean_(mean), stddev_(stddev) {
#ifndef __SYCL_DEVICE_ONLY__
        if (stddev <= RealType(0)) {
            throw oneapi::math::invalid_argument("rng", "gaussian", "stddev <= 0");
        }
#endif
    }

    RealType mean() const {
        return mean_;
    }

    RealType stddev() const {
        return stddev_;
    }

    param_type param() const {
        return param_type(mean_, stddev_);
    }

    void param(const param_type& pt) {
#ifndef __SYCL_DEVICE_ONLY__
        if (pt.stddev_ <= RealType(0)) {
            throw oneapi::math::invalid_argument("rng", "gaussian", "stddev <= 0");
        }
#endif
        mean_ = pt.mean_;
        stddev_ = pt.stddev_;
    }

protected:
    template <typename EngineType>
    __attribute__((always_inline)) inline auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, RealType,
                                  sycl::vec<RealType, EngineType::vec_size>>::type {
        if constexpr (EngineType::vec_size == 1) {
            return generate_single(engine);
        }
        else {
            RealType stddev = stddev_ * sqrt2<RealType>();
            sycl::vec<RealType, EngineType::vec_size> res;
            sycl::vec<RealType, EngineType::vec_size> u =
                engine.generate(RealType(-1), RealType(1));
            for (std::int32_t i = 0; i < EngineType::vec_size; i++) {
                res[i] = erf_inv_wrapper(u[i]);
            }
            return res * stddev + mean_;
        }
    }

    template <typename EngineType>
    __attribute__((always_inline)) inline RealType generate_single(EngineType& engine) {
        RealType stddev = stddev_ * sqrt2<RealType>();
        RealType u = engine.generate_single(RealType(-1), RealType(1));
        return sycl::fma(erf_inv_wrapper(u), stddev, mean_);
    }

    RealType mean_;
    RealType stddev_;
};
#endif

} // namespace oneapi::math::rng::device::detail

#endif // ONEMATH_RNG_DEVICE_GAUSSIAN_IMPL_HPP_
