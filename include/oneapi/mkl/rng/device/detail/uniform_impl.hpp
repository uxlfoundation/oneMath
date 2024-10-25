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

#ifndef _MKL_RNG_DEVICE_UNIFORM_IMPL_HPP_
#define _MKL_RNG_DEVICE_UNIFORM_IMPL_HPP_

#include <limits>
#include <cmath>
#include "engine_base.hpp"

namespace oneapi::mkl::rng::device::detail {

static inline std::uint64_t umul_hi_64(const std::uint64_t a, const std::uint64_t b) {
    const std::uint64_t a_lo = a & 0xFFFFFFFFULL;
    const std::uint64_t a_hi = a >> 32;
    const std::uint64_t b_lo = b & 0xFFFFFFFFULL;
    const std::uint64_t b_hi = b >> 32;

    const std::uint64_t ab_hi = a_hi * b_hi;
    const std::uint64_t ab_lo = a_lo * b_lo;
    const std::uint64_t ab_md = a_hi * b_lo;
    const std::uint64_t ba_md = b_hi * a_lo;

    const std::uint64_t bias =
        ((ab_md & 0xFFFFFFFFULL) + (ba_md & 0xFFFFFFFFULL) + (ab_lo >> 32)) >> 32;

    return ab_hi + (ab_md >> 32) + (ba_md >> 32) + bias;
}

template <typename EngineType, typename Generator>
static inline void generate_leftover(std::uint64_t range, Generator generate, std::uint64_t& res_64,
                                     std::uint64_t& leftover) {
    if constexpr (std::is_same_v<EngineType, mcg31m1<EngineType::vec_size>>) {
        std::uint32_t res_1 = generate();
        std::uint32_t res_2 = generate();
        std::uint32_t res_3 = generate();
        res_64 = (static_cast<std::uint64_t>(res_3) << 62) +
                 (static_cast<std::uint64_t>(res_2) << 31) + res_1;
    }
    else {
        std::uint32_t res_1 = generate();
        std::uint32_t res_2 = generate();
        res_64 = (static_cast<std::uint64_t>(res_2) << 32) + res_1;
    }

    leftover = res_64 * range;
}

template <typename Type, typename Method>
class distribution_base<oneapi::mkl::rng::device::uniform<Type, Method>> {
public:
    struct param_type {
        param_type(Type a, Type b) : a_(a), b_(b) {}
        Type a_;
        Type b_;
    };

    distribution_base(Type a, Type b) : a_(a), b_(b) {
#ifndef __SYCL_DEVICE_ONLY__
        if (a >= b) {
            throw oneapi::mkl::invalid_argument("rng", "uniform", "a >= b");
        }
#endif
    }

    Type a() const {
        return a_;
    }

    Type b() const {
        return b_;
    }

    param_type param() const {
        return param_type(a_, b_);
    }

    void param(const param_type& pt) {
#ifndef __SYCL_DEVICE_ONLY__
        if (pt.a_ >= pt.b_) {
            throw oneapi::mkl::invalid_argument("rng", "uniform", "a >= b");
        }
#endif
        a_ = pt.a_;
        b_ = pt.b_;
    }

protected:
    template <typename FpType, typename OutType, typename EngineType>
    OutType generate_single_int(EngineType& engine) {
        sycl::vec<FpType, EngineType::vec_size> res_fp;
        res_fp = engine.generate(static_cast<FpType>(a_), static_cast<FpType>(b_));
        res_fp = sycl::floor(res_fp);
        OutType res = res_fp.template convert<Type>();
        return res;
    }

    template <typename EngineType>
    auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, Type,
                                  sycl::vec<Type, EngineType::vec_size>>::type {
        using OutType = typename std::conditional<EngineType::vec_size == 1, Type,
                                                  sycl::vec<Type, EngineType::vec_size>>::type;
        using FpType =
            typename std::conditional<std::is_same<Method, uniform_method::accurate>::value, double,
                                      float>::type;
        OutType res;
        if constexpr (std::is_integral<Type>::value) {
            if constexpr (std::is_same_v<Type, std::int32_t> ||
                          std::is_same_v<Type, std::uint32_t>) {
                return generate_single_int<FpType, OutType>(engine);
            }
            else {
                // Lemire's sample rejection method to exclude bias for uniform numbers
                // https://arxiv.org/abs/1805.10941

                constexpr std::uint64_t uint_max64 = std::numeric_limits<std::uint64_t>::max();
                constexpr std::uint64_t uint_max32 = std::numeric_limits<std::uint32_t>::max();

                std::uint64_t range = b_ - a_;
                std::uint64_t threshold = (uint_max64 - range) % range;

                if (range <= uint_max32)
                    return generate_single_int<FpType, OutType>(engine);

                if constexpr (EngineType::vec_size == 1) {
                    std::uint32_t res_1, res_2;
                    std::uint64_t res_64, leftover;

                    generate_leftover<EngineType>(
                        range, [&engine]() { return engine.generate(); }, res_64, leftover);

                    if (range == uint_max64)
                        return res_64;

                    while (leftover < threshold) {
                        generate_leftover<EngineType>(
                            range, [&engine]() { return engine.generate(); }, res_64, leftover);
                    }

                    res = a_ + umul_hi_64(res_64, range);

                    return res;
                }
                else {
                    std::uint64_t leftover;

                    sycl::vec<std::uint32_t, EngineType::vec_size> res_1 = engine.generate();
                    sycl::vec<std::uint32_t, EngineType::vec_size> res_2 = engine.generate();
                    sycl::vec<std::uint64_t, EngineType::vec_size> res_64;

                    if constexpr (std::is_same_v<EngineType, mcg31m1<EngineType::vec_size>>) {
                        sycl::vec<std::uint32_t, EngineType::vec_size> res_3 = engine.generate();

                        for (int i = 0; i < EngineType::vec_size; i++) {
                            res_64[i] = (static_cast<std::uint64_t>(res_3[i]) << 62) +
                                        (static_cast<std::uint64_t>(res_2[i]) << 31) + res_1[i];
                        }
                    }
                    else {
                        if constexpr (EngineType::vec_size == 3) {
                            res_64[0] = (static_cast<std::uint64_t>(res_1[1]) << 32) +
                                        static_cast<std::uint64_t>(res_1[0]);
                            res_64[1] = (static_cast<std::uint64_t>(res_2[0]) << 32) +
                                        static_cast<std::uint64_t>(res_1[2]);
                            res_64[2] = (static_cast<std::uint64_t>(res_2[2]) << 32) +
                                        static_cast<std::uint64_t>(res_2[1]);
                        }
                        else {
                            for (int i = 0; i < EngineType::vec_size / 2; i++) {
                                res_64[i] = (static_cast<std::uint64_t>(res_1[2 * i + 1]) << 32) +
                                            static_cast<std::uint64_t>(res_1[2 * i]);
                                res_64[i + EngineType::vec_size / 2] =
                                    (static_cast<std::uint64_t>(res_2[2 * i + 1]) << 32) +
                                    static_cast<std::uint64_t>(res_2[2 * i]);
                            }
                        }
                    }

                    if (range == uint_max64)
                        return res_64.template convert<Type>();

                    for (int i = 0; i < EngineType::vec_size; i++) {
                        leftover = res_64[i] * range;

                        while (leftover < threshold) {
                            generate_leftover<EngineType>(
                                range, [&engine]() { return engine.generate_single(); }, res_64[i],
                                leftover);
                        }

                        res[i] = a_ + umul_hi_64(res_64[i], range);
                    }

                    return res;
                }
            }
        }
        else {
            res = engine.generate(a_, b_);
            if constexpr (std::is_same<Method, uniform_method::accurate>::value) {
#ifndef __HIPSYCL__
                res = sycl::fmax(res, a_);
                res = sycl::fmin(res, b_);
#else
                // a workaround for hipSYCL (AdaptiveCpp)
                if constexpr (EngineType::vec_size == 1) {
                    res = sycl::fmax(res, a_);
                    res = sycl::fmin(res, b_);
                }
                else {
                    for (int i = 0; i < EngineType::vec_size; i++) {
                        res[i] = sycl::fmax(res[i], a_);
                        res[i] = sycl::fmin(res[i], b_);
                    }
                }
#endif
            }
        }

        return res;
    }

    template <typename EngineType>
    Type generate_single(EngineType& engine) {
        using FpType =
            typename std::conditional<std::is_same<Method, uniform_method::accurate>::value, double,
                                      float>::type;
        Type res;
        if constexpr (std::is_integral<Type>::value) {
            if constexpr (std::is_same_v<Type, std::int32_t> ||
                          std::is_same_v<Type, std::uint32_t>) {
                FpType res_fp =
                    engine.generate_single(static_cast<FpType>(a_), static_cast<FpType>(b_));
                res_fp = sycl::floor(res_fp);
                res = static_cast<Type>(res_fp);
                return res;
            }
            else {
                // Lemire's sample rejection method to exclude bias for uniform numbers
                // https://arxiv.org/abs/1805.10941

                constexpr std::uint64_t uint_max64 = std::numeric_limits<std::uint64_t>::max();
                constexpr std::uint64_t uint_max32 = std::numeric_limits<std::uint32_t>::max();

                std::uint64_t range = b_ - a_;
                std::uint64_t threshold = (uint_max64 - range) % range;

                if (range <= uint_max32) {
                    FpType res_fp =
                        engine.generate_single(static_cast<FpType>(a_), static_cast<FpType>(b_));
                    res_fp = sycl::floor(res_fp);
                    res = static_cast<Type>(res_fp);
                    return res;
                }

                std::uint32_t res_1, res_2;
                std::uint64_t res_64, leftover;

                generate_leftover<EngineType>(
                    range, [&engine]() { return engine.generate_single(); }, res_64, leftover);

                if (range == uint_max64)
                    return res_64;

                while (leftover < threshold) {
                    generate_leftover<EngineType>(
                        range, [&engine]() { return engine.generate_single(); }, res_64, leftover);
                }

                res = a_ + umul_hi_64(res_64, range);

                return res;
            }
        }
        else {
            res = engine.generate_single(a_, b_);
            if constexpr (std::is_same<Method, uniform_method::accurate>::value) {
#ifndef __HIPSYCL__
                res = sycl::fmax(res, a_);
                res = sycl::fmin(res, b_);
#else
                // a workaround for hipSYCL (AdaptiveCpp)
                if constexpr (EngineType::vec_size == 1) {
                    res = sycl::fmax(res, a_);
                    res = sycl::fmin(res, b_);
                }
                else {
                    for (int i = 0; i < EngineType::vec_size; i++) {
                        res[i] = sycl::fmax(res[i], a_);
                        res[i] = sycl::fmin(res[i], b_);
                    }
                }
#endif
            }
        }

        return res;
    }

    Type a_;
    Type b_;
};

} // namespace oneapi::mkl::rng::device::detail

#endif // _MKL_RNG_DEVICE_UNIFORM_IMPL_HPP_
