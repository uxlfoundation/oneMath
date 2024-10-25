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

#ifndef _MKL_RNG_DEVICE_DISTRIBUTIONS_HPP_
#define _MKL_RNG_DEVICE_DISTRIBUTIONS_HPP_

#include <limits>

#include "oneapi/mkl/rng/device/detail/distribution_base.hpp"
#include "oneapi/mkl/rng/device/functions.hpp"

namespace oneapi::mkl::rng::device {

// CONTINUOUS AND DISCRETE RANDOM NUMBER DISTRIBUTIONS

// Class template oneapi::mkl::rng::device::uniform
//
// Represents continuous and discrete uniform random number distribution
//
// Supported types:
//      float
//      double
//      std::int32_t
//      std::uint32_t
//
// Supported methods:
//      oneapi::mkl::rng::device::uniform_method::standard
//      oneapi::mkl::rng::device::uniform_method::accurate
//
// Input arguments:
//      a - left bound. 0.0 by default
//      b - right bound. 1.0 by default (for std::(u)int32_t std::numeric_limits<std::int32_t>::max()
//          is used for accurate method and 2^23 is used for standard method)
//
// Note: using (un)signed integer uniform distribution with uniform_method::standard method may
// cause incorrect statistics of the produced random numbers (due to rounding error) if
// (abs(b - a) > 2^23) || (abs(b) > 2^23) || (abs(a) > 2^23)
// Please use uniform_method::accurate method instead
//
template <typename Type, typename Method>
class uniform : detail::distribution_base<uniform<Type, Method>> {
public:
    static_assert(std::is_same<Method, uniform_method::standard>::value ||
                      std::is_same<Method, uniform_method::accurate>::value,
                  "oneMKL: rng/uniform: method is incorrect");

    static_assert(std::is_same<Type, float>::value || std::is_same<Type, double>::value ||
                      std::is_same<Type, std::int32_t>::value ||
                      std::is_same<Type, std::uint32_t>::value ||
                      std::is_same<Type, std::int64_t>::value ||
                      std::is_same<Type, std::uint64_t>::value,
                  "oneMKL: rng/uniform: type is not supported");

    using method_type = Method;
    using result_type = Type;
    using param_type = typename detail::distribution_base<uniform<Type, Method>>::param_type;

    uniform()
            : detail::distribution_base<uniform<Type, Method>>(
                  Type(0.0),
                  std::is_integral<Type>::value
                      ? ((std::is_same_v<Type, std::uint64_t> || std::is_same_v<Type, std::int64_t>)
                             ? (std::numeric_limits<Type>::max)()
                             : (std::is_same<Method, uniform_method::standard>::value
                                    ? (1 << 23)
                                    : (std::numeric_limits<Type>::max)()))
                      : Type(1.0)) {}

    explicit uniform(Type a, Type b) : detail::distribution_base<uniform<Type, Method>>(a, b) {}
    explicit uniform(const param_type& pt)
            : detail::distribution_base<uniform<Type, Method>>(pt.a_, pt.b_) {}

    Type a() const {
        return detail::distribution_base<uniform<Type, Method>>::a();
    }

    Type b() const {
        return detail::distribution_base<uniform<Type, Method>>::b();
    }

    param_type param() const {
        return detail::distribution_base<uniform<Type, Method>>::param();
    }

    void param(const param_type& pt) {
        detail::distribution_base<uniform<Type, Method>>::param(pt);
    }

private:
    template <typename Distr, typename Engine>
    friend auto generate(Distr& distr, Engine& engine) ->
        typename std::conditional<Engine::vec_size == 1, typename Distr::result_type,
                                  sycl::vec<typename Distr::result_type, Engine::vec_size>>::type;

    template <typename Distr, typename Engine>
    friend typename Distr::result_type generate_single(Distr& distr, Engine& engine);
};

// Class template oneapi::mkl::rng::device::gaussian
//
// Represents continuous normal random number distribution
//
// Supported types:
//      float
//      double
//
// Supported methods:
//      oneapi::mkl::rng::device::gaussian_method::box_muller2
//      oneapi::mkl::rng::device::gaussian_method::icdf
//
// Input arguments:
//      mean   - mean. 0 by default
//      stddev - standard deviation. 1.0 by default
//
template <typename RealType, typename Method>
class gaussian : detail::distribution_base<gaussian<RealType, Method>> {
public:
    static_assert(std::is_same<Method, gaussian_method::box_muller2>::value
#if MKL_RNG_USE_BINARY_CODE
                      || std::is_same<Method, gaussian_method::icdf>::value
#endif
                  ,
                  "oneMKL: rng/gaussian: method is incorrect");
#if !MKL_RNG_USE_BINARY_CODE
    static_assert(!std::is_same<Method, gaussian_method::icdf>::value, "icdf method not supported");
#endif
    static_assert(std::is_same<RealType, float>::value || std::is_same<RealType, double>::value,
                  "oneMKL: rng/gaussian: type is not supported");

    using method_type = Method;
    using result_type = RealType;
    using param_type = typename detail::distribution_base<gaussian<RealType, Method>>::param_type;

    gaussian()
            : detail::distribution_base<gaussian<RealType, Method>>(RealType(0.0), RealType(1.0)) {}

    explicit gaussian(RealType mean, RealType stddev)
            : detail::distribution_base<gaussian<RealType, Method>>(mean, stddev) {}
    explicit gaussian(const param_type& pt)
            : detail::distribution_base<gaussian<RealType, Method>>(pt.mean_, pt.stddev_) {}

    RealType mean() const {
        return detail::distribution_base<gaussian<RealType, Method>>::mean();
    }

    RealType stddev() const {
        return detail::distribution_base<gaussian<RealType, Method>>::stddev();
    }

    param_type param() const {
        return detail::distribution_base<gaussian<RealType, Method>>::param();
    }

    void param(const param_type& pt) {
        detail::distribution_base<gaussian<RealType, Method>>::param(pt);
    }

    template <typename Distr, typename Engine>
    friend auto generate(Distr& distr, Engine& engine) ->
        typename std::conditional<Engine::vec_size == 1, typename Distr::result_type,
                                  sycl::vec<typename Distr::result_type, Engine::vec_size>>::type;
    template <typename Distr, typename Engine>
    friend typename Distr::result_type generate_single(Distr& distr, Engine& engine);
};

// Class template oneapi::mkl::rng::device::lognormal
//
// Represents continuous lognormal random number distribution
//
// Supported types:
//      float
//      double
//
// Supported methods:
//      oneapi::mkl::rng::device::lognormal_method::box_muller2
//
// Input arguments:
//      m     - mean of the subject normal distribution. 0.0 by default
//      s     - standard deviation of the subject normal distribution. 1.0 by default
//      displ - displacement. 0.0 by default
//      scale - scalefactor. 1.0 by default
//
template <typename RealType, typename Method>
class lognormal : detail::distribution_base<lognormal<RealType, Method>> {
public:
    static_assert(std::is_same<Method, lognormal_method::box_muller2>::value,
                  "oneMKL: rng/lognormal: method is incorrect");

    static_assert(std::is_same<RealType, float>::value || std::is_same<RealType, double>::value,
                  "oneMKL: rng/lognormal: type is not supported");

    using method_type = Method;
    using result_type = RealType;
    using param_type = typename detail::distribution_base<lognormal<RealType, Method>>::param_type;

    lognormal()
            : detail::distribution_base<lognormal<RealType, Method>>(
                  RealType(0.0), RealType(1.0), RealType(0.0), RealType(1.0)) {}

    explicit lognormal(RealType m, RealType s, RealType displ = RealType(0.0),
                       RealType scale = RealType(1.0))
            : detail::distribution_base<lognormal<RealType, Method>>(m, s, displ, scale) {}
    explicit lognormal(const param_type& pt)
            : detail::distribution_base<lognormal<RealType, Method>>(pt.m_, pt.s_, pt.displ_,
                                                                     pt.scale_) {}

    RealType m() const {
        return detail::distribution_base<lognormal<RealType, Method>>::m();
    }

    RealType s() const {
        return detail::distribution_base<lognormal<RealType, Method>>::s();
    }

    RealType displ() const {
        return detail::distribution_base<lognormal<RealType, Method>>::displ();
    }

    RealType scale() const {
        return detail::distribution_base<lognormal<RealType, Method>>::scale();
    }

    param_type param() const {
        return detail::distribution_base<lognormal<RealType, Method>>::param();
    }

    void param(const param_type& pt) {
        detail::distribution_base<lognormal<RealType, Method>>::param(pt);
    }

    template <typename Distr, typename Engine>
    friend auto generate(Distr& distr, Engine& engine) ->
        typename std::conditional<Engine::vec_size == 1, typename Distr::result_type,
                                  sycl::vec<typename Distr::result_type, Engine::vec_size>>::type;
    template <typename Distr, typename Engine>
    friend typename Distr::result_type generate_single(Distr& distr, Engine& engine);
};

// Class template oneapi::mkl::rng::device::beta
//
// Represents continuous beta random number distribution
//
// Supported types:
//      float
//      double
//
// Supported methods:
//      oneapi::mkl::rng::device::beta_method::cja
//      oneapi::mkl::rng::device::beta_method::cja_accurate
//
// Input arguments:
//      p - shape. 1.0 by default
//      q - shape. 0.0 by default
//      a - displacement. 1.0 by default
//      b - scalefactor. 1.0 by default
//
template <typename RealType, typename Method>
class beta : detail::distribution_base<beta<RealType, Method>> {
public:
    static_assert(std::is_same<Method, beta_method::cja>::value ||
                      std::is_same<Method, beta_method::cja_accurate>::value,
                  "oneMKL: rng/beta: method is incorrect");

    static_assert(std::is_same<RealType, float>::value || std::is_same<RealType, double>::value,
                  "oneMKL: rng/beta: type is not supported");

    using method_type = Method;
    using result_type = RealType;
    using param_type = typename detail::distribution_base<beta<RealType, Method>>::param_type;

    beta()
            : detail::distribution_base<beta<RealType, Method>>(RealType(1.0), RealType(1.0),
                                                                RealType(0.0), RealType(1.0)) {}

    explicit beta(RealType p, RealType q, RealType a, RealType b)
            : detail::distribution_base<beta<RealType, Method>>(p, q, a, b) {}

    explicit beta(const param_type& pt)
            : detail::distribution_base<beta<RealType, Method>>(pt.p_, pt.q_, pt.a_, pt.b_) {}

    RealType p() const {
        return detail::distribution_base<beta<RealType, Method>>::p();
    }

    RealType q() const {
        return detail::distribution_base<beta<RealType, Method>>::q();
    }

    RealType a() const {
        return detail::distribution_base<beta<RealType, Method>>::a();
    }

    RealType b() const {
        return detail::distribution_base<beta<RealType, Method>>::b();
    }

    param_type param() const {
        return detail::distribution_base<beta<RealType, Method>>::param();
    }

    std::size_t count_rejected_numbers() const {
        return detail::distribution_base<beta<RealType, Method>>::count_rejected_numbers();
    }

    void param(const param_type& pt) {
        detail::distribution_base<beta<RealType, Method>>::param(pt);
    }

    template <typename Distr, typename Engine>
    friend auto generate(Distr& distr, Engine& engine) ->
        typename std::conditional<Engine::vec_size == 1, typename Distr::result_type,
                                  sycl::vec<typename Distr::result_type, Engine::vec_size>>::type;
    template <typename Distr, typename Engine>
    friend typename Distr::result_type generate_single(Distr& distr, Engine& engine);
};

// Class template oneapi::mkl::rng::device::gamma
//
// Represents continuous gamma random number distribution
//
// Supported types:
//      float
//      double
//
// Supported methods:
//      oneapi::mkl::rng::device::gamma_method::marsaglia
//      oneapi::mkl::rng::device::gamma_method::marsaglia_accurate
//
// Input arguments:
//      alpha - shape. 1.0 by default
//      a - displacement. 0.0 by default
//      beta - scalefactor. 1.0 by default
//
template <typename RealType, typename Method>
class gamma : detail::distribution_base<gamma<RealType, Method>> {
public:
    static_assert(std::is_same<Method, gamma_method::marsaglia>::value ||
                      std::is_same<Method, gamma_method::marsaglia_accurate>::value,
                  "oneMKL: rng/gamma: method is incorrect");

    static_assert(std::is_same<RealType, float>::value || std::is_same<RealType, double>::value,
                  "oneMKL: rng/gamma: type is not supported");

    using method_type = Method;
    using result_type = RealType;
    using param_type = typename detail::distribution_base<gamma<RealType, Method>>::param_type;

    gamma()
            : detail::distribution_base<gamma<RealType, Method>>(RealType(1.0), RealType(0.0),
                                                                 RealType(1.0)) {}

    explicit gamma(RealType alpha, RealType a, RealType beta)
            : detail::distribution_base<gamma<RealType, Method>>(alpha, a, beta) {}

    explicit gamma(const param_type& pt)
            : detail::distribution_base<gamma<RealType, Method>>(pt.alpha_, pt.a_, pt.beta_) {}

    RealType alpha() const {
        return detail::distribution_base<gamma<RealType, Method>>::alpha();
    }

    RealType a() const {
        return detail::distribution_base<gamma<RealType, Method>>::a();
    }

    RealType beta() const {
        return detail::distribution_base<gamma<RealType, Method>>::beta();
    }

    std::size_t count_rejected_numbers() const {
        return detail::distribution_base<gamma<RealType, Method>>::count_rejected_numbers();
    }

    param_type param() const {
        return detail::distribution_base<gamma<RealType, Method>>::param();
    }

    void param(const param_type& pt) {
        detail::distribution_base<gamma<RealType, Method>>::param(pt);
    }

    template <typename Distr, typename Engine>
    friend auto generate(Distr& distr, Engine& engine) ->
        typename std::conditional<Engine::vec_size == 1, typename Distr::result_type,
                                  sycl::vec<typename Distr::result_type, Engine::vec_size>>::type;
    template <typename Distr, typename Engine>
    friend typename Distr::result_type generate_single(Distr& distr, Engine& engine);
};

// Class template oneapi::mkl::rng::device::uniform_bits
//
// Represents discrete uniform bits random number distribution
//
// Supported types:
//      std::uint32_t
//      std::uint64_t
//
template <typename UIntType>
class uniform_bits : detail::distribution_base<uniform_bits<UIntType>> {
public:
    static_assert(std::is_same<UIntType, std::uint32_t>::value ||
                      std::is_same<UIntType, std::uint64_t>::value,
                  "oneMKL: rng/uniform_bits: type is not supported");
    using result_type = UIntType;

private:
    template <typename Distr, typename Engine>
    friend auto generate(Distr& distr, Engine& engine) ->
        typename std::conditional<Engine::vec_size == 1, typename Distr::result_type,
                                  sycl::vec<typename Distr::result_type, Engine::vec_size>>::type;

    template <typename Distr, typename Engine>
    friend typename Distr::result_type generate_single(Distr& distr, Engine& engine);
};

// Class template oneapi::mkl::rng::device::bits
//
// Represents bits of underlying random number engine
//
// Supported types:
//      std::uint32_t for philox4x32x10, mrg32k3a and mcg31m1
//      std::uint64_t for mcg59 only
//
template <typename UIntType>
class bits : detail::distribution_base<bits<UIntType>> {
public:
    static_assert(std::is_same<UIntType, std::uint32_t>::value ||
                      std::is_same<UIntType, std::uint64_t>::value,
                  "oneMKL: rng/bits: type is not supported");
    using result_type = UIntType;

private:
    template <typename Distr, typename Engine>
    friend auto generate(Distr& distr, Engine& engine) ->
        typename std::conditional<Engine::vec_size == 1, typename Distr::result_type,
                                  sycl::vec<typename Distr::result_type, Engine::vec_size>>::type;

    template <typename Distr, typename Engine>
    friend typename Distr::result_type generate_single(Distr& distr, Engine& engine);
};

// Class template oneapi::mkl::rng::device::exponential
//
// Represents continuous exponential random number distribution
//
// Supported types:
//      float
//      double
//
// Supported methods:
//      oneapi::mkl::rng::device::exponential_method::icdf
//      oneapi::mkl::rng::device::exponential_method::icdf_accurate
//
// Input arguments:
//      displ - displacement. 0.0 by default
//      scale - scalefactor. 1.0 by default
//
template <typename RealType, typename Method>
class exponential : detail::distribution_base<exponential<RealType, Method>> {
public:
    static_assert(std::is_same<Method, exponential_method::icdf>::value ||
                      std::is_same<Method, exponential_method::icdf_accurate>::value,
                  "oneMKL: rng/exponential: method is incorrect");

    static_assert(std::is_same<RealType, float>::value || std::is_same<RealType, double>::value,
                  "oneMKL: rng/exponential: type is not supported");

    using method_type = Method;
    using result_type = RealType;
    using param_type =
        typename detail::distribution_base<exponential<RealType, Method>>::param_type;

    exponential()
            : detail::distribution_base<exponential<RealType, Method>>(RealType(0.0),
                                                                       RealType(1.0)) {}

    explicit exponential(RealType a, RealType beta)
            : detail::distribution_base<exponential<RealType, Method>>(a, beta) {}

    explicit exponential(const param_type& pt)
            : detail::distribution_base<exponential<RealType, Method>>(pt.a_, pt.beta_) {}

    RealType a() const {
        return detail::distribution_base<exponential<RealType, Method>>::a();
    }

    RealType beta() const {
        return detail::distribution_base<exponential<RealType, Method>>::beta();
    }

    param_type param() const {
        return detail::distribution_base<exponential<RealType, Method>>::param();
    }

    void param(const param_type& pt) {
        detail::distribution_base<exponential<RealType, Method>>::param(pt);
    }

    template <typename Distr, typename Engine>
    friend auto generate(Distr& distr, Engine& engine) ->
        typename std::conditional<Engine::vec_size == 1, typename Distr::result_type,
                                  sycl::vec<typename Distr::result_type, Engine::vec_size>>::type;
    template <typename Distr, typename Engine>
    friend typename Distr::result_type generate_single(Distr& distr, Engine& engine);
};

// Class template oneapi::mkl::rng::device::poisson
//
// Represents discrete poisson random number distribution
//
// Supported types:
//      std::int32_t
//      std::uint32_t
//
// Supported methods:
//      oneapi::mkl::rng::device::poisson_method::devroye
//
// Input arguments:
//      lambda - mean value. 1.0 by default
//
template <typename IntType, typename Method>
class poisson : detail::distribution_base<poisson<IntType, Method>> {
public:
    static_assert(std::is_same<Method, poisson_method::devroye>::value,
                  "oneMKL: rng/poisson: method is incorrect");

    static_assert(std::is_same<IntType, std::int32_t>::value ||
                      std::is_same<IntType, std::uint32_t>::value,
                  "oneMKL: rng/poisson: type is not supported");

    using method_type = Method;
    using result_type = IntType;
    using param_type = typename detail::distribution_base<poisson<IntType, Method>>::param_type;

    poisson() : detail::distribution_base<poisson<IntType, Method>>(0.5) {}

    explicit poisson(double lambda) : detail::distribution_base<poisson<IntType, Method>>(lambda) {}
    explicit poisson(const param_type& pt)
            : detail::distribution_base<poisson<IntType, Method>>(pt.lambda_) {}

    double lambda() const {
        return detail::distribution_base<poisson<IntType, Method>>::lambda();
    }

    param_type param() const {
        return detail::distribution_base<poisson<IntType, Method>>::param();
    }

    void param(const param_type& pt) {
        detail::distribution_base<poisson<IntType, Method>>::param(pt);
    }

    template <typename Distr, typename Engine>
    friend auto generate(Distr& distr, Engine& engine) ->
        typename std::conditional<Engine::vec_size == 1, typename Distr::result_type,
                                  sycl::vec<typename Distr::result_type, Engine::vec_size>>::type;
    template <typename Distr, typename Engine>
    friend typename Distr::result_type generate_single(Distr& distr, Engine& engine);
};

// Class template oneapi::mkl::rng::device::bernoulli
//
// Represents discrete Bernoulli random number distribution
//
// Supported types:
//      std::uint32_t
//      std::int32_t
//
// Supported methods:
//      oneapi::mkl::rng::bernoulli_method::icdf;
//
// Input arguments:
//      p - success probablity of a trial. 0.5 by default
//
template <typename IntType, typename Method>
class bernoulli : detail::distribution_base<bernoulli<IntType, Method>> {
public:
    static_assert(std::is_same<Method, bernoulli_method::icdf>::value,
                  "oneMKL: rng/bernoulli: method is incorrect");

    static_assert(std::is_same<IntType, std::int32_t>::value ||
                      std::is_same<IntType, std::uint32_t>::value ||
                      std::is_same<IntType, std::int16_t>::value ||
                      std::is_same<IntType, std::uint16_t>::value ||
                      std::is_same<IntType, std::int8_t>::value ||
                      std::is_same<IntType, std::uint8_t>::value,
                  "oneMKL: rng/bernoulli: type is not supported");

    using method_type = Method;
    using result_type = IntType;
    using param_type = typename detail::distribution_base<bernoulli<IntType, Method>>::param_type;

    bernoulli() : detail::distribution_base<bernoulli<IntType, Method>>(0.5f) {}

    explicit bernoulli(float p) : detail::distribution_base<bernoulli<IntType, Method>>(p) {}
    explicit bernoulli(const param_type& pt)
            : detail::distribution_base<bernoulli<IntType, Method>>(pt.p_) {}

    float p() const {
        return detail::distribution_base<bernoulli<IntType, Method>>::p();
    }

    param_type param() const {
        return detail::distribution_base<bernoulli<IntType, Method>>::param();
    }

    void param(const param_type& pt) {
        detail::distribution_base<bernoulli<IntType, Method>>::param(pt);
    }

    template <typename Distr, typename Engine>
    friend auto generate(Distr& distr, Engine& engine) ->
        typename std::conditional<Engine::vec_size == 1, typename Distr::result_type,
                                  sycl::vec<typename Distr::result_type, Engine::vec_size>>::type;
    template <typename Distr, typename Engine>
    friend typename Distr::result_type generate_single(Distr& distr, Engine& engine);
};

} // namespace oneapi::mkl::rng::device

#endif // _MKL_RNG_DEVICE_DISTRIBUTIONS_HPP_
