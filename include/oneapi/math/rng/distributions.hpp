/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef _ONEMATH_RNG_DISTRIBUTIONS_HPP_
#define _ONEMATH_RNG_DISTRIBUTIONS_HPP_

#include <cstdint>
#include <limits>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/exceptions.hpp"

namespace oneapi {
namespace math {
namespace rng {

// Class template oneapi::math::rng::uniform
//
// Represents continuous and discrete uniform random number distribution
//
// Supported types:
//      float
//      double
//      std::int32_t
//
// Supported methods:
//      oneapi::math::rng::uniform_method::standard
//      oneapi::math::rng::uniform_method::accurate - for float and double types only
//
// Input arguments:
//      a - left bound. 0.0 by default
//      b - right bound. 1.0 by default (std::numeric_limits<std::int32_t>::max() for std::int32_t)

namespace uniform_method {
struct standard {};
struct accurate {};
using by_default = standard;
} // namespace uniform_method

template <typename Type = float, typename Method = uniform_method::by_default>
class uniform {
public:
    static_assert(std::is_same<Method, uniform_method::standard>::value ||
                      (std::is_same<Method, uniform_method::accurate>::value &&
                       !std::is_same<Type, std::int32_t>::value),
                  "rng uniform distribution method is incorrect");

    static_assert(std::is_same<Type, float>::value || std::is_same<Type, double>::value,
                  "rng uniform distribution type is not supported");

    using method_type = Method;
    using result_type = Type;

    uniform() : uniform(static_cast<Type>(0.0f), static_cast<Type>(1.0f)) {}

    explicit uniform(Type a, Type b) : a_(a), b_(b) {
        if (a >= b) {
            throw oneapi::math::invalid_argument("rng", "uniform",
                                                 "parameters are incorrect, a >= b");
        }
    }

    Type a() const {
        return a_;
    }

    Type b() const {
        return b_;
    }

private:
    Type a_;
    Type b_;
};

template <typename Method>
class uniform<std::int32_t, Method> {
public:
    using method_type = Method;
    using result_type = std::int32_t;

    uniform() : uniform(0, std::numeric_limits<std::int32_t>::max()) {}

    explicit uniform(std::int32_t a, std::int32_t b) : a_(a), b_(b) {
        if (a >= b) {
            throw oneapi::math::invalid_argument("rng", "uniform",
                                                 "parameters are incorrect, a >= b");
        }
    }

    std::int32_t a() const {
        return a_;
    }

    std::int32_t b() const {
        return b_;
    }

private:
    std::int32_t a_;
    std::int32_t b_;
};

// Class template oneapi::math::rng::gaussian
//
// Represents continuous normal random number distribution
//
// Supported types:
//      float
//      double
//
// Supported methods:
//      oneapi::math::rng::gaussian_method::box_muller2
//      oneapi::math::rng::gaussian_method::icdf
//
// Input arguments:
//      mean   - mean. 0 by default
//      stddev - standard deviation. 1.0 by default

namespace gaussian_method {
struct icdf {};
struct box_muller2 {};
using by_default = box_muller2;
} // namespace gaussian_method

template <typename RealType = float, typename Method = gaussian_method::by_default>
class gaussian {
public:
    static_assert(std::is_same<Method, gaussian_method::icdf>::value ||
                      std::is_same<Method, gaussian_method::box_muller2>::value,
                  "rng gaussian distribution method is incorrect");

    static_assert(std::is_same<RealType, float>::value || std::is_same<RealType, double>::value,
                  "rng gaussian distribution type is not supported");

    using method_type = Method;
    using result_type = RealType;

    gaussian() : gaussian(static_cast<RealType>(0.0), static_cast<RealType>(1.0)) {}

    explicit gaussian(RealType mean, RealType stddev) : mean_(mean), stddev_(stddev) {
        if (stddev <= static_cast<RealType>(0.0)) {
            throw oneapi::math::invalid_argument("rng", "gaussian",
                                                 "stddev parameter is incorrect, stddev <= 0.0");
        }
    }

    RealType mean() const {
        return mean_;
    }

    RealType stddev() const {
        return stddev_;
    }

private:
    RealType mean_;
    RealType stddev_;
};

// Class template oneapi::math::rng::lognormal
//
// Represents continuous lognormal random number distribution
//
// Supported types:
//      float
//      double
//
// Supported methods:
//      oneapi::math::rng::lognormal_method::box_muller2
//      oneapi::math::rng::lognormal_method::icdf
//
// Input arguments:
//      m     - mean of the subject normal distribution. 0.0 by default
//      s     - standard deviation of the subject normal distribution. 1.0 by default
//      displ - displacement. 0.0 by default
//      scale - scalefactor. 1.0 by default

namespace lognormal_method {
struct icdf {};
struct box_muller2 {};
using by_default = box_muller2;
} // namespace lognormal_method

template <typename RealType = float, typename Method = lognormal_method::by_default>
class lognormal {
public:
    static_assert(std::is_same<Method, lognormal_method::box_muller2>::value ||
                      std::is_same<Method, lognormal_method::icdf>::value,
                  "rng lognormal distribution method is incorrect");

    static_assert(std::is_same<RealType, float>::value || std::is_same<RealType, double>::value,
                  "rng lognormal distribution type is not supported");

    using method_type = Method;
    using result_type = RealType;

    lognormal()
            : lognormal(static_cast<RealType>(0.0), static_cast<RealType>(1.0),
                        static_cast<RealType>(0.0), static_cast<RealType>(1.0)) {}

    explicit lognormal(RealType m, RealType s, RealType displ = static_cast<RealType>(0.0),
                       RealType scale = static_cast<RealType>(1.0))
            : m_(m),
              s_(s),
              displ_(displ),
              scale_(scale) {
        if (s <= static_cast<RealType>(0.0)) {
            throw oneapi::math::invalid_argument("rng", "lognormal", "s <= 0");
        }
        if (scale <= static_cast<RealType>(0.0)) {
            throw oneapi::math::invalid_argument("rng", "lognormal", "scale <= 0");
        }
    }

    RealType m() const {
        return m_;
    }

    RealType s() const {
        return s_;
    }

    RealType displ() const {
        return displ_;
    }

    RealType scale() const {
        return scale_;
    }

private:
    RealType m_;
    RealType s_;
    RealType displ_;
    RealType scale_;
};

// Class template oneapi::math::rng::bernoulli
//
// Represents discrete Bernoulli random number distribution
//
// Supported types:
//      std::uint32_t
//      std::int32_t
//
// Supported methods:
//      oneapi::math::rng::bernoulli_method::icdf;
//
// Input arguments:
//      p - success probability of a trial. 0.5 by default

namespace bernoulli_method {
struct icdf {};
using by_default = icdf;
} // namespace bernoulli_method

template <typename IntType = std::uint32_t, typename Method = bernoulli_method::by_default>
class bernoulli {
public:
    static_assert(std::is_same<Method, bernoulli_method::icdf>::value,
                  "rng bernoulli method is incorrect");

    static_assert(std::is_same<IntType, std::int32_t>::value ||
                      std::is_same<IntType, std::uint32_t>::value,
                  "rng bernoulli type is not supported");

    using method_type = Method;
    using result_type = IntType;

    bernoulli() : bernoulli(0.5f) {}

    explicit bernoulli(float p) : p_(p) {
        if ((p > 1.0f) || (p < 0.0f)) {
            throw oneapi::math::invalid_argument("rng", "bernoulli", "p > 1 or p < 0");
        }
    }

    float p() const {
        return p_;
    }

private:
    float p_;
};

// Class template oneapi::math::rng::poisson
//
// Represents discrete Poisson random number distribution
//
// Supported types:
//      std::int32_t
//
// Supported methods:
//      oneapi::math::rng::poisson_method::gaussian_icdf_based
//
// Input arguments:
//      lambda - distribution parameter. 0.5 by default

namespace poisson_method {
struct gaussian_icdf_based {};
using by_default = gaussian_icdf_based;
} // namespace poisson_method

template <typename IntType = std::int32_t, typename Method = poisson_method::by_default>
class poisson {
public:
    static_assert(std::is_same<Method, poisson_method::gaussian_icdf_based>::value,
                  "rng poisson method is incorrect");

    static_assert(std::is_same<IntType, std::int32_t>::value ||
                      std::is_same<IntType, std::uint32_t>::value,
                  "rng poisson type is not supported");

    using method_type = Method;
    using result_type = IntType;

    poisson() : poisson(0.5) {}

    explicit poisson(double lambda) : lambda_(lambda) {
        if ((lambda <= 0.0)) {
            throw oneapi::math::invalid_argument("rng", "poisson", "lamdba < 0");
        }
    }

    double lambda() const {
        return lambda_;
    }

private:
    double lambda_;
};

// Class template oneapi::math::rng::bits
//
// Represents bits of underlying random number engine
//
// Supported types:
//      std::uint32_t
//

template <typename UIntType = std::uint32_t>
class bits {
public:
    static_assert(std::is_same<UIntType, std::uint32_t>::value, "rng bits type is not supported");
    using result_type = UIntType;
};

} // namespace rng
} // namespace math
} // namespace oneapi

#endif //_ONEMATH_RNG_DISTRIBUTIONS_HPP_
