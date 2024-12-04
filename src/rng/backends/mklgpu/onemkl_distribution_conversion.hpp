/*******************************************************************************
* Copyright Codeplay Software Ltd
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

#ifndef _ONEMATH_SRC_RNG_ONEMKL_DISTRIBUTION_CONVERSION_HPP_
#define _ONEMATH_SRC_RNG_ONEMKL_DISTRIBUTION_CONVERSION_HPP_

// Convert oneMath RNG distribution types to Intel(R) oneMKL equivalents

#include <mkl/rng.hpp>

#include "common_onemkl_conversion.hpp"
#include "oneapi/math/rng/distributions.hpp"

namespace oneapi {
namespace math {
namespace rng {
namespace detail {

template <class Method>
struct convert_method_t;

template <>
struct convert_method_t<uniform_method::standard> {
    using type = oneapi::mkl::rng::uniform_method::standard;
};

template <>
struct convert_method_t<uniform_method::accurate> {
    using type = oneapi::mkl::rng::uniform_method::accurate;
};

template <>
struct convert_method_t<gaussian_method::icdf> {
    using type = oneapi::mkl::rng::gaussian_method::icdf;
};

template <>
struct convert_method_t<gaussian_method::box_muller2> {
    using type = oneapi::mkl::rng::gaussian_method::box_muller2;
};

template <>
struct convert_method_t<lognormal_method::icdf> {
    using type = oneapi::mkl::rng::lognormal_method::icdf;
};

template <>
struct convert_method_t<lognormal_method::box_muller2> {
    using type = oneapi::mkl::rng::lognormal_method::box_muller2;
};

template <>
struct convert_method_t<bernoulli_method::icdf> {
    using type = oneapi::mkl::rng::bernoulli_method::icdf;
};

template <>
struct convert_method_t<poisson_method::gaussian_icdf_based> {
    using type = oneapi::mkl::rng::poisson_method::gaussian_icdf_based;
};

template <class DistributionT>
struct convert_distrib_t;

template <class T, class Method>
struct convert_distrib_t<uniform<T, Method>> {
    auto operator()(uniform<T, Method> distribution) {
        using onemkl_method_t = typename convert_method_t<Method>::type;
        return oneapi::mkl::rng::uniform<T, onemkl_method_t>(distribution.a(), distribution.b());
    }
};

template <class T, class Method>
struct convert_distrib_t<gaussian<T, Method>> {
    auto operator()(gaussian<T, Method> distribution) {
        using onemkl_method_t = typename convert_method_t<Method>::type;
        return oneapi::mkl::rng::gaussian<T, onemkl_method_t>(distribution.mean(),
                                                              distribution.stddev());
    }
};

template <class T, class Method>
struct convert_distrib_t<lognormal<T, Method>> {
    auto operator()(lognormal<T, Method> distribution) {
        using onemkl_method_t = typename convert_method_t<Method>::type;
        return oneapi::mkl::rng::lognormal<T, onemkl_method_t>(
            distribution.m(), distribution.s(), distribution.displ(), distribution.scale());
    }
};

template <class T, class Method>
struct convert_distrib_t<bernoulli<T, Method>> {
    auto operator()(bernoulli<T, Method> distribution) {
        using onemkl_method_t = typename convert_method_t<Method>::type;
        return oneapi::mkl::rng::bernoulli<T, onemkl_method_t>(distribution.p());
    }
};

template <class T, class Method>
struct convert_distrib_t<poisson<T, Method>> {
    auto operator()(poisson<T, Method> distribution) {
        using onemkl_method_t = typename convert_method_t<Method>::type;
        return oneapi::mkl::rng::poisson<T, onemkl_method_t>(distribution.lambda());
    }
};

template <class DistributionT>
inline auto get_onemkl_distribution(DistributionT distribution) {
    return convert_distrib_t<DistributionT>()(distribution);
}

} // namespace detail
} // namespace rng
} // namespace math
} // namespace oneapi

#endif // _ONEMATH_SRC_RNG_ONEMKL_DISTRIBUTION_CONVERSION_HPP_
