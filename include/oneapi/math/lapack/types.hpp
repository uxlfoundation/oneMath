/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#pragma once

#include <complex>
#include <cstdint>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

namespace oneapi {
namespace math {
namespace lapack {
namespace internal {

// auxilary type aliases and forward declarations
template <bool, typename T = void>
struct enable_if;
template <typename T>
struct is_fp;
template <typename T>
struct is_rfp;
template <typename T>
struct is_cfp;

// auxilary typechecking templates
template <typename T>
struct enable_if<true, T> {
    using type = T;
};

template <>
struct is_fp<float> {
    static constexpr bool value{ true };
};
template <>
struct is_fp<double> {
    static constexpr bool value{ true };
};
template <>
struct is_fp<std::complex<float>> {
    static constexpr bool value{ true };
};
template <>
struct is_fp<std::complex<double>> {
    static constexpr bool value{ true };
};

template <>
struct is_rfp<float> {
    static constexpr bool value{ true };
};
template <>
struct is_rfp<double> {
    static constexpr bool value{ true };
};

template <>
struct is_cfp<std::complex<float>> {
    static constexpr bool value{ true };
};
template <>
struct is_cfp<std::complex<double>> {
    static constexpr bool value{ true };
};

template <typename fp>
using is_floating_point = typename enable_if<is_fp<fp>::value>::type*;
template <typename fp>
using is_real_floating_point = typename enable_if<is_rfp<fp>::value>::type*;
template <typename fp>
using is_complex_floating_point = typename enable_if<is_cfp<fp>::value>::type*;

} // namespace internal
} // namespace lapack
} // namespace math
} // namespace oneapi
