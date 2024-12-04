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

#ifndef ONEMATH_RNG_DEVICE_VM_WRAPPERS_HPP_
#define ONEMATH_RNG_DEVICE_VM_WRAPPERS_HPP_

#include <cmath>

namespace oneapi::math::rng::device::detail {

template <typename DataType>
static inline DataType sqrt_wrapper(DataType a) {
    return sycl::sqrt(a);
}

template <typename DataType>
static inline DataType sinpi_wrapper(DataType a) {
    return sycl::sinpi(a);
}

template <typename DataType>
static inline DataType cospi_wrapper(DataType a) {
    return sycl::cospi(a);
}

template <typename DataType>
static inline DataType sincospi_wrapper(DataType a, DataType& b) {
    b = sycl::cospi(a);
    return sycl::sinpi(a);
}

template <typename DataType>
static inline DataType ln_wrapper(DataType a) {
    if (a == DataType(0)) {
        if constexpr (std::is_same_v<DataType, double>)
            return -0x1.74385446D71C3P+9; // ln(0.494065e-323) = -744.440072
        else
            return -0x1.9D1DA0P+6f; // ln(0.14012984e-44) = -103.278929
    }
    return sycl::log(a);
}

template <typename DataType>
static inline DataType pow_wrapper(DataType a, DataType b) {
    return sycl::pow(a, b);
}

template <typename DataType>
static inline DataType powr_wrapper(DataType a, DataType b) {
    return sycl::powr(a, b);
}

template <typename DataType>
static inline DataType exp_wrapper(DataType a) {
    return sycl::exp(a);
}
} // namespace oneapi::math::rng::device::detail

#endif // ONEMATH_RNG_DEVICE_VM_WRAPPERS_HPP_
