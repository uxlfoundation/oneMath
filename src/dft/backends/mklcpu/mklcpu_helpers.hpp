/*******************************************************************************
* Copyright Codeplay Software Ltd.
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

#ifndef _ONEMATH_DFT_SRC_MKLCPU_HELPERS_HPP_
#define _ONEMATH_DFT_SRC_MKLCPU_HELPERS_HPP_

#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/dft/detail/types_impl.hpp"

// Intel(R) oneMKL header
#include <mkl_dfti.h>

namespace oneapi::math::dft::mklcpu::detail {

template <typename K, typename H, typename F>
static inline auto host_task_internal(H& cgh, F f, int) -> decltype(cgh.host_task(f)) {
    return cgh.host_task(f);
}

template <typename K, typename H, typename F>
static inline void host_task(H& cgh, F f) {
    (void)host_task_internal<K>(cgh, f, 0);
}

template <typename Desc>
class kernel_name {};

/// Convert domain to equivalent backend native value.
inline constexpr DFTI_CONFIG_VALUE to_mklcpu(dft::detail::domain dom) {
    if (dom == dft::detail::domain::REAL) {
        return DFTI_REAL;
    }
    else {
        return DFTI_COMPLEX;
    }
}

/// Convert precision to equivalent backend native value.
inline constexpr DFTI_CONFIG_VALUE to_mklcpu(dft::detail::precision dom) {
    if (dom == dft::detail::precision::SINGLE) {
        return DFTI_SINGLE;
    }
    else {
        return DFTI_DOUBLE;
    }
}

/// Convert a config_param to equivalent backend native value.
inline constexpr DFTI_CONFIG_PARAM to_mklcpu(dft::detail::config_param param) {
    using iparam = dft::detail::config_param;
    switch (param) {
        case iparam::FORWARD_DOMAIN: return DFTI_FORWARD_DOMAIN;
        case iparam::DIMENSION: return DFTI_DIMENSION;
        case iparam::LENGTHS: return DFTI_LENGTHS;
        case iparam::PRECISION: return DFTI_PRECISION;
        case iparam::FORWARD_SCALE: return DFTI_FORWARD_SCALE;
        case iparam::NUMBER_OF_TRANSFORMS: return DFTI_NUMBER_OF_TRANSFORMS;
        case iparam::COMPLEX_STORAGE: return DFTI_COMPLEX_STORAGE;
        case iparam::REAL_STORAGE: return DFTI_REAL_STORAGE;
        case iparam::CONJUGATE_EVEN_STORAGE: return DFTI_CONJUGATE_EVEN_STORAGE;
        case iparam::FWD_DISTANCE: return DFTI_FWD_DISTANCE;
        case iparam::BWD_DISTANCE: return DFTI_BWD_DISTANCE;
        case iparam::WORKSPACE: return DFTI_WORKSPACE;
        case iparam::ORDERING: return DFTI_ORDERING;
        case iparam::TRANSPOSE: return DFTI_TRANSPOSE;
        case iparam::PACKED_FORMAT: return DFTI_PACKED_FORMAT;
        case iparam::COMMIT_STATUS: return DFTI_COMMIT_STATUS;
        default:
            throw math::invalid_argument("dft", "MKLCPU descriptor set_value()",
                                         "Invalid config param.");
            return static_cast<DFTI_CONFIG_PARAM>(0);
    }
}

/** Convert a config_value to the backend's native value. Throw on invalid input.
 * @tparam Param The config param the value is for.
 * @param value The config value to convert.
**/
template <dft::detail::config_param Param>
inline constexpr int to_mklcpu(dft::detail::config_value value);

template <>
inline constexpr int to_mklcpu<dft::detail::config_param::COMPLEX_STORAGE>(
    dft::detail::config_value value) {
    if (value == dft::detail::config_value::COMPLEX_COMPLEX) {
        return DFTI_COMPLEX_COMPLEX;
    }
    else if (value == dft::detail::config_value::REAL_REAL) {
        return DFTI_REAL_REAL;
    }
    else {
        throw math::invalid_argument("dft", "MKLCPU descriptor set_value()",
                                     "Invalid config value for complex storage.");
        return 0;
    }
}

template <>
inline constexpr int to_mklcpu<dft::detail::config_param::REAL_STORAGE>(
    dft::detail::config_value value) {
    if (value == dft::detail::config_value::REAL_REAL) {
        return DFTI_REAL_REAL;
    }
    else {
        throw math::invalid_argument("dft", "MKLCPU descriptor set_value()",
                                     "Invalid config value for real storage.");
        return 0;
    }
}
template <>
inline constexpr int to_mklcpu<dft::detail::config_param::CONJUGATE_EVEN_STORAGE>(
    dft::detail::config_value value) {
    if (value == dft::detail::config_value::COMPLEX_COMPLEX) {
        return DFTI_COMPLEX_COMPLEX;
    }
    else {
        throw math::invalid_argument("dft", "MKLCPU descriptor set_value()",
                                     "Invalid config value for conjugate even storage.");
        return 0;
    }
}

template <>
inline constexpr int to_mklcpu<dft::detail::config_param::PLACEMENT>(
    dft::detail::config_value value) {
    if (value == dft::detail::config_value::INPLACE) {
        return DFTI_INPLACE;
    }
    else if (value == dft::detail::config_value::NOT_INPLACE) {
        return DFTI_NOT_INPLACE;
    }
    else {
        throw math::invalid_argument("dft", "MKLCPU descriptor set_value()",
                                     "Invalid config value for inplace.");
        return 0;
    }
}

template <>
inline constexpr int to_mklcpu<dft::detail::config_param::PACKED_FORMAT>(
    dft::detail::config_value value) {
    if (value == dft::detail::config_value::CCE_FORMAT) {
        return DFTI_CCE_FORMAT;
    }
    else {
        throw math::invalid_argument("dft", "MKLCPU descriptor set_value()",
                                     "Invalid config value for packed format.");
        return 0;
    }
}

using mklcpu_desc_t = DFTI_DESCRIPTOR_HANDLE;

template <typename AccType>
typename AccType::value_type* acc_to_ptr(AccType acc) {
    // no need to decorate the pointer with the address space for mklcpu since its just getting passed to the a host function.
    return acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

} // namespace oneapi::math::dft::mklcpu::detail

#endif // _ONEMATH_DFT_SRC_MKLCPU_HELPERS_HPP_
