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

#ifndef _ONEMATH_DFT_SRC_MKLGPU_HELPERS_HPP_
#define _ONEMATH_DFT_SRC_MKLGPU_HELPERS_HPP_

#include "oneapi/math/detail/exceptions.hpp"
#include "oneapi/math/dft/detail/types_impl.hpp"

// Intel(R) oneMKL headers
#include <mkl_version.h>
#if INTEL_MKL_VERSION < 20250000
#include <mkl/dfti.hpp>
namespace oneapi::math::dft::mklgpu::detail {
constexpr int committed = DFTI_COMMITTED;
constexpr int uncommitted = DFTI_UNCOMMITTED;
} // namespace oneapi::math::dft::mklgpu::detail
#else
#include <mkl/dft.hpp>
namespace oneapi::math::dft::mklgpu::detail {
constexpr auto committed = oneapi::mkl::dft::config_value::COMMITTED;
constexpr auto uncommitted = oneapi::mkl::dft::config_value::UNCOMMITTED;
} // namespace oneapi::math::dft::mklgpu::detail
#endif

namespace oneapi {
namespace math {
namespace dft {
namespace mklgpu {
namespace detail {

/// Convert domain to equivalent backend native value.
inline constexpr oneapi::mkl::dft::domain to_mklgpu(dft::detail::domain dom) {
    if (dom == dft::detail::domain::REAL) {
        return oneapi::mkl::dft::domain::REAL;
    }
    else {
        return oneapi::mkl::dft::domain::COMPLEX;
    }
}

/// Convert precision to equivalent backend native value.
inline constexpr oneapi::mkl::dft::precision to_mklgpu(dft::detail::precision dom) {
    if (dom == dft::detail::precision::SINGLE) {
        return oneapi::mkl::dft::precision::SINGLE;
    }
    else {
        return oneapi::mkl::dft::precision::DOUBLE;
    }
}

template <dft::detail::config_param Param>
struct to_mklgpu_impl;

/** Convert a config_value to the backend's native value. Throw on invalid input.
 * @tparam Param The config param the value is for.
 * @param value The config value to convert.
**/
template <dft::detail::config_param Param>
inline constexpr auto to_mklgpu(dft::detail::config_value value) {
    return to_mklgpu_impl<Param>{}(value);
}

#if INTEL_MKL_VERSION < 20250000
template <>
struct to_mklgpu_impl<dft::detail::config_param::PLACEMENT> {
    inline constexpr auto operator()(dft::detail::config_value value) -> int {
        switch (value) {
            case dft::detail::config_value::INPLACE: return DFTI_INPLACE;
            case dft::detail::config_value::NOT_INPLACE: return DFTI_NOT_INPLACE;
            default:
                throw math::invalid_argument("dft", "MKLGPU descriptor set_value()",
                                             "Invalid config value for inplace.");
        }
    }
};
#else
template <>
struct to_mklgpu_impl<dft::detail::config_param::PLACEMENT> {
    inline constexpr auto operator()(dft::detail::config_value value) {
        switch (value) {
            case dft::detail::config_value::INPLACE: return oneapi::mkl::dft::config_value::INPLACE;
            case dft::detail::config_value::NOT_INPLACE:
                return oneapi::mkl::dft::config_value::NOT_INPLACE;
            default:
                throw math::invalid_argument("dft", "MKLGPU descriptor set_value()",
                                             "Invalid config value for inplace.");
        }
    }
};
#endif

template <>
struct to_mklgpu_impl<dft::detail::config_param::WORKSPACE_PLACEMENT> {
    inline constexpr auto operator()(dft::detail::config_value value) {
        switch (value) {
            case dft::detail::config_value::WORKSPACE_AUTOMATIC:
                return oneapi::mkl::dft::config_value::WORKSPACE_INTERNAL;
            case dft::detail::config_value::WORKSPACE_EXTERNAL:
                return oneapi::mkl::dft::config_value::WORKSPACE_EXTERNAL;
            default:
                throw math::invalid_argument("dft", "MKLGPU descriptor set_value()",
                                             "Invalid config value for inplace.");
        }
    }
};
} // namespace detail
} // namespace mklgpu
} // namespace dft
} // namespace math
} // namespace oneapi

#endif // _ONEMATH_DFT_SRC_MKLGPU_HELPERS_HPP_
