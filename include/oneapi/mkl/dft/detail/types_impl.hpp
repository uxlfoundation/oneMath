/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef _ONEMKL_DETAIL_TYPES_IMPL_HPP_
#define _ONEMKL_DETAIL_TYPES_IMPL_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include <cstdint>
#include <vector>
#include <type_traits>
#include <complex>

namespace oneapi {
namespace mkl {
namespace dft {
namespace detail {

typedef long DFT_ERROR;

enum class precision { SINGLE, DOUBLE };

template <precision prec>
struct precision_t {
    using real_t = std::conditional_t<prec == precision::SINGLE, float, double>;
};

enum class domain { REAL, COMPLEX };

// Forward declarations
template <precision prec, domain dom>
class commit_impl;

template <precision prec, domain dom>
class descriptor;

template <class... T>
constexpr bool always_false = false;

template <typename descriptor_type>
struct descriptor_info {
    static_assert(always_false<descriptor_type>, "Not a valid descriptor type");
};

template <>
struct descriptor_info<descriptor<precision::SINGLE, domain::REAL>> {
    using scalar_type = float;
    using forward_type = float;
    using backward_type = std::complex<float>;
};
template <>
struct descriptor_info<descriptor<precision::SINGLE, domain::COMPLEX>> {
    using scalar_type = float;
    using forward_type = std::complex<float>;
    using backward_type = std::complex<float>;
};
template <>
struct descriptor_info<descriptor<precision::DOUBLE, domain::REAL>> {
    using scalar_type = double;
    using forward_type = double;
    using backward_type = std::complex<double>;
};
template <>
struct descriptor_info<descriptor<precision::DOUBLE, domain::COMPLEX>> {
    using scalar_type = double;
    using forward_type = std::complex<double>;
    using backward_type = std::complex<double>;
};

// Get the scalar type associated with a descriptor.
template <class descriptor_t>
using descriptor_scalar_t = typename descriptor_info<descriptor_t>::scalar_type;

template <typename T>
constexpr bool is_complex_dft = false;
template <precision Prec>
constexpr bool is_complex_dft<descriptor<Prec, domain::COMPLEX>> = true;

template <typename T>
constexpr bool is_complex = false;
template <typename T>
constexpr bool is_complex<std::complex<T>> = true;

template <typename T, typename... Ts>
using is_one_of = typename std::bool_constant<(std::is_same_v<T, Ts> || ...)>;

template <typename descriptor_type, typename T>
using valid_compute_arg = typename std::bool_constant<
    (std::is_same_v<descriptor_scalar_t<descriptor_type>, float> &&
     is_one_of<T, float, sycl::float2, sycl::float4, std::complex<float>>::value) ||
    (std::is_same_v<descriptor_scalar_t<descriptor_type>, double> &&
     is_one_of<T, double, sycl::double2, sycl::double4, std::complex<double>>::value)>;

template <class descriptor_t, typename data_t>
constexpr bool valid_ip_realreal_impl =
    is_complex_dft<descriptor_t> && std::is_same_v<descriptor_scalar_t<descriptor_t>, data_t>;

// compute the range of a reinterpreted buffer
template <typename In, typename Out>
std::size_t reinterpret_range(std::size_t size) {
    if constexpr (sizeof(In) >= sizeof(Out)) {
        static_assert(sizeof(In) % sizeof(Out) == 0);
        return size * (sizeof(In) / sizeof(Out));
    }
    else {
        static_assert(sizeof(Out) % sizeof(In) == 0);
        if (size % (sizeof(Out) / sizeof(In))) {
            throw std::runtime_error("buffer cannot be evenly divived into the expected type");
        }
        return size / (sizeof(Out) / sizeof(In));
    }
}

enum class config_param {
    FORWARD_DOMAIN,
    DIMENSION,
    LENGTHS,
    PRECISION,

    FORWARD_SCALE,
    BACKWARD_SCALE,

    NUMBER_OF_TRANSFORMS,

    COMPLEX_STORAGE,
    REAL_STORAGE,
    CONJUGATE_EVEN_STORAGE,

    PLACEMENT,

    INPUT_STRIDES [[deprecated("Use FWD/BWD_STRIDES")]],
    OUTPUT_STRIDES [[deprecated("Use FWD/BWD_STRIDES")]],

    FWD_DISTANCE,
    BWD_DISTANCE,

    WORKSPACE,
    WORKSPACE_PLACEMENT,
    WORKSPACE_EXTERNAL_BYTES,
    ORDERING,
    TRANSPOSE,
    PACKED_FORMAT,
    COMMIT_STATUS,

    FWD_STRIDES,
    BWD_STRIDES
};

enum class config_value {
    // for config_param::COMMIT_STATUS
    COMMITTED,
    UNCOMMITTED,

    // for config_param::COMPLEX_STORAGE,
    //     config_param::REAL_STORAGE and
    //     config_param::CONJUGATE_EVEN_STORAGE
    COMPLEX_COMPLEX,
    REAL_COMPLEX,
    REAL_REAL,

    // for config_param::PLACEMENT
    INPLACE,
    NOT_INPLACE,

    // for config_param::ORDERING
    ORDERED,
    BACKWARD_SCRAMBLED,

    // Allow/avoid certain usages
    ALLOW,
    AVOID,
    NONE,

    // for config_param::PACKED_FORMAT for storing conjugate-even finite sequence in real containers
    CCE_FORMAT,

    // For config_param::WORKSPACE_PLACEMENT
    WORKSPACE_AUTOMATIC,
    WORKSPACE_EXTERNAL
};

template <precision prec, domain dom>
class dft_values {
private:
    using real_t = typename precision_t<prec>::real_t;

public:
    std::vector<std::int64_t> input_strides;
    std::vector<std::int64_t> output_strides;
    std::vector<std::int64_t> fwd_strides;
    std::vector<std::int64_t> bwd_strides;
    real_t bwd_scale;
    real_t fwd_scale;
    std::int64_t number_of_transforms;
    std::int64_t fwd_dist;
    std::int64_t bwd_dist;
    config_value placement;
    config_value complex_storage;
    config_value real_storage;
    config_value conj_even_storage;
    config_value workspace;
    config_value workspace_placement;
    config_value ordering;
    bool transpose;
    config_value packed_format;
    std::vector<std::int64_t> dimensions;
};

} // namespace detail
} // namespace dft
} // namespace mkl
} // namespace oneapi

#endif //_ONEMKL_DETAIL_TYPES_IMPL_HPP_
