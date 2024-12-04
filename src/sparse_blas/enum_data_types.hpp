/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

#ifndef _ONEMATH_SRC_SPARSE_BLAS_ENUM_DATA_TYPES_HPP_
#define _ONEMATH_SRC_SPARSE_BLAS_ENUM_DATA_TYPES_HPP_

#include <string>

namespace oneapi::math::sparse::detail {

enum data_type { none, int32, int64, real_fp32, real_fp64, complex_fp32, complex_fp64 };

inline std::string data_type_to_str(data_type data_type) {
    switch (data_type) {
        case none: return "none";
        case int32: return "int32";
        case int64: return "int64";
        case real_fp32: return "real_fp32";
        case real_fp64: return "real_fp64";
        case complex_fp32: return "complex_fp32";
        case complex_fp64: return "complex_fp64";
        default: return "unknown";
    }
}

template <typename T>
constexpr data_type get_data_type() {
    if constexpr (std::is_same_v<T, std::int32_t>) {
        return data_type::int32;
    }
    else if constexpr (std::is_same_v<T, std::int64_t>) {
        return data_type::int64;
    }
    else if constexpr (std::is_same_v<T, float>) {
        return data_type::real_fp32;
    }
    else if constexpr (std::is_same_v<T, double>) {
        return data_type::real_fp64;
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return data_type::complex_fp32;
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
        return data_type::complex_fp64;
    }
    else {
        static_assert(false, "Unsupported type");
    }
}

} // namespace oneapi::math::sparse::detail

#endif // _ONEMATH_SRC_SPARSE_BLAS_ENUM_DATA_TYPES_HPP_
