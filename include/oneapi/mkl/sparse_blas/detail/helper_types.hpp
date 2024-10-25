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

#ifndef _ONEMKL_SPARSE_BLAS_DETAIL_HELPER_TYPES_HPP_
#define _ONEMKL_SPARSE_BLAS_DETAIL_HELPER_TYPES_HPP_

#include <complex>
#include <cstdint>
#include <type_traits>

namespace oneapi {
namespace mkl {
namespace sparse {
namespace detail {

template <typename dataType>
inline constexpr bool is_fp_supported_v =
    std::is_same_v<dataType, float> || std::is_same_v<dataType, double> ||
    std::is_same_v<dataType, std::complex<float>> || std::is_same_v<dataType, std::complex<double>>;

template <typename indexType>
inline constexpr bool is_int_supported_v =
    std::is_same_v<indexType, std::int32_t> || std::is_same_v<indexType, std::int64_t>;

template <typename dataType, typename indexType>
inline constexpr bool are_fp_int_supported_v =
    is_fp_supported_v<dataType> && is_int_supported_v<indexType>;

} // namespace detail
} // namespace sparse
} // namespace mkl
} // namespace oneapi

#endif // _ONEMKL_SPARSE_BLAS_DETAIL_HELPER_TYPES_HPP_
