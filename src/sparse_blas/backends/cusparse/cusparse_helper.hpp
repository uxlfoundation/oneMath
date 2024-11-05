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
#ifndef _ONEMATH_SPARSE_BLAS_BACKENDS_CUSPARSE_HELPER_HPP_
#define _ONEMATH_SPARSE_BLAS_BACKENDS_CUSPARSE_HELPER_HPP_

#include <complex>
#include <cstdint>
#include <limits>
#include <string>

#include <cusparse.h>

#include "oneapi/math/sparse_blas/types.hpp"
#include "sparse_blas/enum_data_types.hpp"
#include "sparse_blas/sycl_helper.hpp"
#include "cusparse_error.hpp"

namespace oneapi::math::sparse::cusparse::detail {

using namespace oneapi::math::sparse::detail;

template <typename T>
struct CudaEnumType;
template <>
struct CudaEnumType<float> {
    static constexpr cudaDataType_t value = CUDA_R_32F;
};
template <>
struct CudaEnumType<double> {
    static constexpr cudaDataType_t value = CUDA_R_64F;
};
template <>
struct CudaEnumType<std::complex<float>> {
    static constexpr cudaDataType_t value = CUDA_C_32F;
};
template <>
struct CudaEnumType<std::complex<double>> {
    static constexpr cudaDataType_t value = CUDA_C_64F;
};

template <typename T>
struct CudaIndexEnumType;
template <>
struct CudaIndexEnumType<std::int32_t> {
    static constexpr cusparseIndexType_t value = CUSPARSE_INDEX_32I;
};
template <>
struct CudaIndexEnumType<std::int64_t> {
    static constexpr cusparseIndexType_t value = CUSPARSE_INDEX_64I;
};

template <typename E>
inline std::string cast_enum_to_str(E e) {
    return std::to_string(static_cast<char>(e));
}

inline cudaDataType_t get_cuda_value_type(data_type onemath_data_type) {
    switch (onemath_data_type) {
        case data_type::real_fp32: return CUDA_R_32F;
        case data_type::real_fp64: return CUDA_R_64F;
        case data_type::complex_fp32: return CUDA_C_32F;
        case data_type::complex_fp64: return CUDA_C_64F;
        default:
            throw oneapi::math::invalid_argument(
                "sparse_blas", "get_cuda_value_type",
                "Invalid data type: " + cast_enum_to_str(onemath_data_type));
    }
}

inline cusparseOrder_t get_cuda_order(layout l) {
    switch (l) {
        case layout::row_major: return CUSPARSE_ORDER_ROW;
        case layout::col_major: return CUSPARSE_ORDER_COL;
        default:
            throw oneapi::math::invalid_argument("sparse_blas", "get_cuda_order",
                                                 "Unknown layout: " + cast_enum_to_str(l));
    }
}

inline cusparseIndexBase_t get_cuda_index_base(index_base index) {
    switch (index) {
        case index_base::zero: return CUSPARSE_INDEX_BASE_ZERO;
        case index_base::one: return CUSPARSE_INDEX_BASE_ONE;
        default:
            throw oneapi::math::invalid_argument("sparse_blas", "get_cuda_index_base",
                                                 "Unknown index_base: " + cast_enum_to_str(index));
    }
}

/// Return the CUDA transpose operation from a oneMath type.
/// Do not conjugate for real types to avoid an invalid argument.
inline cusparseOperation_t get_cuda_operation(data_type type, transpose op) {
    switch (op) {
        case transpose::nontrans: return CUSPARSE_OPERATION_NON_TRANSPOSE;
        case transpose::trans: return CUSPARSE_OPERATION_TRANSPOSE;
        case transpose::conjtrans:
            return (type == data_type::complex_fp32 || type == data_type::complex_fp64)
                       ? CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
                       : CUSPARSE_OPERATION_TRANSPOSE;
        default:
            throw oneapi::math::invalid_argument(
                "sparse_blas", "get_cuda_operation",
                "Unknown transpose operation: " + cast_enum_to_str(op));
    }
}

inline auto get_cuda_uplo(uplo uplo_val) {
    switch (uplo_val) {
        case uplo::upper: return CUSPARSE_FILL_MODE_UPPER;
        case uplo::lower: return CUSPARSE_FILL_MODE_LOWER;
        default:
            throw oneapi::math::invalid_argument("sparse_blas", "get_cuda_uplo",
                                                 "Unknown uplo: " + cast_enum_to_str(uplo_val));
    }
}

inline auto get_cuda_diag(diag diag_val) {
    switch (diag_val) {
        case diag::nonunit: return CUSPARSE_DIAG_TYPE_NON_UNIT;
        case diag::unit: return CUSPARSE_DIAG_TYPE_UNIT;
        default:
            throw oneapi::math::invalid_argument("sparse_blas", "get_cuda_diag",
                                                 "Unknown diag: " + cast_enum_to_str(diag_val));
    }
}

inline void set_matrix_attributes(const std::string& func_name, cusparseSpMatDescr_t cu_a,
                                  oneapi::math::sparse::matrix_view A_view) {
    auto cu_fill_mode = get_cuda_uplo(A_view.uplo_view);
    auto status = cusparseSpMatSetAttribute(cu_a, CUSPARSE_SPMAT_FILL_MODE, &cu_fill_mode,
                                            sizeof(cu_fill_mode));
    check_status(status, func_name + "/set_uplo");

    auto cu_diag_type = get_cuda_diag(A_view.diag_view);
    status = cusparseSpMatSetAttribute(cu_a, CUSPARSE_SPMAT_DIAG_TYPE, &cu_diag_type,
                                       sizeof(cu_diag_type));
    check_status(status, func_name + "/set_diag");
}

/**
 * cuSPARSE requires to set the pointer mode for scalars parameters (typically alpha and beta).
 */
inline void set_pointer_mode(cusparseHandle_t cu_handle, bool is_ptr_host_accessible) {
    cusparseSetPointerMode(cu_handle, is_ptr_host_accessible ? CUSPARSE_POINTER_MODE_HOST
                                                             : CUSPARSE_POINTER_MODE_DEVICE);
}

} // namespace oneapi::math::sparse::cusparse::detail

#endif //_ONEMATH_SPARSE_BLAS_BACKENDS_CUSPARSE_HELPER_HPP_
