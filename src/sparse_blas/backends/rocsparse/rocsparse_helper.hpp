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
#ifndef _ONEMKL_SPARSE_BLAS_BACKENDS_ROCSPARSE_HELPER_HPP_
#define _ONEMKL_SPARSE_BLAS_BACKENDS_ROCSPARSE_HELPER_HPP_

#include <complex>
#include <cstdint>
#include <limits>
#include <string>

#include <rocsparse/rocsparse.h>

#include "oneapi/mkl/sparse_blas/types.hpp"
#include "sparse_blas/enum_data_types.hpp"
#include "sparse_blas/sycl_helper.hpp"
#include "rocsparse_error.hpp"

namespace oneapi::mkl::sparse::rocsparse::detail {

using namespace oneapi::mkl::sparse::detail;

template <typename T>
struct RocEnumType;
template <>
struct RocEnumType<float> {
    static constexpr rocsparse_datatype value = rocsparse_datatype_f32_r;
};
template <>
struct RocEnumType<double> {
    static constexpr rocsparse_datatype value = rocsparse_datatype_f64_r;
};
template <>
struct RocEnumType<std::complex<float>> {
    static constexpr rocsparse_datatype value = rocsparse_datatype_f32_c;
};
template <>
struct RocEnumType<std::complex<double>> {
    static constexpr rocsparse_datatype value = rocsparse_datatype_f64_c;
};

template <typename T>
struct RocIndexEnumType;
template <>
struct RocIndexEnumType<std::int32_t> {
    static constexpr rocsparse_indextype value = rocsparse_indextype_i32;
};
template <>
struct RocIndexEnumType<std::int64_t> {
    static constexpr rocsparse_indextype value = rocsparse_indextype_i64;
};

template <typename E>
inline std::string cast_enum_to_str(E e) {
    return std::to_string(static_cast<char>(e));
}

inline auto get_roc_value_type(data_type onemkl_data_type) {
    switch (onemkl_data_type) {
        case data_type::real_fp32: return rocsparse_datatype_f32_r;
        case data_type::real_fp64: return rocsparse_datatype_f64_r;
        case data_type::complex_fp32: return rocsparse_datatype_f32_c;
        case data_type::complex_fp64: return rocsparse_datatype_f64_c;
        default:
            throw oneapi::mkl::invalid_argument(
                "sparse_blas", "get_roc_value_type",
                "Invalid data type: " + cast_enum_to_str(onemkl_data_type));
    }
}

inline auto get_roc_order(layout l) {
    switch (l) {
        case layout::row_major: return rocsparse_order_row;
        case layout::col_major: return rocsparse_order_column;
        default:
            throw oneapi::mkl::invalid_argument("sparse_blas", "get_roc_order",
                                                "Unknown layout: " + cast_enum_to_str(l));
    }
}

inline auto get_roc_index_base(index_base index) {
    switch (index) {
        case index_base::zero: return rocsparse_index_base_zero;
        case index_base::one: return rocsparse_index_base_one;
        default:
            throw oneapi::mkl::invalid_argument("sparse_blas", "get_roc_index_base",
                                                "Unknown index_base: " + cast_enum_to_str(index));
    }
}

inline auto get_roc_operation(transpose op) {
    switch (op) {
        case transpose::nontrans: return rocsparse_operation_none;
        case transpose::trans: return rocsparse_operation_transpose;
        case transpose::conjtrans: return rocsparse_operation_conjugate_transpose;
        default:
            throw oneapi::mkl::invalid_argument(
                "sparse_blas", "get_roc_operation",
                "Unknown transpose operation: " + cast_enum_to_str(op));
    }
}

inline auto get_roc_uplo(uplo uplo_val) {
    switch (uplo_val) {
        case uplo::upper: return rocsparse_fill_mode_upper;
        case uplo::lower: return rocsparse_fill_mode_lower;
        default:
            throw oneapi::mkl::invalid_argument("sparse_blas", "get_roc_uplo",
                                                "Unknown uplo: " + cast_enum_to_str(uplo_val));
    }
}

inline auto get_roc_diag(diag diag_val) {
    switch (diag_val) {
        case diag::nonunit: return rocsparse_diag_type_non_unit;
        case diag::unit: return rocsparse_diag_type_unit;
        default:
            throw oneapi::mkl::invalid_argument("sparse_blas", "get_roc_diag",
                                                "Unknown diag: " + cast_enum_to_str(diag_val));
    }
}

inline void set_matrix_attributes(const std::string& func_name, rocsparse_spmat_descr roc_a,
                                  oneapi::mkl::sparse::matrix_view A_view) {
    auto roc_fill_mode = get_roc_uplo(A_view.uplo_view);
    auto status = rocsparse_spmat_set_attribute(roc_a, rocsparse_spmat_fill_mode, &roc_fill_mode,
                                                sizeof(roc_fill_mode));
    check_status(status, func_name + "/set_uplo");

    auto roc_diag_type = get_roc_diag(A_view.diag_view);
    status = rocsparse_spmat_set_attribute(roc_a, rocsparse_spmat_diag_type, &roc_diag_type,
                                           sizeof(roc_diag_type));
    check_status(status, func_name + "/set_diag");
}

/**
 * rocSPARSE requires to set the pointer mode for scalars parameters (typically alpha and beta).
 */
inline void set_pointer_mode(rocsparse_handle roc_handle, bool is_ptr_accessible_on_host) {
    rocsparse_set_pointer_mode(roc_handle, is_ptr_accessible_on_host
                                               ? rocsparse_pointer_mode_host
                                               : rocsparse_pointer_mode_device);
}

} // namespace oneapi::mkl::sparse::rocsparse::detail

#endif //_ONEMKL_SPARSE_BLAS_BACKENDS_ROCSPARSE_HELPER_HPP_
