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

#ifndef _ONEMATH_SRC_SPARSE_BLAS_COMMON_OP_VERIFICATION_HPP_
#define _ONEMATH_SRC_SPARSE_BLAS_COMMON_OP_VERIFICATION_HPP_

#include <string>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/sparse_blas/types.hpp"
#include "macros.hpp"

namespace oneapi::math::sparse::detail {

/// Throw an exception if the scalar is not accessible in the host
inline void check_ptr_is_host_accessible(const std::string& function_name,
                                         const std::string& scalar_name,
                                         bool is_ptr_accessible_on_host) {
    if (!is_ptr_accessible_on_host) {
        throw math::invalid_argument(
            "sparse_blas", function_name,
            "Scalar " + scalar_name + " must be accessible on the host for buffer functions.");
    }
}

template <typename InternalSparseMatHandleT>
void check_valid_spmm_common(const std::string& function_name, matrix_view A_view,
                             InternalSparseMatHandleT internal_A_handle,
                             dense_matrix_handle_t B_handle, dense_matrix_handle_t C_handle,
                             bool is_alpha_host_accessible, bool is_beta_host_accessible) {
    THROW_IF_NULLPTR(function_name, internal_A_handle);
    THROW_IF_NULLPTR(function_name, B_handle);
    THROW_IF_NULLPTR(function_name, C_handle);

    check_all_containers_compatible(function_name, internal_A_handle, B_handle, C_handle);
    if (internal_A_handle->all_use_buffer()) {
        check_ptr_is_host_accessible("spmm", "alpha", is_alpha_host_accessible);
        check_ptr_is_host_accessible("spmm", "beta", is_beta_host_accessible);
    }
    if (is_alpha_host_accessible != is_beta_host_accessible) {
        throw math::invalid_argument(
            "sparse_blas", function_name,
            "Alpha and beta must both be placed on host memory or device memory.");
    }
    if (B_handle->dense_layout != C_handle->dense_layout) {
        throw math::invalid_argument("sparse_blas", function_name,
                                     "B and C matrices must use the same layout.");
    }

    if (A_view.type_view != matrix_descr::general) {
        throw math::invalid_argument("sparse_blas", function_name,
                                     "Matrix view's `type_view` must be `matrix_descr::general`.");
    }

    if (A_view.diag_view != oneapi::math::diag::nonunit) {
        throw math::invalid_argument("sparse_blas", function_name,
                                     "Matrix's diag_view must be `nonunit`.");
    }
}

template <typename InternalSparseMatHandleT>
void check_valid_spmv_common(const std::string& function_name, oneapi::math::transpose /*opA*/,
                             matrix_view A_view, InternalSparseMatHandleT internal_A_handle,
                             dense_vector_handle_t x_handle, dense_vector_handle_t y_handle,
                             bool is_alpha_host_accessible, bool is_beta_host_accessible) {
    THROW_IF_NULLPTR(function_name, internal_A_handle);
    THROW_IF_NULLPTR(function_name, x_handle);
    THROW_IF_NULLPTR(function_name, y_handle);

    check_all_containers_compatible(function_name, internal_A_handle, x_handle, y_handle);
    if (internal_A_handle->all_use_buffer()) {
        check_ptr_is_host_accessible("spmv", "alpha", is_alpha_host_accessible);
        check_ptr_is_host_accessible("spmv", "beta", is_beta_host_accessible);
    }
    if (is_alpha_host_accessible != is_beta_host_accessible) {
        throw math::invalid_argument(
            "sparse_blas", function_name,
            "Alpha and beta must both be placed on host memory or device memory.");
    }
    if (A_view.type_view == matrix_descr::diagonal) {
        throw math::invalid_argument("sparse_blas", function_name,
                                     "Matrix view's `type_view` cannot be diagonal.");
    }

    if (A_view.type_view != matrix_descr::triangular &&
        A_view.diag_view == oneapi::math::diag::unit) {
        throw math::invalid_argument(
            "sparse_blas", function_name,
            "`diag_view::unit` can only be used with `type_view::triangular`.");
    }
}

template <typename InternalSparseMatHandleT>
void check_valid_spsv_common(const std::string& function_name, matrix_view A_view,
                             InternalSparseMatHandleT internal_A_handle,
                             dense_vector_handle_t x_handle, dense_vector_handle_t y_handle,
                             bool is_alpha_host_accessible) {
    THROW_IF_NULLPTR(function_name, internal_A_handle);
    THROW_IF_NULLPTR(function_name, x_handle);
    THROW_IF_NULLPTR(function_name, y_handle);

    check_all_containers_compatible(function_name, internal_A_handle, x_handle, y_handle);
    if (A_view.type_view != matrix_descr::triangular) {
        throw math::invalid_argument(
            "sparse_blas", function_name,
            "Matrix view's `type_view` must be `matrix_descr::triangular`.");
    }

    if (internal_A_handle->all_use_buffer()) {
        check_ptr_is_host_accessible("spsv", "alpha", is_alpha_host_accessible);
    }
}

} // namespace oneapi::math::sparse::detail

#endif // _ONEMATH_SRC_SPARSE_BLAS_COMMON_OP_VERIFICATION_HPP_