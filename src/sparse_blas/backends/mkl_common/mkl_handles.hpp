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

#ifndef _ONEMKL_SRC_SPARSE_BLAS_BACKENDS_MKL_COMMON_MKL_HANDLES_HPP_
#define _ONEMKL_SRC_SPARSE_BLAS_BACKENDS_MKL_COMMON_MKL_HANDLES_HPP_

// MKLCPU and MKLGPU backends include
// This include defines its own oneapi::mkl::sparse namespace with some of the
// types that are used here: matrix_handle_t, index_base, transpose, uplo, diag.
#include <oneapi/mkl/spblas.hpp>

#include "sparse_blas/generic_container.hpp"
#include "sparse_blas/macros.hpp"
#include "sparse_blas/sycl_helper.hpp"

namespace oneapi::mkl::sparse {

// Complete the definition of incomplete types dense_vector_handle and
// dense_matrix_handle as they don't exist in oneMKL backends yet.

struct dense_vector_handle : public detail::generic_dense_vector_handle<void*> {
    template <typename T>
    dense_vector_handle(T* value_ptr, std::int64_t size)
            : detail::generic_dense_vector_handle<void*>(nullptr, value_ptr, size) {}

    template <typename T>
    dense_vector_handle(const sycl::buffer<T, 1> value_buffer, std::int64_t size)
            : detail::generic_dense_vector_handle<void*>(nullptr, value_buffer, size) {}
};

struct dense_matrix_handle : public detail::generic_dense_matrix_handle<void*> {
    template <typename T>
    dense_matrix_handle(T* value_ptr, std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,
                        layout dense_layout)
            : detail::generic_dense_matrix_handle<void*>(nullptr, value_ptr, num_rows, num_cols, ld,
                                                         dense_layout) {}

    template <typename T>
    dense_matrix_handle(const sycl::buffer<T, 1> value_buffer, std::int64_t num_rows,
                        std::int64_t num_cols, std::int64_t ld, layout dense_layout)
            : detail::generic_dense_matrix_handle<void*>(nullptr, value_buffer, num_rows, num_cols,
                                                         ld, dense_layout) {}
};

} // namespace oneapi::mkl::sparse

namespace oneapi::mkl::sparse::detail {

/**
 * Internal sparse_matrix_handle type for MKLCPU and MKLGPU backends.
 * Here \p matrix_handle_t is the type of the backend's handle.
 * The user-facing incomplete type matrix_handle_t must be kept incomplete.
 * Internally matrix_handle_t is reinterpret_cast as
 * oneapi::mkl::sparse::detail::sparse_matrix_handle which holds another
 * matrix_handle_t for the backend handle.
 */
using sparse_matrix_handle = detail::generic_sparse_handle<matrix_handle_t>;

/// Cast to oneMKL's interface handle type
inline auto get_internal_handle(matrix_handle_t handle) {
    return reinterpret_cast<sparse_matrix_handle*>(handle);
}

} // namespace oneapi::mkl::sparse::detail

#endif // _ONEMKL_SRC_SPARSE_BLAS_BACKENDS_MKL_COMMON_MKL_HANDLES_HPP_
