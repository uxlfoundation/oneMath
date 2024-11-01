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

#ifndef _ONEMKL_SRC_SPARSE_BLAS_BACKENDS_ROCSPARSE_HANDLES_HPP_
#define _ONEMKL_SRC_SPARSE_BLAS_BACKENDS_ROCSPARSE_HANDLES_HPP_

#include <rocsparse/rocsparse.h>

#include "sparse_blas/generic_container.hpp"

namespace oneapi::mkl::sparse {

// Complete the definition of incomplete types dense_vector_handle, dense_matrix_handle and matrix_handle.

struct dense_vector_handle : public detail::generic_dense_vector_handle<rocsparse_dnvec_descr> {
    template <typename T>
    dense_vector_handle(rocsparse_dnvec_descr roc_descr, T* value_ptr, std::int64_t size)
            : detail::generic_dense_vector_handle<rocsparse_dnvec_descr>(roc_descr, value_ptr,
                                                                         size) {}

    template <typename T>
    dense_vector_handle(rocsparse_dnvec_descr roc_descr, const sycl::buffer<T, 1> value_buffer,
                        std::int64_t size)
            : detail::generic_dense_vector_handle<rocsparse_dnvec_descr>(roc_descr, value_buffer,
                                                                         size) {}
};

struct dense_matrix_handle : public detail::generic_dense_matrix_handle<rocsparse_dnmat_descr> {
    template <typename T>
    dense_matrix_handle(rocsparse_dnmat_descr roc_descr, T* value_ptr, std::int64_t num_rows,
                        std::int64_t num_cols, std::int64_t ld, layout dense_layout)
            : detail::generic_dense_matrix_handle<rocsparse_dnmat_descr>(
                  roc_descr, value_ptr, num_rows, num_cols, ld, dense_layout) {}

    template <typename T>
    dense_matrix_handle(rocsparse_dnmat_descr roc_descr, const sycl::buffer<T, 1> value_buffer,
                        std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,
                        layout dense_layout)
            : detail::generic_dense_matrix_handle<rocsparse_dnmat_descr>(
                  roc_descr, value_buffer, num_rows, num_cols, ld, dense_layout) {}
};

struct matrix_handle : public detail::generic_sparse_handle<rocsparse_spmat_descr> {
    // A matrix handle should only be used once per operation to be safe with the rocSPARSE backend.
    // An operation can store information in the handle. See details in https://github.com/ROCm/rocSPARSE/issues/332.
private:
    bool used = false;

public:
    template <typename fpType, typename intType>
    matrix_handle(rocsparse_spmat_descr roc_descr, intType* row_ptr, intType* col_ptr,
                  fpType* value_ptr, detail::sparse_format format, std::int64_t num_rows,
                  std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index)
            : detail::generic_sparse_handle<rocsparse_spmat_descr>(
                  roc_descr, row_ptr, col_ptr, value_ptr, format, num_rows, num_cols, nnz, index) {}

    template <typename fpType, typename intType>
    matrix_handle(rocsparse_spmat_descr roc_descr, const sycl::buffer<intType, 1> row_buffer,
                  const sycl::buffer<intType, 1> col_buffer,
                  const sycl::buffer<fpType, 1> value_buffer, detail::sparse_format format,
                  std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                  oneapi::mkl::index_base index)
            : detail::generic_sparse_handle<rocsparse_spmat_descr>(roc_descr, row_buffer,
                                                                   col_buffer, value_buffer, format,
                                                                   num_rows, num_cols, nnz, index) {
    }

    void check_valid_handle(const std::string& function_name) {
        if (used) {
            throw mkl::unimplemented(
                "sparse_blas", function_name,
                "The backend does not support re-using the same sparse matrix handle in multiple operations.");
        }
        if (this->format == detail::sparse_format::COO &&
            !this->has_matrix_property(matrix_property::sorted)) {
            throw mkl::unimplemented(
                "sparse_blas", function_name,
                "The backend does not support unsorted COO format. Use `set_matrix_property` to set the property `matrix_property::sorted`");
        }
    }

    void mark_used() {
        used = true;
    }
};

} // namespace oneapi::mkl::sparse

#endif // _ONEMKL_SRC_SPARSE_BLAS_BACKENDS_ROCSPARSE_HANDLES_HPP_
