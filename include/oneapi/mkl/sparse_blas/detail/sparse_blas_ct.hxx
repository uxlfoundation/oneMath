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

// This file is meant to be included in each backend sparse_blas_ct.hpp files
// Each function calls the implementation from onemkl_sparse_blas_backends.hxx

#ifndef BACKEND
#error "BACKEND is not defined"
#endif

// Dense vector
template <typename dataType>
std::enable_if_t<detail::is_fp_supported_v<dataType>> init_dense_vector(
    backend_selector<backend::BACKEND> selector, dense_vector_handle_t* p_dvhandle,
    std::int64_t size, sycl::buffer<dataType, 1> val) {
    BACKEND::init_dense_vector(selector.get_queue(), p_dvhandle, size, val);
}
template <typename dataType>
std::enable_if_t<detail::is_fp_supported_v<dataType>> init_dense_vector(
    backend_selector<backend::BACKEND> selector, dense_vector_handle_t* p_dvhandle,
    std::int64_t size, dataType* val) {
    BACKEND::init_dense_vector(selector.get_queue(), p_dvhandle, size, val);
}

template <typename dataType>
std::enable_if_t<detail::is_fp_supported_v<dataType>> set_dense_vector_data(
    backend_selector<backend::BACKEND> selector, dense_vector_handle_t dvhandle, std::int64_t size,
    sycl::buffer<dataType, 1> val) {
    BACKEND::set_dense_vector_data(selector.get_queue(), dvhandle, size, val);
}
template <typename dataType>
std::enable_if_t<detail::is_fp_supported_v<dataType>> set_dense_vector_data(
    backend_selector<backend::BACKEND> selector, dense_vector_handle_t dvhandle, std::int64_t size,
    dataType* val) {
    BACKEND::set_dense_vector_data(selector.get_queue(), dvhandle, size, val);
}

inline sycl::event release_dense_vector(backend_selector<backend::BACKEND> selector,
                                        dense_vector_handle_t dvhandle,
                                        const std::vector<sycl::event>& dependencies = {}) {
    return BACKEND::release_dense_vector(selector.get_queue(), dvhandle, dependencies);
}

// Dense matrix
template <typename dataType>
std::enable_if_t<detail::is_fp_supported_v<dataType>> init_dense_matrix(
    backend_selector<backend::BACKEND> selector, dense_matrix_handle_t* p_dmhandle,
    std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld, layout dense_layout,
    sycl::buffer<dataType, 1> val) {
    BACKEND::init_dense_matrix(selector.get_queue(), p_dmhandle, num_rows, num_cols, ld,
                               dense_layout, val);
}
template <typename dataType>
std::enable_if_t<detail::is_fp_supported_v<dataType>> init_dense_matrix(
    backend_selector<backend::BACKEND> selector, dense_matrix_handle_t* p_dmhandle,
    std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld, layout dense_layout,
    dataType* val) {
    BACKEND::init_dense_matrix(selector.get_queue(), p_dmhandle, num_rows, num_cols, ld,
                               dense_layout, val);
}

template <typename dataType>
std::enable_if_t<detail::is_fp_supported_v<dataType>> set_dense_matrix_data(
    backend_selector<backend::BACKEND> selector, dense_matrix_handle_t dmhandle,
    std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld, layout dense_layout,
    sycl::buffer<dataType, 1> val) {
    BACKEND::set_dense_matrix_data(selector.get_queue(), dmhandle, num_rows, num_cols, ld,
                                   dense_layout, val);
}
template <typename dataType>
std::enable_if_t<detail::is_fp_supported_v<dataType>> set_dense_matrix_data(
    backend_selector<backend::BACKEND> selector, dense_matrix_handle_t dmhandle,
    std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld, layout dense_layout,
    dataType* val) {
    BACKEND::set_dense_matrix_data(selector.get_queue(), dmhandle, num_rows, num_cols, ld,
                                   dense_layout, val);
}

inline sycl::event release_dense_matrix(backend_selector<backend::BACKEND> selector,
                                        dense_matrix_handle_t dmhandle,
                                        const std::vector<sycl::event>& dependencies = {}) {
    return BACKEND::release_dense_matrix(selector.get_queue(), dmhandle, dependencies);
}

// COO matrix
template <typename dataType, typename indexType>
std::enable_if_t<detail::are_fp_int_supported_v<dataType, indexType>> init_coo_matrix(
    backend_selector<backend::BACKEND> selector, matrix_handle_t* p_smhandle, std::int64_t num_rows,
    std::int64_t num_cols, std::int64_t nnz, index_base index, sycl::buffer<indexType, 1> row_ind,
    sycl::buffer<indexType, 1> col_ind, sycl::buffer<dataType, 1> val) {
    BACKEND::init_coo_matrix(selector.get_queue(), p_smhandle, num_rows, num_cols, nnz, index,
                             row_ind, col_ind, val);
}
template <typename dataType, typename indexType>
std::enable_if_t<detail::are_fp_int_supported_v<dataType, indexType>> init_coo_matrix(
    backend_selector<backend::BACKEND> selector, matrix_handle_t* p_smhandle, std::int64_t num_rows,
    std::int64_t num_cols, std::int64_t nnz, index_base index, indexType* row_ind,
    indexType* col_ind, dataType* val) {
    BACKEND::init_coo_matrix(selector.get_queue(), p_smhandle, num_rows, num_cols, nnz, index,
                             row_ind, col_ind, val);
}

template <typename dataType, typename indexType>
std::enable_if_t<detail::are_fp_int_supported_v<dataType, indexType>> set_coo_matrix_data(
    backend_selector<backend::BACKEND> selector, matrix_handle_t smhandle, std::int64_t num_rows,
    std::int64_t num_cols, std::int64_t nnz, index_base index, sycl::buffer<indexType, 1> row_ind,
    sycl::buffer<indexType, 1> col_ind, sycl::buffer<dataType, 1> val) {
    BACKEND::set_coo_matrix_data(selector.get_queue(), smhandle, num_rows, num_cols, nnz, index,
                                 row_ind, col_ind, val);
}
template <typename dataType, typename indexType>
std::enable_if_t<detail::are_fp_int_supported_v<dataType, indexType>> set_coo_matrix_data(
    backend_selector<backend::BACKEND> selector, matrix_handle_t smhandle, std::int64_t num_rows,
    std::int64_t num_cols, std::int64_t nnz, index_base index, indexType* row_ind,
    indexType* col_ind, dataType* val) {
    BACKEND::set_coo_matrix_data(selector.get_queue(), smhandle, num_rows, num_cols, nnz, index,
                                 row_ind, col_ind, val);
}

// CSR matrix
template <typename dataType, typename indexType>
std::enable_if_t<detail::are_fp_int_supported_v<dataType, indexType>> init_csr_matrix(
    backend_selector<backend::BACKEND> selector, matrix_handle_t* p_smhandle, std::int64_t num_rows,
    std::int64_t num_cols, std::int64_t nnz, index_base index, sycl::buffer<indexType, 1> row_ptr,
    sycl::buffer<indexType, 1> col_ind, sycl::buffer<dataType, 1> val) {
    BACKEND::init_csr_matrix(selector.get_queue(), p_smhandle, num_rows, num_cols, nnz, index,
                             row_ptr, col_ind, val);
}
template <typename dataType, typename indexType>
std::enable_if_t<detail::are_fp_int_supported_v<dataType, indexType>> init_csr_matrix(
    backend_selector<backend::BACKEND> selector, matrix_handle_t* p_smhandle, std::int64_t num_rows,
    std::int64_t num_cols, std::int64_t nnz, index_base index, indexType* row_ptr,
    indexType* col_ind, dataType* val) {
    BACKEND::init_csr_matrix(selector.get_queue(), p_smhandle, num_rows, num_cols, nnz, index,
                             row_ptr, col_ind, val);
}

template <typename dataType, typename indexType>
std::enable_if_t<detail::are_fp_int_supported_v<dataType, indexType>> set_csr_matrix_data(
    backend_selector<backend::BACKEND> selector, matrix_handle_t smhandle, std::int64_t num_rows,
    std::int64_t num_cols, std::int64_t nnz, index_base index, sycl::buffer<indexType, 1> row_ptr,
    sycl::buffer<indexType, 1> col_ind, sycl::buffer<dataType, 1> val) {
    BACKEND::set_csr_matrix_data(selector.get_queue(), smhandle, num_rows, num_cols, nnz, index,
                                 row_ptr, col_ind, val);
}
template <typename dataType, typename indexType>
std::enable_if_t<detail::are_fp_int_supported_v<dataType, indexType>> set_csr_matrix_data(
    backend_selector<backend::BACKEND> selector, matrix_handle_t smhandle, std::int64_t num_rows,
    std::int64_t num_cols, std::int64_t nnz, index_base index, indexType* row_ptr,
    indexType* col_ind, dataType* val) {
    BACKEND::set_csr_matrix_data(selector.get_queue(), smhandle, num_rows, num_cols, nnz, index,
                                 row_ptr, col_ind, val);
}

// Common sparse matrix functions
inline sycl::event release_sparse_matrix(backend_selector<backend::BACKEND> selector,
                                         matrix_handle_t smhandle,
                                         const std::vector<sycl::event>& dependencies = {}) {
    return BACKEND::release_sparse_matrix(selector.get_queue(), smhandle, dependencies);
}

inline bool set_matrix_property(backend_selector<backend::BACKEND> selector,
                                matrix_handle_t smhandle, matrix_property property) {
    return BACKEND::set_matrix_property(selector.get_queue(), smhandle, property);
}

// SPMM
inline void init_spmm_descr(backend_selector<backend::BACKEND> selector,
                            spmm_descr_t* p_spmm_descr) {
    BACKEND::init_spmm_descr(selector.get_queue(), p_spmm_descr);
}

inline sycl::event release_spmm_descr(backend_selector<backend::BACKEND> selector,
                                      spmm_descr_t spmm_descr,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return BACKEND::release_spmm_descr(selector.get_queue(), spmm_descr, dependencies);
}

inline void spmm_buffer_size(backend_selector<backend::BACKEND> selector,
                             oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                             const void* alpha, matrix_view A_view, matrix_handle_t A_handle,
                             dense_matrix_handle_t B_handle, const void* beta,
                             dense_matrix_handle_t C_handle, spmm_alg alg, spmm_descr_t spmm_descr,
                             std::size_t& temp_buffer_size) {
    BACKEND::spmm_buffer_size(selector.get_queue(), opA, opB, alpha, A_view, A_handle, B_handle,
                              beta, C_handle, alg, spmm_descr, temp_buffer_size);
}

inline void spmm_optimize(backend_selector<backend::BACKEND> selector, oneapi::mkl::transpose opA,
                          oneapi::mkl::transpose opB, const void* alpha, matrix_view A_view,
                          matrix_handle_t A_handle, dense_matrix_handle_t B_handle,
                          const void* beta, dense_matrix_handle_t C_handle, spmm_alg alg,
                          spmm_descr_t spmm_descr, sycl::buffer<std::uint8_t, 1> workspace) {
    BACKEND::spmm_optimize(selector.get_queue(), opA, opB, alpha, A_view, A_handle, B_handle, beta,
                           C_handle, alg, spmm_descr, workspace);
}

inline sycl::event spmm_optimize(backend_selector<backend::BACKEND> selector,
                                 oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                                 const void* alpha, matrix_view A_view, matrix_handle_t A_handle,
                                 dense_matrix_handle_t B_handle, const void* beta,
                                 dense_matrix_handle_t C_handle, spmm_alg alg,
                                 spmm_descr_t spmm_descr, void* workspace,
                                 const std::vector<sycl::event>& dependencies = {}) {
    return BACKEND::spmm_optimize(selector.get_queue(), opA, opB, alpha, A_view, A_handle, B_handle,
                                  beta, C_handle, alg, spmm_descr, workspace, dependencies);
}

inline sycl::event spmm(backend_selector<backend::BACKEND> selector, oneapi::mkl::transpose opA,
                        oneapi::mkl::transpose opB, const void* alpha, matrix_view A_view,
                        matrix_handle_t A_handle, dense_matrix_handle_t B_handle, const void* beta,
                        dense_matrix_handle_t C_handle, spmm_alg alg, spmm_descr_t spmm_descr,
                        const std::vector<sycl::event>& dependencies = {}) {
    return BACKEND::spmm(selector.get_queue(), opA, opB, alpha, A_view, A_handle, B_handle, beta,
                         C_handle, alg, spmm_descr, dependencies);
}

// SPMV
inline void init_spmv_descr(backend_selector<backend::BACKEND> selector,
                            spmv_descr_t* p_spmv_descr) {
    BACKEND::init_spmv_descr(selector.get_queue(), p_spmv_descr);
}

inline sycl::event release_spmv_descr(backend_selector<backend::BACKEND> selector,
                                      spmv_descr_t spmv_descr,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return BACKEND::release_spmv_descr(selector.get_queue(), spmv_descr, dependencies);
}

inline void spmv_buffer_size(backend_selector<backend::BACKEND> selector,
                             oneapi::mkl::transpose opA, const void* alpha, matrix_view A_view,
                             matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                             const void* beta, dense_vector_handle_t y_handle, spmv_alg alg,
                             spmv_descr_t spmv_descr, std::size_t& temp_buffer_size) {
    BACKEND::spmv_buffer_size(selector.get_queue(), opA, alpha, A_view, A_handle, x_handle, beta,
                              y_handle, alg, spmv_descr, temp_buffer_size);
}

inline void spmv_optimize(backend_selector<backend::BACKEND> selector, oneapi::mkl::transpose opA,
                          const void* alpha, matrix_view A_view, matrix_handle_t A_handle,
                          dense_vector_handle_t x_handle, const void* beta,
                          dense_vector_handle_t y_handle, spmv_alg alg, spmv_descr_t spmv_descr,
                          sycl::buffer<std::uint8_t, 1> workspace) {
    BACKEND::spmv_optimize(selector.get_queue(), opA, alpha, A_view, A_handle, x_handle, beta,
                           y_handle, alg, spmv_descr, workspace);
}

inline sycl::event spmv_optimize(backend_selector<backend::BACKEND> selector,
                                 oneapi::mkl::transpose opA, const void* alpha, matrix_view A_view,
                                 matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                                 const void* beta, dense_vector_handle_t y_handle, spmv_alg alg,
                                 spmv_descr_t spmv_descr, void* workspace,
                                 const std::vector<sycl::event>& dependencies = {}) {
    return BACKEND::spmv_optimize(selector.get_queue(), opA, alpha, A_view, A_handle, x_handle,
                                  beta, y_handle, alg, spmv_descr, workspace, dependencies);
}

inline sycl::event spmv(backend_selector<backend::BACKEND> selector, oneapi::mkl::transpose opA,
                        const void* alpha, matrix_view A_view, matrix_handle_t A_handle,
                        dense_vector_handle_t x_handle, const void* beta,
                        dense_vector_handle_t y_handle, spmv_alg alg, spmv_descr_t spmv_descr,
                        const std::vector<sycl::event>& dependencies = {}) {
    return BACKEND::spmv(selector.get_queue(), opA, alpha, A_view, A_handle, x_handle, beta,
                         y_handle, alg, spmv_descr, dependencies);
}

// SPSV
inline void init_spsv_descr(backend_selector<backend::BACKEND> selector,
                            spsv_descr_t* p_spsv_descr) {
    BACKEND::init_spsv_descr(selector.get_queue(), p_spsv_descr);
}

inline sycl::event release_spsv_descr(backend_selector<backend::BACKEND> selector,
                                      spsv_descr_t spsv_descr,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return BACKEND::release_spsv_descr(selector.get_queue(), spsv_descr, dependencies);
}

inline void spsv_buffer_size(backend_selector<backend::BACKEND> selector,
                             oneapi::mkl::transpose opA, const void* alpha, matrix_view A_view,
                             matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                             dense_vector_handle_t y_handle, spsv_alg alg, spsv_descr_t spsv_descr,
                             std::size_t& temp_buffer_size) {
    BACKEND::spsv_buffer_size(selector.get_queue(), opA, alpha, A_view, A_handle, x_handle,
                              y_handle, alg, spsv_descr, temp_buffer_size);
}

inline void spsv_optimize(backend_selector<backend::BACKEND> selector, oneapi::mkl::transpose opA,
                          const void* alpha, matrix_view A_view, matrix_handle_t A_handle,
                          dense_vector_handle_t x_handle, dense_vector_handle_t y_handle,
                          spsv_alg alg, spsv_descr_t spsv_descr,
                          sycl::buffer<std::uint8_t, 1> workspace) {
    BACKEND::spsv_optimize(selector.get_queue(), opA, alpha, A_view, A_handle, x_handle, y_handle,
                           alg, spsv_descr, workspace);
}

inline sycl::event spsv_optimize(backend_selector<backend::BACKEND> selector,
                                 oneapi::mkl::transpose opA, const void* alpha, matrix_view A_view,
                                 matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                                 dense_vector_handle_t y_handle, spsv_alg alg,
                                 spsv_descr_t spsv_descr, void* workspace,
                                 const std::vector<sycl::event>& dependencies = {}) {
    return BACKEND::spsv_optimize(selector.get_queue(), opA, alpha, A_view, A_handle, x_handle,
                                  y_handle, alg, spsv_descr, workspace, dependencies);
}

inline sycl::event spsv(backend_selector<backend::BACKEND> selector, oneapi::mkl::transpose opA,
                        const void* alpha, matrix_view A_view, matrix_handle_t A_handle,
                        dense_vector_handle_t x_handle, dense_vector_handle_t y_handle,
                        spsv_alg alg, spsv_descr_t spsv_descr,
                        const std::vector<sycl::event>& dependencies = {}) {
    return BACKEND::spsv(selector.get_queue(), opA, alpha, A_view, A_handle, x_handle, y_handle,
                         alg, spsv_descr, dependencies);
}
