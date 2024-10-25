/***************************************************************************
*  Copyright(C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0(the "License");
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

// This file is meant to be included in each backend onemkl_sparse_blas_BACKEND.hpp files.
// It is used to exports each symbol to the onemkl_sparse_blas_BACKEND library.

// Dense vector
template <typename dataType>
ONEMKL_EXPORT void init_dense_vector(sycl::queue& queue, dense_vector_handle_t* p_dvhandle,
                                     std::int64_t size, sycl::buffer<dataType, 1> val);
template <typename dataType>
ONEMKL_EXPORT void init_dense_vector(sycl::queue& queue, dense_vector_handle_t* p_dvhandle,
                                     std::int64_t size, dataType* val);

template <typename dataType>
ONEMKL_EXPORT void set_dense_vector_data(sycl::queue& queue, dense_vector_handle_t dvhandle,
                                         std::int64_t size, sycl::buffer<dataType, 1> val);
template <typename dataType>
ONEMKL_EXPORT void set_dense_vector_data(sycl::queue& queue, dense_vector_handle_t dvhandle,
                                         std::int64_t size, dataType* val);

ONEMKL_EXPORT sycl::event release_dense_vector(sycl::queue& queue, dense_vector_handle_t dvhandle,
                                               const std::vector<sycl::event>& dependencies = {});

// Dense matrix
template <typename dataType>
ONEMKL_EXPORT void init_dense_matrix(sycl::queue& queue, dense_matrix_handle_t* p_dmhandle,
                                     std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,
                                     layout dense_layout, sycl::buffer<dataType, 1> val);
template <typename dataType>
ONEMKL_EXPORT void init_dense_matrix(sycl::queue& queue, dense_matrix_handle_t* p_dmhandle,
                                     std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,
                                     layout dense_layout, dataType* val);

template <typename dataType>
ONEMKL_EXPORT void set_dense_matrix_data(sycl::queue& queue, dense_matrix_handle_t dmhandle,
                                         std::int64_t num_rows, std::int64_t num_cols,
                                         std::int64_t ld, layout dense_layout,
                                         sycl::buffer<dataType, 1> val);
template <typename dataType>
ONEMKL_EXPORT void set_dense_matrix_data(sycl::queue& queue, dense_matrix_handle_t dmhandle,
                                         std::int64_t num_rows, std::int64_t num_cols,
                                         std::int64_t ld, layout dense_layout, dataType* val);

ONEMKL_EXPORT sycl::event release_dense_matrix(sycl::queue& queue, dense_matrix_handle_t dmhandle,
                                               const std::vector<sycl::event>& dependencies = {});

// COO matrix
template <typename dataType, typename indexType>
ONEMKL_EXPORT void init_coo_matrix(sycl::queue& queue, matrix_handle_t* p_smhandle,
                                   std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                                   index_base index, sycl::buffer<indexType, 1> row_ind,
                                   sycl::buffer<indexType, 1> col_ind,
                                   sycl::buffer<dataType, 1> val);
template <typename dataType, typename indexType>
ONEMKL_EXPORT void init_coo_matrix(sycl::queue& queue, matrix_handle_t* p_smhandle,
                                   std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                                   index_base index, indexType* row_ind, indexType* col_ind,
                                   dataType* val);

template <typename dataType, typename indexType>
ONEMKL_EXPORT void set_coo_matrix_data(sycl::queue& queue, matrix_handle_t smhandle,
                                       std::int64_t num_rows, std::int64_t num_cols,
                                       std::int64_t nnz, index_base index,
                                       sycl::buffer<indexType, 1> row_ind,
                                       sycl::buffer<indexType, 1> col_ind,
                                       sycl::buffer<dataType, 1> val);
template <typename dataType, typename indexType>
ONEMKL_EXPORT void set_coo_matrix_data(sycl::queue& queue, matrix_handle_t smhandle,
                                       std::int64_t num_rows, std::int64_t num_cols,
                                       std::int64_t nnz, index_base index, indexType* row_ind,
                                       indexType* col_ind, dataType* val);

// CSR matrix
template <typename dataType, typename indexType>
ONEMKL_EXPORT void init_csr_matrix(sycl::queue& queue, matrix_handle_t* p_smhandle,
                                   std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                                   index_base index, sycl::buffer<indexType, 1> row_ptr,
                                   sycl::buffer<indexType, 1> col_ind,
                                   sycl::buffer<dataType, 1> val);
template <typename dataType, typename indexType>
ONEMKL_EXPORT void init_csr_matrix(sycl::queue& queue, matrix_handle_t* p_smhandle,
                                   std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                                   index_base index, indexType* row_ptr, indexType* col_ind,
                                   dataType* val);

template <typename dataType, typename indexType>
ONEMKL_EXPORT void set_csr_matrix_data(sycl::queue& queue, matrix_handle_t smhandle,
                                       std::int64_t num_rows, std::int64_t num_cols,
                                       std::int64_t nnz, index_base index,
                                       sycl::buffer<indexType, 1> row_ptr,
                                       sycl::buffer<indexType, 1> col_ind,
                                       sycl::buffer<dataType, 1> val);
template <typename dataType, typename indexType>
ONEMKL_EXPORT void set_csr_matrix_data(sycl::queue& queue, matrix_handle_t smhandle,
                                       std::int64_t num_rows, std::int64_t num_cols,
                                       std::int64_t nnz, index_base index, indexType* row_ptr,
                                       indexType* col_ind, dataType* val);

// Common sparse matrix functions
ONEMKL_EXPORT sycl::event release_sparse_matrix(sycl::queue& queue, matrix_handle_t smhandle,
                                                const std::vector<sycl::event>& dependencies = {});

bool set_matrix_property(sycl::queue& queue, matrix_handle_t smhandle, matrix_property property);

// SPMM
ONEMKL_EXPORT void init_spmm_descr(sycl::queue& queue, spmm_descr_t* p_spmm_descr);

ONEMKL_EXPORT sycl::event release_spmm_descr(sycl::queue& queue, spmm_descr_t spmm_descr,
                                             const std::vector<sycl::event>& dependencies = {});

ONEMKL_EXPORT void spmm_buffer_size(sycl::queue& queue, oneapi::mkl::transpose opA,
                                    oneapi::mkl::transpose opB, const void* alpha,
                                    matrix_view A_view, matrix_handle_t A_handle,
                                    dense_matrix_handle_t B_handle, const void* beta,
                                    dense_matrix_handle_t C_handle, spmm_alg alg,
                                    spmm_descr_t spmm_descr, std::size_t& temp_buffer_size);

ONEMKL_EXPORT void spmm_optimize(sycl::queue& queue, oneapi::mkl::transpose opA,
                                 oneapi::mkl::transpose opB, const void* alpha, matrix_view A_view,
                                 matrix_handle_t A_handle, dense_matrix_handle_t B_handle,
                                 const void* beta, dense_matrix_handle_t C_handle, spmm_alg alg,
                                 spmm_descr_t spmm_descr, sycl::buffer<std::uint8_t, 1> workspace);

ONEMKL_EXPORT sycl::event spmm_optimize(sycl::queue& queue, oneapi::mkl::transpose opA,
                                        oneapi::mkl::transpose opB, const void* alpha,
                                        matrix_view A_view, matrix_handle_t A_handle,
                                        dense_matrix_handle_t B_handle, const void* beta,
                                        dense_matrix_handle_t C_handle, spmm_alg alg,
                                        spmm_descr_t spmm_descr, void* workspace,
                                        const std::vector<sycl::event>& dependencies = {});

ONEMKL_EXPORT sycl::event spmm(sycl::queue& queue, oneapi::mkl::transpose opA,
                               oneapi::mkl::transpose opB, const void* alpha, matrix_view A_view,
                               matrix_handle_t A_handle, dense_matrix_handle_t B_handle,
                               const void* beta, dense_matrix_handle_t C_handle, spmm_alg alg,
                               spmm_descr_t spmm_descr,
                               const std::vector<sycl::event>& dependencies = {});

// SPMV
ONEMKL_EXPORT void init_spmv_descr(sycl::queue& queue, spmv_descr_t* p_spmv_descr);

ONEMKL_EXPORT sycl::event release_spmv_descr(sycl::queue& queue, spmv_descr_t spmv_descr,
                                             const std::vector<sycl::event>& dependencies = {});

ONEMKL_EXPORT void spmv_buffer_size(sycl::queue& queue, oneapi::mkl::transpose opA,
                                    const void* alpha, matrix_view A_view, matrix_handle_t A_handle,
                                    dense_vector_handle_t x_handle, const void* beta,
                                    dense_vector_handle_t y_handle, spmv_alg alg,
                                    spmv_descr_t spmv_descr, std::size_t& temp_buffer_size);

ONEMKL_EXPORT void spmv_optimize(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                                 matrix_view A_view, matrix_handle_t A_handle,
                                 dense_vector_handle_t x_handle, const void* beta,
                                 dense_vector_handle_t y_handle, spmv_alg alg,
                                 spmv_descr_t spmv_descr, sycl::buffer<std::uint8_t, 1> workspace);

ONEMKL_EXPORT sycl::event spmv_optimize(sycl::queue& queue, oneapi::mkl::transpose opA,
                                        const void* alpha, matrix_view A_view,
                                        matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                                        const void* beta, dense_vector_handle_t y_handle,
                                        spmv_alg alg, spmv_descr_t spmv_descr, void* workspace,
                                        const std::vector<sycl::event>& dependencies = {});

ONEMKL_EXPORT sycl::event spmv(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                               matrix_view A_view, matrix_handle_t A_handle,
                               dense_vector_handle_t x_handle, const void* beta,
                               dense_vector_handle_t y_handle, spmv_alg alg,
                               spmv_descr_t spmv_descr,
                               const std::vector<sycl::event>& dependencies = {});

// SPSV
ONEMKL_EXPORT void init_spsv_descr(sycl::queue& queue, spsv_descr_t* p_spsv_descr);

ONEMKL_EXPORT sycl::event release_spsv_descr(sycl::queue& queue, spsv_descr_t spsv_descr,
                                             const std::vector<sycl::event>& dependencies = {});

ONEMKL_EXPORT void spsv_buffer_size(sycl::queue& queue, oneapi::mkl::transpose opA,
                                    const void* alpha, matrix_view A_view, matrix_handle_t A_handle,
                                    dense_vector_handle_t x_handle, dense_vector_handle_t y_handle,
                                    spsv_alg alg, spsv_descr_t spsv_descr,
                                    std::size_t& temp_buffer_size);

ONEMKL_EXPORT void spsv_optimize(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                                 matrix_view A_view, matrix_handle_t A_handle,
                                 dense_vector_handle_t x_handle, dense_vector_handle_t y_handle,
                                 spsv_alg alg, spsv_descr_t spsv_descr,
                                 sycl::buffer<std::uint8_t, 1> workspace);

ONEMKL_EXPORT sycl::event spsv_optimize(sycl::queue& queue, oneapi::mkl::transpose opA,
                                        const void* alpha, matrix_view A_view,
                                        matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                                        dense_vector_handle_t y_handle, spsv_alg alg,
                                        spsv_descr_t spsv_descr, void* workspace,
                                        const std::vector<sycl::event>& dependencies = {});

ONEMKL_EXPORT sycl::event spsv(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                               matrix_view A_view, matrix_handle_t A_handle,
                               dense_vector_handle_t x_handle, dense_vector_handle_t y_handle,
                               spsv_alg alg, spsv_descr_t spsv_descr,
                               const std::vector<sycl::event>& dependencies = {});
