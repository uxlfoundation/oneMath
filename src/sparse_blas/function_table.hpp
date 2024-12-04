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

#ifndef _ONEMATH_SPARSE_BLAS_FUNCTION_TABLE_HPP_
#define _ONEMATH_SPARSE_BLAS_FUNCTION_TABLE_HPP_

#include "oneapi/math/sparse_blas/types.hpp"
#include "sparse_blas/macros.hpp"

// Dense vector
#define DEFINE_DENSE_VECTOR_FUNCS(FP_TYPE, FP_SUFFIX)                                  \
    void (*init_dense_vector_buffer##FP_SUFFIX)(                                       \
        sycl::queue & queue, oneapi::math::sparse::dense_vector_handle_t * p_dvhandle, \
        std::int64_t size, sycl::buffer<FP_TYPE, 1> val);                              \
    void (*init_dense_vector_usm##FP_SUFFIX)(                                          \
        sycl::queue & queue, oneapi::math::sparse::dense_vector_handle_t * p_dvhandle, \
        std::int64_t size, FP_TYPE* val);                                              \
    void (*set_dense_vector_data_buffer##FP_SUFFIX)(                                   \
        sycl::queue & queue, oneapi::math::sparse::dense_vector_handle_t dvhandle,     \
        std::int64_t size, sycl::buffer<FP_TYPE, 1> val);                              \
    void (*set_dense_vector_data_usm##FP_SUFFIX)(                                      \
        sycl::queue & queue, oneapi::math::sparse::dense_vector_handle_t dvhandle,     \
        std::int64_t size, FP_TYPE* val)

// Dense matrix
#define DEFINE_DENSE_MATRIX_FUNCS(FP_TYPE, FP_SUFFIX)                                  \
    void (*init_dense_matrix_buffer##FP_SUFFIX)(                                       \
        sycl::queue & queue, oneapi::math::sparse::dense_matrix_handle_t * p_dmhandle, \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,                 \
        oneapi::math::layout dense_layout, sycl::buffer<FP_TYPE, 1> val);              \
    void (*init_dense_matrix_usm##FP_SUFFIX)(                                          \
        sycl::queue & queue, oneapi::math::sparse::dense_matrix_handle_t * p_dmhandle, \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,                 \
        oneapi::math::layout dense_layout, FP_TYPE* val);                              \
    void (*set_dense_matrix_data_buffer##FP_SUFFIX)(                                   \
        sycl::queue & queue, oneapi::math::sparse::dense_matrix_handle_t dmhandle,     \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,                 \
        oneapi::math::layout dense_layout, sycl::buffer<FP_TYPE, 1> val);              \
    void (*set_dense_matrix_data_usm##FP_SUFFIX)(                                      \
        sycl::queue & queue, oneapi::math::sparse::dense_matrix_handle_t dmhandle,     \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,                 \
        oneapi::math::layout dense_layout, FP_TYPE* val)

// COO matrix
#define DEFINE_COO_MATRIX_FUNCS(FP_TYPE, FP_SUFFIX, INT_TYPE, INT_SUFFIX)                    \
    void (*init_coo_matrix_buffer##FP_SUFFIX##INT_SUFFIX)(                                   \
        sycl::queue & queue, oneapi::math::sparse::matrix_handle_t * p_smhandle,             \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,                      \
        oneapi::math::index_base index, sycl::buffer<INT_TYPE, 1> row_ind,                   \
        sycl::buffer<INT_TYPE, 1> col_ind, sycl::buffer<FP_TYPE, 1> val);                    \
    void (*init_coo_matrix_usm##FP_SUFFIX##INT_SUFFIX)(                                      \
        sycl::queue & queue, oneapi::math::sparse::matrix_handle_t * p_smhandle,             \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,                      \
        oneapi::math::index_base index, INT_TYPE* row_ind, INT_TYPE* col_ind, FP_TYPE* val); \
    void (*set_coo_matrix_data_buffer##FP_SUFFIX##INT_SUFFIX)(                               \
        sycl::queue & queue, oneapi::math::sparse::matrix_handle_t smhandle,                 \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,                      \
        oneapi::math::index_base index, sycl::buffer<INT_TYPE, 1> row_ind,                   \
        sycl::buffer<INT_TYPE, 1> col_ind, sycl::buffer<FP_TYPE, 1> val);                    \
    void (*set_coo_matrix_data_usm##FP_SUFFIX##INT_SUFFIX)(                                  \
        sycl::queue & queue, oneapi::math::sparse::matrix_handle_t smhandle,                 \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,                      \
        oneapi::math::index_base index, INT_TYPE* row_ind, INT_TYPE* col_ind, FP_TYPE* val)

// CSR matrix
#define DEFINE_CSR_MATRIX_FUNCS(FP_TYPE, FP_SUFFIX, INT_TYPE, INT_SUFFIX)                    \
    void (*init_csr_matrix_buffer##FP_SUFFIX##INT_SUFFIX)(                                   \
        sycl::queue & queue, oneapi::math::sparse::matrix_handle_t * p_smhandle,             \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,                      \
        oneapi::math::index_base index, sycl::buffer<INT_TYPE, 1> row_ptr,                   \
        sycl::buffer<INT_TYPE, 1> col_ind, sycl::buffer<FP_TYPE, 1> val);                    \
    void (*init_csr_matrix_usm##FP_SUFFIX##INT_SUFFIX)(                                      \
        sycl::queue & queue, oneapi::math::sparse::matrix_handle_t * p_smhandle,             \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,                      \
        oneapi::math::index_base index, INT_TYPE* row_ptr, INT_TYPE* col_ind, FP_TYPE* val); \
    void (*set_csr_matrix_data_buffer##FP_SUFFIX##INT_SUFFIX)(                               \
        sycl::queue & queue, oneapi::math::sparse::matrix_handle_t smhandle,                 \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,                      \
        oneapi::math::index_base index, sycl::buffer<INT_TYPE, 1> row_ptr,                   \
        sycl::buffer<INT_TYPE, 1> col_ind, sycl::buffer<FP_TYPE, 1> val);                    \
    void (*set_csr_matrix_data_usm##FP_SUFFIX##INT_SUFFIX)(                                  \
        sycl::queue & queue, oneapi::math::sparse::matrix_handle_t smhandle,                 \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,                      \
        oneapi::math::index_base index, INT_TYPE* row_ptr, INT_TYPE* col_ind, FP_TYPE* val)

typedef struct {
    int version;

    // Dense vector
    FOR_EACH_FP_TYPE(DEFINE_DENSE_VECTOR_FUNCS);
    sycl::event (*release_dense_vector)(sycl::queue& queue,
                                        oneapi::math::sparse::dense_vector_handle_t dvhandle,
                                        const std::vector<sycl::event>& dependencies);

    // Dense matrix
    FOR_EACH_FP_TYPE(DEFINE_DENSE_MATRIX_FUNCS);
    sycl::event (*release_dense_matrix)(sycl::queue& queue,
                                        oneapi::math::sparse::dense_matrix_handle_t dmhandle,
                                        const std::vector<sycl::event>& dependencies);

    // COO matrix
    FOR_EACH_FP_AND_INT_TYPE(DEFINE_COO_MATRIX_FUNCS);

    // CSR matrix
    FOR_EACH_FP_AND_INT_TYPE(DEFINE_CSR_MATRIX_FUNCS);

    // Common sparse matrix functions
    sycl::event (*release_sparse_matrix)(sycl::queue& queue,
                                         oneapi::math::sparse::matrix_handle_t smhandle,
                                         const std::vector<sycl::event>& dependencies);

    bool (*set_matrix_property)(sycl::queue& queue, oneapi::math::sparse::matrix_handle_t smhandle,
                                oneapi::math::sparse::matrix_property property);

    // SPMM
    void (*init_spmm_descr)(sycl::queue& queue, oneapi::math::sparse::spmm_descr_t* p_spmm_descr);

    sycl::event (*release_spmm_descr)(sycl::queue& queue,
                                      oneapi::math::sparse::spmm_descr_t spmm_descr,
                                      const std::vector<sycl::event>& dependencies);

    void (*spmm_buffer_size)(sycl::queue& queue, oneapi::math::transpose opA,
                             oneapi::math::transpose opB, const void* alpha,
                             oneapi::math::sparse::matrix_view A_view,
                             oneapi::math::sparse::matrix_handle_t A_handle,
                             oneapi::math::sparse::dense_matrix_handle_t B_handle, const void* beta,
                             oneapi::math::sparse::dense_matrix_handle_t C_handle,
                             oneapi::math::sparse::spmm_alg alg,
                             oneapi::math::sparse::spmm_descr_t spmm_descr,
                             std::size_t& temp_buffer_size);

    void (*spmm_optimize_buffer)(
        sycl::queue& queue, oneapi::math::transpose opA, oneapi::math::transpose opB,
        const void* alpha, oneapi::math::sparse::matrix_view A_view,
        oneapi::math::sparse::matrix_handle_t A_handle,
        oneapi::math::sparse::dense_matrix_handle_t B_handle, const void* beta,
        oneapi::math::sparse::dense_matrix_handle_t C_handle, oneapi::math::sparse::spmm_alg alg,
        oneapi::math::sparse::spmm_descr_t spmm_descr, sycl::buffer<std::uint8_t, 1> workspace);

    sycl::event (*spmm_optimize_usm)(sycl::queue& queue, oneapi::math::transpose opA,
                                     oneapi::math::transpose opB, const void* alpha,
                                     oneapi::math::sparse::matrix_view A_view,
                                     oneapi::math::sparse::matrix_handle_t A_handle,
                                     oneapi::math::sparse::dense_matrix_handle_t B_handle,
                                     const void* beta,
                                     oneapi::math::sparse::dense_matrix_handle_t C_handle,
                                     oneapi::math::sparse::spmm_alg alg,
                                     oneapi::math::sparse::spmm_descr_t spmm_descr, void* workspace,
                                     const std::vector<sycl::event>& dependencies);

    sycl::event (*spmm)(sycl::queue& queue, oneapi::math::transpose opA,
                        oneapi::math::transpose opB, const void* alpha,
                        oneapi::math::sparse::matrix_view A_view,
                        oneapi::math::sparse::matrix_handle_t A_handle,
                        oneapi::math::sparse::dense_matrix_handle_t B_handle, const void* beta,
                        oneapi::math::sparse::dense_matrix_handle_t C_handle,
                        oneapi::math::sparse::spmm_alg alg,
                        oneapi::math::sparse::spmm_descr_t spmm_descr,
                        const std::vector<sycl::event>& dependencies);

    // SPMV
    void (*init_spmv_descr)(sycl::queue& queue, oneapi::math::sparse::spmv_descr_t* p_spmv_descr);

    sycl::event (*release_spmv_descr)(sycl::queue& queue,
                                      oneapi::math::sparse::spmv_descr_t spmv_descr,
                                      const std::vector<sycl::event>& dependencies);

    void (*spmv_buffer_size)(sycl::queue& queue, oneapi::math::transpose opA, const void* alpha,
                             oneapi::math::sparse::matrix_view A_view,
                             oneapi::math::sparse::matrix_handle_t A_handle,
                             oneapi::math::sparse::dense_vector_handle_t x_handle, const void* beta,
                             oneapi::math::sparse::dense_vector_handle_t y_handle,
                             oneapi::math::sparse::spmv_alg alg,
                             oneapi::math::sparse::spmv_descr_t spmv_descr,
                             std::size_t& temp_buffer_size);

    void (*spmv_optimize_buffer)(
        sycl::queue& queue, oneapi::math::transpose opA, const void* alpha,
        oneapi::math::sparse::matrix_view A_view, oneapi::math::sparse::matrix_handle_t A_handle,
        oneapi::math::sparse::dense_vector_handle_t x_handle, const void* beta,
        oneapi::math::sparse::dense_vector_handle_t y_handle, oneapi::math::sparse::spmv_alg alg,
        oneapi::math::sparse::spmv_descr_t spmv_descr, sycl::buffer<std::uint8_t, 1> workspace);

    sycl::event (*spmv_optimize_usm)(sycl::queue& queue, oneapi::math::transpose opA,
                                     const void* alpha, oneapi::math::sparse::matrix_view A_view,
                                     oneapi::math::sparse::matrix_handle_t A_handle,
                                     oneapi::math::sparse::dense_vector_handle_t x_handle,
                                     const void* beta,
                                     oneapi::math::sparse::dense_vector_handle_t y_handle,
                                     oneapi::math::sparse::spmv_alg alg,
                                     oneapi::math::sparse::spmv_descr_t spmv_descr, void* workspace,
                                     const std::vector<sycl::event>& dependencies);

    sycl::event (*spmv)(sycl::queue& queue, oneapi::math::transpose opA, const void* alpha,
                        oneapi::math::sparse::matrix_view A_view,
                        oneapi::math::sparse::matrix_handle_t A_handle,
                        oneapi::math::sparse::dense_vector_handle_t x_handle, const void* beta,
                        oneapi::math::sparse::dense_vector_handle_t y_handle,
                        oneapi::math::sparse::spmv_alg alg,
                        oneapi::math::sparse::spmv_descr_t spmv_descr,
                        const std::vector<sycl::event>& dependencies);

    // SPSV
    void (*init_spsv_descr)(sycl::queue& queue, oneapi::math::sparse::spsv_descr_t* p_spsv_descr);

    sycl::event (*release_spsv_descr)(sycl::queue& queue,
                                      oneapi::math::sparse::spsv_descr_t spsv_descr,
                                      const std::vector<sycl::event>& dependencies);

    void (*spsv_buffer_size)(sycl::queue& queue, oneapi::math::transpose opA, const void* alpha,
                             oneapi::math::sparse::matrix_view A_view,
                             oneapi::math::sparse::matrix_handle_t A_handle,
                             oneapi::math::sparse::dense_vector_handle_t x_handle,
                             oneapi::math::sparse::dense_vector_handle_t y_handle,
                             oneapi::math::sparse::spsv_alg alg,
                             oneapi::math::sparse::spsv_descr_t spsv_descr,
                             std::size_t& temp_buffer_size);

    void (*spsv_optimize_buffer)(sycl::queue& queue, oneapi::math::transpose opA, const void* alpha,
                                 oneapi::math::sparse::matrix_view A_view,
                                 oneapi::math::sparse::matrix_handle_t A_handle,
                                 oneapi::math::sparse::dense_vector_handle_t x_handle,
                                 oneapi::math::sparse::dense_vector_handle_t y_handle,
                                 oneapi::math::sparse::spsv_alg alg,
                                 oneapi::math::sparse::spsv_descr_t spsv_descr,
                                 sycl::buffer<std::uint8_t, 1> workspace);

    sycl::event (*spsv_optimize_usm)(sycl::queue& queue, oneapi::math::transpose opA,
                                     const void* alpha, oneapi::math::sparse::matrix_view A_view,
                                     oneapi::math::sparse::matrix_handle_t A_handle,
                                     oneapi::math::sparse::dense_vector_handle_t x_handle,
                                     oneapi::math::sparse::dense_vector_handle_t y_handle,
                                     oneapi::math::sparse::spsv_alg alg,
                                     oneapi::math::sparse::spsv_descr_t spsv_descr, void* workspace,
                                     const std::vector<sycl::event>& dependencies);

    sycl::event (*spsv)(sycl::queue& queue, oneapi::math::transpose opA, const void* alpha,
                        oneapi::math::sparse::matrix_view A_view,
                        oneapi::math::sparse::matrix_handle_t A_handle,
                        oneapi::math::sparse::dense_vector_handle_t x_handle,
                        oneapi::math::sparse::dense_vector_handle_t y_handle,
                        oneapi::math::sparse::spsv_alg alg,
                        oneapi::math::sparse::spsv_descr_t spsv_descr,
                        const std::vector<sycl::event>& dependencies);
} sparse_blas_function_table_t;

#undef DEFINE_DENSE_VECTOR_FUNCS
#undef DEFINE_DENSE_MATRIX_FUNCS
#undef DEFINE_COO_MATRIX_FUNCS
#undef DEFINE_CSR_MATRIX_FUNCS

#endif // _ONEMATH_SPARSE_BLAS_FUNCTION_TABLE_HPP_
