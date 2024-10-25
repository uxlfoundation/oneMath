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

#include "oneapi/mkl/sparse_blas/detail/sparse_blas_rt.hpp"

#include "function_table_initializer.hpp"
#include "sparse_blas/function_table.hpp"
#include "sparse_blas/macros.hpp"
#include "oneapi/mkl/detail/get_device_id.hpp"

namespace oneapi::mkl::sparse {

static oneapi::mkl::detail::table_initializer<mkl::domain::sparse_blas,
                                              sparse_blas_function_table_t>
    function_tables;

// Dense vector
#define DEFINE_DENSE_VECTOR_FUNCS(FP_TYPE, FP_SUFFIX)                                             \
    template <>                                                                                   \
    void init_dense_vector(sycl::queue& queue, dense_vector_handle_t* p_dvhandle,                 \
                           std::int64_t size, sycl::buffer<FP_TYPE, 1> val) {                     \
        auto libkey = get_device_id(queue);                                                       \
        function_tables[{ libkey, queue }].init_dense_vector_buffer##FP_SUFFIX(queue, p_dvhandle, \
                                                                               size, val);        \
    }                                                                                             \
    template <>                                                                                   \
    void init_dense_vector(sycl::queue& queue, dense_vector_handle_t* p_dvhandle,                 \
                           std::int64_t size, FP_TYPE* val) {                                     \
        auto libkey = get_device_id(queue);                                                       \
        function_tables[{ libkey, queue }].init_dense_vector_usm##FP_SUFFIX(queue, p_dvhandle,    \
                                                                            size, val);           \
    }                                                                                             \
    template <>                                                                                   \
    void set_dense_vector_data(sycl::queue& queue, dense_vector_handle_t dvhandle,                \
                               std::int64_t size, sycl::buffer<FP_TYPE, 1> val) {                 \
        auto libkey = get_device_id(queue);                                                       \
        function_tables[{ libkey, queue }].set_dense_vector_data_buffer##FP_SUFFIX(               \
            queue, dvhandle, size, val);                                                          \
    }                                                                                             \
    template <>                                                                                   \
    void set_dense_vector_data(sycl::queue& queue, dense_vector_handle_t dvhandle,                \
                               std::int64_t size, FP_TYPE* val) {                                 \
        auto libkey = get_device_id(queue);                                                       \
        function_tables[{ libkey, queue }].set_dense_vector_data_usm##FP_SUFFIX(queue, dvhandle,  \
                                                                                size, val);       \
    }
FOR_EACH_FP_TYPE(DEFINE_DENSE_VECTOR_FUNCS);
#undef DEFINE_DENSE_VECTOR_FUNCS

sycl::event release_dense_vector(sycl::queue& queue, dense_vector_handle_t dvhandle,
                                 const std::vector<sycl::event>& dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[{ libkey, queue }].release_dense_vector(queue, dvhandle, dependencies);
}

// Dense matrix
#define DEFINE_DENSE_MATRIX_FUNCS(FP_TYPE, FP_SUFFIX)                                         \
    template <>                                                                               \
    void init_dense_matrix(sycl::queue& queue, dense_matrix_handle_t* p_dmhandle,             \
                           std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,     \
                           layout dense_layout, sycl::buffer<FP_TYPE, 1> val) {               \
        auto libkey = get_device_id(queue);                                                   \
        function_tables[{ libkey, queue }].init_dense_matrix_buffer##FP_SUFFIX(               \
            queue, p_dmhandle, num_rows, num_cols, ld, dense_layout, val);                    \
    }                                                                                         \
    template <>                                                                               \
    void init_dense_matrix(sycl::queue& queue, dense_matrix_handle_t* p_dmhandle,             \
                           std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,     \
                           layout dense_layout, FP_TYPE* val) {                               \
        auto libkey = get_device_id(queue);                                                   \
        function_tables[{ libkey, queue }].init_dense_matrix_usm##FP_SUFFIX(                  \
            queue, p_dmhandle, num_rows, num_cols, ld, dense_layout, val);                    \
    }                                                                                         \
    template <>                                                                               \
    void set_dense_matrix_data(sycl::queue& queue, dense_matrix_handle_t dmhandle,            \
                               std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld, \
                               layout dense_layout, sycl::buffer<FP_TYPE, 1> val) {           \
        auto libkey = get_device_id(queue);                                                   \
        function_tables[{ libkey, queue }].set_dense_matrix_data_buffer##FP_SUFFIX(           \
            queue, dmhandle, num_rows, num_cols, ld, dense_layout, val);                      \
    }                                                                                         \
    template <>                                                                               \
    void set_dense_matrix_data(sycl::queue& queue, dense_matrix_handle_t dmhandle,            \
                               std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld, \
                               layout dense_layout, FP_TYPE* val) {                           \
        auto libkey = get_device_id(queue);                                                   \
        function_tables[{ libkey, queue }].set_dense_matrix_data_usm##FP_SUFFIX(              \
            queue, dmhandle, num_rows, num_cols, ld, dense_layout, val);                      \
    }
FOR_EACH_FP_TYPE(DEFINE_DENSE_MATRIX_FUNCS);
#undef DEFINE_DENSE_MATRIX_FUNCS

sycl::event release_dense_matrix(sycl::queue& queue, dense_matrix_handle_t dmhandle,
                                 const std::vector<sycl::event>& dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[{ libkey, queue }].release_dense_matrix(queue, dmhandle, dependencies);
}

// COO matrix
#define DEFINE_COO_MATRIX_FUNCS(FP_TYPE, FP_SUFFIX, INT_TYPE, INT_SUFFIX)                          \
    template <>                                                                                    \
    void init_coo_matrix(sycl::queue& queue, matrix_handle_t* p_smhandle, std::int64_t num_rows,   \
                         std::int64_t num_cols, std::int64_t nnz, index_base index,                \
                         sycl::buffer<INT_TYPE, 1> row_ind, sycl::buffer<INT_TYPE, 1> col_ind,     \
                         sycl::buffer<FP_TYPE, 1> val) {                                           \
        auto libkey = get_device_id(queue);                                                        \
        function_tables[{ libkey, queue }].init_coo_matrix_buffer##FP_SUFFIX##INT_SUFFIX(          \
            queue, p_smhandle, num_rows, num_cols, nnz, index, row_ind, col_ind, val);             \
    }                                                                                              \
    template <>                                                                                    \
    void init_coo_matrix(sycl::queue& queue, matrix_handle_t* p_smhandle, std::int64_t num_rows,   \
                         std::int64_t num_cols, std::int64_t nnz, index_base index,                \
                         INT_TYPE* row_ind, INT_TYPE* col_ind, FP_TYPE* val) {                     \
        auto libkey = get_device_id(queue);                                                        \
        function_tables[{ libkey, queue }].init_coo_matrix_usm##FP_SUFFIX##INT_SUFFIX(             \
            queue, p_smhandle, num_rows, num_cols, nnz, index, row_ind, col_ind, val);             \
    }                                                                                              \
    template <>                                                                                    \
    void set_coo_matrix_data(sycl::queue& queue, matrix_handle_t smhandle, std::int64_t num_rows,  \
                             std::int64_t num_cols, std::int64_t nnz, index_base index,            \
                             sycl::buffer<INT_TYPE, 1> row_ind, sycl::buffer<INT_TYPE, 1> col_ind, \
                             sycl::buffer<FP_TYPE, 1> val) {                                       \
        auto libkey = get_device_id(queue);                                                        \
        function_tables[{ libkey, queue }].set_coo_matrix_data_buffer##FP_SUFFIX##INT_SUFFIX(      \
            queue, smhandle, num_rows, num_cols, nnz, index, row_ind, col_ind, val);               \
    }                                                                                              \
    template <>                                                                                    \
    void set_coo_matrix_data(sycl::queue& queue, matrix_handle_t smhandle, std::int64_t num_rows,  \
                             std::int64_t num_cols, std::int64_t nnz, index_base index,            \
                             INT_TYPE* row_ind, INT_TYPE* col_ind, FP_TYPE* val) {                 \
        auto libkey = get_device_id(queue);                                                        \
        function_tables[{ libkey, queue }].set_coo_matrix_data_usm##FP_SUFFIX##INT_SUFFIX(         \
            queue, smhandle, num_rows, num_cols, nnz, index, row_ind, col_ind, val);               \
    }
FOR_EACH_FP_AND_INT_TYPE(DEFINE_COO_MATRIX_FUNCS);
#undef DEFINE_COO_MATRIX_FUNCS

// CSR matrix
#define DEFINE_INIT_CSR_MATRIX_FUNCS(FP_TYPE, FP_SUFFIX, INT_TYPE, INT_SUFFIX)                     \
    template <>                                                                                    \
    void init_csr_matrix(sycl::queue& queue, matrix_handle_t* p_smhandle, std::int64_t num_rows,   \
                         std::int64_t num_cols, std::int64_t nnz, index_base index,                \
                         sycl::buffer<INT_TYPE, 1> row_ptr, sycl::buffer<INT_TYPE, 1> col_ind,     \
                         sycl::buffer<FP_TYPE, 1> val) {                                           \
        auto libkey = get_device_id(queue);                                                        \
        function_tables[{ libkey, queue }].init_csr_matrix_buffer##FP_SUFFIX##INT_SUFFIX(          \
            queue, p_smhandle, num_rows, num_cols, nnz, index, row_ptr, col_ind, val);             \
    }                                                                                              \
    template <>                                                                                    \
    void init_csr_matrix(sycl::queue& queue, matrix_handle_t* p_smhandle, std::int64_t num_rows,   \
                         std::int64_t num_cols, std::int64_t nnz, index_base index,                \
                         INT_TYPE* row_ptr, INT_TYPE* col_ind, FP_TYPE* val) {                     \
        auto libkey = get_device_id(queue);                                                        \
        function_tables[{ libkey, queue }].init_csr_matrix_usm##FP_SUFFIX##INT_SUFFIX(             \
            queue, p_smhandle, num_rows, num_cols, nnz, index, row_ptr, col_ind, val);             \
    }                                                                                              \
    template <>                                                                                    \
    void set_csr_matrix_data(sycl::queue& queue, matrix_handle_t smhandle, std::int64_t num_rows,  \
                             std::int64_t num_cols, std::int64_t nnz, index_base index,            \
                             sycl::buffer<INT_TYPE, 1> row_ptr, sycl::buffer<INT_TYPE, 1> col_ind, \
                             sycl::buffer<FP_TYPE, 1> val) {                                       \
        auto libkey = get_device_id(queue);                                                        \
        function_tables[{ libkey, queue }].set_csr_matrix_data_buffer##FP_SUFFIX##INT_SUFFIX(      \
            queue, smhandle, num_rows, num_cols, nnz, index, row_ptr, col_ind, val);               \
    }                                                                                              \
    template <>                                                                                    \
    void set_csr_matrix_data(sycl::queue& queue, matrix_handle_t smhandle, std::int64_t num_rows,  \
                             std::int64_t num_cols, std::int64_t nnz, index_base index,            \
                             INT_TYPE* row_ptr, INT_TYPE* col_ind, FP_TYPE* val) {                 \
        auto libkey = get_device_id(queue);                                                        \
        function_tables[{ libkey, queue }].set_csr_matrix_data_usm##FP_SUFFIX##INT_SUFFIX(         \
            queue, smhandle, num_rows, num_cols, nnz, index, row_ptr, col_ind, val);               \
    }
FOR_EACH_FP_AND_INT_TYPE(DEFINE_INIT_CSR_MATRIX_FUNCS);
#undef DEFINE_INIT_CSR_MATRIX_FUNCS

// Common sparse matrix functions
sycl::event release_sparse_matrix(sycl::queue& queue, matrix_handle_t smhandle,
                                  const std::vector<sycl::event>& dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[{ libkey, queue }].release_sparse_matrix(queue, smhandle, dependencies);
}

bool set_matrix_property(sycl::queue& queue, matrix_handle_t smhandle, matrix_property property) {
    auto libkey = get_device_id(queue);
    return function_tables[{ libkey, queue }].set_matrix_property(queue, smhandle, property);
}

// SPMM
void init_spmm_descr(sycl::queue& queue, spmm_descr_t* p_spmm_descr) {
    auto libkey = get_device_id(queue);
    function_tables[{ libkey, queue }].init_spmm_descr(queue, p_spmm_descr);
}

sycl::event release_spmm_descr(sycl::queue& queue, spmm_descr_t spmm_descr,
                               const std::vector<sycl::event>& dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[{ libkey, queue }].release_spmm_descr(queue, spmm_descr, dependencies);
}

void spmm_buffer_size(sycl::queue& queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                      const void* alpha, matrix_view A_view, matrix_handle_t A_handle,
                      dense_matrix_handle_t B_handle, const void* beta,
                      dense_matrix_handle_t C_handle, spmm_alg alg, spmm_descr_t spmm_descr,
                      std::size_t& temp_buffer_size) {
    auto libkey = get_device_id(queue);
    function_tables[{ libkey, queue }].spmm_buffer_size(queue, opA, opB, alpha, A_view, A_handle,
                                                        B_handle, beta, C_handle, alg, spmm_descr,
                                                        temp_buffer_size);
}

void spmm_optimize(sycl::queue& queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                   const void* alpha, matrix_view A_view, matrix_handle_t A_handle,
                   dense_matrix_handle_t B_handle, const void* beta, dense_matrix_handle_t C_handle,
                   spmm_alg alg, spmm_descr_t spmm_descr, sycl::buffer<std::uint8_t, 1> workspace) {
    auto libkey = get_device_id(queue);
    function_tables[{ libkey, queue }].spmm_optimize_buffer(queue, opA, opB, alpha, A_view,
                                                            A_handle, B_handle, beta, C_handle, alg,
                                                            spmm_descr, workspace);
}

sycl::event spmm_optimize(sycl::queue& queue, oneapi::mkl::transpose opA,
                          oneapi::mkl::transpose opB, const void* alpha, matrix_view A_view,
                          matrix_handle_t A_handle, dense_matrix_handle_t B_handle,
                          const void* beta, dense_matrix_handle_t C_handle, spmm_alg alg,
                          spmm_descr_t spmm_descr, void* workspace,
                          const std::vector<sycl::event>& dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[{ libkey, queue }].spmm_optimize_usm(
        queue, opA, opB, alpha, A_view, A_handle, B_handle, beta, C_handle, alg, spmm_descr,
        workspace, dependencies);
}

sycl::event spmm(sycl::queue& queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                 const void* alpha, matrix_view A_view, matrix_handle_t A_handle,
                 dense_matrix_handle_t B_handle, const void* beta, dense_matrix_handle_t C_handle,
                 spmm_alg alg, spmm_descr_t spmm_descr,
                 const std::vector<sycl::event>& dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[{ libkey, queue }].spmm(queue, opA, opB, alpha, A_view, A_handle,
                                                   B_handle, beta, C_handle, alg, spmm_descr,
                                                   dependencies);
}

// SPMV
void init_spmv_descr(sycl::queue& queue, spmv_descr_t* p_spmv_descr) {
    auto libkey = get_device_id(queue);
    function_tables[{ libkey, queue }].init_spmv_descr(queue, p_spmv_descr);
}

sycl::event release_spmv_descr(sycl::queue& queue, spmv_descr_t spmv_descr,
                               const std::vector<sycl::event>& dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[{ libkey, queue }].release_spmv_descr(queue, spmv_descr, dependencies);
}

void spmv_buffer_size(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                      matrix_view A_view, matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                      const void* beta, dense_vector_handle_t y_handle, spmv_alg alg,
                      spmv_descr_t spmv_descr, std::size_t& temp_buffer_size) {
    auto libkey = get_device_id(queue);
    function_tables[{ libkey, queue }].spmv_buffer_size(queue, opA, alpha, A_view, A_handle,
                                                        x_handle, beta, y_handle, alg, spmv_descr,
                                                        temp_buffer_size);
}

void spmv_optimize(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                   matrix_view A_view, matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                   const void* beta, dense_vector_handle_t y_handle, spmv_alg alg,
                   spmv_descr_t spmv_descr, sycl::buffer<std::uint8_t, 1> workspace) {
    auto libkey = get_device_id(queue);
    function_tables[{ libkey, queue }].spmv_optimize_buffer(
        queue, opA, alpha, A_view, A_handle, x_handle, beta, y_handle, alg, spmv_descr, workspace);
}

sycl::event spmv_optimize(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                          matrix_view A_view, matrix_handle_t A_handle,
                          dense_vector_handle_t x_handle, const void* beta,
                          dense_vector_handle_t y_handle, spmv_alg alg, spmv_descr_t spmv_descr,
                          void* workspace, const std::vector<sycl::event>& dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[{ libkey, queue }].spmv_optimize_usm(
        queue, opA, alpha, A_view, A_handle, x_handle, beta, y_handle, alg, spmv_descr, workspace,
        dependencies);
}

sycl::event spmv(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                 matrix_view A_view, matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                 const void* beta, dense_vector_handle_t y_handle, spmv_alg alg,
                 spmv_descr_t spmv_descr, const std::vector<sycl::event>& dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[{ libkey, queue }].spmv(queue, opA, alpha, A_view, A_handle, x_handle,
                                                   beta, y_handle, alg, spmv_descr, dependencies);
}

// SPSV
void init_spsv_descr(sycl::queue& queue, spsv_descr_t* p_spsv_descr) {
    auto libkey = get_device_id(queue);
    function_tables[{ libkey, queue }].init_spsv_descr(queue, p_spsv_descr);
}

sycl::event release_spsv_descr(sycl::queue& queue, spsv_descr_t spsv_descr,
                               const std::vector<sycl::event>& dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[{ libkey, queue }].release_spsv_descr(queue, spsv_descr, dependencies);
}

void spsv_buffer_size(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                      matrix_view A_view, matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                      dense_vector_handle_t y_handle, spsv_alg alg, spsv_descr_t spsv_descr,
                      std::size_t& temp_buffer_size) {
    auto libkey = get_device_id(queue);
    function_tables[{ libkey, queue }].spsv_buffer_size(
        queue, opA, alpha, A_view, A_handle, x_handle, y_handle, alg, spsv_descr, temp_buffer_size);
}

void spsv_optimize(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                   matrix_view A_view, matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                   dense_vector_handle_t y_handle, spsv_alg alg, spsv_descr_t spsv_descr,
                   sycl::buffer<std::uint8_t, 1> workspace) {
    auto libkey = get_device_id(queue);
    function_tables[{ libkey, queue }].spsv_optimize_buffer(
        queue, opA, alpha, A_view, A_handle, x_handle, y_handle, alg, spsv_descr, workspace);
}

sycl::event spsv_optimize(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                          matrix_view A_view, matrix_handle_t A_handle,
                          dense_vector_handle_t x_handle, dense_vector_handle_t y_handle,
                          spsv_alg alg, spsv_descr_t spsv_descr, void* workspace,
                          const std::vector<sycl::event>& dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[{ libkey, queue }].spsv_optimize_usm(queue, opA, alpha, A_view, A_handle,
                                                                x_handle, y_handle, alg, spsv_descr,
                                                                workspace, dependencies);
}

sycl::event spsv(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                 matrix_view A_view, matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                 dense_vector_handle_t y_handle, spsv_alg alg, spsv_descr_t spsv_descr,
                 const std::vector<sycl::event>& dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[{ libkey, queue }].spsv(queue, opA, alpha, A_view, A_handle, x_handle,
                                                   y_handle, alg, spsv_descr, dependencies);
}

} // namespace oneapi::mkl::sparse
