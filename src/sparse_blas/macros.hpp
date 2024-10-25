/*******************************************************************************
* Copyright 2023 Codeplay Software Ltd.
*
* (*Licensed under the Apache License, Version 2.0 )(the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#ifndef _ONEMKL_SPARSE_BLAS_MACROS_HPP_
#define _ONEMKL_SPARSE_BLAS_MACROS_HPP_

#define FOR_EACH_FP_TYPE(DEFINE_MACRO)      \
    DEFINE_MACRO(float, _rf);               \
    DEFINE_MACRO(double, _rd);              \
    DEFINE_MACRO(std::complex<float>, _cf); \
    DEFINE_MACRO(std::complex<double>, _cd)

#define FOR_EACH_FP_AND_INT_TYPE_HELPER(DEFINE_MACRO, INT_TYPE, INT_SUFFIX) \
    DEFINE_MACRO(float, _rf, INT_TYPE, INT_SUFFIX);                         \
    DEFINE_MACRO(double, _rd, INT_TYPE, INT_SUFFIX);                        \
    DEFINE_MACRO(std::complex<float>, _cf, INT_TYPE, INT_SUFFIX);           \
    DEFINE_MACRO(std::complex<double>, _cd, INT_TYPE, INT_SUFFIX)

#define FOR_EACH_FP_AND_INT_TYPE(DEFINE_MACRO)                         \
    FOR_EACH_FP_AND_INT_TYPE_HELPER(DEFINE_MACRO, std::int32_t, _i32); \
    FOR_EACH_FP_AND_INT_TYPE_HELPER(DEFINE_MACRO, std::int64_t, _i64)

#define INSTANTIATE_DENSE_VECTOR_FUNCS(FP_TYPE, FP_SUFFIX)                            \
    template void init_dense_vector<FP_TYPE>(                                         \
        sycl::queue & queue, oneapi::mkl::sparse::dense_vector_handle_t * p_dvhandle, \
        std::int64_t size, sycl::buffer<FP_TYPE, 1> val);                             \
    template void init_dense_vector<FP_TYPE>(                                         \
        sycl::queue & queue, oneapi::mkl::sparse::dense_vector_handle_t * p_dvhandle, \
        std::int64_t size, FP_TYPE* val);                                             \
    template void set_dense_vector_data<FP_TYPE>(                                     \
        sycl::queue & queue, oneapi::mkl::sparse::dense_vector_handle_t dvhandle,     \
        std::int64_t size, sycl::buffer<FP_TYPE, 1> val);                             \
    template void set_dense_vector_data<FP_TYPE>(                                     \
        sycl::queue & queue, oneapi::mkl::sparse::dense_vector_handle_t dvhandle,     \
        std::int64_t size, FP_TYPE* val)

#define INSTANTIATE_DENSE_MATRIX_FUNCS(FP_TYPE, FP_SUFFIX)                            \
    template void init_dense_matrix<FP_TYPE>(                                         \
        sycl::queue & queue, oneapi::mkl::sparse::dense_matrix_handle_t * p_dmhandle, \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,                \
        oneapi::mkl::layout dense_layout, sycl::buffer<FP_TYPE, 1> val);              \
    template void init_dense_matrix<FP_TYPE>(                                         \
        sycl::queue & queue, oneapi::mkl::sparse::dense_matrix_handle_t * p_dmhandle, \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,                \
        oneapi::mkl::layout dense_layout, FP_TYPE* val);                              \
    template void set_dense_matrix_data<FP_TYPE>(                                     \
        sycl::queue & queue, oneapi::mkl::sparse::dense_matrix_handle_t dmhandle,     \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,                \
        oneapi::mkl::layout dense_layout, sycl::buffer<FP_TYPE, 1> val);              \
    template void set_dense_matrix_data<FP_TYPE>(                                     \
        sycl::queue & queue, oneapi::mkl::sparse::dense_matrix_handle_t dmhandle,     \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,                \
        oneapi::mkl::layout dense_layout, FP_TYPE* val)

#define INSTANTIATE_COO_MATRIX_FUNCS(FP_TYPE, FP_SUFFIX, INT_TYPE, INT_SUFFIX)                     \
    template void init_coo_matrix<FP_TYPE, INT_TYPE>(                                              \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t * p_smhandle,                    \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,                            \
        oneapi::mkl::index_base index, sycl::buffer<INT_TYPE, 1> row_ind,                          \
        sycl::buffer<INT_TYPE, 1> col_ind, sycl::buffer<FP_TYPE, 1> val);                          \
    template void init_coo_matrix<FP_TYPE, INT_TYPE>(                                              \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t * p_smhandle,                    \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,                            \
        oneapi::mkl::index_base index, INT_TYPE* row_ind, INT_TYPE* col_ind, FP_TYPE* val);        \
    template void set_coo_matrix_data<FP_TYPE, INT_TYPE>(                                          \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t smhandle, std::int64_t num_rows, \
        std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index,                    \
        sycl::buffer<INT_TYPE, 1> row_ind, sycl::buffer<INT_TYPE, 1> col_ind,                      \
        sycl::buffer<FP_TYPE, 1> val);                                                             \
    template void set_coo_matrix_data<FP_TYPE, INT_TYPE>(                                          \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t smhandle, std::int64_t num_rows, \
        std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index, INT_TYPE* row_ind, \
        INT_TYPE* col_ind, FP_TYPE* val)

#define INSTANTIATE_CSR_MATRIX_FUNCS(FP_TYPE, FP_SUFFIX, INT_TYPE, INT_SUFFIX)                     \
    template void init_csr_matrix<FP_TYPE, INT_TYPE>(                                              \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t * p_smhandle,                    \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,                            \
        oneapi::mkl::index_base index, sycl::buffer<INT_TYPE, 1> row_ptr,                          \
        sycl::buffer<INT_TYPE, 1> col_ind, sycl::buffer<FP_TYPE, 1> val);                          \
    template void init_csr_matrix<FP_TYPE, INT_TYPE>(                                              \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t * p_smhandle,                    \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,                            \
        oneapi::mkl::index_base index, INT_TYPE* row_ptr, INT_TYPE* col_ind, FP_TYPE* val);        \
    template void set_csr_matrix_data<FP_TYPE, INT_TYPE>(                                          \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t smhandle, std::int64_t num_rows, \
        std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index,                    \
        sycl::buffer<INT_TYPE, 1> row_ptr, sycl::buffer<INT_TYPE, 1> col_ind,                      \
        sycl::buffer<FP_TYPE, 1> val);                                                             \
    template void set_csr_matrix_data<FP_TYPE, INT_TYPE>(                                          \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t smhandle, std::int64_t num_rows, \
        std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index, INT_TYPE* row_ptr, \
        INT_TYPE* col_ind, FP_TYPE* val)

#define THROW_IF_NULLPTR(FUNC_NAME, PTR)                                       \
    if (!(PTR)) {                                                              \
        throw mkl::uninitialized("sparse_blas", FUNC_NAME,                     \
                                 std::string(#PTR) + " must not be nullptr."); \
    }

#define CHECK_DESCR_MATCH(descr, argument, optimize_func_name)                                    \
    do {                                                                                          \
        if (descr->last_optimized_##argument != argument) {                                       \
            throw mkl::invalid_argument(                                                          \
                "sparse_blas", __func__,                                                          \
                #argument " argument must match with the previous call to " #optimize_func_name); \
        }                                                                                         \
    } while (0)

#endif // _ONEMKL_SPARSE_BLAS_MACROS_HPP_
