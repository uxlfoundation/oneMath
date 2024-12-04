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

/*
This file lists functions matching those required by sparse_blas_function_table_t in
src/sparse_blas/function_table.hpp.

To use this:

#define WRAPPER_VERSION <Wrapper version number>
#define BACKEND         <Backend name eg. mklgpu>

extern "C" sparse_blas_function_table_t onemath_sparse_blas_table = {
    WRAPPER_VERSION,
#include "sparse_blas/backends/backend_wrappers.cxx"
};

Changes to this file should be matched to changes in sparse_blas/function_table.hpp. The required
function template instantiations must be added to backend_sparse_blas_instantiations.cxx.
*/

#define REPEAT_FOR_EACH_FP_TYPE(DEFINE_MACRO) \
    DEFINE_MACRO()                            \
    DEFINE_MACRO()                            \
    DEFINE_MACRO()                            \
    DEFINE_MACRO()

#define REPEAT_FOR_EACH_FP_AND_INT_TYPE(DEFINE_MACRO) \
    REPEAT_FOR_EACH_FP_TYPE(DEFINE_MACRO)             \
    REPEAT_FOR_EACH_FP_TYPE(DEFINE_MACRO)

// clang-format off
// Dense vector
#define LIST_DENSE_VECTOR_FUNCS() \
oneapi::math::sparse::BACKEND::init_dense_vector, \
oneapi::math::sparse::BACKEND::init_dense_vector, \
oneapi::math::sparse::BACKEND::set_dense_vector_data, \
oneapi::math::sparse::BACKEND::set_dense_vector_data,
REPEAT_FOR_EACH_FP_TYPE(LIST_DENSE_VECTOR_FUNCS)
#undef LIST_DENSE_VECTOR_FUNCS
oneapi::math::sparse::BACKEND::release_dense_vector,

// Dense matrix
#define LIST_DENSE_MATRIX_FUNCS() \
oneapi::math::sparse::BACKEND::init_dense_matrix, \
oneapi::math::sparse::BACKEND::init_dense_matrix, \
oneapi::math::sparse::BACKEND::set_dense_matrix_data, \
oneapi::math::sparse::BACKEND::set_dense_matrix_data,
REPEAT_FOR_EACH_FP_TYPE(LIST_DENSE_MATRIX_FUNCS)
#undef LIST_DENSE_MATRIX_FUNCS
oneapi::math::sparse::BACKEND::release_dense_matrix,

// COO matrix
#define LIST_COO_MATRIX_FUNCS() \
oneapi::math::sparse::BACKEND::init_coo_matrix, \
oneapi::math::sparse::BACKEND::init_coo_matrix, \
oneapi::math::sparse::BACKEND::set_coo_matrix_data, \
oneapi::math::sparse::BACKEND::set_coo_matrix_data,
REPEAT_FOR_EACH_FP_AND_INT_TYPE(LIST_COO_MATRIX_FUNCS)
#undef LIST_COO_MATRIX_FUNCS

// CSR matrix
#define LIST_CSR_MATRIX_FUNCS() \
oneapi::math::sparse::BACKEND::init_csr_matrix, \
oneapi::math::sparse::BACKEND::init_csr_matrix, \
oneapi::math::sparse::BACKEND::set_csr_matrix_data, \
oneapi::math::sparse::BACKEND::set_csr_matrix_data,
REPEAT_FOR_EACH_FP_AND_INT_TYPE(LIST_CSR_MATRIX_FUNCS)
#undef LIST_CSR_MATRIX_FUNCS

// Common sparse matrix functions
oneapi::math::sparse::BACKEND::release_sparse_matrix,
oneapi::math::sparse::BACKEND::set_matrix_property,

// SPMM
oneapi::math::sparse::BACKEND::init_spmm_descr,
oneapi::math::sparse::BACKEND::release_spmm_descr,
oneapi::math::sparse::BACKEND::spmm_buffer_size,
oneapi::math::sparse::BACKEND::spmm_optimize,
oneapi::math::sparse::BACKEND::spmm_optimize,
oneapi::math::sparse::BACKEND::spmm,

// SPMV
oneapi::math::sparse::BACKEND::init_spmv_descr,
oneapi::math::sparse::BACKEND::release_spmv_descr,
oneapi::math::sparse::BACKEND::spmv_buffer_size,
oneapi::math::sparse::BACKEND::spmv_optimize,
oneapi::math::sparse::BACKEND::spmv_optimize,
oneapi::math::sparse::BACKEND::spmv,

// SPSV
oneapi::math::sparse::BACKEND::init_spsv_descr,
oneapi::math::sparse::BACKEND::release_spsv_descr,
oneapi::math::sparse::BACKEND::spsv_buffer_size,
oneapi::math::sparse::BACKEND::spsv_optimize,
oneapi::math::sparse::BACKEND::spsv_optimize,
oneapi::math::sparse::BACKEND::spsv,

    // clang-format on
