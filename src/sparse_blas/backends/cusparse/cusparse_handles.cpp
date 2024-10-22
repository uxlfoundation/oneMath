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

#include "oneapi/mkl/sparse_blas/detail/cusparse/onemkl_sparse_blas_cusparse.hpp"

#include "cusparse_error.hpp"
#include "cusparse_helper.hpp"
#include "cusparse_handles.hpp"
#include "cusparse_task.hpp"
#include "sparse_blas/macros.hpp"

namespace oneapi::mkl::sparse::cusparse {

/**
 * In this file CusparseScopedContextHandler are used to ensure that a cusparseHandle_t is created before any other cuSPARSE call, as required by the specification.
*/

// Dense vector
template <typename fpType>
void init_dense_vector(sycl::queue &queue, dense_vector_handle_t *p_dvhandle, std::int64_t size,
                       sycl::buffer<fpType, 1> val) {
    auto event = queue.submit([&](sycl::handler &cgh) {
        auto acc = val.template get_access<sycl::access::mode::read_write>(cgh);
        submit_host_task(cgh, queue, [=](sycl::interop_handle ih) {
            // Ensure that a cusparse handle is created before any other cuSPARSE function is called.
            CusparseScopedContextHandler(queue, ih).get_handle(queue);
            auto cuda_value_type = CudaEnumType<fpType>::value;
            cusparseDnVecDescr_t cu_dvhandle;
            CUSPARSE_ERR_FUNC(cusparseCreateDnVec, &cu_dvhandle, size, get_mem(ih, acc),
                              cuda_value_type);
            *p_dvhandle = new dense_vector_handle(cu_dvhandle, val, size);
        });
    });
    event.wait_and_throw();
}

template <typename fpType>
void init_dense_vector(sycl::queue &queue, dense_vector_handle_t *p_dvhandle, std::int64_t size,
                       fpType *val) {
    auto event = queue.submit([&](sycl::handler &cgh) {
        submit_host_task(cgh, queue, [=](sycl::interop_handle ih) {
            // Ensure that a cusparse handle is created before any other cuSPARSE function is called.
            CusparseScopedContextHandler(queue, ih).get_handle(queue);
            auto cuda_value_type = CudaEnumType<fpType>::value;
            cusparseDnVecDescr_t cu_dvhandle;
            CUSPARSE_ERR_FUNC(cusparseCreateDnVec, &cu_dvhandle, size, val, cuda_value_type);
            *p_dvhandle = new dense_vector_handle(cu_dvhandle, val, size);
        });
    });
    event.wait_and_throw();
}

template <typename fpType>
void set_dense_vector_data(sycl::queue &queue, dense_vector_handle_t dvhandle, std::int64_t size,
                           sycl::buffer<fpType, 1> val) {
    detail::check_can_reset_value_handle<fpType>(__func__, dvhandle, true);
    auto event = queue.submit([&](sycl::handler &cgh) {
        auto acc = val.template get_access<sycl::access::mode::read_write>(cgh);
        submit_host_task(cgh, queue, [=](sycl::interop_handle ih) {
            if (dvhandle->size != size) {
                CUSPARSE_ERR_FUNC(cusparseDestroyDnVec, dvhandle->backend_handle);
                auto cuda_value_type = CudaEnumType<fpType>::value;
                CUSPARSE_ERR_FUNC(cusparseCreateDnVec, &dvhandle->backend_handle, size,
                                  get_mem(ih, acc), cuda_value_type);
                dvhandle->size = size;
            }
            else {
                CUSPARSE_ERR_FUNC(cusparseDnVecSetValues, dvhandle->backend_handle,
                                  get_mem(ih, acc));
            }
            dvhandle->set_buffer(val);
        });
    });
    event.wait_and_throw();
}

template <typename fpType>
void set_dense_vector_data(sycl::queue &, dense_vector_handle_t dvhandle, std::int64_t size,
                           fpType *val) {
    detail::check_can_reset_value_handle<fpType>(__func__, dvhandle, false);
    if (dvhandle->size != size) {
        CUSPARSE_ERR_FUNC(cusparseDestroyDnVec, dvhandle->backend_handle);
        auto cuda_value_type = CudaEnumType<fpType>::value;
        CUSPARSE_ERR_FUNC(cusparseCreateDnVec, &dvhandle->backend_handle, size, val,
                          cuda_value_type);
        dvhandle->size = size;
    }
    else {
        CUSPARSE_ERR_FUNC(cusparseDnVecSetValues, dvhandle->backend_handle, val);
    }
    dvhandle->set_usm_ptr(val);
}

FOR_EACH_FP_TYPE(INSTANTIATE_DENSE_VECTOR_FUNCS);

sycl::event release_dense_vector(sycl::queue &queue, dense_vector_handle_t dvhandle,
                                 const std::vector<sycl::event> &dependencies) {
    // Use dispatch_submit_impl_fp to ensure the backend's handle is kept alive as long as the buffer is used
    auto functor = [=](sycl::interop_handle) {
        CUSPARSE_ERR_FUNC(cusparseDestroyDnVec, dvhandle->backend_handle);
        delete dvhandle;
    };
    return dispatch_submit_impl_fp(__func__, queue, dependencies, functor, dvhandle);
}

// Dense matrix
template <typename fpType>
void init_dense_matrix(sycl::queue &queue, dense_matrix_handle_t *p_dmhandle, std::int64_t num_rows,
                       std::int64_t num_cols, std::int64_t ld, layout dense_layout,
                       sycl::buffer<fpType, 1> val) {
    auto event = queue.submit([&](sycl::handler &cgh) {
        auto acc = val.template get_access<sycl::access::mode::read_write>(cgh);
        submit_host_task(cgh, queue, [=](sycl::interop_handle ih) {
            // Ensure that a cusparse handle is created before any other cuSPARSE function is called.
            CusparseScopedContextHandler(queue, ih).get_handle(queue);
            auto cuda_value_type = CudaEnumType<fpType>::value;
            auto cuda_order = get_cuda_order(dense_layout);
            cusparseDnMatDescr_t cu_dmhandle;
            CUSPARSE_ERR_FUNC(cusparseCreateDnMat, &cu_dmhandle, num_rows, num_cols, ld,
                              get_mem(ih, acc), cuda_value_type, cuda_order);
            *p_dmhandle =
                new dense_matrix_handle(cu_dmhandle, val, num_rows, num_cols, ld, dense_layout);
        });
    });
    event.wait_and_throw();
}

template <typename fpType>
void init_dense_matrix(sycl::queue &queue, dense_matrix_handle_t *p_dmhandle, std::int64_t num_rows,
                       std::int64_t num_cols, std::int64_t ld, layout dense_layout, fpType *val) {
    auto event = queue.submit([&](sycl::handler &cgh) {
        submit_host_task(cgh, queue, [=](sycl::interop_handle ih) {
            // Ensure that a cusparse handle is created before any other cuSPARSE function is called.
            CusparseScopedContextHandler(queue, ih).get_handle(queue);
            auto cuda_value_type = CudaEnumType<fpType>::value;
            auto cuda_order = get_cuda_order(dense_layout);
            cusparseDnMatDescr_t cu_dmhandle;
            CUSPARSE_ERR_FUNC(cusparseCreateDnMat, &cu_dmhandle, num_rows, num_cols, ld, val,
                              cuda_value_type, cuda_order);
            *p_dmhandle =
                new dense_matrix_handle(cu_dmhandle, val, num_rows, num_cols, ld, dense_layout);
        });
    });
    event.wait_and_throw();
}

template <typename fpType>
void set_dense_matrix_data(sycl::queue &queue, dense_matrix_handle_t dmhandle,
                           std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,
                           oneapi::mkl::layout dense_layout, sycl::buffer<fpType, 1> val) {
    detail::check_can_reset_value_handle<fpType>(__func__, dmhandle, true);
    auto event = queue.submit([&](sycl::handler &cgh) {
        auto acc = val.template get_access<sycl::access::mode::read_write>(cgh);
        submit_host_task(cgh, queue, [=](sycl::interop_handle ih) {
            if (dmhandle->num_rows != num_rows || dmhandle->num_cols != num_cols ||
                dmhandle->ld != ld || dmhandle->dense_layout != dense_layout) {
                CUSPARSE_ERR_FUNC(cusparseDestroyDnMat, dmhandle->backend_handle);
                auto cuda_value_type = CudaEnumType<fpType>::value;
                auto cuda_order = get_cuda_order(dense_layout);
                CUSPARSE_ERR_FUNC(cusparseCreateDnMat, &dmhandle->backend_handle, num_rows,
                                  num_cols, ld, get_mem(ih, acc), cuda_value_type, cuda_order);
                dmhandle->num_rows = num_rows;
                dmhandle->num_cols = num_cols;
                dmhandle->ld = ld;
                dmhandle->dense_layout = dense_layout;
            }
            else {
                CUSPARSE_ERR_FUNC(cusparseDnMatSetValues, dmhandle->backend_handle,
                                  get_mem(ih, acc));
            }
            dmhandle->set_buffer(val);
        });
    });
    event.wait_and_throw();
}

template <typename fpType>
void set_dense_matrix_data(sycl::queue &, dense_matrix_handle_t dmhandle, std::int64_t num_rows,
                           std::int64_t num_cols, std::int64_t ld, oneapi::mkl::layout dense_layout,
                           fpType *val) {
    detail::check_can_reset_value_handle<fpType>(__func__, dmhandle, false);
    if (dmhandle->num_rows != num_rows || dmhandle->num_cols != num_cols || dmhandle->ld != ld ||
        dmhandle->dense_layout != dense_layout) {
        CUSPARSE_ERR_FUNC(cusparseDestroyDnMat, dmhandle->backend_handle);
        auto cuda_value_type = CudaEnumType<fpType>::value;
        auto cuda_order = get_cuda_order(dense_layout);
        CUSPARSE_ERR_FUNC(cusparseCreateDnMat, &dmhandle->backend_handle, num_rows, num_cols, ld,
                          val, cuda_value_type, cuda_order);
        dmhandle->num_rows = num_rows;
        dmhandle->num_cols = num_cols;
        dmhandle->ld = ld;
        dmhandle->dense_layout = dense_layout;
    }
    else {
        CUSPARSE_ERR_FUNC(cusparseDnMatSetValues, dmhandle->backend_handle, val);
    }
    dmhandle->set_usm_ptr(val);
}

FOR_EACH_FP_TYPE(INSTANTIATE_DENSE_MATRIX_FUNCS);

sycl::event release_dense_matrix(sycl::queue &queue, dense_matrix_handle_t dmhandle,
                                 const std::vector<sycl::event> &dependencies) {
    // Use dispatch_submit_impl_fp to ensure the backend's handle is kept alive as long as the buffer is used
    auto functor = [=](sycl::interop_handle) {
        CUSPARSE_ERR_FUNC(cusparseDestroyDnMat, dmhandle->backend_handle);
        delete dmhandle;
    };
    return dispatch_submit_impl_fp(__func__, queue, dependencies, functor, dmhandle);
}

// COO matrix
template <typename fpType, typename intType>
void init_coo_matrix(sycl::queue &queue, matrix_handle_t *p_smhandle, std::int64_t num_rows,
                     std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index,
                     sycl::buffer<intType, 1> row_ind, sycl::buffer<intType, 1> col_ind,
                     sycl::buffer<fpType, 1> val) {
    auto event = queue.submit([&](sycl::handler &cgh) {
        auto row_acc = row_ind.template get_access<sycl::access::mode::read_write>(cgh);
        auto col_acc = col_ind.template get_access<sycl::access::mode::read_write>(cgh);
        auto val_acc = val.template get_access<sycl::access::mode::read_write>(cgh);
        submit_host_task(cgh, queue, [=](sycl::interop_handle ih) {
            // Ensure that a cusparse handle is created before any other cuSPARSE function is called.
            CusparseScopedContextHandler(queue, ih).get_handle(queue);
            auto cuda_index_type = CudaIndexEnumType<intType>::value;
            auto cuda_index_base = get_cuda_index_base(index);
            auto cuda_value_type = CudaEnumType<fpType>::value;
            cusparseSpMatDescr_t cu_smhandle;
            CUSPARSE_ERR_FUNC(cusparseCreateCoo, &cu_smhandle, num_rows, num_cols, nnz,
                              get_mem(ih, row_acc), get_mem(ih, col_acc), get_mem(ih, val_acc),
                              cuda_index_type, cuda_index_base, cuda_value_type);
            *p_smhandle = new matrix_handle(cu_smhandle, row_ind, col_ind, val, detail::sparse_format::COO,
                                            num_rows, num_cols, nnz, index);
        });
    });
    event.wait_and_throw();
}

template <typename fpType, typename intType>
void init_coo_matrix(sycl::queue &queue, matrix_handle_t *p_smhandle, std::int64_t num_rows,
                     std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index,
                     intType *row_ind, intType *col_ind, fpType *val) {
    auto event = queue.submit([&](sycl::handler &cgh) {
        submit_host_task(cgh, queue, [=](sycl::interop_handle ih) {
            // Ensure that a cusparse handle is created before any other cuSPARSE function is called.
            CusparseScopedContextHandler(queue, ih).get_handle(queue);
            auto cuda_index_type = CudaIndexEnumType<intType>::value;
            auto cuda_index_base = get_cuda_index_base(index);
            auto cuda_value_type = CudaEnumType<fpType>::value;
            cusparseSpMatDescr_t cu_smhandle;
            CUSPARSE_ERR_FUNC(cusparseCreateCoo, &cu_smhandle, num_rows, num_cols, nnz, row_ind,
                              col_ind, val, cuda_index_type, cuda_index_base, cuda_value_type);
            *p_smhandle = new matrix_handle(cu_smhandle, row_ind, col_ind, val, detail::sparse_format::COO,
                                            num_rows, num_cols, nnz, index);
        });
    });
    event.wait_and_throw();
}

template <typename fpType, typename intType>
void set_coo_matrix_data(sycl::queue &queue, matrix_handle_t smhandle, std::int64_t num_rows,
                         std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index,
                         sycl::buffer<intType, 1> row_ind, sycl::buffer<intType, 1> col_ind,
                         sycl::buffer<fpType, 1> val) {
    detail::check_can_reset_sparse_handle<fpType, intType>(__func__, smhandle, true);
    auto event = queue.submit([&](sycl::handler &cgh) {
        auto row_acc = row_ind.template get_access<sycl::access::mode::read_write>(cgh);
        auto col_acc = col_ind.template get_access<sycl::access::mode::read_write>(cgh);
        auto val_acc = val.template get_access<sycl::access::mode::read_write>(cgh);
        submit_host_task(cgh, queue, [=](sycl::interop_handle ih) {
            if (smhandle->num_rows != num_rows || smhandle->num_cols != num_cols ||
                smhandle->nnz != nnz || smhandle->index != index) {
                CUSPARSE_ERR_FUNC(cusparseDestroySpMat, smhandle->backend_handle);
                auto cuda_index_type = CudaIndexEnumType<intType>::value;
                auto cuda_index_base = get_cuda_index_base(index);
                auto cuda_value_type = CudaEnumType<fpType>::value;
                CUSPARSE_ERR_FUNC(cusparseCreateCoo, &smhandle->backend_handle, num_rows, num_cols,
                                  nnz, get_mem(ih, row_acc), get_mem(ih, col_acc),
                                  get_mem(ih, val_acc), cuda_index_type, cuda_index_base,
                                  cuda_value_type);
                smhandle->num_rows = num_rows;
                smhandle->num_cols = num_cols;
                smhandle->nnz = nnz;
                smhandle->index = index;
            }
            else {
                CUSPARSE_ERR_FUNC(cusparseCooSetPointers, smhandle->backend_handle,
                                  get_mem(ih, row_acc), get_mem(ih, col_acc), get_mem(ih, val_acc));
            }
            smhandle->row_container.set_buffer(row_ind);
            smhandle->col_container.set_buffer(col_ind);
            smhandle->value_container.set_buffer(val);
        });
    });
    event.wait_and_throw();
}

template <typename fpType, typename intType>
void set_coo_matrix_data(sycl::queue &, matrix_handle_t smhandle, std::int64_t num_rows,
                         std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index,
                         intType *row_ind, intType *col_ind, fpType *val) {
    detail::check_can_reset_sparse_handle<fpType, intType>(__func__, smhandle, false);
    if (smhandle->num_rows != num_rows || smhandle->num_cols != num_cols || smhandle->nnz != nnz ||
        smhandle->index != index) {
        CUSPARSE_ERR_FUNC(cusparseDestroySpMat, smhandle->backend_handle);
        auto cuda_index_type = CudaIndexEnumType<intType>::value;
        auto cuda_index_base = get_cuda_index_base(index);
        auto cuda_value_type = CudaEnumType<fpType>::value;
        CUSPARSE_ERR_FUNC(cusparseCreateCoo, &smhandle->backend_handle, num_rows, num_cols, nnz,
                          row_ind, col_ind, val, cuda_index_type, cuda_index_base, cuda_value_type);
        smhandle->num_rows = num_rows;
        smhandle->num_cols = num_cols;
        smhandle->nnz = nnz;
        smhandle->index = index;
    }
    else {
        CUSPARSE_ERR_FUNC(cusparseCooSetPointers, smhandle->backend_handle, row_ind, col_ind, val);
    }
    smhandle->row_container.set_usm_ptr(row_ind);
    smhandle->col_container.set_usm_ptr(col_ind);
    smhandle->value_container.set_usm_ptr(val);
}

FOR_EACH_FP_AND_INT_TYPE(INSTANTIATE_COO_MATRIX_FUNCS);

// CSR matrix
template <typename fpType, typename intType>
void init_csr_matrix(sycl::queue &queue, matrix_handle_t *p_smhandle, std::int64_t num_rows,
                     std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index,
                     sycl::buffer<intType, 1> row_ptr, sycl::buffer<intType, 1> col_ind,
                     sycl::buffer<fpType, 1> val) {
    auto event = queue.submit([&](sycl::handler &cgh) {
        auto row_acc = row_ptr.template get_access<sycl::access::mode::read_write>(cgh);
        auto col_acc = col_ind.template get_access<sycl::access::mode::read_write>(cgh);
        auto val_acc = val.template get_access<sycl::access::mode::read_write>(cgh);
        submit_host_task(cgh, queue, [=](sycl::interop_handle ih) {
            // Ensure that a cusparse handle is created before any other cuSPARSE function is called.
            CusparseScopedContextHandler(queue, ih).get_handle(queue);
            auto cuda_index_type = CudaIndexEnumType<intType>::value;
            auto cuda_index_base = get_cuda_index_base(index);
            auto cuda_value_type = CudaEnumType<fpType>::value;
            cusparseSpMatDescr_t cu_smhandle;
            CUSPARSE_ERR_FUNC(cusparseCreateCsr, &cu_smhandle, num_rows, num_cols, nnz,
                              get_mem(ih, row_acc), get_mem(ih, col_acc), get_mem(ih, val_acc),
                              cuda_index_type, cuda_index_type, cuda_index_base, cuda_value_type);
            *p_smhandle = new matrix_handle(cu_smhandle, row_ptr, col_ind, val, detail::sparse_format::CSR,
                                            num_rows, num_cols, nnz, index);
        });
    });
    event.wait_and_throw();
}

template <typename fpType, typename intType>
void init_csr_matrix(sycl::queue &queue, matrix_handle_t *p_smhandle, std::int64_t num_rows,
                     std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index,
                     intType *row_ptr, intType *col_ind, fpType *val) {
    auto event = queue.submit([&](sycl::handler &cgh) {
        submit_host_task(cgh, queue, [=](sycl::interop_handle ih) {
            // Ensure that a cusparse handle is created before any other cuSPARSE function is called.
            CusparseScopedContextHandler(queue, ih).get_handle(queue);
            auto cuda_index_type = CudaIndexEnumType<intType>::value;
            auto cuda_index_base = get_cuda_index_base(index);
            auto cuda_value_type = CudaEnumType<fpType>::value;
            cusparseSpMatDescr_t cu_smhandle;
            CUSPARSE_ERR_FUNC(cusparseCreateCsr, &cu_smhandle, num_rows, num_cols, nnz, row_ptr,
                              col_ind, val, cuda_index_type, cuda_index_type, cuda_index_base,
                              cuda_value_type);
            *p_smhandle = new matrix_handle(cu_smhandle, row_ptr, col_ind, val, detail::sparse_format::CSR,
                                            num_rows, num_cols, nnz, index);
        });
    });
    event.wait_and_throw();
}

template <typename fpType, typename intType>
void set_csr_matrix_data(sycl::queue &queue, matrix_handle_t smhandle, std::int64_t num_rows,
                         std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index,
                         sycl::buffer<intType, 1> row_ptr, sycl::buffer<intType, 1> col_ind,
                         sycl::buffer<fpType, 1> val) {
    detail::check_can_reset_sparse_handle<fpType, intType>(__func__, smhandle, true);
    auto event = queue.submit([&](sycl::handler &cgh) {
        auto row_acc = row_ptr.template get_access<sycl::access::mode::read_write>(cgh);
        auto col_acc = col_ind.template get_access<sycl::access::mode::read_write>(cgh);
        auto val_acc = val.template get_access<sycl::access::mode::read_write>(cgh);
        submit_host_task(cgh, queue, [=](sycl::interop_handle ih) {
            if (smhandle->num_rows != num_rows || smhandle->num_cols != num_cols ||
                smhandle->nnz != nnz || smhandle->index != index) {
                CUSPARSE_ERR_FUNC(cusparseDestroySpMat, smhandle->backend_handle);
                auto cuda_index_type = CudaIndexEnumType<intType>::value;
                auto cuda_index_base = get_cuda_index_base(index);
                auto cuda_value_type = CudaEnumType<fpType>::value;
                CUSPARSE_ERR_FUNC(cusparseCreateCsr, &smhandle->backend_handle, num_rows, num_cols,
                                  nnz, get_mem(ih, row_acc), get_mem(ih, col_acc),
                                  get_mem(ih, val_acc), cuda_index_type, cuda_index_type,
                                  cuda_index_base, cuda_value_type);
                smhandle->num_rows = num_rows;
                smhandle->num_cols = num_cols;
                smhandle->nnz = nnz;
                smhandle->index = index;
            }
            else {
                CUSPARSE_ERR_FUNC(cusparseCsrSetPointers, smhandle->backend_handle,
                                  get_mem(ih, row_acc), get_mem(ih, col_acc), get_mem(ih, val_acc));
            }
            smhandle->row_container.set_buffer(row_ptr);
            smhandle->col_container.set_buffer(col_ind);
            smhandle->value_container.set_buffer(val);
        });
    });
    event.wait_and_throw();
}

template <typename fpType, typename intType>
void set_csr_matrix_data(sycl::queue &, matrix_handle_t smhandle, std::int64_t num_rows,
                         std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index,
                         intType *row_ptr, intType *col_ind, fpType *val) {
    detail::check_can_reset_sparse_handle<fpType, intType>(__func__, smhandle, false);
    if (smhandle->num_rows != num_rows || smhandle->num_cols != num_cols || smhandle->nnz != nnz ||
        smhandle->index != index) {
        CUSPARSE_ERR_FUNC(cusparseDestroySpMat, smhandle->backend_handle);
        auto cuda_index_type = CudaIndexEnumType<intType>::value;
        auto cuda_index_base = get_cuda_index_base(index);
        auto cuda_value_type = CudaEnumType<fpType>::value;
        CUSPARSE_ERR_FUNC(cusparseCreateCsr, &smhandle->backend_handle, num_rows, num_cols, nnz,
                          row_ptr, col_ind, val, cuda_index_type, cuda_index_type, cuda_index_base,
                          cuda_value_type);
        smhandle->num_rows = num_rows;
        smhandle->num_cols = num_cols;
        smhandle->nnz = nnz;
        smhandle->index = index;
    }
    else {
        CUSPARSE_ERR_FUNC(cusparseCsrSetPointers, smhandle->backend_handle, row_ptr, col_ind, val);
    }
    smhandle->row_container.set_usm_ptr(row_ptr);
    smhandle->col_container.set_usm_ptr(col_ind);
    smhandle->value_container.set_usm_ptr(val);
}

FOR_EACH_FP_AND_INT_TYPE(INSTANTIATE_CSR_MATRIX_FUNCS);

sycl::event release_sparse_matrix(sycl::queue &queue, matrix_handle_t smhandle,
                                  const std::vector<sycl::event> &dependencies) {
    // Use dispatch_submit to ensure the backend's handle is kept alive as long as the buffers are used
    auto functor = [=](sycl::interop_handle) {
        CUSPARSE_ERR_FUNC(cusparseDestroySpMat, smhandle->backend_handle);
        delete smhandle;
    };
    return dispatch_submit(__func__, queue, dependencies, functor, smhandle);
}

// Matrix property
bool set_matrix_property(sycl::queue &, matrix_handle_t smhandle, matrix_property property) {
    // No equivalent in cuSPARSE
    // Store the matrix property internally for future usages
    smhandle->set_matrix_property(property);
    return false;
}

} // namespace oneapi::mkl::sparse::cusparse
