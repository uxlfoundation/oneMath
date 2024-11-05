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

// Dense vector
template <typename fpType>
void init_dense_vector(sycl::queue& /*queue*/, dense_vector_handle_t* p_dvhandle, std::int64_t size,
                       sycl::buffer<fpType, 1> val) {
    *p_dvhandle = new dense_vector_handle(val, size);
}

template <typename fpType>
void init_dense_vector(sycl::queue& /*queue*/, dense_vector_handle_t* p_dvhandle, std::int64_t size,
                       fpType* val) {
    *p_dvhandle = new dense_vector_handle(val, size);
}

template <typename fpType>
void set_dense_vector_data(sycl::queue& /*queue*/, dense_vector_handle_t dvhandle,
                           std::int64_t size, sycl::buffer<fpType, 1> val) {
    detail::check_can_reset_value_handle<fpType>(__func__, dvhandle, true);
    dvhandle->size = size;
    dvhandle->set_buffer(val);
}

template <typename fpType>
void set_dense_vector_data(sycl::queue& /*queue*/, dense_vector_handle_t dvhandle,
                           std::int64_t size, fpType* val) {
    detail::check_can_reset_value_handle<fpType>(__func__, dvhandle, false);
    dvhandle->size = size;
    dvhandle->set_usm_ptr(val);
}

FOR_EACH_FP_TYPE(INSTANTIATE_DENSE_VECTOR_FUNCS);

sycl::event release_dense_vector(sycl::queue& queue, dense_vector_handle_t dvhandle,
                                 const std::vector<sycl::event>& dependencies) {
    return detail::submit_release(queue, dvhandle, dependencies);
}

// Dense matrix
template <typename fpType>
void init_dense_matrix(sycl::queue& /*queue*/, dense_matrix_handle_t* p_dmhandle,
                       std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,
                       layout dense_layout, sycl::buffer<fpType, 1> val) {
    *p_dmhandle = new dense_matrix_handle(val, num_rows, num_cols, ld, dense_layout);
}

template <typename fpType>
void init_dense_matrix(sycl::queue& /*queue*/, dense_matrix_handle_t* p_dmhandle,
                       std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,
                       layout dense_layout, fpType* val) {
    *p_dmhandle = new dense_matrix_handle(val, num_rows, num_cols, ld, dense_layout);
}

template <typename fpType>
void set_dense_matrix_data(sycl::queue& /*queue*/, dense_matrix_handle_t dmhandle,
                           std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,
                           layout dense_layout, sycl::buffer<fpType, 1> val) {
    detail::check_can_reset_value_handle<fpType>(__func__, dmhandle, true);
    dmhandle->num_rows = num_rows;
    dmhandle->num_cols = num_cols;
    dmhandle->ld = ld;
    dmhandle->dense_layout = dense_layout;
    dmhandle->set_buffer(val);
}

template <typename fpType>
void set_dense_matrix_data(sycl::queue& /*queue*/, dense_matrix_handle_t dmhandle,
                           std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,
                           layout dense_layout, fpType* val) {
    detail::check_can_reset_value_handle<fpType>(__func__, dmhandle, false);
    dmhandle->num_rows = num_rows;
    dmhandle->num_cols = num_cols;
    dmhandle->ld = ld;
    dmhandle->dense_layout = dense_layout;
    dmhandle->set_usm_ptr(val);
}

FOR_EACH_FP_TYPE(INSTANTIATE_DENSE_MATRIX_FUNCS);

sycl::event release_dense_matrix(sycl::queue& queue, dense_matrix_handle_t dmhandle,
                                 const std::vector<sycl::event>& dependencies) {
    return detail::submit_release(queue, dmhandle, dependencies);
}

// COO matrix
template <typename fpType, typename intType>
void init_coo_matrix(sycl::queue& queue, oneapi::math::sparse::matrix_handle_t* p_smhandle,
                     std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                     index_base index, sycl::buffer<intType, 1> row_ind,
                     sycl::buffer<intType, 1> col_ind, sycl::buffer<fpType, 1> val) {
    oneapi::mkl::sparse::matrix_handle_t onemkl_handle;
    RETHROW_ONEMKL_EXCEPTIONS(oneapi::mkl::sparse::init_matrix_handle(&onemkl_handle));
    auto internal_smhandle = new detail::sparse_matrix_handle(onemkl_handle, row_ind, col_ind, val,
                                                              detail::sparse_format::COO, num_rows,
                                                              num_cols, nnz, index);
    // The backend handle must use the buffers from the internal handle as they will be kept alive until the handle is released.
    RETHROW_ONEMKL_EXCEPTIONS(oneapi::mkl::sparse::set_coo_data(
        queue, onemkl_handle, static_cast<intType>(num_rows), static_cast<intType>(num_cols),
        static_cast<intType>(nnz), detail::get_onemkl_index_base(index),
        internal_smhandle->row_container.get_buffer<intType>(),
        internal_smhandle->col_container.get_buffer<intType>(),
        internal_smhandle->value_container.get_buffer<fpType>()));
    *p_smhandle = reinterpret_cast<oneapi::math::sparse::matrix_handle_t>(internal_smhandle);
}

template <typename fpType, typename intType>
void init_coo_matrix(sycl::queue& queue, oneapi::math::sparse::matrix_handle_t* p_smhandle,
                     std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                     index_base index, intType* row_ind, intType* col_ind, fpType* val) {
    oneapi::mkl::sparse::matrix_handle_t onemkl_handle;
    RETHROW_ONEMKL_EXCEPTIONS(oneapi::mkl::sparse::init_matrix_handle(&onemkl_handle));
    auto internal_smhandle = new detail::sparse_matrix_handle(onemkl_handle, row_ind, col_ind, val,
                                                              detail::sparse_format::COO, num_rows,
                                                              num_cols, nnz, index);
    sycl::event event;
    RETHROW_ONEMKL_EXCEPTIONS(event = oneapi::mkl::sparse::set_coo_data(
                                  queue, onemkl_handle, static_cast<intType>(num_rows),
                                  static_cast<intType>(num_cols), static_cast<intType>(nnz),
                                  detail::get_onemkl_index_base(index), row_ind, col_ind, val));
    event.wait_and_throw();
    *p_smhandle = reinterpret_cast<oneapi::math::sparse::matrix_handle_t>(internal_smhandle);
}

template <typename fpType, typename intType>
void set_coo_matrix_data(sycl::queue& queue, oneapi::math::sparse::matrix_handle_t smhandle,
                         std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                         index_base index, sycl::buffer<intType, 1> row_ind,
                         sycl::buffer<intType, 1> col_ind, sycl::buffer<fpType, 1> val) {
    auto internal_smhandle = detail::get_internal_handle(smhandle);
    detail::check_can_reset_sparse_handle<fpType, intType>(__func__, internal_smhandle, true);
    internal_smhandle->num_rows = num_rows;
    internal_smhandle->num_cols = num_cols;
    internal_smhandle->nnz = nnz;
    internal_smhandle->index = index;
    internal_smhandle->row_container.set_buffer(row_ind);
    internal_smhandle->col_container.set_buffer(col_ind);
    internal_smhandle->value_container.set_buffer(val);
    // The backend handle must use the buffers from the internal handle as they will be kept alive until the handle is released.
    RETHROW_ONEMKL_EXCEPTIONS(oneapi::mkl::sparse::set_coo_data(
        queue, internal_smhandle->backend_handle, static_cast<intType>(num_rows),
        static_cast<intType>(num_cols), static_cast<intType>(nnz),
        detail::get_onemkl_index_base(index),
        internal_smhandle->row_container.get_buffer<intType>(),
        internal_smhandle->col_container.get_buffer<intType>(),
        internal_smhandle->value_container.get_buffer<fpType>()));
}

template <typename fpType, typename intType>
void set_coo_matrix_data(sycl::queue& queue, oneapi::math::sparse::matrix_handle_t smhandle,
                         std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                         index_base index, intType* row_ind, intType* col_ind, fpType* val) {
    auto internal_smhandle = detail::get_internal_handle(smhandle);
    detail::check_can_reset_sparse_handle<fpType, intType>(__func__, internal_smhandle, false);
    internal_smhandle->num_rows = num_rows;
    internal_smhandle->num_cols = num_cols;
    internal_smhandle->nnz = nnz;
    internal_smhandle->index = index;
    internal_smhandle->row_container.set_usm_ptr(row_ind);
    internal_smhandle->col_container.set_usm_ptr(col_ind);
    internal_smhandle->value_container.set_usm_ptr(val);
    sycl::event event;
    RETHROW_ONEMKL_EXCEPTIONS(event = oneapi::mkl::sparse::set_coo_data(
                                  queue, internal_smhandle->backend_handle,
                                  static_cast<intType>(num_rows), static_cast<intType>(num_cols),
                                  static_cast<intType>(nnz), detail::get_onemkl_index_base(index),
                                  row_ind, col_ind, val));
    event.wait_and_throw();
}

FOR_EACH_FP_AND_INT_TYPE(INSTANTIATE_COO_MATRIX_FUNCS);

// CSR matrix
template <typename fpType, typename intType>
void init_csr_matrix(sycl::queue& queue, oneapi::math::sparse::matrix_handle_t* p_smhandle,
                     std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                     index_base index, sycl::buffer<intType, 1> row_ptr,
                     sycl::buffer<intType, 1> col_ind, sycl::buffer<fpType, 1> val) {
    oneapi::mkl::sparse::matrix_handle_t onemkl_handle;
    RETHROW_ONEMKL_EXCEPTIONS(oneapi::mkl::sparse::init_matrix_handle(&onemkl_handle));
    auto internal_smhandle = new detail::sparse_matrix_handle(onemkl_handle, row_ptr, col_ind, val,
                                                              detail::sparse_format::CSR, num_rows,
                                                              num_cols, nnz, index);
    // The backend deduces nnz from row_ptr.
    // The backend handle must use the buffers from the internal handle as they will be kept alive until the handle is released.
    RETHROW_ONEMKL_EXCEPTIONS(oneapi::mkl::sparse::set_csr_data(
        queue, onemkl_handle, static_cast<intType>(num_rows), static_cast<intType>(num_cols),
        detail::get_onemkl_index_base(index),
        internal_smhandle->row_container.get_buffer<intType>(),
        internal_smhandle->col_container.get_buffer<intType>(),
        internal_smhandle->value_container.get_buffer<fpType>()));
    *p_smhandle = reinterpret_cast<oneapi::math::sparse::matrix_handle_t>(internal_smhandle);
}

template <typename fpType, typename intType>
void init_csr_matrix(sycl::queue& queue, oneapi::math::sparse::matrix_handle_t* p_smhandle,
                     std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                     index_base index, intType* row_ptr, intType* col_ind, fpType* val) {
    oneapi::mkl::sparse::matrix_handle_t onemkl_handle;
    RETHROW_ONEMKL_EXCEPTIONS(oneapi::mkl::sparse::init_matrix_handle(&onemkl_handle));
    auto internal_smhandle = new detail::sparse_matrix_handle(onemkl_handle, row_ptr, col_ind, val,
                                                              detail::sparse_format::CSR, num_rows,
                                                              num_cols, nnz, index);
    // The backend deduces nnz from row_ptr.
    sycl::event event;
    RETHROW_ONEMKL_EXCEPTIONS(event = oneapi::mkl::sparse::set_csr_data(
                                  queue, onemkl_handle, static_cast<intType>(num_rows),
                                  static_cast<intType>(num_cols),
                                  detail::get_onemkl_index_base(index), row_ptr, col_ind, val));
    event.wait_and_throw();
    *p_smhandle = reinterpret_cast<oneapi::math::sparse::matrix_handle_t>(internal_smhandle);
}

template <typename fpType, typename intType>
void set_csr_matrix_data(sycl::queue& queue, oneapi::math::sparse::matrix_handle_t smhandle,
                         std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                         index_base index, sycl::buffer<intType, 1> row_ptr,
                         sycl::buffer<intType, 1> col_ind, sycl::buffer<fpType, 1> val) {
    auto internal_smhandle = detail::get_internal_handle(smhandle);
    detail::check_can_reset_sparse_handle<fpType, intType>(__func__, internal_smhandle, true);
    internal_smhandle->num_rows = num_rows;
    internal_smhandle->num_cols = num_cols;
    internal_smhandle->nnz = nnz;
    internal_smhandle->index = index;
    internal_smhandle->row_container.set_buffer(row_ptr);
    internal_smhandle->col_container.set_buffer(col_ind);
    internal_smhandle->value_container.set_buffer(val);
    // The backend deduces nnz from row_ptr.
    // The backend handle must use the buffers from the internal handle as they will be kept alive until the handle is released.
    RETHROW_ONEMKL_EXCEPTIONS(oneapi::mkl::sparse::set_csr_data(
        queue, internal_smhandle->backend_handle, static_cast<intType>(num_rows),
        static_cast<intType>(num_cols), detail::get_onemkl_index_base(index),
        internal_smhandle->row_container.get_buffer<intType>(),
        internal_smhandle->col_container.get_buffer<intType>(),
        internal_smhandle->value_container.get_buffer<fpType>()));
}

template <typename fpType, typename intType>
void set_csr_matrix_data(sycl::queue& queue, oneapi::math::sparse::matrix_handle_t smhandle,
                         std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                         index_base index, intType* row_ptr, intType* col_ind, fpType* val) {
    auto internal_smhandle = detail::get_internal_handle(smhandle);
    detail::check_can_reset_sparse_handle<fpType, intType>(__func__, internal_smhandle, false);
    internal_smhandle->num_rows = num_rows;
    internal_smhandle->num_cols = num_cols;
    internal_smhandle->nnz = nnz;
    internal_smhandle->index = index;
    internal_smhandle->row_container.set_usm_ptr(row_ptr);
    internal_smhandle->col_container.set_usm_ptr(col_ind);
    internal_smhandle->value_container.set_usm_ptr(val);
    // The backend deduces nnz from row_ptr.
    sycl::event event;
    RETHROW_ONEMKL_EXCEPTIONS(event = oneapi::mkl::sparse::set_csr_data(
                                  queue, internal_smhandle->backend_handle,
                                  static_cast<intType>(num_rows), static_cast<intType>(num_cols),
                                  detail::get_onemkl_index_base(index), row_ptr, col_ind, val));
    event.wait_and_throw();
}

FOR_EACH_FP_AND_INT_TYPE(INSTANTIATE_CSR_MATRIX_FUNCS);

// Common sparse matrix functions
sycl::event release_sparse_matrix(sycl::queue& queue,
                                  oneapi::math::sparse::matrix_handle_t smhandle,
                                  const std::vector<sycl::event>& dependencies) {
    auto internal_smhandle = detail::get_internal_handle(smhandle);
    // Asynchronously release the backend's handle followed by the internal handle.
    sycl::event event;
    RETHROW_ONEMKL_EXCEPTIONS(event = oneapi::mkl::sparse::release_matrix_handle(
                                  queue, &internal_smhandle->backend_handle, dependencies));
    return detail::submit_release(queue, internal_smhandle, { event });
}

bool set_matrix_property(sycl::queue& /*queue*/, oneapi::math::sparse::matrix_handle_t smhandle,
                         matrix_property property) {
    auto internal_smhandle = detail::get_internal_handle(smhandle);
    // Store the matrix property internally for better error checking
    internal_smhandle->set_matrix_property(property);
    // Set the matrix property on the backend handle
    // Backend and oneMath types for the property don't match
    switch (property) {
        case matrix_property::symmetric:
            RETHROW_ONEMKL_EXCEPTIONS(oneapi::mkl::sparse::set_matrix_property(
                internal_smhandle->backend_handle, oneapi::mkl::sparse::property::symmetric));
            return true;
        case matrix_property::sorted:
            RETHROW_ONEMKL_EXCEPTIONS(oneapi::mkl::sparse::set_matrix_property(
                internal_smhandle->backend_handle, oneapi::mkl::sparse::property::sorted));
            return true;
        default: return false;
    }
}
