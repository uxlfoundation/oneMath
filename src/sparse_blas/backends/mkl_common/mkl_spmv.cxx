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

// In this file functions and types using the namespace oneapi::mkl::sparse:: refer to the backend's namespace for better readability.

namespace oneapi::mkl::sparse {

struct spmv_descr {
    bool buffer_size_called = false;
    bool optimized_called = false;
    oneapi::mkl::transpose last_optimized_opA;
    matrix_view last_optimized_A_view;
    matrix_handle_t last_optimized_A_handle;
    dense_vector_handle_t last_optimized_x_handle;
    dense_vector_handle_t last_optimized_y_handle;
    spmv_alg last_optimized_alg;
};

} // namespace oneapi::mkl::sparse

namespace oneapi::mkl::sparse::BACKEND {

void init_spmv_descr(sycl::queue& /*queue*/, spmv_descr_t* p_spmv_descr) {
    *p_spmv_descr = new spmv_descr();
}

sycl::event release_spmv_descr(sycl::queue& queue, spmv_descr_t spmv_descr,
                               const std::vector<sycl::event>& dependencies) {
    return detail::submit_release(queue, spmv_descr, dependencies);
}

void check_valid_spmv(const std::string& function_name, oneapi::mkl::transpose opA,
                      matrix_view A_view, matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                      dense_vector_handle_t y_handle, bool is_alpha_host_accessible,
                      bool is_beta_host_accessible) {
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    detail::check_valid_spmv_common(__func__, opA, A_view, internal_A_handle, x_handle, y_handle,
                                    is_alpha_host_accessible, is_beta_host_accessible);

    if ((A_view.type_view == matrix_descr::symmetric ||
         A_view.type_view == matrix_descr::hermitian) &&
        opA == oneapi::mkl::transpose::conjtrans) {
        throw mkl::unimplemented(
            "sparse_blas", function_name,
            "The backend does not support Symmetric or Hermitian matrix with `conjtrans`.");
    }
}

void spmv_buffer_size(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                      matrix_view A_view, matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                      const void* beta, dense_vector_handle_t y_handle, spmv_alg /*alg*/,
                      spmv_descr_t spmv_descr, std::size_t& temp_buffer_size) {
    // TODO: Add support for external workspace once the close-source oneMKL backend supports it.
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    check_valid_spmv(__func__, opA, A_view, A_handle, x_handle, y_handle, is_alpha_host_accessible,
                     is_beta_host_accessible);
    temp_buffer_size = 0;
    spmv_descr->buffer_size_called = true;
}

inline void common_spmv_optimize(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                                 matrix_view A_view, matrix_handle_t A_handle,
                                 dense_vector_handle_t x_handle, const void* beta,
                                 dense_vector_handle_t y_handle, spmv_alg alg,
                                 spmv_descr_t spmv_descr) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    check_valid_spmv("spmv_optimize", opA, A_view, A_handle, x_handle, y_handle,
                     is_alpha_host_accessible, is_beta_host_accessible);
    if (!spmv_descr->buffer_size_called) {
        throw mkl::uninitialized("sparse_blas", "spmv_optimize",
                                 "spmv_buffer_size must be called before spmv_optimize.");
    }
    spmv_descr->optimized_called = true;
    spmv_descr->last_optimized_opA = opA;
    spmv_descr->last_optimized_A_view = A_view;
    spmv_descr->last_optimized_A_handle = A_handle;
    spmv_descr->last_optimized_x_handle = x_handle;
    spmv_descr->last_optimized_y_handle = y_handle;
    spmv_descr->last_optimized_alg = alg;
}

void spmv_optimize(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                   matrix_view A_view, matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                   const void* beta, dense_vector_handle_t y_handle, spmv_alg alg,
                   spmv_descr_t spmv_descr, sycl::buffer<std::uint8_t, 1> /*workspace*/) {
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    if (!internal_A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    common_spmv_optimize(queue, opA, alpha, A_view, A_handle, x_handle, beta, y_handle, alg,
                         spmv_descr);
    if (alg == spmv_alg::no_optimize_alg) {
        return;
    }
    internal_A_handle->can_be_reset = false;
    if (A_view.type_view == matrix_descr::triangular) {
        oneapi::mkl::sparse::optimize_trmv(queue, A_view.uplo_view, opA, A_view.diag_view,
                                           internal_A_handle->backend_handle);
    }
    else if (A_view.type_view == matrix_descr::symmetric ||
             A_view.type_view == matrix_descr::hermitian) {
        // No optimize_symv currently
        return;
    }
    else {
        oneapi::mkl::sparse::optimize_gemv(queue, opA, internal_A_handle->backend_handle);
    }
}

sycl::event spmv_optimize(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                          matrix_view A_view, matrix_handle_t A_handle,
                          dense_vector_handle_t x_handle, const void* beta,
                          dense_vector_handle_t y_handle, spmv_alg alg, spmv_descr_t spmv_descr,
                          void* /*workspace*/, const std::vector<sycl::event>& dependencies) {
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    if (internal_A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    common_spmv_optimize(queue, opA, alpha, A_view, A_handle, x_handle, beta, y_handle, alg,
                         spmv_descr);
    if (alg == spmv_alg::no_optimize_alg) {
        return detail::collapse_dependencies(queue, dependencies);
    }
    internal_A_handle->can_be_reset = false;
    if (A_view.type_view == matrix_descr::triangular) {
        return oneapi::mkl::sparse::optimize_trmv(queue, A_view.uplo_view, opA, A_view.diag_view,
                                                  internal_A_handle->backend_handle, dependencies);
    }
    else if (A_view.type_view == matrix_descr::symmetric ||
             A_view.type_view == matrix_descr::hermitian) {
        return detail::collapse_dependencies(queue, dependencies);
    }
    else {
        return oneapi::mkl::sparse::optimize_gemv(queue, opA, internal_A_handle->backend_handle,
                                                  dependencies);
    }
}

template <typename T>
sycl::event internal_spmv(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                          matrix_view A_view, matrix_handle_t A_handle,
                          dense_vector_handle_t x_handle, const void* beta,
                          dense_vector_handle_t y_handle, spmv_alg /*alg*/,
                          spmv_descr_t /*spmv_descr*/, const std::vector<sycl::event>& dependencies,
                          bool is_alpha_host_accessible, bool is_beta_host_accessible) {
    T host_alpha =
        detail::get_scalar_on_host(queue, static_cast<const T*>(alpha), is_alpha_host_accessible);
    T host_beta =
        detail::get_scalar_on_host(queue, static_cast<const T*>(beta), is_beta_host_accessible);
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    internal_A_handle->can_be_reset = false;
    auto backend_handle = internal_A_handle->backend_handle;
    if (internal_A_handle->all_use_buffer()) {
        auto x_buffer = x_handle->get_buffer<T>();
        auto y_buffer = y_handle->get_buffer<T>();
        if (A_view.type_view == matrix_descr::triangular) {
            oneapi::mkl::sparse::trmv(queue, A_view.uplo_view, opA, A_view.diag_view, host_alpha,
                                      backend_handle, x_buffer, host_beta, y_buffer);
        }
        else if (A_view.type_view == matrix_descr::symmetric ||
                 A_view.type_view == matrix_descr::hermitian) {
            oneapi::mkl::sparse::symv(queue, A_view.uplo_view, host_alpha, backend_handle, x_buffer,
                                      host_beta, y_buffer);
        }
        else {
            oneapi::mkl::sparse::gemv(queue, opA, host_alpha, backend_handle, x_buffer, host_beta,
                                      y_buffer);
        }
        // Dependencies are not used for buffers
        return {};
    }
    else {
        auto x_usm = x_handle->get_usm_ptr<T>();
        auto y_usm = y_handle->get_usm_ptr<T>();
        if (A_view.type_view == matrix_descr::triangular) {
            return oneapi::mkl::sparse::trmv(queue, A_view.uplo_view, opA, A_view.diag_view,
                                             host_alpha, backend_handle, x_usm, host_beta, y_usm,
                                             dependencies);
        }
        else if (A_view.type_view == matrix_descr::symmetric ||
                 A_view.type_view == matrix_descr::hermitian) {
            return oneapi::mkl::sparse::symv(queue, A_view.uplo_view, host_alpha, backend_handle,
                                             x_usm, host_beta, y_usm, dependencies);
        }
        else {
            return oneapi::mkl::sparse::gemv(queue, opA, host_alpha, backend_handle, x_usm,
                                             host_beta, y_usm, dependencies);
        }
    }
}

sycl::event spmv(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                 matrix_view A_view, matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                 const void* beta, dense_vector_handle_t y_handle, spmv_alg alg,
                 spmv_descr_t spmv_descr, const std::vector<sycl::event>& dependencies) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    check_valid_spmv(__func__, opA, A_view, A_handle, x_handle, y_handle, is_alpha_host_accessible,
                     is_beta_host_accessible);

    if (!spmv_descr->optimized_called) {
        throw mkl::uninitialized("sparse_blas", __func__,
                                 "spmv_optimize must be called before spmv.");
    }
    CHECK_DESCR_MATCH(spmv_descr, opA, "spmv_optimize");
    CHECK_DESCR_MATCH(spmv_descr, A_view, "spmv_optimize");
    CHECK_DESCR_MATCH(spmv_descr, A_handle, "spmv_optimize");
    CHECK_DESCR_MATCH(spmv_descr, x_handle, "spmv_optimize");
    CHECK_DESCR_MATCH(spmv_descr, y_handle, "spmv_optimize");
    CHECK_DESCR_MATCH(spmv_descr, alg, "spmv_optimize");

    auto value_type = detail::get_internal_handle(A_handle)->get_value_type();
    DISPATCH_MKL_OPERATION("spmv", value_type, internal_spmv, queue, opA, alpha, A_view, A_handle,
                           x_handle, beta, y_handle, alg, spmv_descr, dependencies,
                           is_alpha_host_accessible, is_beta_host_accessible);
}

} // namespace oneapi::mkl::sparse::BACKEND
