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

#include "oneapi/mkl/sparse_blas/detail/rocsparse/onemkl_sparse_blas_rocsparse.hpp"

#include "sparse_blas/backends/rocsparse/rocsparse_error.hpp"
#include "sparse_blas/backends/rocsparse/rocsparse_handles.hpp"
#include "sparse_blas/backends/rocsparse/rocsparse_helper.hpp"
#include "sparse_blas/backends/rocsparse/rocsparse_task.hpp"
#include "sparse_blas/backends/rocsparse/rocsparse_scope_handle.hpp"
#include "sparse_blas/common_op_verification.hpp"
#include "sparse_blas/macros.hpp"
#include "sparse_blas/matrix_view_comparison.hpp"
#include "sparse_blas/sycl_helper.hpp"

namespace oneapi::mkl::sparse {

// Complete the definition of the incomplete type
struct spmm_descr {
    // Cache the hipStream_t and global handle to avoid relying on RocsparseScopedContextHandler to retrieve them.
    hipStream_t hip_stream;
    rocsparse_handle roc_handle;

    detail::generic_container workspace;
    std::size_t temp_buffer_size = 0;
    bool buffer_size_called = false;
    bool optimized_called = false;
    oneapi::mkl::transpose last_optimized_opA;
    oneapi::mkl::transpose last_optimized_opB;
    oneapi::mkl::sparse::matrix_view last_optimized_A_view;
    oneapi::mkl::sparse::matrix_handle_t last_optimized_A_handle;
    oneapi::mkl::sparse::dense_matrix_handle_t last_optimized_B_handle;
    oneapi::mkl::sparse::dense_matrix_handle_t last_optimized_C_handle;
    oneapi::mkl::sparse::spmm_alg last_optimized_alg;
};

} // namespace oneapi::mkl::sparse

namespace oneapi::mkl::sparse::rocsparse {

namespace detail {

inline auto get_roc_spmm_alg(spmm_alg alg) {
    switch (alg) {
        case spmm_alg::coo_alg1: return rocsparse_spmm_alg_coo_segmented;
        case spmm_alg::coo_alg2: return rocsparse_spmm_alg_coo_atomic;
        case spmm_alg::coo_alg3: return rocsparse_spmm_alg_coo_segmented_atomic;
        case spmm_alg::csr_alg1: return rocsparse_spmm_alg_csr;
        case spmm_alg::csr_alg2: return rocsparse_spmm_alg_csr_row_split;
        case spmm_alg::csr_alg3: return rocsparse_spmm_alg_csr_merge;
        default: return rocsparse_spmm_alg_default;
    }
}

void check_valid_spmm(const std::string& function_name, matrix_view A_view,
                      matrix_handle_t A_handle, dense_matrix_handle_t B_handle,
                      dense_matrix_handle_t C_handle, bool is_alpha_host_accessible,
                      bool is_beta_host_accessible) {
    check_valid_spmm_common(function_name, A_view, A_handle, B_handle, C_handle,
                            is_alpha_host_accessible, is_beta_host_accessible);
    A_handle->check_valid_handle(function_name);
}

inline void common_spmm_optimize(
    oneapi::mkl::transpose opA, oneapi::mkl::transpose opB, bool is_alpha_host_accessible,
    oneapi::mkl::sparse::matrix_view A_view, oneapi::mkl::sparse::matrix_handle_t A_handle,
    oneapi::mkl::sparse::dense_matrix_handle_t B_handle, bool is_beta_host_accessible,
    oneapi::mkl::sparse::dense_matrix_handle_t C_handle, oneapi::mkl::sparse::spmm_alg alg,
    oneapi::mkl::sparse::spmm_descr_t spmm_descr) {
    check_valid_spmm("spmm_optimize", A_view, A_handle, B_handle, C_handle,
                     is_alpha_host_accessible, is_beta_host_accessible);
    if (!spmm_descr->buffer_size_called) {
        throw mkl::uninitialized("sparse_blas", "spmm_optimize",
                                 "spmm_buffer_size must be called before spmm_optimize.");
    }
    spmm_descr->optimized_called = true;
    spmm_descr->last_optimized_opA = opA;
    spmm_descr->last_optimized_opB = opB;
    spmm_descr->last_optimized_A_view = A_view;
    spmm_descr->last_optimized_A_handle = A_handle;
    spmm_descr->last_optimized_B_handle = B_handle;
    spmm_descr->last_optimized_C_handle = C_handle;
    spmm_descr->last_optimized_alg = alg;
}

void spmm_optimize_impl(rocsparse_handle roc_handle, oneapi::mkl::transpose opA,
                        oneapi::mkl::transpose opB, const void* alpha,
                        oneapi::mkl::sparse::matrix_handle_t A_handle,
                        oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void* beta,
                        oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                        oneapi::mkl::sparse::spmm_alg alg, std::size_t buffer_size,
                        void* workspace_ptr, bool is_alpha_host_accessible) {
    auto roc_a = A_handle->backend_handle;
    auto roc_b = B_handle->backend_handle;
    auto roc_c = C_handle->backend_handle;
    auto roc_op_a = get_roc_operation(opA);
    auto roc_op_b = get_roc_operation(opB);
    auto roc_type = get_roc_value_type(A_handle->value_container.data_type);
    auto roc_alg = get_roc_spmm_alg(alg);
    set_pointer_mode(roc_handle, is_alpha_host_accessible);
    // rocsparse_spmm_stage_preprocess stage is blocking
    auto status =
        rocsparse_spmm(roc_handle, roc_op_a, roc_op_b, alpha, roc_a, roc_b, beta, roc_c, roc_type,
                       roc_alg, rocsparse_spmm_stage_preprocess, &buffer_size, workspace_ptr);
    check_status(status, "spmm_optimize");
}

} // namespace detail

void init_spmm_descr(sycl::queue& /*queue*/, spmm_descr_t* p_spmm_descr) {
    *p_spmm_descr = new spmm_descr();
}

sycl::event release_spmm_descr(sycl::queue& queue, spmm_descr_t spmm_descr,
                               const std::vector<sycl::event>& dependencies) {
    if (!spmm_descr) {
        return detail::collapse_dependencies(queue, dependencies);
    }

    auto release_functor = [=]() {
        spmm_descr->roc_handle = nullptr;
        spmm_descr->last_optimized_A_handle = nullptr;
        spmm_descr->last_optimized_B_handle = nullptr;
        spmm_descr->last_optimized_C_handle = nullptr;
        delete spmm_descr;
    };

    // Use dispatch_submit to ensure the descriptor is kept alive as long as the buffers are used
    // dispatch_submit can only be used if the descriptor's handles are valid
    if (spmm_descr->last_optimized_A_handle &&
        spmm_descr->last_optimized_A_handle->all_use_buffer() &&
        spmm_descr->last_optimized_B_handle && spmm_descr->last_optimized_C_handle &&
        spmm_descr->workspace.use_buffer()) {
        auto dispatch_functor = [=](sycl::interop_handle, sycl::accessor<std::uint8_t>) {
            release_functor();
        };
        return detail::dispatch_submit(
            __func__, queue, dispatch_functor, spmm_descr->last_optimized_A_handle,
            spmm_descr->workspace.get_buffer<std::uint8_t>(), spmm_descr->last_optimized_B_handle,
            spmm_descr->last_optimized_C_handle);
    }

    // Release used if USM is used or if the descriptor has been released before spmm_optimize has succeeded
    sycl::event event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.host_task(release_functor);
    });
    return event;
}

void spmm_buffer_size(sycl::queue& queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                      const void* alpha, oneapi::mkl::sparse::matrix_view A_view,
                      oneapi::mkl::sparse::matrix_handle_t A_handle,
                      oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void* beta,
                      oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                      oneapi::mkl::sparse::spmm_alg alg,
                      oneapi::mkl::sparse::spmm_descr_t spmm_descr, std::size_t& temp_buffer_size) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    detail::check_valid_spmm(__func__, A_view, A_handle, B_handle, C_handle,
                             is_alpha_host_accessible, is_beta_host_accessible);
    bool is_in_order_queue = queue.is_in_order();
    auto functor = [=, &temp_buffer_size](sycl::interop_handle ih) {
        detail::RocsparseScopedContextHandler sc(queue, ih);
        auto [roc_handle, hip_stream] = sc.get_handle_and_stream(queue);
        spmm_descr->roc_handle = roc_handle;
        spmm_descr->hip_stream = hip_stream;
        auto roc_a = A_handle->backend_handle;
        auto roc_b = B_handle->backend_handle;
        auto roc_c = C_handle->backend_handle;
        auto roc_op_a = detail::get_roc_operation(opA);
        auto roc_op_b = detail::get_roc_operation(opB);
        auto roc_type = detail::get_roc_value_type(A_handle->value_container.data_type);
        auto roc_alg = detail::get_roc_spmm_alg(alg);
        detail::set_pointer_mode(roc_handle, is_alpha_host_accessible);
        auto status = rocsparse_spmm(roc_handle, roc_op_a, roc_op_b, alpha, roc_a, roc_b, beta,
                                     roc_c, roc_type, roc_alg, rocsparse_spmm_stage_buffer_size,
                                     &temp_buffer_size, nullptr);
        detail::check_status(status, __func__);
        detail::synchronize_if_needed(is_in_order_queue, hip_stream);
    };
    auto event = detail::dispatch_submit(__func__, queue, functor, A_handle, B_handle, C_handle);
    event.wait_and_throw();
    spmm_descr->temp_buffer_size = temp_buffer_size;
    spmm_descr->buffer_size_called = true;
}

void spmm_optimize(sycl::queue& queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                   const void* alpha, oneapi::mkl::sparse::matrix_view A_view,
                   oneapi::mkl::sparse::matrix_handle_t A_handle,
                   oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void* beta,
                   oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                   oneapi::mkl::sparse::spmm_alg alg, oneapi::mkl::sparse::spmm_descr_t spmm_descr,
                   sycl::buffer<std::uint8_t, 1> workspace) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    if (!A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    detail::common_spmm_optimize(opA, opB, is_alpha_host_accessible, A_view, A_handle, B_handle,
                                 is_beta_host_accessible, C_handle, alg, spmm_descr);
    // Copy the buffer to extend its lifetime until the descriptor is free'd.
    spmm_descr->workspace.set_buffer_untyped(workspace);
    if (alg == oneapi::mkl::sparse::spmm_alg::no_optimize_alg) {
        return;
    }
    std::size_t buffer_size = spmm_descr->temp_buffer_size;

    // The accessor can only be created if the buffer size is greater than 0
    if (buffer_size > 0) {
        auto functor = [=](sycl::interop_handle ih, sycl::accessor<std::uint8_t> workspace_acc) {
            auto roc_handle = spmm_descr->roc_handle;
            auto workspace_ptr = detail::get_mem(ih, workspace_acc);
            detail::spmm_optimize_impl(roc_handle, opA, opB, alpha, A_handle, B_handle, beta,
                                       C_handle, alg, buffer_size, workspace_ptr,
                                       is_alpha_host_accessible);
        };

        detail::dispatch_submit(__func__, queue, functor, A_handle, workspace, B_handle, C_handle);
    }
    else {
        auto functor = [=](sycl::interop_handle) {
            auto roc_handle = spmm_descr->roc_handle;
            detail::spmm_optimize_impl(roc_handle, opA, opB, alpha, A_handle, B_handle, beta,
                                       C_handle, alg, buffer_size, nullptr,
                                       is_alpha_host_accessible);
        };

        detail::dispatch_submit(__func__, queue, functor, A_handle, B_handle, C_handle);
    }
}

sycl::event spmm_optimize(sycl::queue& queue, oneapi::mkl::transpose opA,
                          oneapi::mkl::transpose opB, const void* alpha,
                          oneapi::mkl::sparse::matrix_view A_view,
                          oneapi::mkl::sparse::matrix_handle_t A_handle,
                          oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void* beta,
                          oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                          oneapi::mkl::sparse::spmm_alg alg,
                          oneapi::mkl::sparse::spmm_descr_t spmm_descr, void* workspace,
                          const std::vector<sycl::event>& dependencies) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    if (A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    detail::common_spmm_optimize(opA, opB, is_alpha_host_accessible, A_view, A_handle, B_handle,
                                 is_beta_host_accessible, C_handle, alg, spmm_descr);
    spmm_descr->workspace.usm_ptr = workspace;
    if (alg == oneapi::mkl::sparse::spmm_alg::no_optimize_alg) {
        return detail::collapse_dependencies(queue, dependencies);
    }
    std::size_t buffer_size = spmm_descr->temp_buffer_size;
    auto functor = [=](sycl::interop_handle) {
        auto roc_handle = spmm_descr->roc_handle;
        detail::spmm_optimize_impl(roc_handle, opA, opB, alpha, A_handle, B_handle, beta, C_handle,
                                   alg, buffer_size, workspace, is_alpha_host_accessible);
    };

    return detail::dispatch_submit(__func__, queue, dependencies, functor, A_handle, B_handle,
                                   C_handle);
}

sycl::event spmm(sycl::queue& queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                 const void* alpha, oneapi::mkl::sparse::matrix_view A_view,
                 oneapi::mkl::sparse::matrix_handle_t A_handle,
                 oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void* beta,
                 oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                 oneapi::mkl::sparse::spmm_alg alg, oneapi::mkl::sparse::spmm_descr_t spmm_descr,
                 const std::vector<sycl::event>& dependencies) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    if (A_handle->all_use_buffer() != spmm_descr->workspace.use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    if (!spmm_descr->optimized_called) {
        throw mkl::uninitialized(
            "sparse_blas", __func__,
            "spmm_optimize must be called with the same arguments before spmm.");
    }
    CHECK_DESCR_MATCH(spmm_descr, opA, "spmm_optimize");
    CHECK_DESCR_MATCH(spmm_descr, opB, "spmm_optimize");
    CHECK_DESCR_MATCH(spmm_descr, A_view, "spmm_optimize");
    CHECK_DESCR_MATCH(spmm_descr, A_handle, "spmm_optimize");
    CHECK_DESCR_MATCH(spmm_descr, B_handle, "spmm_optimize");
    CHECK_DESCR_MATCH(spmm_descr, C_handle, "spmm_optimize");
    CHECK_DESCR_MATCH(spmm_descr, alg, "spmm_optimize");
    detail::check_valid_spmm(__func__, A_view, A_handle, B_handle, C_handle,
                             is_alpha_host_accessible, is_beta_host_accessible);
    A_handle->mark_used();
    bool is_in_order_queue = queue.is_in_order();
    auto compute_functor = [=](void* workspace_ptr) {
        auto roc_handle = spmm_descr->roc_handle;
        auto hip_stream = spmm_descr->hip_stream;
        auto buffer_size = spmm_descr->temp_buffer_size;
        auto roc_a = A_handle->backend_handle;
        auto roc_b = B_handle->backend_handle;
        auto roc_c = C_handle->backend_handle;
        auto roc_op_a = detail::get_roc_operation(opA);
        auto roc_op_b = detail::get_roc_operation(opB);
        auto roc_type = detail::get_roc_value_type(A_handle->value_container.data_type);
        auto roc_alg = detail::get_roc_spmm_alg(alg);
        detail::set_pointer_mode(roc_handle, is_alpha_host_accessible);
        auto status = rocsparse_spmm(roc_handle, roc_op_a, roc_op_b, alpha, roc_a, roc_b, beta,
                                     roc_c, roc_type, roc_alg, rocsparse_spmm_stage_compute,
                                     &buffer_size, workspace_ptr);
        detail::check_status(status, __func__);
        detail::synchronize_if_needed(is_in_order_queue, hip_stream);
    };
    // The accessor can only be created if the buffer size is greater than 0
    if (A_handle->all_use_buffer() && spmm_descr->temp_buffer_size > 0) {
        auto functor_buffer = [=](sycl::interop_handle ih,
                                  sycl::accessor<std::uint8_t> workspace_acc) {
            auto workspace_ptr = detail::get_mem(ih, workspace_acc);
            compute_functor(workspace_ptr);
        };
        return detail::dispatch_submit_native_ext(__func__, queue, functor_buffer, A_handle,
                                                  spmm_descr->workspace.get_buffer<std::uint8_t>(),
                                                  B_handle, C_handle);
    }
    else {
        // The same dispatch_submit can be used for USM or buffers if no
        // workspace accessor is needed.
        // workspace_ptr will be a nullptr in the latter case.
        auto workspace_ptr = spmm_descr->workspace.usm_ptr;
        auto functor_usm = [=](sycl::interop_handle) {
            compute_functor(workspace_ptr);
        };
        return detail::dispatch_submit_native_ext(__func__, queue, dependencies, functor_usm,
                                                  A_handle, B_handle, C_handle);
    }
}

} // namespace oneapi::mkl::sparse::rocsparse
