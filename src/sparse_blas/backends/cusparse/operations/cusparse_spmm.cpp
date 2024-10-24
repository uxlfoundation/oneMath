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

#include "sparse_blas/backends/cusparse/cusparse_error.hpp"
#include "sparse_blas/backends/cusparse/cusparse_helper.hpp"
#include "sparse_blas/backends/cusparse/cusparse_task.hpp"
#include "sparse_blas/backends/cusparse/cusparse_handles.hpp"
#include "sparse_blas/common_op_verification.hpp"
#include "sparse_blas/macros.hpp"
#include "sparse_blas/matrix_view_comparison.hpp"
#include "sparse_blas/sycl_helper.hpp"

namespace oneapi::mkl::sparse {

// Complete the definition of the incomplete type
struct spmm_descr {
    // Cache the CUstream and global handle to avoid relying on CusparseScopedContextHandler to retrieve them.
    // cuSPARSE seem to implicitly require to use the same CUstream for a whole operation (buffer_size, optimization and computation steps).
    // This is needed as the default SYCL queue is out-of-order which can have a different CUstream for each host_task or native_command.
    CUstream cu_stream;
    cusparseHandle_t cu_handle;

    detail::generic_container workspace;
    std::size_t temp_buffer_size = 0;
    bool buffer_size_called = false;
    bool optimized_called = false;
    oneapi::mkl::transpose last_optimized_opA;
    oneapi::mkl::transpose last_optimized_opB;
    matrix_view last_optimized_A_view;
    matrix_handle_t last_optimized_A_handle;
    dense_matrix_handle_t last_optimized_B_handle;
    dense_matrix_handle_t last_optimized_C_handle;
    spmm_alg last_optimized_alg;
};

} // namespace oneapi::mkl::sparse

namespace oneapi::mkl::sparse::cusparse {

void init_spmm_descr(sycl::queue& /*queue*/, spmm_descr_t* p_spmm_descr) {
    *p_spmm_descr = new spmm_descr();
}

sycl::event release_spmm_descr(sycl::queue& queue, spmm_descr_t spmm_descr,
                               const std::vector<sycl::event>& dependencies) {
    if (!spmm_descr) {
        return detail::collapse_dependencies(queue, dependencies);
    }

    auto release_functor = [=]() {
        spmm_descr->cu_handle = nullptr;
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
        return dispatch_submit(
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

inline auto get_cuda_spmm_alg(spmm_alg alg) {
    switch (alg) {
        case spmm_alg::coo_alg1: return CUSPARSE_SPMM_COO_ALG1;
        case spmm_alg::coo_alg2: return CUSPARSE_SPMM_COO_ALG2;
        case spmm_alg::coo_alg3: return CUSPARSE_SPMM_COO_ALG3;
        case spmm_alg::coo_alg4: return CUSPARSE_SPMM_COO_ALG4;
        case spmm_alg::csr_alg1: return CUSPARSE_SPMM_CSR_ALG1;
        case spmm_alg::csr_alg2: return CUSPARSE_SPMM_CSR_ALG2;
        case spmm_alg::csr_alg3: return CUSPARSE_SPMM_CSR_ALG3;
        default: return CUSPARSE_SPMM_ALG_DEFAULT;
    }
}

void check_valid_spmm(const std::string& function_name, oneapi::mkl::transpose opA,
                      oneapi::mkl::transpose opB, matrix_view A_view, matrix_handle_t A_handle,
                      dense_matrix_handle_t B_handle, dense_matrix_handle_t C_handle,
                      bool is_alpha_host_accessible, bool is_beta_host_accessible, spmm_alg alg) {
    detail::check_valid_spmm_common(function_name, A_view, A_handle, B_handle, C_handle,
                                    is_alpha_host_accessible, is_beta_host_accessible);
    check_valid_matrix_properties(function_name, A_handle);
    if (alg == spmm_alg::csr_alg3 && opA != oneapi::mkl::transpose::nontrans) {
        throw mkl::unimplemented(
            "sparse_blas", function_name,
            "The backend does not support spmm with the algorithm `spmm_alg::csr_alg3` if `opA` is not `transpose::nontrans`.");
    }
    if (alg == spmm_alg::csr_alg3 && opB == oneapi::mkl::transpose::conjtrans) {
        throw mkl::unimplemented(
            "sparse_blas", function_name,
            "The backend does not support spmm with the algorithm `spmm_alg::csr_alg3` if `opB` is `transpose::conjtrans`.");
    }
    if (alg == spmm_alg::csr_alg3 && opB == oneapi::mkl::transpose::trans &&
        A_handle->get_value_type() == detail::data_type::real_fp64) {
        // TODO: Remove once the issue is fixed: https://forums.developer.nvidia.com/t/cusparse-spmm-sample-failing-with-misaligned-address/311022
        throw mkl::unimplemented(
            "sparse_blas", function_name,
            "The backend does not support spmm with the algorithm `spmm_alg::csr_alg3` if `opB` is `transpose::trans` and the real fp64 precision is used.");
    }
}

void spmm_buffer_size(sycl::queue& queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                      const void* alpha, matrix_view A_view, matrix_handle_t A_handle,
                      dense_matrix_handle_t B_handle, const void* beta,
                      dense_matrix_handle_t C_handle, spmm_alg alg, spmm_descr_t spmm_descr,
                      std::size_t& temp_buffer_size) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    check_valid_spmm(__func__, opA, opB, A_view, A_handle, B_handle, C_handle,
                     is_alpha_host_accessible, is_beta_host_accessible, alg);
    auto functor = [=, &temp_buffer_size](sycl::interop_handle ih) {
        CusparseScopedContextHandler sc(queue, ih);
        auto [cu_handle, cu_stream] = sc.get_handle_and_stream(queue);
        spmm_descr->cu_handle = cu_handle;
        spmm_descr->cu_stream = cu_stream;
        auto cu_a = A_handle->backend_handle;
        auto cu_b = B_handle->backend_handle;
        auto cu_c = C_handle->backend_handle;
        auto type = A_handle->value_container.data_type;
        auto cu_op_a = get_cuda_operation(type, opA);
        auto cu_op_b = get_cuda_operation(type, opB);
        auto cu_type = get_cuda_value_type(type);
        auto cu_alg = get_cuda_spmm_alg(alg);
        set_pointer_mode(cu_handle, is_alpha_host_accessible);
        auto status = cusparseSpMM_bufferSize(cu_handle, cu_op_a, cu_op_b, alpha, cu_a, cu_b, beta,
                                              cu_c, cu_type, cu_alg, &temp_buffer_size);
        check_status(status, __func__);
    };
    auto event = dispatch_submit(__func__, queue, functor, A_handle, B_handle, C_handle);
    event.wait_and_throw();
    spmm_descr->temp_buffer_size = temp_buffer_size;
    spmm_descr->buffer_size_called = true;
}

inline void common_spmm_optimize(oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                                 bool is_alpha_host_accessible, matrix_view A_view,
                                 matrix_handle_t A_handle, dense_matrix_handle_t B_handle,
                                 bool is_beta_host_accessible, dense_matrix_handle_t C_handle,
                                 spmm_alg alg, spmm_descr_t spmm_descr) {
    check_valid_spmm("spmm_optimize", opA, opB, A_view, A_handle, B_handle, C_handle,
                     is_alpha_host_accessible, is_beta_host_accessible, alg);
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

void spmm_optimize_impl(cusparseHandle_t cu_handle, oneapi::mkl::transpose opA,
                        oneapi::mkl::transpose opB, const void* alpha, matrix_handle_t A_handle,
                        dense_matrix_handle_t B_handle, const void* beta,
                        dense_matrix_handle_t C_handle, spmm_alg alg, void* workspace_ptr,
                        bool is_alpha_host_accessible) {
    auto cu_a = A_handle->backend_handle;
    auto cu_b = B_handle->backend_handle;
    auto cu_c = C_handle->backend_handle;
    auto type = A_handle->value_container.data_type;
    auto cu_op_a = get_cuda_operation(type, opA);
    auto cu_op_b = get_cuda_operation(type, opB);
    auto cu_type = get_cuda_value_type(type);
    auto cu_alg = get_cuda_spmm_alg(alg);
    set_pointer_mode(cu_handle, is_alpha_host_accessible);
    auto status = cusparseSpMM_preprocess(cu_handle, cu_op_a, cu_op_b, alpha, cu_a, cu_b, beta,
                                          cu_c, cu_type, cu_alg, workspace_ptr);
    check_status(status, "spmm_optimize");
}

void spmm_optimize(sycl::queue& queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                   const void* alpha, matrix_view A_view, matrix_handle_t A_handle,
                   dense_matrix_handle_t B_handle, const void* beta, dense_matrix_handle_t C_handle,
                   spmm_alg alg, spmm_descr_t spmm_descr, sycl::buffer<std::uint8_t, 1> workspace) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    if (!A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    common_spmm_optimize(opA, opB, is_alpha_host_accessible, A_view, A_handle, B_handle,
                         is_beta_host_accessible, C_handle, alg, spmm_descr);
    // Copy the buffer to extend its lifetime until the descriptor is free'd.
    spmm_descr->workspace.set_buffer_untyped(workspace);
    if (alg == spmm_alg::no_optimize_alg || workspace.size() == 0) {
        // cusparseSpMM_preprocess cannot be called if the workspace is empty
        return;
    }
    auto functor = [=](sycl::interop_handle ih, sycl::accessor<std::uint8_t> workspace_acc) {
        auto cu_handle = spmm_descr->cu_handle;
        auto workspace_ptr = get_mem(ih, workspace_acc);
        spmm_optimize_impl(cu_handle, opA, opB, alpha, A_handle, B_handle, beta, C_handle, alg,
                           workspace_ptr, is_alpha_host_accessible);
    };

    dispatch_submit(__func__, queue, functor, A_handle, workspace, B_handle, C_handle);
}

sycl::event spmm_optimize(sycl::queue& queue, oneapi::mkl::transpose opA,
                          oneapi::mkl::transpose opB, const void* alpha, matrix_view A_view,
                          matrix_handle_t A_handle, dense_matrix_handle_t B_handle,
                          const void* beta, dense_matrix_handle_t C_handle, spmm_alg alg,
                          spmm_descr_t spmm_descr, void* workspace,
                          const std::vector<sycl::event>& dependencies) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    if (A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    common_spmm_optimize(opA, opB, is_alpha_host_accessible, A_view, A_handle, B_handle,
                         is_beta_host_accessible, C_handle, alg, spmm_descr);
    spmm_descr->workspace.usm_ptr = workspace;
    if (alg == spmm_alg::no_optimize_alg || workspace == nullptr) {
        // cusparseSpMM_preprocess cannot be called if the workspace is empty
        return detail::collapse_dependencies(queue, dependencies);
    }
    auto functor = [=](sycl::interop_handle) {
        auto cu_handle = spmm_descr->cu_handle;
        spmm_optimize_impl(cu_handle, opA, opB, alpha, A_handle, B_handle, beta, C_handle, alg,
                           workspace, is_alpha_host_accessible);
    };

    return dispatch_submit(__func__, queue, dependencies, functor, A_handle, B_handle, C_handle);
}

sycl::event spmm(sycl::queue& queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                 const void* alpha, matrix_view A_view, matrix_handle_t A_handle,
                 dense_matrix_handle_t B_handle, const void* beta, dense_matrix_handle_t C_handle,
                 spmm_alg alg, spmm_descr_t spmm_descr,
                 const std::vector<sycl::event>& dependencies) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    check_valid_spmm(__func__, opA, opB, A_view, A_handle, B_handle, C_handle,
                     is_alpha_host_accessible, is_beta_host_accessible, alg);
    if (A_handle->all_use_buffer() != spmm_descr->workspace.use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }

    if (!spmm_descr->optimized_called) {
        throw mkl::uninitialized("sparse_blas", __func__,
                                 "spmm_optimize must be called before spmm.");
    }
    CHECK_DESCR_MATCH(spmm_descr, opA, "spmm_optimize");
    CHECK_DESCR_MATCH(spmm_descr, opB, "spmm_optimize");
    CHECK_DESCR_MATCH(spmm_descr, A_view, "spmm_optimize");
    CHECK_DESCR_MATCH(spmm_descr, A_handle, "spmm_optimize");
    CHECK_DESCR_MATCH(spmm_descr, B_handle, "spmm_optimize");
    CHECK_DESCR_MATCH(spmm_descr, C_handle, "spmm_optimize");
    CHECK_DESCR_MATCH(spmm_descr, alg, "spmm_optimize");

    auto compute_functor = [=](void* workspace_ptr) {
        auto cu_handle = spmm_descr->cu_handle;
        auto cu_a = A_handle->backend_handle;
        auto cu_b = B_handle->backend_handle;
        auto cu_c = C_handle->backend_handle;
        auto type = A_handle->value_container.data_type;
        auto cu_op_a = get_cuda_operation(type, opA);
        auto cu_op_b = get_cuda_operation(type, opB);
        auto cu_type = get_cuda_value_type(type);
        auto cu_alg = get_cuda_spmm_alg(alg);
        set_pointer_mode(cu_handle, is_alpha_host_accessible);
        auto status = cusparseSpMM(cu_handle, cu_op_a, cu_op_b, alpha, cu_a, cu_b, beta, cu_c,
                                   cu_type, cu_alg, workspace_ptr);
        check_status(status, __func__);
#ifndef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
        auto cu_stream = spmm_descr->cu_stream;
        CUDA_ERROR_FUNC(cuStreamSynchronize, cu_stream);
#endif
    };
    if (A_handle->all_use_buffer() && spmm_descr->temp_buffer_size > 0) {
        // The accessor can only be created if the buffer size is greater than 0
        auto functor_buffer = [=](sycl::interop_handle ih,
                                  sycl::accessor<std::uint8_t> workspace_acc) {
            auto workspace_ptr = get_mem(ih, workspace_acc);
            compute_functor(workspace_ptr);
        };
        return dispatch_submit_native_ext(__func__, queue, functor_buffer, A_handle,
                                          spmm_descr->workspace.get_buffer<std::uint8_t>(),
                                          B_handle, C_handle);
    }
    else {
        // The same dispatch_submit can be used for USM or buffers if no
        // workspace accessor is needed, workspace_ptr will be a nullptr in the
        // latter case.
        auto workspace_ptr = spmm_descr->workspace.usm_ptr;
        auto functor_usm = [=](sycl::interop_handle) {
            compute_functor(workspace_ptr);
        };
        return dispatch_submit_native_ext(__func__, queue, dependencies, functor_usm, A_handle,
                                          B_handle, C_handle);
    }
}

} // namespace oneapi::mkl::sparse::cusparse
