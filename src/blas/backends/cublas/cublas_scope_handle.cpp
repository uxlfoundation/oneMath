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
#include "cublas_scope_handle.hpp"
#if __has_include(<sycl/detail/common.hpp>)
#include <sycl/detail/common.hpp>
#else
#include <CL/sycl/detail/common.hpp>
#endif

namespace oneapi {
namespace mkl {
namespace blas {
namespace cublas {

/**
 * Inserts a new element in the map if its key is unique. This new element
 * is constructed in place using args as the arguments for the construction
 * of a value_type (which is an object of a pair type). The insertion only
 * takes place if no other element in the container has a key equivalent to
 * the one being emplaced (keys in a map container are unique).
 */
thread_local cublas_handle<CUcontext> CublasScopedContextHandler::handle_helper =
    cublas_handle<CUcontext>{};

CublasScopedContextHandler::CublasScopedContextHandler(sycl::queue queue, sycl::interop_handle& ih)
        : ih(ih) {
    placedContext_ = new sycl::context(queue.get_context());
    auto cudaDevice = ih.get_native_device<sycl::backend::ext_oneapi_cuda>();
    CUresult err;
    CUcontext desired;
    CUDA_ERROR_FUNC(cuDevicePrimaryCtxRetain, err, &desired, cudaDevice);
}

CublasScopedContextHandler::~CublasScopedContextHandler() noexcept(false) {
    delete placedContext_;
}

void ContextCallback(void* userData) {
    auto* ptr = static_cast<std::atomic<cublasHandle_t>*>(userData);
    if (!ptr) {
        return;
    }
    auto handle = ptr->exchange(nullptr);
    if (handle != nullptr) {
        cublasStatus_t err1;
        CUBLAS_ERROR_FUNC(cublasDestroy, err1, handle);
        handle = nullptr;
    }
    else {
        // if the handle is nullptr it means the handle was already destroyed by
        // the cublas_handle destructor and we're free to delete the atomic
        // object.
        delete ptr;
    }
}

cublasHandle_t CublasScopedContextHandler::get_handle(const sycl::queue& queue) {
    auto cudaDevice = ih.get_native_device<sycl::backend::ext_oneapi_cuda>();
    CUresult cuErr;
    CUcontext desiredCtx;
    CUDA_ERROR_FUNC(cuDevicePrimaryCtxRetain, cuErr, &desiredCtx, cudaDevice);
    CUstream streamId = get_stream(queue);
    cublasStatus_t err;
    auto it = handle_helper.cublas_handle_mapper_.find(desiredCtx);
    if (it != handle_helper.cublas_handle_mapper_.end()) {
        if (it->second == nullptr) {
            handle_helper.cublas_handle_mapper_.erase(it);
        }
        else {
            auto handle = it->second->load();
            if (handle != nullptr) {
                cudaStream_t currentStreamId;
                CUBLAS_ERROR_FUNC(cublasGetStream, err, handle, &currentStreamId);
                if (currentStreamId != streamId) {
                    CUBLAS_ERROR_FUNC(cublasSetStream, err, handle, streamId);
                }
                return handle;
            }
            else {
                handle_helper.cublas_handle_mapper_.erase(it);
            }
        }
    }

    cublasHandle_t handle;

    CUBLAS_ERROR_FUNC(cublasCreate, err, &handle);
    CUBLAS_ERROR_FUNC(cublasSetStream, err, handle, streamId);

    auto insert_iter = handle_helper.cublas_handle_mapper_.insert(
        std::make_pair(desiredCtx, new std::atomic<cublasHandle_t>(handle)));

    sycl::detail::pi::contextSetExtendedDeleter(*placedContext_, ContextCallback,
                                                insert_iter.first->second);

    return handle;
}

CUstream CublasScopedContextHandler::get_stream(const sycl::queue& queue) {
    return sycl::get_native<sycl::backend::ext_oneapi_cuda>(queue);
}
sycl::context CublasScopedContextHandler::get_context(const sycl::queue& queue) {
    return queue.get_context();
}

} // namespace cublas
} // namespace blas
} // namespace mkl
} // namespace oneapi
