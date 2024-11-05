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

/**
 * @file Similar to cublas_scope_handle.cpp
*/

#include "cusparse_scope_handle.hpp"

namespace oneapi::math::sparse::cusparse::detail {

/**
 * Inserts a new element in the map if its key is unique. This new element
 * is constructed in place using args as the arguments for the construction
 * of a value_type (which is an object of a pair type). The insertion only
 * takes place if no other element in the container has a key equivalent to
 * the one being emplaced (keys in a map container are unique).
 */
#ifdef ONEAPI_ONEMATH_PI_INTERFACE_REMOVED
thread_local cusparse_global_handle<ur_context_handle_t>
    CusparseScopedContextHandler::handle_helper = cusparse_global_handle<ur_context_handle_t>{};
#else
thread_local cusparse_global_handle<pi_context> CusparseScopedContextHandler::handle_helper =
    cusparse_global_handle<pi_context>{};
#endif

CusparseScopedContextHandler::CusparseScopedContextHandler(sycl::queue queue,
                                                           sycl::interop_handle& ih)
        : ih(ih),
          needToRecover_(false) {
    placedContext_ = new sycl::context(queue.get_context());
    auto cudaDevice = ih.get_native_device<sycl::backend::ext_oneapi_cuda>();
    CUcontext desired;
    CUDA_ERROR_FUNC(cuCtxGetCurrent, &original_);
    CUDA_ERROR_FUNC(cuDevicePrimaryCtxRetain, &desired, cudaDevice);
    if (original_ != desired) {
        // Sets the desired context as the active one for the thread
        CUDA_ERROR_FUNC(cuCtxSetCurrent, desired);
        // No context is installed and the suggested context is primary
        // This is the most common case. We can activate the context in the
        // thread and leave it there until all the PI context referring to the
        // same underlying CUDA primary context are destroyed. This emulates
        // the behaviour of the CUDA runtime api, and avoids costly context
        // switches. No action is required on this side of the if.
        needToRecover_ = !(original_ == nullptr);
    }
}

CusparseScopedContextHandler::~CusparseScopedContextHandler() noexcept(false) {
    if (needToRecover_) {
        CUDA_ERROR_FUNC(cuCtxSetCurrent, original_);
    }
    delete placedContext_;
}

void ContextCallback(void* userData) {
    auto* ptr = static_cast<std::atomic<cusparseHandle_t>*>(userData);
    if (!ptr) {
        return;
    }
    auto handle = ptr->exchange(nullptr);
    if (handle != nullptr) {
        CUSPARSE_ERR_FUNC(cusparseDestroy, handle);
        handle = nullptr;
    }
    else {
        // if the handle is nullptr it means the handle was already destroyed by
        // the cusparse_global_handle destructor and we're free to delete the atomic
        // object.
        delete ptr;
    }
}

std::pair<cusparseHandle_t, CUstream> CusparseScopedContextHandler::get_handle_and_stream(
    const sycl::queue& queue) {
    auto cudaDevice = ih.get_native_device<sycl::backend::ext_oneapi_cuda>();
    CUcontext desired;
    CUDA_ERROR_FUNC(cuDevicePrimaryCtxRetain, &desired, cudaDevice);
#ifdef ONEAPI_ONEMATH_PI_INTERFACE_REMOVED
    auto piPlacedContext_ = reinterpret_cast<ur_context_handle_t>(desired);
#else
    auto piPlacedContext_ = reinterpret_cast<pi_context>(desired);
#endif
    CUstream streamId = get_stream(queue);
    auto it = handle_helper.cusparse_global_handle_mapper_.find(piPlacedContext_);
    if (it != handle_helper.cusparse_global_handle_mapper_.end()) {
        if (it->second == nullptr) {
            handle_helper.cusparse_global_handle_mapper_.erase(it);
        }
        else {
            auto handle = it->second->load();
            if (handle != nullptr) {
                cudaStream_t currentStreamId;
                CUSPARSE_ERR_FUNC(cusparseGetStream, handle, &currentStreamId);
                if (currentStreamId != streamId) {
                    CUSPARSE_ERR_FUNC(cusparseSetStream, handle, streamId);
                }
                return { handle, streamId };
            }
            else {
                handle_helper.cusparse_global_handle_mapper_.erase(it);
            }
        }
    }

    cusparseHandle_t handle;
    CUSPARSE_ERR_FUNC(cusparseCreate, &handle);
    CUSPARSE_ERR_FUNC(cusparseSetStream, handle, streamId);

    auto insert_iter = handle_helper.cusparse_global_handle_mapper_.insert(
        std::make_pair(piPlacedContext_, new std::atomic<cusparseHandle_t>(handle)));

    sycl::detail::pi::contextSetExtendedDeleter(*placedContext_, ContextCallback,
                                                insert_iter.first->second);

    return { handle, streamId };
}

cusparseHandle_t CusparseScopedContextHandler::get_handle(const sycl::queue& queue) {
    return get_handle_and_stream(queue).first;
}

CUstream CusparseScopedContextHandler::get_stream(const sycl::queue& queue) {
    return sycl::get_native<sycl::backend::ext_oneapi_cuda>(queue);
}

sycl::context CusparseScopedContextHandler::get_context(const sycl::queue& queue) {
    return queue.get_context();
}

} // namespace oneapi::math::sparse::cusparse::detail
