/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Copyright 2022 Intel Corporation
*
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
#include "rocsolver_scope_handle.hpp"
#if __has_include(<sycl/detail/common.hpp>)
#include <sycl/detail/common.hpp>
#else
#include <CL/sycl/detail/common.hpp>
#endif

namespace oneapi {
namespace mkl {
namespace lapack {
namespace rocsolver {

/**
 * Inserts a new element in the map if its key is unique. This new element
 * is constructed in place using args as the arguments for the construction
 * of a value_type (which is an object of a pair type). The insertion only
 * takes place if no other element in the container has a key equivalent to
 * the one being emplaced (keys in a map container are unique).
 */
#ifdef ONEMKL_PI_INTERFACE_REMOVED
thread_local rocsolver_handle<ur_context_handle_t> RocsolverScopedContextHandler::handle_helper =
    rocsolver_handle<ur_context_handle_t>{};
#else
thread_local rocsolver_handle<pi_context> RocsolverScopedContextHandler::handle_helper =
    rocsolver_handle<pi_context>{};
#endif

RocsolverScopedContextHandler::RocsolverScopedContextHandler(sycl::queue queue,
                                                             sycl::interop_handle& ih)
        : ih(ih),
          needToRecover_(false) {
    placedContext_ = new sycl::context(queue.get_context());
    auto hipDevice = ih.get_native_device<sycl::backend::ext_oneapi_hip>();
    hipError_t err;
    hipCtx_t desired;
    HIP_ERROR_FUNC(hipCtxGetCurrent, err, &original_);
    HIP_ERROR_FUNC(hipDevicePrimaryCtxRetain, err, &desired, hipDevice);
    if (original_ != desired) {
        // Sets the desired context as the active one for the thread
        HIP_ERROR_FUNC(hipCtxSetCurrent, err, desired);
        // No context is installed and the suggested context is primary
        // This is the most common case. We can activate the context in the
        // thread and leave it there until all the PI context referring to the
        // same underlying rocblas primary context are destroyed. This emulates
        // the behaviour of the rocblas runtime api, and avoids costly context
        // switches. No action is required on this side of the if.
        needToRecover_ = !(original_ == nullptr);
    }
}

RocsolverScopedContextHandler::~RocsolverScopedContextHandler() noexcept(false) {
    if (needToRecover_) {
        hipError_t err;
        HIP_ERROR_FUNC(hipCtxSetCurrent, err, original_);
    }
    delete placedContext_;
}

void ContextCallback(void* userData) {
    auto* ptr = static_cast<std::atomic<rocblas_handle>*>(userData);
    if (!ptr) {
        return;
    }
    auto handle = ptr->exchange(nullptr);
    if (handle != nullptr) {
        rocblas_status err1;
        ROCSOLVER_ERROR_FUNC(rocblas_destroy_handle, err1, handle);
        handle = nullptr;
    }
    else {
        // if the handle is nullptr it means the handle was already destroyed by
        // the rocblas_handle destructor and we're free to delete the atomic
        // object.
        delete ptr;
    }
}

rocblas_handle RocsolverScopedContextHandler::get_handle(const sycl::queue& queue) {
    auto hipDevice = ih.get_native_device<sycl::backend::ext_oneapi_hip>();
    hipError_t hipErr;
    hipCtx_t desired;
    HIP_ERROR_FUNC(hipDevicePrimaryCtxRetain, hipErr, &desired, hipDevice);
#ifdef ONEMKL_PI_INTERFACE_REMOVED
    auto piPlacedContext_ = reinterpret_cast<ur_context_handle_t>(desired);
#else
    auto piPlacedContext_ = reinterpret_cast<pi_context>(desired);
#endif
    hipStream_t streamId = get_stream(queue);
    rocblas_status err;
    auto it = handle_helper.rocsolver_handle_mapper_.find(piPlacedContext_);
    if (it != handle_helper.rocsolver_handle_mapper_.end()) {
        if (it->second == nullptr) {
            handle_helper.rocsolver_handle_mapper_.erase(it);
        }
        else {
            auto handle = it->second->load();
            if (handle != nullptr) {
                hipStream_t currentStreamId;
                ROCSOLVER_ERROR_FUNC(rocblas_get_stream, err, handle, &currentStreamId);
                if (currentStreamId != streamId) {
                    ROCSOLVER_ERROR_FUNC(rocblas_set_stream, err, handle, streamId);
                }
                return handle;
            }
            else {
                handle_helper.rocsolver_handle_mapper_.erase(it);
            }
        }
    }

    rocblas_handle handle;

    ROCSOLVER_ERROR_FUNC(rocblas_create_handle, err, &handle);
    ROCSOLVER_ERROR_FUNC(rocblas_set_stream, err, handle, streamId);

    auto insert_iter = handle_helper.rocsolver_handle_mapper_.insert(
        std::make_pair(piPlacedContext_, new std::atomic<rocblas_handle>(handle)));

    sycl::detail::pi::contextSetExtendedDeleter(*placedContext_, ContextCallback,
                                                insert_iter.first->second);

    return handle;
}

hipStream_t RocsolverScopedContextHandler::get_stream(const sycl::queue& queue) {
    return sycl::get_native<sycl::backend::ext_oneapi_hip>(queue);
}
sycl::context RocsolverScopedContextHandler::get_context(const sycl::queue& queue) {
    return queue.get_context();
}

} // namespace rocsolver
} // namespace lapack
} // namespace mkl
} // namespace oneapi
