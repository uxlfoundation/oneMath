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
 * @file Similar to rocblas_scope_handle.cpp
*/

#include "rocsparse_scope_handle.hpp"

namespace oneapi::mkl::sparse::rocsparse::detail {

/**
 * Inserts a new element in the map if its key is unique. This new element
 * is constructed in place using args as the arguments for the construction
 * of a value_type (which is an object of a pair type). The insertion only
 * takes place if no other element in the container has a key equivalent to
 * the one being emplaced (keys in a map container are unique).
 */
#ifdef ONEAPI_ONEMKL_PI_INTERFACE_REMOVED
thread_local rocsparse_handle_container<ur_context_handle_t>
    RocsparseScopedContextHandler::handle_helper =
        rocsparse_handle_container<ur_context_handle_t>{};
#else
thread_local rocsparse_handle_container<pi_context> RocsparseScopedContextHandler::handle_helper =
    rocsparse_handle_container<pi_context>{};
#endif

// Disable warning for deprecated hipCtxGetCurrent and similar hip runtime functions
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

RocsparseScopedContextHandler::RocsparseScopedContextHandler(sycl::queue queue,
                                                             sycl::interop_handle& ih)
        : ih(ih),
          needToRecover_(false) {
    placedContext_ = new sycl::context(queue.get_context());
    auto hipDevice = ih.get_native_device<sycl::backend::ext_oneapi_hip>();
    hipCtx_t desired;
    HIP_ERROR_FUNC(hipCtxGetCurrent, &original_);
    HIP_ERROR_FUNC(hipDevicePrimaryCtxRetain, &desired, hipDevice);
    if (original_ != desired) {
        // Sets the desired context as the active one for the thread
        HIP_ERROR_FUNC(hipCtxSetCurrent, desired);
        // No context is installed and the suggested context is primary
        // This is the most common case. We can activate the context in the
        // thread and leave it there until all the PI context referring to the
        // same underlying rocsparse primary context are destroyed. This emulates
        // the behaviour of the rocsparse runtime api, and avoids costly context
        // switches. No action is required on this side of the if.
        needToRecover_ = !(original_ == nullptr);
    }
}

RocsparseScopedContextHandler::~RocsparseScopedContextHandler() noexcept(false) {
    if (needToRecover_) {
        HIP_ERROR_FUNC(hipCtxSetCurrent, original_);
    }
    delete placedContext_;
}

void ContextCallback(void* userData) {
    auto* atomic_ptr = static_cast<std::atomic<rocsparse_handle>*>(userData);
    if (!atomic_ptr) {
        return;
    }
    auto handle = atomic_ptr->exchange(nullptr);
    if (handle != nullptr) {
        ROCSPARSE_ERR_FUNC(rocsparse_destroy_handle, handle);
    }
    delete atomic_ptr;
}

std::pair<rocsparse_handle, hipStream_t> RocsparseScopedContextHandler::get_handle_and_stream(
    const sycl::queue& queue) {
    auto hipDevice = ih.get_native_device<sycl::backend::ext_oneapi_hip>();
    hipCtx_t desired;
    HIP_ERROR_FUNC(hipDevicePrimaryCtxRetain, &desired, hipDevice);
#ifdef ONEAPI_ONEMKL_PI_INTERFACE_REMOVED
    auto piPlacedContext_ = reinterpret_cast<ur_context_handle_t>(desired);
#else
    auto piPlacedContext_ = reinterpret_cast<pi_context>(desired);
#endif
    hipStream_t streamId = sycl::get_native<sycl::backend::ext_oneapi_hip>(queue);
    auto it = handle_helper.rocsparse_handle_container_mapper_.find(piPlacedContext_);
    if (it != handle_helper.rocsparse_handle_container_mapper_.end()) {
        if (it->second == nullptr) {
            handle_helper.rocsparse_handle_container_mapper_.erase(it);
        }
        else {
            auto handle = it->second->load();
            if (handle != nullptr) {
                hipStream_t currentStreamId;
                ROCSPARSE_ERR_FUNC(rocsparse_get_stream, handle, &currentStreamId);
                if (currentStreamId != streamId) {
                    ROCSPARSE_ERR_FUNC(rocsparse_set_stream, handle, streamId);
                }
                return { handle, streamId };
            }
            else {
                handle_helper.rocsparse_handle_container_mapper_.erase(it);
            }
        }
    }

    rocsparse_handle handle;
    ROCSPARSE_ERR_FUNC(rocsparse_create_handle, &handle);
    ROCSPARSE_ERR_FUNC(rocsparse_set_stream, handle, streamId);

    auto atomic_ptr = new std::atomic<rocsparse_handle>(handle);
    handle_helper.rocsparse_handle_container_mapper_.insert(
        std::make_pair(piPlacedContext_, atomic_ptr));

    sycl::detail::pi::contextSetExtendedDeleter(*placedContext_, ContextCallback, atomic_ptr);
    return { handle, streamId };
}

#pragma clang diagnostic pop

} // namespace oneapi::mkl::sparse::rocsparse::detail
