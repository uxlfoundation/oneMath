/***************************************************************************
*  Copyright 2020-2022 Intel Corporation
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
#include "rocblas_scope_handle.hpp"

namespace oneapi {
namespace math {
namespace blas {
namespace rocblas {

template <typename T>
rocblas_handle_container<T>::~rocblas_handle_container() noexcept(false) {
    for (auto& handle_pair : rocblas_handle_container_mapper_) {
        rocblas_status err;
        if (handle_pair.second != nullptr) {
            auto handle = handle_pair.second->exchange(nullptr);
            if (handle != nullptr) {
                ROCBLAS_ERROR_FUNC(rocblas_destroy_handle, err, handle);
                handle = nullptr;
            }
            else {
                delete handle_pair.second;
            }
            handle_pair.second = nullptr;
        }
    }
    rocblas_handle_container_mapper_.clear();
}

/**
 * Inserts a new element in the map if its key is unique. This new element
 * is constructed in place using args as the arguments for the construction
 * of a value_type (which is an object of a pair type). The insertion only
 * takes place if no other element in the container has a key equivalent to
 * the one being emplaced (keys in a map container are unique).
 */
#ifdef ONEMATH_PI_INTERFACE_REMOVED
thread_local rocblas_handle_container<ur_context_handle_t>
    RocblasScopedContextHandler::handle_helper = rocblas_handle_container<ur_context_handle_t>{};
#else
thread_local rocblas_handle_container<pi_context> RocblasScopedContextHandler::handle_helper =
    rocblas_handle_container<pi_context>{};
#endif

RocblasScopedContextHandler::RocblasScopedContextHandler(sycl::queue queue,
                                                         sycl::interop_handle& ih)
        : interop_h(ih),
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

RocblasScopedContextHandler::~RocblasScopedContextHandler() noexcept(false) {
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
        ROCBLAS_ERROR_FUNC(rocblas_destroy_handle, err1, handle);
        handle = nullptr;
    }
    else {
        // if the handle is nullptr it means the handle was already destroyed by
        // the rocblas_handle destructor and we're free to delete the atomic
        // object.
        delete ptr;
    }
}

rocblas_handle RocblasScopedContextHandler::get_handle(const sycl::queue& queue) {
    auto hipDevice = interop_h.get_native_device<sycl::backend::ext_oneapi_hip>();
    hipError_t hipErr;
    hipCtx_t desired;
    HIP_ERROR_FUNC(hipDevicePrimaryCtxRetain, hipErr, &desired, hipDevice);
#ifdef ONEMATH_PI_INTERFACE_REMOVED
    auto piPlacedContext_ = reinterpret_cast<ur_context_handle_t>(desired);
#else
    auto piPlacedContext_ = reinterpret_cast<pi_context>(desired);
#endif
    hipStream_t streamId = get_stream(queue);
    rocblas_status err;
    auto it = handle_helper.rocblas_handle_container_mapper_.find(piPlacedContext_);
    if (it != handle_helper.rocblas_handle_container_mapper_.end()) {
        if (it->second == nullptr) {
            handle_helper.rocblas_handle_container_mapper_.erase(it);
        }
        else {
            auto handle = it->second->load();
            if (handle != nullptr) {
                hipStream_t currentStreamId;
                ROCBLAS_ERROR_FUNC(rocblas_get_stream, err, handle, &currentStreamId);
                if (currentStreamId != streamId) {
                    ROCBLAS_ERROR_FUNC(rocblas_set_stream, err, handle, streamId);
                }
                return handle;
            }
            else {
                handle_helper.rocblas_handle_container_mapper_.erase(it);
            }
        }
    }

    rocblas_handle handle;

    ROCBLAS_ERROR_FUNC(rocblas_create_handle, err, &handle);
    ROCBLAS_ERROR_FUNC(rocblas_set_stream, err, handle, streamId);

    auto insert_iter = handle_helper.rocblas_handle_container_mapper_.insert(
        std::make_pair(piPlacedContext_, new std::atomic<rocblas_handle>(handle)));

    sycl::detail::pi::contextSetExtendedDeleter(*placedContext_, ContextCallback,
                                                insert_iter.first->second);
    return handle;
}

hipStream_t RocblasScopedContextHandler::get_stream(const sycl::queue& queue) {
    return sycl::get_native<sycl::backend::ext_oneapi_hip>(queue);
}
sycl::context RocblasScopedContextHandler::get_context(const sycl::queue& queue) {
    return queue.get_context();
}

} // namespace rocblas
} // namespace blas
} // namespace math
} // namespace oneapi
