/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Copyright (C) 2022 Heidelberg University, Engineering Mathematics and Computing Lab (EMCL) and Computing Centre (URZ)
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

#include "rocblas_scope_handle_hipsycl.hpp"

namespace oneapi {
namespace math {
namespace blas {
namespace rocblas {

rocblas_handle_container::~rocblas_handle_container() noexcept(false) {
    for (auto& handle_pair : rocblas_handle_mapper_) {
        rocblas_status err;
        if (handle_pair.second != nullptr) {
            auto handle = handle_pair.second->exchange(nullptr);
            if (handle != nullptr) {
                ROCBLAS_ERROR_FUNC(rocblas_destroy_handle, err, handle);
                handle = nullptr;
            }
            delete handle_pair.second;
            handle_pair.second = nullptr;
        }
    }
    rocblas_handle_mapper_.clear();
}

thread_local rocblas_handle_container RocblasScopedContextHandler::handle_helper =
    rocblas_handle_container{};

RocblasScopedContextHandler::RocblasScopedContextHandler(sycl::queue queue,
                                                         sycl::interop_handle& ih)
        : interop_h(ih) {}

rocblas_handle RocblasScopedContextHandler::get_handle(const sycl::queue& queue) {
    sycl::device device = queue.get_device();
    int current_device = interop_h.get_native_device<sycl::backend::hip>();
    hipStream_t streamId = get_stream(queue);
    rocblas_status err;
    auto it = handle_helper.rocblas_handle_mapper_.find(current_device);
    if (it != handle_helper.rocblas_handle_mapper_.end()) {
        if (it->second == nullptr) {
            handle_helper.rocblas_handle_mapper_.erase(it);
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
                handle_helper.rocblas_handle_mapper_.erase(it);
            }
        }
    }
    rocblas_handle handle;

    ROCBLAS_ERROR_FUNC(rocblas_create_handle, err, &handle);
    ROCBLAS_ERROR_FUNC(rocblas_set_stream, err, handle, streamId);

    auto insert_iter = handle_helper.rocblas_handle_mapper_.insert(
        std::make_pair(current_device, new std::atomic<rocblas_handle>(handle)));
    return handle;
}

hipStream_t RocblasScopedContextHandler::get_stream(const sycl::queue& queue) {
    return interop_h.get_native_queue<sycl::backend::hip>();
}

} // namespace rocblas
} // namespace blas
} // namespace math
} // namespace oneapi