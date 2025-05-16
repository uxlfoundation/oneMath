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

namespace oneapi {
namespace math {
namespace blas {
namespace cublas {

/**
 * Inserts a new element in the map if its key is unique. This new element
 * is constructed in place using args as the arguments for the construction
 * of a value_type (which is an object of a pair type). The insertion only
 * takes place if no other element in the container has a key equivalent to
 * the one being emplaced (keys in a map container are unique).
 */
thread_local cublas_handle CublasScopedContextHandler::handle_helper = cublas_handle{};

CublasScopedContextHandler::CublasScopedContextHandler(sycl::interop_handle& ih) : ih(ih) {}

cublasHandle_t CublasScopedContextHandler::get_handle() {
    CUdevice device = ih.get_native_device<sycl::backend::ext_oneapi_cuda>();
    CUstream streamId = get_stream();
    cublasStatus_t err;

    auto it = handle_helper.cublas_handle_mapper_.find(device);
    if (it != handle_helper.cublas_handle_mapper_.end()) {
        cublasHandle_t nativeHandle = it->second;
        cudaStream_t currentStreamId;
        CUBLAS_ERROR_FUNC(cublasGetStream, err, nativeHandle, &currentStreamId);
        if (currentStreamId != streamId) {
            CUBLAS_ERROR_FUNC(cublasSetStream, err, nativeHandle, streamId);
        }
        return nativeHandle;
    }

    cublasHandle_t nativeHandle;
    CUBLAS_ERROR_FUNC(cublasCreate, err, &nativeHandle);
    CUBLAS_ERROR_FUNC(cublasSetStream, err, nativeHandle, streamId);

    auto insert_iter =
        handle_helper.cublas_handle_mapper_.insert(std::make_pair(device, nativeHandle));

    return nativeHandle;
}

CUstream CublasScopedContextHandler::get_stream() {
    return ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
}
} // namespace cublas
} // namespace blas
} // namespace math
} // namespace oneapi
