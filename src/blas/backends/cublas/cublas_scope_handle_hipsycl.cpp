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
#include "cublas_scope_handle_hipsycl.hpp"
#include "cublas_handle.hpp"

namespace oneapi {
namespace math {
namespace blas {
namespace cublas {

thread_local cublas_handle CublasScopedContextHandler::handle_helper = cublas_handle{};

CublasScopedContextHandler::CublasScopedContextHandler(sycl::interop_handle& ih) : interop_h(ih) {}

cublasHandle_t CublasScopedContextHandler::get_handle() {
    CUdevice current_device = interop_h.get_native_device<sycl::backend::cuda>();
    CUstream streamId = get_stream();
    cublasStatus_t err;
    auto it = handle_helper.cublas_handle_mapper_.find(current_device);
    if (it != handle_helper.cublas_handle_mapper_.end()) {
        cublasHandle_t handle = it->second;
        cudaStream_t currentStreamId;
        CUBLAS_ERROR_FUNC(cublasGetStream, err, handle, &currentStreamId);
        if (currentStreamId != streamId) {
            CUBLAS_ERROR_FUNC(cublasSetStream, err, handle, streamId);
        }
        return handle;
    }
    cublasHandle_t handle;

    CUBLAS_ERROR_FUNC(cublasCreate, err, &handle);
    CUBLAS_ERROR_FUNC(cublasSetStream, err, handle, streamId);

    auto insert_iter =
        handle_helper.cublas_handle_mapper_.insert(std::make_pair(current_device, handle));
    return handle;
}

CUstream CublasScopedContextHandler::get_stream() {
    return interop_h.get_native_queue<sycl::backend::cuda>();
}

} // namespace cublas
} // namespace blas
} // namespace math
} // namespace oneapi
