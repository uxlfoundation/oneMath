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
#ifndef CUBLAS_HANDLE_HPP
#define CUBLAS_HANDLE_HPP
#include <unordered_map>
#include "cublas_helper.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace cublas {

struct cublas_handle {
    using handle_container_t = std::unordered_map<CUdevice, cublasHandle_t>;
    handle_container_t cublas_handle_mapper_{};
    ~cublas_handle() noexcept(false) {
        CUresult err;
        CUcontext original;
        CUDA_ERROR_FUNC(cuCtxGetCurrent, err, &original);
        for (auto& handle_pair : cublas_handle_mapper_) {
            CUcontext desired;
            CUDA_ERROR_FUNC(cuDevicePrimaryCtxRetain, err, &desired, handle_pair.first);
            if (original != desired) {
                // Sets the desired context as the active one for the thread in order to destroy its corresponding cublasHandle_t.
                CUDA_ERROR_FUNC(cuCtxSetCurrent, err, desired);
            }
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC(cublasDestroy, err, handle_pair.second);
        }
        cublas_handle_mapper_.clear();
    }
};

} // namespace cublas
} // namespace blas
} // namespace mkl
} // namespace oneapi

#endif // CUBLAS_HANDLE_HPP
