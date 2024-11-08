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

namespace oneapi {
namespace mkl {
namespace blas {
namespace cublas {

template <typename T>
struct cublas_handle {
    using handle_container_t = std::unordered_map<T, cublasHandle_t>;
    handle_container_t cublas_handle_mapper_{};
    ~cublas_handle() noexcept(false) {
        for (auto& handle_pair : cublas_handle_mapper_) {
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
