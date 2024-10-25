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
#ifndef ROCSOLVER_HANDLE_HPP
#define ROCSOLVER_HANDLE_HPP
#include <atomic>
#include <unordered_map>

namespace oneapi {
namespace mkl {
namespace lapack {
namespace rocsolver {

template <typename T>
struct rocsolver_handle {
    using handle_container_t = std::unordered_map<T, std::atomic<rocblas_handle>*>;
    handle_container_t rocsolver_handle_mapper_{};
    ~rocsolver_handle() noexcept(false) {
        for (auto& handle_pair : rocsolver_handle_mapper_) {
            rocblas_status err;
            if (handle_pair.second != nullptr) {
                auto handle = handle_pair.second->exchange(nullptr);
                if (handle != nullptr) {
                    ROCSOLVER_ERROR_FUNC(rocblas_destroy_handle, err, handle);
                    handle = nullptr;
                }
                else {
                    // if the handle is nullptr it means the handle was already
                    // destroyed by the ContextCallback and we're free to delete the
                    // atomic object.
                    delete handle_pair.second;
                }

                handle_pair.second = nullptr;
            }
        }
        rocsolver_handle_mapper_.clear();
    }
};

} // namespace rocsolver
} // namespace lapack
} // namespace mkl
} // namespace oneapi

#endif // ROCSOLVER_HANDLE_HPP
