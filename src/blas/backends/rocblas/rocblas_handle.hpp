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
#ifndef _ROCBLAS_HANDLE_HPP_
#define _ROCBLAS_HANDLE_HPP_
#include <atomic>
#include <unordered_map>
#include "rocblas_helper.hpp"

namespace oneapi {
namespace math {
namespace blas {
namespace rocblas {

template <typename T>
struct rocblas_handle_ {
    using handle_container_t = std::unordered_map<T, std::atomic<rocblas_handle>*>;
    handle_container_t rocblas_handle_mapper_{};
    ~rocblas_handle_() noexcept(false) {
        for (auto& handle_pair : rocblas_handle_mapper_) {
            rocblas_status err;
            if (handle_pair.second != nullptr) {
                auto handle = handle_pair.second->exchange(nullptr);
                if (handle != nullptr) {
                    ROCBLAS_ERROR_FUNC(rocblas_destroy_handle, err, handle);
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
        rocblas_handle_mapper_.clear();
    }
};

} // namespace rocblas
} // namespace blas
} // namespace math
} // namespace oneapi

#endif // _ROCBLAS_HANDLE_HPP_
