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
#ifndef CUSOLVER_HANDLE_HPP
#define CUSOLVER_HANDLE_HPP
#include <atomic>
#include <unordered_map>

namespace oneapi {
namespace math {
namespace lapack {
namespace cusolver {

template <typename T>
struct cusolver_handle {
    using handle_container_t = std::unordered_map<T, std::atomic<cusolverDnHandle_t>*>;
    handle_container_t cusolver_handle_mapper_{};
    ~cusolver_handle() noexcept(false) {
        for (auto& handle_pair : cusolver_handle_mapper_) {
            cusolverStatus_t err;
            if (handle_pair.second != nullptr) {
                auto handle = handle_pair.second->exchange(nullptr);
                if (handle != nullptr) {
                    CUSOLVER_ERROR_FUNC(cusolverDnDestroy, err, handle);
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
        cusolver_handle_mapper_.clear();
    }
};

} // namespace cusolver
} // namespace lapack
} // namespace math
} // namespace oneapi

#endif // CUSOLVER_HANDLE_HPP
