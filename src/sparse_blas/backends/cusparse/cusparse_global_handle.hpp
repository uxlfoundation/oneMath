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

#ifndef _ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_GLOBAL_HANDLE_HPP_
#define _ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_GLOBAL_HANDLE_HPP_

/**
 * @file Similar to blas_handle.hpp
 * Provides a map from a ur_context_handle_t (or equivalent) to a cusparseHandle_t.
 * @see cusparse_scope_handle.hpp
*/

#include <atomic>
#include <unordered_map>

namespace oneapi::mkl::sparse::cusparse {

template <typename T>
struct cusparse_global_handle {
    using handle_container_t = std::unordered_map<T, std::atomic<cusparseHandle_t>*>;
    handle_container_t cusparse_global_handle_mapper_{};

    ~cusparse_global_handle() noexcept(false) {
        for (auto& handle_pair : cusparse_global_handle_mapper_) {
            if (handle_pair.second != nullptr) {
                auto handle = handle_pair.second->exchange(nullptr);
                if (handle != nullptr) {
                    CUSPARSE_ERR_FUNC(cusparseDestroy, handle);
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
        cusparse_global_handle_mapper_.clear();
    }
};

} // namespace oneapi::mkl::sparse::cusparse

#endif // _ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_GLOBAL_HANDLE_HPP_
