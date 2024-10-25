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
#ifndef _ROCBLAS_SCOPED_HANDLE_HPP_
#define _ROCBLAS_SCOPED_HANDLE_HPP_
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <memory>
#include <thread>
#include <unordered_map>
#include "rocblas_helper.hpp"
namespace oneapi {
namespace mkl {
namespace blas {
namespace rocblas {

struct rocblas_handle_container {
    using handle_container_t = std::unordered_map<int, std::atomic<rocblas_handle>*>;
    handle_container_t rocblas_handle_mapper_{};
    ~rocblas_handle_container() noexcept(false);
};

class RocblasScopedContextHandler {
    sycl::interop_handle interop_h;
    static thread_local rocblas_handle_container handle_helper;
    sycl::context get_context(const sycl::queue& queue);
    hipStream_t get_stream(const sycl::queue& queue);

public:
    RocblasScopedContextHandler(sycl::queue queue, sycl::interop_handle& ih);

    rocblas_handle get_handle(const sycl::queue& queue);

    // This is a work-around function for reinterpret_casting the memory. This
    // will be fixed when SYCL-2020 has been implemented for Pi backend.
    template <typename T, typename U>
    inline T get_mem(U acc) {
        return reinterpret_cast<T>(interop_h.get_native_mem<sycl::backend::hip>(acc));
    }
};

} // namespace rocblas
} // namespace blas
} // namespace mkl
} // namespace oneapi
#endif //_ROCBLAS_SCOPED_HANDLE_HPP_
