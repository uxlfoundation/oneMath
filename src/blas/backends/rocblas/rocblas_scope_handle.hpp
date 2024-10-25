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
#ifndef _ROCBLAS_SCOPED_HANDLE_HPP_
#define _ROCBLAS_SCOPED_HANDLE_HPP_
#include <CL/sycl.hpp>
#include <memory>
#include <thread>
#include <atomic>
#include <unordered_map>
#include "rocblas_helper.hpp"

// After Plugin Interface removal in DPC++ ur.hpp is the new include
#if __has_include(<sycl/detail/ur.hpp>)
#include <sycl/detail/ur.hpp>
#ifndef ONEMKL_PI_INTERFACE_REMOVED
#define ONEMKL_PI_INTERFACE_REMOVED
#endif
#elif __has_include(<sycl/detail/pi.hpp>)
#include <sycl/detail/pi.hpp>
#else
#include <CL/sycl/detail/pi.hpp>
#endif

namespace oneapi {
namespace mkl {
namespace blas {
namespace rocblas {

template <typename T>
struct rocblas_handle_container {
    using handle_container_t = std::unordered_map<T, std::atomic<rocblas_handle>*>;
    handle_container_t rocblas_handle_container_mapper_{};
    ~rocblas_handle_container() noexcept(false);
};

class RocblasScopedContextHandler {
    HIPcontext original_;
    sycl::context* placedContext_;
    bool needToRecover_;
    sycl::interop_handle& interop_h;
#ifdef ONEMKL_PI_INTERFACE_REMOVED
    static thread_local rocblas_handle_container<ur_context_handle_t> handle_helper;
#else
    static thread_local rocblas_handle_container<pi_context> handle_helper;
#endif
    sycl::context get_context(const sycl::queue& queue);
    hipStream_t get_stream(const sycl::queue& queue);

public:
    RocblasScopedContextHandler(sycl::queue queue, sycl::interop_handle& ih);
    ~RocblasScopedContextHandler() noexcept(false);

    rocblas_handle get_handle(const sycl::queue& queue);

    // This is a work-around function for reinterpret_casting the memory. This
    // will be fixed when SYCL-2020 has been implemented for Pi backend.
    template <typename T, typename U>
    inline T get_mem(U acc) {
        return reinterpret_cast<T>(interop_h.get_native_mem<sycl::backend::ext_oneapi_hip>(acc));
    }
};

} // namespace rocblas
} // namespace blas
} // namespace mkl
} // namespace oneapi
#endif //_ROCBLAS_SCOPED_HANDLE_HPP_
