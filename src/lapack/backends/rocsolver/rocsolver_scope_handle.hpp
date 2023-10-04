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
#ifndef _ROCSOLVER_SCOPED_HANDLE_HPP_
#define _ROCSOLVER_SCOPED_HANDLE_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <atomic>
#include <memory>
#include <thread>
#include <unordered_map>
#include "rocsolver_helper.hpp"
#include "rocsolver_handle.hpp"

namespace oneapi {
namespace mkl {
namespace lapack {
namespace rocsolver {

class RocsolverScopedContextHandler {
    hipCtx_t original_;
    sycl::context *placedContext_;
    bool needToRecover_;
    sycl::interop_handle &ih;
    static thread_local rocsolver_handle<pi_context> handle_helper;
    hipStream_t get_stream(const sycl::queue &queue);
    sycl::context get_context(const sycl::queue &queue);

public:
    RocsolverScopedContextHandler(sycl::queue queue, sycl::interop_handle &ih);

    ~RocsolverScopedContextHandler() noexcept(false);

    rocblas_handle get_handle(const sycl::queue &queue);
    // This is a work-around function for reinterpret_casting the memory. This
    // will be fixed when SYCL-2020 has been implemented for Pi backend.
    template <typename T, typename U>
    inline T get_mem(U acc) {
#ifdef SYCL_IMPLEMENTATION_ONEAPI
        hipDeviceptr_t hipPtr = ih.get_native_mem<sycl::backend::ext_oneapi_hip>(acc);
#else
        hipDeviceptr_t hipPtr = ih.get_native_mem<sycl::backend::hip>(acc);
#endif
        return reinterpret_cast<T>(hipPtr);
    }
};

} // namespace rocsolver
} // namespace lapack
} // namespace mkl
} // namespace oneapi
#endif //_ROCSOLVER_SCOPED_HANDLE_HPP_
