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
#ifndef _ONEMATH_SPARSE_BLAS_BACKENDS_ROCSPARSE_SCOPE_HANDLE_HPP_
#define _ONEMATH_SPARSE_BLAS_BACKENDS_ROCSPARSE_SCOPE_HANDLE_HPP_

/**
 * @file Similar to rocblas_scope_handle.hpp
*/

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

// After Plugin Interface removal in DPC++ ur.hpp is the new include
#if __has_include(<sycl/detail/ur.hpp>) && !defined(ONEAPI_ONEMATH_PI_INTERFACE_REMOVED)
#define ONEAPI_ONEMATH_PI_INTERFACE_REMOVED
#endif

#include <thread>

#include "rocsparse_error.hpp"
#include "rocsparse_global_handle.hpp"
#include "rocsparse_helper.hpp"

namespace oneapi::math::sparse::rocsparse::detail {

template <typename T>
struct rocsparse_handle_container {
    using handle_container_t = std::unordered_map<T, std::atomic<rocsparse_handle>*>;
    handle_container_t rocsparse_handle_container_mapper_{};

    // Do not free any pointer nor handle in this destructor. The resources are
    // free'd via the PI ContextCallback to ensure the context is still alive.
    ~rocsparse_handle_container() = default;
};

class RocsparseScopedContextHandler {
    HIPcontext original_;
    sycl::context* placedContext_;
    sycl::interop_handle& ih;
    bool needToRecover_;
#ifdef ONEAPI_ONEMATH_PI_INTERFACE_REMOVED
    static thread_local rocsparse_handle_container<ur_context_handle_t> handle_helper;
#else
    static thread_local rocsparse_handle_container<pi_context> handle_helper;
#endif

public:
    RocsparseScopedContextHandler(sycl::queue queue, sycl::interop_handle& ih);
    ~RocsparseScopedContextHandler() noexcept(false);

    std::pair<rocsparse_handle, hipStream_t> get_handle_and_stream(const sycl::queue& queue);

    rocsparse_handle get_handle(const sycl::queue& queue) {
        return get_handle_and_stream(queue).first;
    }
};

// Get the native pointer from an accessor. This is a different pointer than
// what can be retrieved with get_multi_ptr.
template <typename AccT>
inline void* get_mem(sycl::interop_handle ih, AccT acc) {
    auto hipPtr = ih.get_native_mem<sycl::backend::ext_oneapi_hip>(acc);
    return reinterpret_cast<void*>(hipPtr);
}

} // namespace oneapi::math::sparse::rocsparse::detail

#endif //_ONEMATH_SPARSE_BLAS_BACKENDS_ROCSPARSE_SCOPE_HANDLE_HPP_
