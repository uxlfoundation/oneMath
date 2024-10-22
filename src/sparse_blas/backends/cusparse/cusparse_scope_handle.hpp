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
#ifndef _ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_SCOPE_HANDLE_HPP_
#define _ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_SCOPE_HANDLE_HPP_

/**
 * @file Similar to cublas_scope_handle.hpp
*/

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

// After Plugin Interface removal in DPC++ ur.hpp is the new include
#if __has_include(<sycl/detail/ur.hpp>) && !defined(ONEAPI_ONEMKL_PI_INTERFACE_REMOVED)
#define ONEAPI_ONEMKL_PI_INTERFACE_REMOVED
#endif

#include <thread>

#include "cusparse_error.hpp"
#include "cusparse_global_handle.hpp"
#include "cusparse_helper.hpp"

namespace oneapi::mkl::sparse::cusparse {

class CusparseScopedContextHandler {
    CUcontext original_;
    sycl::context *placedContext_;
    sycl::interop_handle &ih;
    bool needToRecover_;

#ifdef ONEAPI_ONEMKL_PI_INTERFACE_REMOVED
    static thread_local cusparse_global_handle<ur_context_handle_t> handle_helper;
#else
    static thread_local cusparse_global_handle<pi_context> handle_helper;
#endif

    CUstream get_stream(const sycl::queue &queue);
    sycl::context get_context(const sycl::queue &queue);

public:
    CusparseScopedContextHandler(sycl::queue queue, sycl::interop_handle &ih);

    ~CusparseScopedContextHandler() noexcept(false);

    /**
     * @brief get_handle: creates the handle by implicitly impose the advice
     * given by nvidia for creating a cusparse_global_handle. (e.g. one cuStream per device
     * per thread).
     * @param queue sycl queue.
     * @return a pair of: cusparseHandle_t a handle to construct cusparse routines; and a CUDA stream
     */
    std::pair<cusparseHandle_t, CUstream> get_handle_and_stream(const sycl::queue &queue);

    /// See get_handle_and_stream
    cusparseHandle_t get_handle(const sycl::queue &queue);
};

// Get the native pointer from an accessor. This is a different pointer than
// what can be retrieved with get_multi_ptr.
template <typename AccT>
inline void *get_mem(sycl::interop_handle ih, AccT acc) {
    auto cudaPtr = ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(acc);
    return reinterpret_cast<void *>(cudaPtr);
}

} // namespace oneapi::mkl::sparse::cusparse

#endif //_ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_SCOPE_HANDLE_HPP_
