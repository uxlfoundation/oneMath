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
#ifndef _CUSOLVER_SCOPED_HANDLE_HPP_
#define _CUSOLVER_SCOPED_HANDLE_HPP_
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#if __has_include(<sycl/context.hpp>)
#if __SYCL_COMPILER_VERSION <= 20220930
#include <sycl/backend/cuda.hpp>
#endif
#include <sycl/context.hpp>
#else
#include <CL/sycl/backend/cuda.hpp>
#include <CL/sycl/context.hpp>
#endif
#include <atomic>
#include <memory>
#include <thread>
#include <unordered_map>
#include "cusolver_helper.hpp"
#include "cusolver_handle.hpp"

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
namespace lapack {
namespace cusolver {

/**
* @brief NVIDIA advise for handle creation for cublas:
https://devtalk.nvidia.com/default/topic/838794/gpu-accelerated libraries/using-cublas-in-different-cuda-streams/
According to NVIDIA: 
1)	The cusolver handles behaviour with different devices is unclear. However, cusolver is based on the cublas API 
    which required that different handles to be used for different devices. So it is assumed that cusolver also 
    requires different handles on different devices: 
https://docs.nvidia.com/cuda/cusolver/index.html#introduction
http://docs.nvidia.com/cuda/cublas/index.html#cublas-context	
2) 	The library is thread safe and can be called form different host threads: 
https://docs.nvidia.com/cuda/cusolver/index.html#thread-safety
3)	It is neither required nor recommended that different handles be used for different streams on the same device,
 using the same host thread.

The advice above is for using cublas with the cuda runtime API. Given that cusolver is based on cublas the advice is 
transferable. The cusolver_scope_handle is based on the oneMKL cublas_scope_handle. The NVIDIA runtime API creates a 
default context for users. The cusolverDnCreate function in uses the context located on top of the stack for each thread. 
Then, the cuSolver routine uses this context for resource allocation/access. Calling a cuSolver function with a handle 
created for context A and memories/queue created for context B results in a segmentation fault. Thus we need to create 
one handle per context and per thread. A context can have multiple streams, so the important thing here is to have one 
cusolverDnHandle_t per driver context and that cuSolver handle can switch between multiple streams created for that context. 
Here, we are dealing with CUDA driver API, therefore, the SYCL-CUDA backend controls the context. If a queue(equivalent of 
CUDA stream) is associated with a context different from the one on top of the thread stack(can be any context which 
associated at any time by either the runtime or user for any specific reason), the context associated with the queue must 
be moved on top of the stack temporarily for the requested routine operations. However, after the cuSolver routine 
execution, the original context must be restored to prevent intervening with the original user/runtime execution set up. 
Here, the RAII type context switch is used to guarantee to recover the original CUDA context. The cuSolver handle allocates 
internal resources, therefore, the handle must be destroyed when the context goes out of scope. This will bind the life of 
cuSolver handle to the SYCL context.
**/

class CusolverScopedContextHandler {
    CUcontext original_;
    sycl::context* placedContext_;
    bool needToRecover_;
    sycl::interop_handle& ih;
#ifdef ONEMKL_PI_INTERFACE_REMOVED
    static thread_local cusolver_handle<ur_context_handle_t> handle_helper;
#else
    static thread_local cusolver_handle<pi_context> handle_helper;
#endif
    CUstream get_stream(const sycl::queue& queue);
    sycl::context get_context(const sycl::queue& queue);

public:
    CusolverScopedContextHandler(sycl::queue queue, sycl::interop_handle& ih);

    ~CusolverScopedContextHandler() noexcept(false);
    /**
   * @brief get_handle: creates the handle by implicitly impose the advice
   * given by nvidia for creating a cusolver_handle. (e.g. one cuStream per device
   * per thread).
   * @param queue sycl queue.
   * @return cusolverDnHandle_t a handle to construct cusolver routines
   */
    cusolverDnHandle_t get_handle(const sycl::queue& queue);
    // This is a work-around function for reinterpret_casting the memory. This
    // will be fixed when SYCL-2020 has been implemented for Pi backend.
    template <typename T, typename U>
    inline T get_mem(U acc) {
        CUdeviceptr cudaPtr = ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(acc);
        return reinterpret_cast<T>(cudaPtr);
    }

    void wait_stream(const sycl::queue& queue) {
        cuStreamSynchronize(get_stream(queue));
    }
};

} // namespace cusolver
} // namespace lapack
} // namespace mkl
} // namespace oneapi
#endif //_CUSOLVER_SCOPED_HANDLE_HPP_
