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
#ifndef CUBLAS_SCOPED_HANDLE_HIPSYCL_HPP
#define CUBLAS_SCOPED_HANDLE_HIPSYCL_HPP
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <memory>
#include <thread>
#include <unordered_map>
#include "cublas_helper.hpp"
#include "cublas_handle.hpp"
namespace oneapi {
namespace mkl {
namespace blas {
namespace cublas {

/**
* @brief NVIDIA advise for handle creation:
https://devtalk.nvidia.com/default/topic/838794/gpu-accelerated libraries/using-cublas-in-different-cuda-streams/
According to NVIDIA: 
1)	It is required that different handles to be used for different devices:
 http://docs.nvidia.com/cuda/cublas/index.html#cublas-context	
2)	It is recommended (but not required, if care is taken) that different handles be used for different host threads: 
http://docs.nvidia.com/cuda/cublas/index.html#thread-safety2changeme
3)	It is neither required nor recommended that different handles be used for different streams on the same device,
 using the same host thread.
However, the 3 above advises are for using cuda runtime API. The NVIDIA runtime API creates a default context for users. 
The createHandle function in cuBLAS uses the context located on top of the stack for each thread. Then, the cuBLAS routine 
uses this context for resource allocation/access. Calling a cuBLAS function with a handle created for context A and 
memories/queue created for context B results in a segmentation fault. Thus we need to create one handle per context 
and per thread. A context can have multiple streams, so the important thing here is to have one cublasHandle per driver 
context and that cuBLAS handle can switch between multiple streams created for that context. Here, we are dealing with 
CUDA driver API, therefore, the SYCL-CUDA backend controls the context. If a queue(equivalent of CUDA stream) is associated 
with a context different from the one on top of the thread stack(can be any context which associated at any time by either 
the runtime or user for any specific reason), the context associated with the queue must be moved on top of the stack 
temporarily for the requested routine operations. However, after the cuBLAS routine execution, the original context must 
be restored to prevent intervening with the original user/runtime execution set up. Here, the RAII type context switch 
is used to guarantee to recover the original CUDA context. The cuBLAS handle allocates internal resources, therefore, 
the handle must be destroyed when the context goes out of scope. This will bind the life of cuBLAS handle to the SYCL context.
**/

class CublasScopedContextHandler {
    sycl::interop_handle interop_h;
    static thread_local cublas_handle<int> handle_helper;
    sycl::context get_context(const sycl::queue& queue);
    CUstream get_stream(const sycl::queue& queue);

public:
    CublasScopedContextHandler(sycl::queue queue, sycl::interop_handle& ih);

    cublasHandle_t get_handle(const sycl::queue& queue);

    // This is a work-around function for reinterpret_casting the memory. This
    // will be fixed when SYCL-2020 has been implemented for Pi backend.
    template <typename T, typename U>
    inline T get_mem(U acc) {
        return reinterpret_cast<T>(interop_h.get_native_mem<sycl::backend::cuda>(acc));
    }
};

} // namespace cublas
} // namespace blas
} // namespace mkl
} // namespace oneapi
#endif //CUBLAS_SCOPED_HANDLE_HIPSYCL_HPP
