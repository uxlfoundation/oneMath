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

#ifndef _MKL_LAPACK_CUSOLVER_TASK_HPP_
#define _MKL_LAPACK_CUSOLVER_TASK_HPP_
#include <cuda.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <complex>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "oneapi/mkl/types.hpp"
#include "cusolver_scope_handle.hpp"

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

template <typename H, typename F>
static inline void host_task_internal(H& cgh, sycl::queue queue, F f) {
#ifdef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
    cgh.ext_codeplay_enqueue_native_command([f, queue](sycl::interop_handle ih) {
#else
    cgh.host_task([f, queue](sycl::interop_handle ih) {
#endif
        auto sc = CusolverScopedContextHandler(queue, ih);
        f(sc);
    });
}

template <typename H, typename F>
static inline void onemkl_cusolver_host_task(H& cgh, sycl::queue queue, F f) {
    (void)host_task_internal(cgh, queue, f);
}

} // namespace cusolver
} // namespace lapack
} // namespace mkl
} // namespace oneapi
#endif // _MKL_LAPACK_CUSOLVER_TASK_HPP_
