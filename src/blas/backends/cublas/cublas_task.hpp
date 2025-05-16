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

#ifndef ONEMATH_BLAS_CUBLAS_TASK_HPP_
#define ONEMATH_BLAS_CUBLAS_TASK_HPP_
#include <cublas_v2.h>
#include <cuda.h>
#include <complex>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "oneapi/math/types.hpp"
#ifndef __HIPSYCL__
#include "cublas_scope_handle.hpp"
#else
#include "cublas_scope_handle_hipsycl.hpp"

namespace sycl {
using interop_handler = sycl::interop_handle;
}
#endif
namespace oneapi {
namespace math {
namespace blas {
namespace cublas {

#ifdef __HIPSYCL__
template <typename H, typename F>
static inline void host_task_internal(H& cgh, F f) {
    cgh.hipSYCL_enqueue_custom_operation([f](sycl::interop_handle ih) {
        auto sc = CublasScopedContextHandler(ih);
        f(sc);
    });
}
#else
template <typename H, typename F>
static inline void host_task_internal(H& cgh, F f) {
#ifdef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
    cgh.ext_codeplay_enqueue_native_command([f](sycl::interop_handle ih) {
#else
    cgh.host_task([f](sycl::interop_handle ih) {
#endif
        auto sc = CublasScopedContextHandler(ih);
        f(sc);
    });
}
#endif
template <typename H, typename F>
static inline void onemath_cublas_host_task(H& cgh, F f) {
    (void)host_task_internal(cgh, f);
}

} // namespace cublas
} // namespace blas
} // namespace math
} // namespace oneapi
#endif // ONEMATH_BLAS_CUBLAS_TASK_HPP_
