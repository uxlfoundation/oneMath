/*******************************************************************************
 * Copyright (C) 2022 Heidelberg University, Engineering Mathematics and Computing Lab (EMCL)
 * and Computing Centre (URZ)
 * cuRAND back-end Copyright (c) 2021, The Regents of the University of
 * California, through Lawrence Berkeley National Laboratory (subject to receipt
 * of any required approvals from the U.S. Dept. of Energy). All rights
 * reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * (1) Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * (2) Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * (3) Neither the name of the University of California, Lawrence Berkeley
 * National Laboratory, U.S. Dept. of Energy nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * You are under no obligation whatsoever to provide any bug fixes, patches,
 * or upgrades to the features, functionality or performance of the source
 * code ("Enhancements") to anyone; however, if you choose to make your
 * Enhancements available either publicly, or directly to Lawrence Berkeley
 * National Laboratory, without imposing a separate written license agreement
 * for such Enhancements, then you hereby grant the following license: a
 * non-exclusive, royalty-free perpetual license to install, use, modify,
 * prepare derivative works, incorporate into other computer software,
 * distribute, and sublicense such enhancements or derivative works thereof,
 * in binary and source code form.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Intellectual Property Office at
 * IPO@lbl.gov.
 *
 * NOTICE.  This Software was developed under funding from the U.S. Department
 * of Energy and the U.S. Government consequently retains certain rights.  As
 * such, the U.S. Government has been granted for itself and others acting on
 * its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 * Software to reproduce, distribute copies to the public, prepare derivative
 * works, and perform publicly and display publicly, and to permit others to do
 * so.
 ******************************************************************************/

#ifndef ONEMATH_RNG_ROCRAND_TASK_HPP_
#define ONEMATH_RNG_ROCRAND_TASK_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "rocrand_helper.hpp"

namespace oneapi {
namespace math {
namespace rng {
namespace rocrand {
#ifdef __HIPSYCL__
template <typename H, typename A, typename E, typename F>
static inline void host_task_internal(H& cgh, A acc, E e, F f) {
    cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle ih) {
        rocrand_status status;
        ROCRAND_CALL(rocrand_set_stream, status, e, ih.get_native_queue<sycl::backend::hip>());
        auto r_ptr =
            reinterpret_cast<typename A::value_type*>(ih.get_native_mem<sycl::backend::hip>(acc));
        f(r_ptr);
    });
}

template <typename H, typename E, typename F>
static inline void host_task_internal(H& cgh, E e, F f) {
    cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle ih) {
        rocrand_status status;
        ROCRAND_CALL(rocrand_set_stream, status, e, ih.get_native_queue<sycl::backend::hip>());
        f(ih);
    });
}
#else
template <typename H, typename A, typename E, typename F>
static inline void host_task_internal(H& cgh, A acc, E e, F f) {
#ifdef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
    cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
#else
    cgh.host_task([=](sycl::interop_handle ih) {
#endif
        rocrand_status status;
        auto stream = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
        ROCRAND_CALL(rocrand_set_stream, status, e, stream);
        auto r_ptr = reinterpret_cast<typename A::value_type*>(
            ih.get_native_mem<sycl::backend::ext_oneapi_hip>(acc));
        f(r_ptr);
#ifndef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
        hipError_t err;
        HIP_ERROR_FUNC(hipStreamSynchronize, err, stream);
#endif
    });
}

template <typename H, typename E, typename F>
static inline void host_task_internal(H& cgh, E e, F f) {
#ifdef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
    cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
#else
    cgh.host_task([=](sycl::interop_handle ih) {
#endif
        rocrand_status status;
        auto stream = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
        ROCRAND_CALL(rocrand_set_stream, status, e, stream);
        f(ih);
#ifndef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
        hipError_t err;
        HIP_ERROR_FUNC(hipStreamSynchronize, err, stream);
#endif
    });
}
#endif
template <typename H, typename A, typename E, typename F>
static inline void onemath_rocrand_host_task(H& cgh, A acc, E e, F f) {
    host_task_internal(cgh, acc, e, f);
}

template <typename H, typename Engine, typename F>
static inline void onemath_rocrand_host_task(H& cgh, Engine e, F f) {
    host_task_internal(cgh, e, f);
}

} // namespace rocrand
} // namespace rng
} // namespace math
} // namespace oneapi

#endif
