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
