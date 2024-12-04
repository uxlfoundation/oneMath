#ifndef ONEMATH_RNG_CURAND_TASK_HPP_
#define ONEMATH_RNG_CURAND_TASK_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "curand_helper.hpp"

namespace oneapi {
namespace math {
namespace rng {
namespace curand {
#ifdef __HIPSYCL__
template <typename H, typename A, typename E, typename F>
static inline void host_task_internal(H& cgh, A acc, E e, F f) {
    cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle ih) {
        curandStatus_t status;
        CURAND_CALL(curandSetStream, status, e, ih.get_native_queue<sycl::backend::cuda>());
        auto r_ptr =
            reinterpret_cast<typename A::value_type*>(ih.get_native_mem<sycl::backend::cuda>(acc));
        f(r_ptr);
    });
}

template <typename H, typename E, typename F>
static inline void host_task_internal(H& cgh, E e, F f) {
    cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle ih) {
        curandStatus_t status;
        CURAND_CALL(curandSetStream, status, e, ih.get_native_queue<sycl::backend::cuda>());
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
        curandStatus_t status;
        auto stream = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        CURAND_CALL(curandSetStream, status, e, stream);
        auto r_ptr = reinterpret_cast<typename A::value_type*>(
            ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(acc));
        f(r_ptr);
#ifndef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
        CUresult err;
        CUDA_ERROR_FUNC(cuStreamSynchronize, err, stream);
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
        curandStatus_t status;
        auto stream = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        CURAND_CALL(curandSetStream, status, e, stream);
        f(ih);
#ifndef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
        CUresult err;
        CUDA_ERROR_FUNC(cuStreamSynchronize, err, stream);
#endif
    });
}
#endif
template <typename H, typename A, typename E, typename F>
static inline void onemath_curand_host_task(H& cgh, A acc, E e, F f) {
    host_task_internal(cgh, acc, e, f);
}

template <typename H, typename Engine, typename F>
static inline void onemath_curand_host_task(H& cgh, Engine e, F f) {
    host_task_internal(cgh, e, f);
}

} // namespace curand
} // namespace rng
} // namespace math
} // namespace oneapi

#endif
