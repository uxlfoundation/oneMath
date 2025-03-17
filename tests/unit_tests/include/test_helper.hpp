/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#ifndef _TEST_HELPER_HPP_
#define _TEST_HELPER_HPP_

#include <iostream>
#include <string>
#include <tuple>
#include <gtest/gtest.h>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math.hpp"
#include "oneapi/math/detail/config.hpp"
#include "oneapi/math/detail/backend_selector.hpp"

#ifdef _WIN64
#include <malloc.h>
#else
#include <stdlib.h>
#endif

#define test_failed  0
#define test_passed  1
#define test_skipped 2

// Note GTEST_SKIP may not print the associated message when using ctest.
// However, running a test binary with the flag `--terse-output` will print them.

#define EXPECT_TRUEORSKIP(a)             \
    do {                                 \
        int res = a;                     \
        if (res == test_skipped)         \
            GTEST_SKIP();                \
        else                             \
            EXPECT_EQ(res, test_passed); \
    } while (0);

// GTEST_SKIP stops the execution of the program.
// This macro lets a test use multiple EXPECT_TRUE_OR_FUTURE_SKIP and mark a test as skipped only once at the end.
#define EXPECT_TRUE_OR_FUTURE_SKIP(a, num_passed, num_skipped) \
    do {                                                       \
        int res = a;                                           \
        if (res == test_skipped)                               \
            ++num_skipped;                                     \
        else {                                                 \
            ++num_passed;                                      \
            EXPECT_EQ(res, test_passed);                       \
        }                                                      \
    } while (0);

#define CHECK_DOUBLE_ON_DEVICE(d)                                        \
    if (d->get_info<sycl::info::device::double_fp_config>().size() == 0) \
    GTEST_SKIP() << "Double precision is not supported on the device"

#ifdef SYCL_EXT_ONEAPI_GRAPH
#define CHECK_GRAPH_ON_DEVICE(d)                   \
    if (!d->has(aspect::ext_oneapi_limited_graph)) \
    GTEST_SKIP() << "SYCL-Graph is not supported on the device"
#else
#define CHECK_GRAPH_ON_DEVICE(d) GTEST_SKIP() << "SYCL-Graph is not supported on the device"
#endif

#if defined(ONEMATH_ENABLE_MKLCPU_BACKEND) || defined(ONEMATH_ENABLE_NETLIB_BACKEND) || \
    defined(ONEMATH_ENABLE_ARMPL_BACKEND)
#ifdef ONEMATH_ENABLE_MKLCPU_BACKEND
#define TEST_RUN_INTELCPU_SELECT_NO_ARGS(q, func) \
    func(oneapi::math::backend_selector<oneapi::math::backend::mklcpu>{ q })
#define TEST_RUN_INTELCPU_SELECT(q, func, ...) \
    func(oneapi::math::backend_selector<oneapi::math::backend::mklcpu>{ q }, __VA_ARGS__)
#elif defined(ONEMATH_ENABLE_NETLIB_BACKEND)
#define TEST_RUN_INTELCPU_SELECT(q, func, ...) \
    func(oneapi::math::backend_selector<oneapi::math::backend::netlib>{ q }, __VA_ARGS__)
#elif defined(ONEMATH_ENABLE_ARMPL_BACKEND)
#define TEST_RUN_INTELCPU_SELECT_NO_ARGS(q, func) \
    func(oneapi::math::backend_selector<oneapi::math::backend::armpl>{ q })
#define TEST_RUN_INTELCPU_SELECT(q, func, ...) \
    func(oneapi::math::backend_selector<oneapi::math::backend::armpl>{ q }, __VA_ARGS__)
#endif
#else
#define TEST_RUN_INTELCPU_SELECT_NO_ARGS(q, func)
#define TEST_RUN_INTELCPU_SELECT(q, func, ...)
#endif

#ifdef ONEMATH_ENABLE_MKLGPU_BACKEND
#define TEST_RUN_INTELGPU_SELECT_NO_ARGS(q, func) \
    func(oneapi::math::backend_selector<oneapi::math::backend::mklgpu>{ q })
#define TEST_RUN_INTELGPU_SELECT(q, func, ...) \
    func(oneapi::math::backend_selector<oneapi::math::backend::mklgpu>{ q }, __VA_ARGS__)
#else
#define TEST_RUN_INTELGPU_SELECT_NO_ARGS(q, func)
#define TEST_RUN_INTELGPU_SELECT(q, func, ...)
#endif

#ifdef ONEMATH_ENABLE_CUBLAS_BACKEND
#define TEST_RUN_NVIDIAGPU_CUBLAS_SELECT(q, func, ...) \
    func(oneapi::math::backend_selector<oneapi::math::backend::cublas>{ q }, __VA_ARGS__)
#else
#define TEST_RUN_NVIDIAGPU_CUBLAS_SELECT(q, func, ...)
#endif
#ifdef ONEMATH_ENABLE_CUSOLVER_BACKEND
#define TEST_RUN_NVIDIAGPU_CUSOLVER_SELECT(q, func, ...) \
    func(oneapi::math::backend_selector<oneapi::math::backend::cusolver>{ q }, __VA_ARGS__)
#else
#define TEST_RUN_NVIDIAGPU_CUSOLVER_SELECT(q, func, ...)
#endif

#ifdef ONEMATH_ENABLE_ROCBLAS_BACKEND
#define TEST_RUN_AMDGPU_ROCBLAS_SELECT(q, func, ...) \
    func(oneapi::math::backend_selector<oneapi::math::backend::rocblas>{ q }, __VA_ARGS__)
#else
#define TEST_RUN_AMDGPU_ROCBLAS_SELECT(q, func, ...)
#endif

#ifdef ONEMATH_ENABLE_CURAND_BACKEND
#define TEST_RUN_NVIDIAGPU_CURAND_SELECT(q, func, ...) \
    func(oneapi::math::backend_selector<oneapi::math::backend::curand>{ q }, __VA_ARGS__)
#else
#define TEST_RUN_NVIDIAGPU_CURAND_SELECT(q, func, ...)
#endif

#ifdef ONEMATH_ENABLE_ROCRAND_BACKEND
#define TEST_RUN_AMDGPU_ROCRAND_SELECT(q, func, ...) \
    func(oneapi::math::backend_selector<oneapi::math::backend::rocrand>{ q }, __VA_ARGS__)
#else
#define TEST_RUN_AMDGPU_ROCRAND_SELECT(q, func, ...)
#endif

#ifdef ONEMATH_ENABLE_ROCSOLVER_BACKEND
#define TEST_RUN_AMDGPU_ROCSOLVER_SELECT(q, func, ...) \
    func(oneapi::math::backend_selector<oneapi::math::backend::rocsolver>{ q }, __VA_ARGS__)
#else
#define TEST_RUN_AMDGPU_ROCSOLVER_SELECT(q, func, ...)
#endif

#ifdef ONEMATH_ENABLE_GENERIC_BLAS_BACKEND
#define TEST_RUN_GENERIC_BLAS_SELECT(q, func, ...) \
    func(oneapi::math::backend_selector<oneapi::math::backend::generic>{ q }, __VA_ARGS__)
#else
#define TEST_RUN_GENERIC_BLAS_SELECT(q, func, ...)
#endif

#ifdef ONEMATH_ENABLE_CUFFT_BACKEND
#define TEST_RUN_NVIDIAGPU_CUFFT_SELECT_NO_ARGS(q, func) \
    func(oneapi::math::backend_selector<oneapi::math::backend::cufft>{ q })
#define TEST_RUN_NVIDIAGPU_CUFFT_SELECT(q, func, ...) \
    func(oneapi::math::backend_selector<oneapi::math::backend::cufft>{ q }, __VA_ARGS__)
#else
#define TEST_RUN_NVIDIAGPU_CUFFT_SELECT_NO_ARGS(q, func)
#define TEST_RUN_NVIDIAGPU_CUFFT_SELECT(q, func, ...)
#endif

#ifdef ONEMATH_ENABLE_ROCFFT_BACKEND
#define TEST_RUN_AMDGPU_ROCFFT_SELECT_NO_ARGS(q, func) \
    func(oneapi::math::backend_selector<oneapi::math::backend::rocfft>{ q })
#define TEST_RUN_AMDGPU_ROCFFT_SELECT(q, func, ...) \
    func(oneapi::math::backend_selector<oneapi::math::backend::rocfft>{ q }, __VA_ARGS__)
#else
#define TEST_RUN_AMDGPU_ROCFFT_SELECT_NO_ARGS(q, func)
#define TEST_RUN_AMDGPU_ROCFFT_SELECT(q, func, ...)
#endif

#ifdef ONEMATH_ENABLE_PORTFFT_BACKEND
#define TEST_RUN_PORTFFT_SELECT_NO_ARGS(q, func) \
    func(oneapi::math::backend_selector<oneapi::math::backend::portfft>{ q })
#define TEST_RUN_PORTFFT_SELECT(q, func, ...) \
    func(oneapi::math::backend_selector<oneapi::math::backend::portfft>{ q }, __VA_ARGS__)
#else
#define TEST_RUN_PORTFFT_SELECT_NO_ARGS(q, func)
#define TEST_RUN_PORTFFT_SELECT(q, func, ...)
#endif

#ifdef ONEMATH_ENABLE_CUSPARSE_BACKEND
#define TEST_RUN_NVIDIAGPU_CUSPARSE_SELECT(q, func, ...) \
    func(oneapi::math::backend_selector<oneapi::math::backend::cusparse>{ q }, __VA_ARGS__)
#else
#define TEST_RUN_NVIDIAGPU_CUSPARSE_SELECT(q, func, ...)
#endif

#ifdef ONEMATH_ENABLE_ROCSPARSE_BACKEND
#define TEST_RUN_AMDGPU_ROCSPARSE_SELECT(q, func, ...) \
    func(oneapi::math::backend_selector<oneapi::math::backend::rocsparse>{ q }, __VA_ARGS__)
#else
#define TEST_RUN_AMDGPU_ROCSPARSE_SELECT(q, func, ...)
#endif

#ifndef __ADAPTIVECPP__
#define CHECK_HOST_OR_CPU(q) q.get_device().is_cpu()
#else
#define CHECK_HOST_OR_CPU(q) q.is_host() || q.get_device().is_cpu()
#endif

#define TEST_RUN_CT_SELECT_NO_ARGS(q, func)                                \
    do {                                                                   \
        if (CHECK_HOST_OR_CPU(q)) {                                        \
            TEST_RUN_INTELCPU_SELECT_NO_ARGS(q, func);                     \
        }                                                                  \
        else if (q.get_device().is_gpu()) {                                \
            unsigned int vendor_id = static_cast<unsigned int>(            \
                q.get_device().get_info<sycl::info::device::vendor_id>()); \
            if (vendor_id == INTEL_ID) {                                   \
                TEST_RUN_INTELGPU_SELECT_NO_ARGS(q, func);                 \
            }                                                              \
            else if (vendor_id == NVIDIA_ID) {                             \
                TEST_RUN_NVIDIAGPU_CUFFT_SELECT_NO_ARGS(q, func);          \
            }                                                              \
            else if (vendor_id == AMD_ID) {                                \
                TEST_RUN_AMDGPU_ROCFFT_SELECT_NO_ARGS(q, func);            \
            }                                                              \
        }                                                                  \
        TEST_RUN_PORTFFT_SELECT_NO_ARGS(q, func);                          \
    } while (0);

#define TEST_RUN_BLAS_CT_SELECT(q, func, ...)                              \
    do {                                                                   \
        if (CHECK_HOST_OR_CPU(q))                                          \
            TEST_RUN_INTELCPU_SELECT(q, func, __VA_ARGS__);                \
        else if (q.get_device().is_gpu()) {                                \
            unsigned int vendor_id = static_cast<unsigned int>(            \
                q.get_device().get_info<sycl::info::device::vendor_id>()); \
            if (vendor_id == INTEL_ID)                                     \
                TEST_RUN_INTELGPU_SELECT(q, func, __VA_ARGS__);            \
            else if (vendor_id == NVIDIA_ID) {                             \
                TEST_RUN_NVIDIAGPU_CUBLAS_SELECT(q, func, __VA_ARGS__);    \
            }                                                              \
            else if (vendor_id == AMD_ID) {                                \
                TEST_RUN_AMDGPU_ROCBLAS_SELECT(q, func, __VA_ARGS__);      \
            }                                                              \
        }                                                                  \
        TEST_RUN_GENERIC_BLAS_SELECT(q, func, __VA_ARGS__);                \
    } while (0);

#define TEST_RUN_RNG_CT_SELECT(q, func, ...)                               \
    do {                                                                   \
        if (CHECK_HOST_OR_CPU(q))                                          \
            TEST_RUN_INTELCPU_SELECT(q, func, __VA_ARGS__);                \
        else if (q.get_device().is_gpu()) {                                \
            unsigned int vendor_id = static_cast<unsigned int>(            \
                q.get_device().get_info<sycl::info::device::vendor_id>()); \
            if (vendor_id == INTEL_ID)                                     \
                TEST_RUN_INTELGPU_SELECT(q, func, __VA_ARGS__);            \
            else if (vendor_id == NVIDIA_ID) {                             \
                TEST_RUN_NVIDIAGPU_CURAND_SELECT(q, func, __VA_ARGS__);    \
            }                                                              \
            else if (vendor_id == AMD_ID) {                                \
                TEST_RUN_AMDGPU_ROCRAND_SELECT(q, func, __VA_ARGS__);      \
            }                                                              \
        }                                                                  \
    } while (0);

#define TEST_RUN_LAPACK_CT_SELECT(q, func, ...)                            \
    do {                                                                   \
        if (CHECK_HOST_OR_CPU(q))                                          \
            TEST_RUN_INTELCPU_SELECT(q, func, __VA_ARGS__);                \
        else if (q.get_device().is_gpu()) {                                \
            unsigned int vendor_id = static_cast<unsigned int>(            \
                q.get_device().get_info<sycl::info::device::vendor_id>()); \
            if (vendor_id == INTEL_ID)                                     \
                TEST_RUN_INTELGPU_SELECT(q, func, __VA_ARGS__);            \
            else if (vendor_id == NVIDIA_ID) {                             \
                TEST_RUN_NVIDIAGPU_CUSOLVER_SELECT(q, func, __VA_ARGS__);  \
            }                                                              \
            else if (vendor_id == AMD_ID) {                                \
                TEST_RUN_AMDGPU_ROCSOLVER_SELECT(q, func, __VA_ARGS__);    \
            }                                                              \
        }                                                                  \
    } while (0);

#define TEST_RUN_SPARSE_CT_SELECT(q, func, ...)                            \
    do {                                                                   \
        if (CHECK_HOST_OR_CPU(q))                                          \
            TEST_RUN_INTELCPU_SELECT(q, func, __VA_ARGS__);                \
        else if (q.get_device().is_gpu()) {                                \
            unsigned int vendor_id = static_cast<unsigned int>(            \
                q.get_device().get_info<sycl::info::device::vendor_id>()); \
            if (vendor_id == INTEL_ID) {                                   \
                TEST_RUN_INTELGPU_SELECT(q, func, __VA_ARGS__);            \
            }                                                              \
            else if (vendor_id == NVIDIA_ID) {                             \
                TEST_RUN_NVIDIAGPU_CUSPARSE_SELECT(q, func, __VA_ARGS__);  \
            }                                                              \
            else if (vendor_id == AMD_ID) {                                \
                TEST_RUN_AMDGPU_ROCSPARSE_SELECT(q, func, __VA_ARGS__);    \
            }                                                              \
        }                                                                  \
    } while (0);

void print_error_code(sycl::exception const& e);

class DeviceNamePrint {
public:
    std::string operator()(testing::TestParamInfo<sycl::device*> dev) const {
        std::string dev_name = dev.param->get_info<sycl::info::device::name>();
        for (std::string::size_type i = 0; i < dev_name.size(); ++i) {
            if (!isalnum(dev_name[i]))
                dev_name[i] = '_';
        }
        if (dev_name.size() == 0)
            dev_name = dev_name.append("_");
        return dev_name;
    }
};

class LayoutDeviceNamePrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<sycl::device*, oneapi::math::layout>> dev) const {
        std::string layout_name = std::get<1>(dev.param) == oneapi::math::layout::col_major
                                      ? "Column_Major"
                                      : "Row_Major";
        std::string dev_name = std::get<0>(dev.param)->get_info<sycl::info::device::name>();
        for (std::string::size_type i = 0; i < dev_name.size(); ++i) {
            if (!isalnum(dev_name[i]))
                dev_name[i] = '_';
        }
        std::string info_name = (layout_name.append("_")).append(dev_name);
        return info_name;
    }
};

/* to accommodate Windows and Linux differences between alligned_alloc and
   _aligned_malloc calls use oneapi::math::aligned_alloc and oneapi::math::aligned_free instead */
namespace oneapi {
namespace math {

static inline void* aligned_alloc(size_t align, size_t size) {
#ifdef _WIN64
    return ::_aligned_malloc(size, align);
#else
    return ::aligned_alloc(align, size);
#endif
}

static inline void aligned_free(void* p) {
#ifdef _WIN64
    ::_aligned_free(p);
#else
    ::free(p);
#endif
}

/* Support for Unified Shared Memory allocations for different backends */
static inline void* malloc_shared(size_t align, size_t size, sycl::device dev, sycl::context ctx) {
    (void)align;
#ifdef _WIN64
    return sycl::malloc_shared(size, dev, ctx);
#else
#if defined(ONEMATH_ENABLE_CUBLAS_BACKEND) || defined(ONEMATH_ENABLE_ROCBLAS_BACKEND)
    return sycl::aligned_alloc_shared(align, size, dev, ctx);
#endif
#if !defined(ONEMATH_ENABLE_CUBLAS_BACKEND) && !defined(ONEMATH_ENABLE_ROCBLAS_BACKEND)
    return sycl::malloc_shared(size, dev, ctx);
#endif
#endif
}

static inline void* malloc_device(size_t align, size_t size, sycl::device dev, sycl::context ctx) {
    (void)align;
#ifdef _WIN64
    return sycl::malloc_device(size, dev, ctx);
#else
#if defined(ONEMATH_ENABLE_CUBLAS_BACKEND) || defined(ONEMATH_ENABLE_ROCBLAS_BACKEND)
    return sycl::aligned_alloc_device(align, size, dev, ctx);
#endif
#if !defined(ONEMATH_ENABLE_CUBLAS_BACKEND) && !defined(ONEMATH_ENABLE_ROCBLAS_BACKEND)
    return sycl::malloc_device(size, dev, ctx);
#endif
#endif
}

static inline void free_shared(void* p, sycl::context ctx) {
    sycl::free(p, ctx);
}

static inline void free_usm(void* p, sycl::context ctx) {
    sycl::free(p, ctx);
}

} // namespace math
} // namespace oneapi

#endif // _TEST_HELPER_HPP_
