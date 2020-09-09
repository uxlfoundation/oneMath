/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include "oneapi/mkl/detail/config.hpp"
#include "oneapi/mkl.hpp"

#ifdef _WIN64
#include <malloc.h>
#else
#include <stdlib.h>
#endif

#define test_failed  0
#define test_passed  1
#define test_skipped 2

#define EXPECT_TRUEORSKIP(a)             \
    do {                                 \
        int res = a;                     \
        if (res == test_skipped)         \
            GTEST_SKIP();                \
        else                             \
            EXPECT_EQ(res, test_passed); \
    } while (0);

#ifdef ENABLE_MKLCPU_BACKEND
#define TEST_RUN_INTELCPU(q, func, args) func<oneapi::mkl::backend::mklcpu> args
#else
#define TEST_RUN_INTELCPU(q, func, args)
#endif

#ifdef ENABLE_MKLGPU_BACKEND
#define TEST_RUN_INTELGPU(q, func, args) func<oneapi::mkl::backend::mklgpu> args
#else
#define TEST_RUN_INTELGPU(q, func, args)
#endif

#ifdef ENABLE_CUBLAS_BACKEND
#define TEST_RUN_NVIDIAGPU(q, func, args) func<oneapi::mkl::backend::cublas> args
#else
#define TEST_RUN_NVIDIAGPU(q, func, args)
#endif

#define TEST_RUN_CT(q, func, args)                                             \
    do {                                                                       \
        if (q.is_host() || q.get_device().is_cpu())                            \
            TEST_RUN_INTELCPU(q, func, args);                                  \
        else if (q.get_device().is_gpu()) {                                    \
            unsigned int vendor_id = static_cast<unsigned int>(                \
                q.get_device().get_info<cl::sycl::info::device::vendor_id>()); \
            if (vendor_id == INTEL_ID)                                         \
                TEST_RUN_INTELGPU(q, func, args);                              \
            else if (vendor_id == NVIDIA_ID)                                   \
                TEST_RUN_NVIDIAGPU(q, func, args);                             \
        }                                                                      \
    } while (0);

class LayoutDeviceNamePrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<cl::sycl::device, oneapi::mkl::layout>> dev) const {
        std::string layout_name = std::get<1>(dev.param) == oneapi::mkl::layout::column_major
                                      ? "Column_Major"
                                      : "Row_Major";
        std::string dev_name = std::get<0>(dev.param).get_info<cl::sycl::info::device::name>();
        for (std::string::size_type i = 0; i < dev_name.size(); ++i) {
            if (!isalnum(dev_name[i]))
                dev_name[i] = '_';
        }
        std::string info_name = (layout_name.append("_")).append(dev_name);
        return info_name;
    }
};

/* to accomodate Windows and Linux differences between alligned_alloc and
   _aligned_malloc calls use oneapi::mkl::aligned_alloc and oneapi::mkl::aligned_free instead */
namespace oneapi {
namespace mkl {

static inline void *aligned_alloc(size_t align, size_t size) {
#ifdef _WIN64
    return ::_aligned_malloc(size, align);
#else
    return ::aligned_alloc(align, size);
#endif
}

static inline void aligned_free(void *p) {
#ifdef _WIN64
    ::_aligned_free(p);
#else
    ::free(p);
#endif
}

/* Support for Unified Shared Memory allocations for different backends */
static inline void *malloc_shared(size_t align, size_t size, cl::sycl::device dev,
                                  cl::sycl::context ctx) {
#ifdef _WIN64
    return cl::sycl::malloc_shared(size, dev, ctx);
#else
#ifdef ENABLE_CUBLAS_BACKEND
    return cl::sycl::aligned_alloc_shared(align, size, dev, ctx);
#else
    return cl::sycl::malloc_shared(size, dev, ctx);
#endif
#endif
}

static inline void free_shared(void *p, cl::sycl::context ctx) {
    cl::sycl::free(p, ctx);
}

} // namespace mkl
} // namespace oneapi

#endif // _TEST_HELPER_HPP_
