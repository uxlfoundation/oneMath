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

#ifndef _ONEMATH_BACKEND_SELECTOR_PREDICATES_HPP_
#define _ONEMATH_BACKEND_SELECTOR_PREDICATES_HPP_

#include <cstdint>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/detail/backends.hpp"
#include "oneapi/math/detail/get_device_id.hpp"

namespace oneapi {
namespace math {

template <backend Backend>
inline void backend_selector_precondition(sycl::queue&) {}

template <>
inline void backend_selector_precondition<backend::netlib>(sycl::queue& queue) {
#ifndef ONEMATH_DISABLE_PREDICATES
#ifdef __HIPSYCL__
    if (!(queue.is_host() || queue.get_device().is_cpu())) {
#else
    if (!queue.get_device().is_cpu()) {
#endif
        throw unsupported_device("",
                                 "backend_selector<backend::" + backend_map[backend::netlib] + ">",
                                 queue.get_device());
    }
#endif
}

template <>
inline void backend_selector_precondition<backend::mklcpu>(sycl::queue& queue) {
#ifndef ONEMATH_DISABLE_PREDICATES
#ifdef __HIPSYCL__
    if (!(queue.is_host() || queue.get_device().is_cpu())) {
#else
    if (!queue.get_device().is_cpu()) {
#endif
        throw unsupported_device("",
                                 "backend_selector<backend::" + backend_map[backend::mklcpu] + ">",
                                 queue.get_device());
    }
#endif
}

template <>
inline void backend_selector_precondition<backend::mklgpu>(sycl::queue& queue) {
#ifndef ONEMATH_DISABLE_PREDICATES
    unsigned int vendor_id =
        static_cast<unsigned int>(queue.get_device().get_info<sycl::info::device::vendor_id>());
    if (!(queue.get_device().is_gpu() && vendor_id == INTEL_ID)) {
        throw unsupported_device("",
                                 "backend_selector<backend::" + backend_map[backend::mklgpu] + ">",
                                 queue.get_device());
    }
#endif
}

template <>
inline void backend_selector_precondition<backend::cublas>(sycl::queue& queue) {
#ifndef ONEMATH_DISABLE_PREDICATES
    unsigned int vendor_id =
        static_cast<unsigned int>(queue.get_device().get_info<sycl::info::device::vendor_id>());
    if (!(queue.get_device().is_gpu() && vendor_id == NVIDIA_ID)) {
        throw unsupported_device("",
                                 "backend_selector<backend::" + backend_map[backend::cublas] + ">",
                                 queue.get_device());
    }
#endif
}

template <>
inline void backend_selector_precondition<backend::cusolver>(sycl::queue& queue) {
#ifndef ONEMATH_DISABLE_PREDICATES
    unsigned int vendor_id =
        static_cast<unsigned int>(queue.get_device().get_info<sycl::info::device::vendor_id>());
    if (!(queue.get_device().is_gpu() && vendor_id == NVIDIA_ID)) {
        throw unsupported_device(
            "", "backend_selector<backend::" + backend_map[backend::cusolver] + ">",
            queue.get_device());
    }
#endif
}

template <>
inline void backend_selector_precondition<backend::rocblas>(sycl::queue& queue) {
#ifndef ONEMATH_DISABLE_PREDICATES
    unsigned int vendor_id =
        static_cast<unsigned int>(queue.get_device().get_info<sycl::info::device::vendor_id>());
    if (!(queue.get_device().is_gpu() && vendor_id == AMD_ID)) {
        throw unsupported_device("",
                                 "backend_selector<backend::" + backend_map[backend::rocblas] + ">",
                                 queue.get_device());
    }
#endif
}

template <>
inline void backend_selector_precondition<backend::rocrand>(sycl::queue& queue) {
#ifndef ONEMATH_DISABLE_PREDICATES
    unsigned int vendor_id =
        static_cast<unsigned int>(queue.get_device().get_info<sycl::info::device::vendor_id>());
    if (!(queue.get_device().is_gpu() && vendor_id == AMD_ID)) {
        throw unsupported_device("",
                                 "backend_selector<backend::" + backend_map[backend::rocrand] + ">",
                                 queue.get_device());
    }
#endif
}

template <>
inline void backend_selector_precondition<backend::rocsolver>(sycl::queue& queue) {
#ifndef ONEMATH_DISABLE_PREDICATES
    unsigned int vendor_id =
        static_cast<unsigned int>(queue.get_device().get_info<sycl::info::device::vendor_id>());
    if (!(queue.get_device().is_gpu() && vendor_id == AMD_ID)) {
        throw unsupported_device(
            "", "backend_selector<backend::" + backend_map[backend::rocsolver] + ">",
            queue.get_device());
    }
#endif
}
} // namespace math
} // namespace oneapi

#endif //_ONEMATH_BACKEND_SELECTOR_PREDICATES_HPP_
