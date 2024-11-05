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

#ifndef _ONEMATH_SRC_SPARSE_BLAS_SYCL_HELPER_HPP_
#define _ONEMATH_SRC_SPARSE_BLAS_SYCL_HELPER_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

namespace oneapi::math::sparse::detail {

/// Return whether a pointer is accessible on the host
template <typename T>
inline bool is_ptr_accessible_on_host(sycl::queue queue, const T* host_or_device_ptr) {
    auto alloc_type = sycl::get_pointer_type(host_or_device_ptr, queue.get_context());
    return alloc_type == sycl::usm::alloc::host || alloc_type == sycl::usm::alloc::shared ||
           alloc_type == sycl::usm::alloc::unknown;
}

/// Return a scalar on the host from a pointer to host or device memory
template <typename T>
inline T get_scalar_on_host(sycl::queue& queue, const T* host_or_device_ptr,
                            bool is_ptr_accessible_on_host) {
    if (is_ptr_accessible_on_host) {
        return *host_or_device_ptr;
    }
    T scalar;
    auto event = queue.copy(host_or_device_ptr, &scalar, 1);
    event.wait_and_throw();
    return scalar;
}

/// Submit the release of \p ptr in a host_task waiting on the dependencies
template <typename T>
sycl::event submit_release(sycl::queue& queue, T* ptr,
                           const std::vector<sycl::event>& dependencies) {
    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.host_task([=]() { delete ptr; });
    });
}

/// Merge multiple event dependencies into one
inline sycl::event collapse_dependencies(sycl::queue& queue,
                                         const std::vector<sycl::event>& dependencies) {
    if (dependencies.empty()) {
        return {};
    }
    else if (dependencies.size() == 1) {
        return dependencies[0];
    }

    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.host_task([=]() {});
    });
}

} // namespace oneapi::math::sparse::detail

#endif // _ONEMATH_SRC_SPARSE_BLAS_SYCL_HELPER_HPP_
