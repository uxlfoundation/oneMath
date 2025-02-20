/*******************************************************************************
* Copyright 2025 SiPearl
* Copyright 2020-2021 Intel Corporation
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

#ifndef _RNG_CPU_COMMON_HPP_
#define _RNG_CPU_COMMON_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#define GET_MULTI_PTR template get_multi_ptr<sycl::access::decorated::yes>().get_raw()
#define __fp16        _Float16
#define INTEGER64     1

#include "armpl.h"

namespace oneapi {
namespace math {
namespace rng {
namespace armpl {

inline int check_armpl_version(armpl_int_t major_req, armpl_int_t minor_req, armpl_int_t build_req,
                               const char* message) {
    armpl_int_t major, minor, build;
    char* tag;
    armplversion(&major, &minor, &build, (const char**)&tag);
    if (major > major_req) {
        return 0;
    }
    else if (major == major_req && minor > minor_req) {
        return 0;
    }
    else if (major == major_req && minor == minor_req && build >= build_req) {
        return 0;
    }
    throw oneapi::math::unimplemented("rng", "version support", message);
}

template <typename K, typename H, typename F>
static inline auto host_task_internal(H& cgh, F f, int) -> decltype(cgh.host_task(f)) {
    return cgh.host_task(f);
}

template <typename K, typename H, typename F>
static inline void host_task_internal(H& cgh, F f, long) {
#ifndef __SYCL_DEVICE_ONLY__
    cgh.template single_task<K>(f);
#endif
}

template <typename K, typename H, typename F>
static inline void host_task(H& cgh, F f) {
    (void)host_task_internal<K>(cgh, f, 0);
}

template <typename Engine, typename Distr>
class kernel_name {};

template <typename Engine, typename Distr>
class kernel_name_usm {};

} // namespace armpl
} // namespace rng
} // namespace math
} // namespace oneapi

#endif //_RNG_CPU_COMMON_HPP_
