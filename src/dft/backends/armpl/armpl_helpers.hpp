/*******************************************************************************
* Copyright 2025 SiPearl
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

#ifndef _ONEMATH_DFT_SRC_ARMPL_HELPERS_HPP_
#define _ONEMATH_DFT_SRC_ARMPL_HELPERS_HPP_

#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/dft/detail/types_impl.hpp"

#include "fftw3.h"

namespace oneapi::math::dft::armpl::detail {

template <typename K, typename H, typename F>
static inline auto host_task_internal(H& cgh, F f, int) -> decltype(cgh.host_task(f)) {
    return cgh.host_task(f);
}

template <typename K, typename H, typename F>
static inline void host_task_internal([[maybe_unused]] H& cgh, [[maybe_unused]] F f, long) {
#ifndef __SYCL_DEVICE_ONLY__
    cgh.template single_task<K>(f);
#endif
}

template <typename K, typename H, typename F>
static inline void host_task(H& cgh, F f) {
    (void)host_task_internal<K>(cgh, f, 0);
}

template <typename Desc>
class kernel_name {};

template <typename AccType>
typename AccType::value_type* acc_to_ptr(AccType acc) {
    // no need to decorate the pointer with the address space for armpl since its just getting passed to the a host function.
    return acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

} // namespace oneapi::math::dft::armpl::detail

#endif // _ONEMATH_DFT_SRC_ARMPL_HELPERS_HPP_
