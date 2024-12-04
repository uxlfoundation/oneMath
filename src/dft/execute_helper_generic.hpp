/*******************************************************************************
* Copyright Codeplay Software Ltd.
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

#ifndef _ONEMATH_DFT_SRC_EXECUTE_HELPER_GENERIC_HPP_
#define _ONEMATH_DFT_SRC_EXECUTE_HELPER_GENERIC_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

namespace oneapi::math::dft::detail {

/** Wrap interop API to launch interop host task.
 * 
 * @tparam HandlerT The command group handler type
 * @tparam FnT The body of the enqueued task
 *
 * Either uses host task interop API, or enqueue native command extension.
 * This extension avoids host synchronization after 
 * the native call is complete.
 */
template <typename HandlerT, typename FnT>
static inline void fft_enqueue_task(HandlerT&& cgh, FnT&& f) {
#ifdef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
    cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
#else
    cgh.host_task([=](sycl::interop_handle ih) {
#endif
        f(std::move(ih));
    });
}

} // namespace oneapi::math::dft::detail

#endif // _ONEMATH_DFT_SRC_EXECUTE_HELPER_GENERIC_HPP_
