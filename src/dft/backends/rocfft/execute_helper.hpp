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

#ifndef _ONEMATH_DFT_SRC_EXECUTE_HELPER_ROCFFT_HPP_
#define _ONEMATH_DFT_SRC_EXECUTE_HELPER_ROCFFT_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/dft/detail/commit_impl.hpp"
#include "oneapi/math/dft/detail/descriptor_impl.hpp"
#include "oneapi/math/dft/types.hpp"
#include "oneapi/math/exceptions.hpp"

#include <hip/hip_runtime.h>
#include <rocfft.h>

namespace oneapi::math::dft::rocfft::detail {

template <dft::precision prec, dft::domain dom>
inline dft::detail::commit_impl<prec, dom>* checked_get_commit(
    dft::detail::descriptor<prec, dom>& desc) {
    auto commit_handle = dft::detail::get_commit(desc);
    if (commit_handle == nullptr || commit_handle->get_backend() != backend::rocfft) {
        throw math::invalid_argument("dft/backends/rocfft", "get_commit",
                                     "DFT descriptor has not been commited for rocFFT");
    }
    return commit_handle;
}

/// Throw an math::invalid_argument if the runtime param in the descriptor does not match
/// the expected value.
template <dft::config_param Param, dft::config_value Expected, typename DescT>
inline auto expect_config(DescT& desc, const char* message) {
    dft::config_value actual{ 0 };
    desc.get_value(Param, &actual);
    if (actual != Expected) {
        throw math::invalid_argument("dft/backends/rocfft", "expect_config", message);
    }
}

template <typename Acc>
inline void* native_mem(sycl::interop_handle& ih, Acc& buf) {
    return ih.get_native_mem<sycl::backend::ext_oneapi_hip>(buf);
}

inline hipStream_t setup_stream(const std::string& func, sycl::interop_handle& ih,
                                rocfft_execution_info info) {
    auto stream = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
    auto result = rocfft_execution_info_set_stream(info, stream);
    if (result != rocfft_status_success) {
        throw oneapi::math::exception(
            "dft/backends/rocfft", func,
            "rocfft_execution_info_set_stream returned " + std::to_string(result));
    }
    return stream;
}

inline void sync_checked(const std::string& func, hipStream_t stream) {
    auto result = hipStreamSynchronize(stream);
    if (result != hipSuccess) {
        throw oneapi::math::exception("dft/backends/rocfft", func,
                                      "hipStreamSynchronize returned " + std::to_string(result));
    }
}

inline void execute_checked(const std::string& func, hipStream_t stream, const rocfft_plan plan,
                            void* in_buffer[], void* out_buffer[], rocfft_execution_info info) {
    auto result = rocfft_execute(plan, in_buffer, out_buffer, info);
    if (result != rocfft_status_success) {
        throw oneapi::math::exception("dft/backends/rocfft", func,
                                      "rocfft_execute returned " + std::to_string(result));
    }
#ifndef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
    // If not using equeue native extension, the host task must wait on the
    // asynchronous operation to complete. Otherwise it report the operation
    // as complete early.
    sync_checked(func, stream);
#endif
}

} // namespace oneapi::math::dft::rocfft::detail

#endif // _ONEMATH_DFT_SRC_EXECUTE_HELPER_ROCFFT_HPP_
