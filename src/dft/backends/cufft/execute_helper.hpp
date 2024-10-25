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

#ifndef _ONEMKL_DFT_SRC_EXECUTE_HELPER_CUFFT_HPP_
#define _ONEMKL_DFT_SRC_EXECUTE_HELPER_CUFFT_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/dft/detail/commit_impl.hpp"
#include "oneapi/mkl/dft/detail/descriptor_impl.hpp"
#include "oneapi/mkl/dft/types.hpp"
#include "oneapi/mkl/exceptions.hpp"

#include <cuda.h>
#include <cufft.h>

namespace oneapi::mkl::dft::cufft::detail {

template <dft::precision prec, dft::domain dom>
inline dft::detail::commit_impl<prec, dom>* checked_get_commit(
    dft::detail::descriptor<prec, dom>& desc) {
    auto commit_handle = dft::detail::get_commit(desc);
    if (commit_handle == nullptr || commit_handle->get_backend() != backend::cufft) {
        throw mkl::invalid_argument("dft/backends/cufft", "get_commit",
                                    "DFT descriptor has not been commited for cuFFT");
    }
    return commit_handle;
}

/// Throw an mkl::invalid_argument if the runtime param in the descriptor does not match
/// the expected value.
template <dft::config_param Param, dft::config_value Expected, typename DescT>
inline auto expect_config(DescT& desc, const char* message) {
    dft::config_value actual{ 0 };
    desc.get_value(Param, &actual);
    if (actual != Expected) {
        throw mkl::invalid_argument("dft/backends/cufft", "expect_config", message);
    }
}

enum class Direction { Forward = CUFFT_FORWARD, Backward = CUFFT_INVERSE };

template <Direction dir, typename forward_data_type>
void cufft_execute(const std::string& func, CUstream stream, cufftHandle plan, void* input,
                   void* output) {
    constexpr bool is_real = std::is_floating_point_v<forward_data_type>;
    using single_type = std::conditional_t<is_real, float, std::complex<float>>;
    constexpr bool is_single = std::is_same_v<forward_data_type, single_type>;

    if constexpr (is_real) {
        if constexpr (dir == Direction::Forward) {
            if constexpr (is_single) {
                auto result = cufftExecR2C(plan, reinterpret_cast<cufftReal*>(input),
                                           reinterpret_cast<cufftComplex*>(output));
                if (result != CUFFT_SUCCESS) {
                    throw oneapi::mkl::exception("dft/backends/cufft", func,
                                                 "cufftExecR2C returned " + std::to_string(result));
                }
            }
            else {
                auto result = cufftExecD2Z(plan, reinterpret_cast<cufftDoubleReal*>(input),
                                           reinterpret_cast<cufftDoubleComplex*>(output));
                if (result != CUFFT_SUCCESS) {
                    throw oneapi::mkl::exception("dft/backends/cufft", func,
                                                 "cufftExecD2Z returned " + std::to_string(result));
                }
            }
        }
        else {
            if constexpr (is_single) {
                auto result = cufftExecC2R(plan, reinterpret_cast<cufftComplex*>(input),
                                           reinterpret_cast<cufftReal*>(output));
                if (result != CUFFT_SUCCESS) {
                    throw oneapi::mkl::exception("dft/backends/cufft", func,
                                                 "cufftExecC2R returned " + std::to_string(result));
                }
            }
            else {
                auto result = cufftExecZ2D(plan, reinterpret_cast<cufftDoubleComplex*>(input),
                                           reinterpret_cast<cufftDoubleReal*>(output));
                if (result != CUFFT_SUCCESS) {
                    throw oneapi::mkl::exception("dft/backends/cufft", func,
                                                 "cufftExecZ2D returned " + std::to_string(result));
                }
            }
        }
    }
    else {
        if constexpr (is_single) {
            auto result =
                cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(input),
                             reinterpret_cast<cufftComplex*>(output), static_cast<int>(dir));
            if (result != CUFFT_SUCCESS) {
                throw oneapi::mkl::exception("dft/backends/cufft", func,
                                             "cufftExecC2C returned " + std::to_string(result));
            }
        }
        else {
            auto result =
                cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(input),
                             reinterpret_cast<cufftDoubleComplex*>(output), static_cast<int>(dir));
            if (result != CUFFT_SUCCESS) {
                throw oneapi::mkl::exception("dft/backends/cufft", func,
                                             "cufftExecZ2Z returned " + std::to_string(result));
            }
        }
    }
#ifndef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
    // If not using the enqueue native extension, the host task must wait on the
    // asynchronous operation to complete. Otherwise it report the operation
    // as complete early.
    auto result = cuStreamSynchronize(stream);
    if (result != CUDA_SUCCESS) {
        throw oneapi::mkl::exception("dft/backends/cufft", func,
                                     "cuStreamSynchronize returned " + std::to_string(result));
    }
#endif
}

inline CUstream setup_stream(const std::string& func, sycl::interop_handle ih, cufftHandle plan) {
    auto stream = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
    auto result = cufftSetStream(plan, stream);
    if (result != CUFFT_SUCCESS) {
        throw oneapi::mkl::exception("dft/backends/cufft", func,
                                     "cufftSetStream returned " + std::to_string(result));
    }
    return stream;
}

} // namespace oneapi::mkl::dft::cufft::detail

#endif // _ONEMKL_DFT_SRC_EXECUTE_HELPER_CUFFT_HPP_
