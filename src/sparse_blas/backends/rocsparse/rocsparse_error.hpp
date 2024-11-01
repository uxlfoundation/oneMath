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

#ifndef _ONEMKL_SPARSE_BLAS_BACKENDS_ROCSPARSE_ERROR_HPP_
#define _ONEMKL_SPARSE_BLAS_BACKENDS_ROCSPARSE_ERROR_HPP_

#include <string>

#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>

#include "oneapi/mkl/exceptions.hpp"

namespace oneapi::mkl::sparse::rocsparse::detail {

inline std::string hip_result_to_str(hipError_t result) {
    switch (result) {
#define ONEMKL_ROCSPARSE_CASE(STATUS) \
    case STATUS: return #STATUS
        ONEMKL_ROCSPARSE_CASE(hipSuccess);
        ONEMKL_ROCSPARSE_CASE(hipErrorInvalidContext);
        ONEMKL_ROCSPARSE_CASE(hipErrorInvalidKernelFile);
        ONEMKL_ROCSPARSE_CASE(hipErrorMemoryAllocation);
        ONEMKL_ROCSPARSE_CASE(hipErrorInitializationError);
        ONEMKL_ROCSPARSE_CASE(hipErrorLaunchFailure);
        ONEMKL_ROCSPARSE_CASE(hipErrorLaunchOutOfResources);
        ONEMKL_ROCSPARSE_CASE(hipErrorInvalidDevice);
        ONEMKL_ROCSPARSE_CASE(hipErrorInvalidValue);
        ONEMKL_ROCSPARSE_CASE(hipErrorInvalidDevicePointer);
        ONEMKL_ROCSPARSE_CASE(hipErrorInvalidMemcpyDirection);
        ONEMKL_ROCSPARSE_CASE(hipErrorUnknown);
        ONEMKL_ROCSPARSE_CASE(hipErrorInvalidResourceHandle);
        ONEMKL_ROCSPARSE_CASE(hipErrorNotReady);
        ONEMKL_ROCSPARSE_CASE(hipErrorNoDevice);
        ONEMKL_ROCSPARSE_CASE(hipErrorPeerAccessAlreadyEnabled);
        ONEMKL_ROCSPARSE_CASE(hipErrorPeerAccessNotEnabled);
        ONEMKL_ROCSPARSE_CASE(hipErrorRuntimeMemory);
        ONEMKL_ROCSPARSE_CASE(hipErrorRuntimeOther);
        ONEMKL_ROCSPARSE_CASE(hipErrorHostMemoryAlreadyRegistered);
        ONEMKL_ROCSPARSE_CASE(hipErrorHostMemoryNotRegistered);
        ONEMKL_ROCSPARSE_CASE(hipErrorMapBufferObjectFailed);
        ONEMKL_ROCSPARSE_CASE(hipErrorTbd);
        default: return "<unknown>";
    }
}

#define HIP_ERROR_FUNC(func, ...)                                                         \
    do {                                                                                  \
        auto res = func(__VA_ARGS__);                                                     \
        if (res != hipSuccess) {                                                          \
            throw oneapi::mkl::exception("sparse_blas", #func,                            \
                                         "hip error: " + detail::hip_result_to_str(res)); \
        }                                                                                 \
    } while (0)

inline std::string rocsparse_status_to_str(rocsparse_status status) {
    switch (status) {
#define ONEMKL_ROCSPARSE_CASE(STATUS) \
    case STATUS: return #STATUS
        ONEMKL_ROCSPARSE_CASE(rocsparse_status_success);
        ONEMKL_ROCSPARSE_CASE(rocsparse_status_invalid_handle);
        ONEMKL_ROCSPARSE_CASE(rocsparse_status_not_implemented);
        ONEMKL_ROCSPARSE_CASE(rocsparse_status_invalid_pointer);
        ONEMKL_ROCSPARSE_CASE(rocsparse_status_invalid_size);
        ONEMKL_ROCSPARSE_CASE(rocsparse_status_memory_error);
        ONEMKL_ROCSPARSE_CASE(rocsparse_status_internal_error);
        ONEMKL_ROCSPARSE_CASE(rocsparse_status_invalid_value);
        ONEMKL_ROCSPARSE_CASE(rocsparse_status_arch_mismatch);
        ONEMKL_ROCSPARSE_CASE(rocsparse_status_zero_pivot);
        ONEMKL_ROCSPARSE_CASE(rocsparse_status_not_initialized);
        ONEMKL_ROCSPARSE_CASE(rocsparse_status_type_mismatch);
        ONEMKL_ROCSPARSE_CASE(rocsparse_status_requires_sorted_storage);
        ONEMKL_ROCSPARSE_CASE(rocsparse_status_thrown_exception);
        ONEMKL_ROCSPARSE_CASE(rocsparse_status_continue);
#undef ONEMKL_ROCSPARSE_CASE
        default: return "<unknown>";
    }
}

inline void check_status(rocsparse_status status, const std::string& function,
                         std::string error_str = "") {
    if (status != rocsparse_status_success) {
        if (!error_str.empty()) {
            error_str += "; ";
        }
        error_str += "rocSPARSE status: " + rocsparse_status_to_str(status);
        switch (status) {
            case rocsparse_status_not_implemented:
                throw oneapi::mkl::unimplemented("sparse_blas", function, error_str);
            case rocsparse_status_invalid_handle:
            case rocsparse_status_invalid_pointer:
            case rocsparse_status_invalid_size:
            case rocsparse_status_invalid_value:
                throw oneapi::mkl::invalid_argument("sparse_blas", function, error_str);
            case rocsparse_status_not_initialized:
                throw oneapi::mkl::uninitialized("sparse_blas", function, error_str);
            default: throw oneapi::mkl::exception("sparse_blas", function, error_str);
        }
    }
}

#define ROCSPARSE_ERR_FUNC(func, ...)        \
    do {                                     \
        auto status = func(__VA_ARGS__);     \
        detail::check_status(status, #func); \
    } while (0)

} // namespace oneapi::mkl::sparse::rocsparse::detail

#endif // _ONEMKL_SPARSE_BLAS_BACKENDS_ROCSPARSE_ERROR_HPP_
