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

#ifndef _ONEMATH_SPARSE_BLAS_BACKENDS_ROCSPARSE_ERROR_HPP_
#define _ONEMATH_SPARSE_BLAS_BACKENDS_ROCSPARSE_ERROR_HPP_

#include <string>

#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>

#include "oneapi/math/exceptions.hpp"

namespace oneapi::math::sparse::rocsparse::detail {

inline std::string hip_result_to_str(hipError_t result) {
    switch (result) {
#define ONEMATH_ROCSPARSE_CASE(STATUS) \
    case STATUS: return #STATUS
        ONEMATH_ROCSPARSE_CASE(hipSuccess);
        ONEMATH_ROCSPARSE_CASE(hipErrorInvalidContext);
        ONEMATH_ROCSPARSE_CASE(hipErrorInvalidKernelFile);
        ONEMATH_ROCSPARSE_CASE(hipErrorMemoryAllocation);
        ONEMATH_ROCSPARSE_CASE(hipErrorInitializationError);
        ONEMATH_ROCSPARSE_CASE(hipErrorLaunchFailure);
        ONEMATH_ROCSPARSE_CASE(hipErrorLaunchOutOfResources);
        ONEMATH_ROCSPARSE_CASE(hipErrorInvalidDevice);
        ONEMATH_ROCSPARSE_CASE(hipErrorInvalidValue);
        ONEMATH_ROCSPARSE_CASE(hipErrorInvalidDevicePointer);
        ONEMATH_ROCSPARSE_CASE(hipErrorInvalidMemcpyDirection);
        ONEMATH_ROCSPARSE_CASE(hipErrorUnknown);
        ONEMATH_ROCSPARSE_CASE(hipErrorInvalidResourceHandle);
        ONEMATH_ROCSPARSE_CASE(hipErrorNotReady);
        ONEMATH_ROCSPARSE_CASE(hipErrorNoDevice);
        ONEMATH_ROCSPARSE_CASE(hipErrorPeerAccessAlreadyEnabled);
        ONEMATH_ROCSPARSE_CASE(hipErrorPeerAccessNotEnabled);
        ONEMATH_ROCSPARSE_CASE(hipErrorRuntimeMemory);
        ONEMATH_ROCSPARSE_CASE(hipErrorRuntimeOther);
        ONEMATH_ROCSPARSE_CASE(hipErrorHostMemoryAlreadyRegistered);
        ONEMATH_ROCSPARSE_CASE(hipErrorHostMemoryNotRegistered);
        ONEMATH_ROCSPARSE_CASE(hipErrorMapBufferObjectFailed);
        ONEMATH_ROCSPARSE_CASE(hipErrorTbd);
        default: return "<unknown>";
    }
}

#define HIP_ERROR_FUNC(func, ...)                                                          \
    do {                                                                                   \
        auto res = func(__VA_ARGS__);                                                      \
        if (res != hipSuccess) {                                                           \
            throw oneapi::math::exception("sparse_blas", #func,                            \
                                          "hip error: " + detail::hip_result_to_str(res)); \
        }                                                                                  \
    } while (0)

inline std::string rocsparse_status_to_str(rocsparse_status status) {
    switch (status) {
#define ONEMATH_ROCSPARSE_CASE(STATUS) \
    case STATUS: return #STATUS
        ONEMATH_ROCSPARSE_CASE(rocsparse_status_success);
        ONEMATH_ROCSPARSE_CASE(rocsparse_status_invalid_handle);
        ONEMATH_ROCSPARSE_CASE(rocsparse_status_not_implemented);
        ONEMATH_ROCSPARSE_CASE(rocsparse_status_invalid_pointer);
        ONEMATH_ROCSPARSE_CASE(rocsparse_status_invalid_size);
        ONEMATH_ROCSPARSE_CASE(rocsparse_status_memory_error);
        ONEMATH_ROCSPARSE_CASE(rocsparse_status_internal_error);
        ONEMATH_ROCSPARSE_CASE(rocsparse_status_invalid_value);
        ONEMATH_ROCSPARSE_CASE(rocsparse_status_arch_mismatch);
        ONEMATH_ROCSPARSE_CASE(rocsparse_status_zero_pivot);
        ONEMATH_ROCSPARSE_CASE(rocsparse_status_not_initialized);
        ONEMATH_ROCSPARSE_CASE(rocsparse_status_type_mismatch);
        ONEMATH_ROCSPARSE_CASE(rocsparse_status_requires_sorted_storage);
        ONEMATH_ROCSPARSE_CASE(rocsparse_status_thrown_exception);
        ONEMATH_ROCSPARSE_CASE(rocsparse_status_continue);
#undef ONEMATH_ROCSPARSE_CASE
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
                throw oneapi::math::unimplemented("sparse_blas", function, error_str);
            case rocsparse_status_invalid_handle:
            case rocsparse_status_invalid_pointer:
            case rocsparse_status_invalid_size:
            case rocsparse_status_invalid_value:
                throw oneapi::math::invalid_argument("sparse_blas", function, error_str);
            case rocsparse_status_not_initialized:
                throw oneapi::math::uninitialized("sparse_blas", function, error_str);
            default: throw oneapi::math::exception("sparse_blas", function, error_str);
        }
    }
}

#define ROCSPARSE_ERR_FUNC(func, ...)        \
    do {                                     \
        auto status = func(__VA_ARGS__);     \
        detail::check_status(status, #func); \
    } while (0)

} // namespace oneapi::math::sparse::rocsparse::detail

#endif // _ONEMATH_SPARSE_BLAS_BACKENDS_ROCSPARSE_ERROR_HPP_
