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

#ifndef _ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_ERROR_HPP_
#define _ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_ERROR_HPP_

#include <string>

#include <cuda.h>
#include <cusparse.h>

#include "oneapi/mkl/exceptions.hpp"

namespace oneapi::mkl::sparse::cusparse::detail {

inline std::string cuda_result_to_str(CUresult result) {
    switch (result) {
#define ONEMKL_CUSPARSE_CASE(STATUS) \
    case STATUS: return #STATUS
        ONEMKL_CUSPARSE_CASE(CUDA_SUCCESS);
        ONEMKL_CUSPARSE_CASE(CUDA_ERROR_NOT_PERMITTED);
        ONEMKL_CUSPARSE_CASE(CUDA_ERROR_INVALID_CONTEXT);
        ONEMKL_CUSPARSE_CASE(CUDA_ERROR_INVALID_DEVICE);
        ONEMKL_CUSPARSE_CASE(CUDA_ERROR_INVALID_VALUE);
        ONEMKL_CUSPARSE_CASE(CUDA_ERROR_OUT_OF_MEMORY);
        ONEMKL_CUSPARSE_CASE(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);
        default: return "<unknown>";
    }
}

#define CUDA_ERROR_FUNC(func, ...)                                                          \
    do {                                                                                    \
        auto res = func(__VA_ARGS__);                                                       \
        if (res != CUDA_SUCCESS) {                                                          \
            throw oneapi::mkl::exception("sparse_blas", #func,                              \
                                         "cuda error: " + detail::cuda_result_to_str(res)); \
        }                                                                                   \
    } while (0)

inline std::string cusparse_status_to_str(cusparseStatus_t status) {
    switch (status) {
#define ONEMKL_CUSPARSE_CASE(STATUS) \
    case STATUS: return #STATUS
        ONEMKL_CUSPARSE_CASE(CUSPARSE_STATUS_SUCCESS);
        ONEMKL_CUSPARSE_CASE(CUSPARSE_STATUS_NOT_INITIALIZED);
        ONEMKL_CUSPARSE_CASE(CUSPARSE_STATUS_ALLOC_FAILED);
        ONEMKL_CUSPARSE_CASE(CUSPARSE_STATUS_INVALID_VALUE);
        ONEMKL_CUSPARSE_CASE(CUSPARSE_STATUS_ARCH_MISMATCH);
        ONEMKL_CUSPARSE_CASE(CUSPARSE_STATUS_EXECUTION_FAILED);
        ONEMKL_CUSPARSE_CASE(CUSPARSE_STATUS_INTERNAL_ERROR);
        ONEMKL_CUSPARSE_CASE(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
        ONEMKL_CUSPARSE_CASE(CUSPARSE_STATUS_NOT_SUPPORTED);
        ONEMKL_CUSPARSE_CASE(CUSPARSE_STATUS_INSUFFICIENT_RESOURCES);
#undef ONEMKL_CUSPARSE_CASE
        default: return "<unknown>";
    }
}

inline void check_status(cusparseStatus_t status, const std::string& function,
                         std::string error_str = "") {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        if (!error_str.empty()) {
            error_str += "; ";
        }
        error_str += "cuSPARSE status: " + cusparse_status_to_str(status);
        switch (status) {
            case CUSPARSE_STATUS_NOT_SUPPORTED:
                throw oneapi::mkl::unimplemented("sparse_blas", function, error_str);
            case CUSPARSE_STATUS_NOT_INITIALIZED:
                throw oneapi::mkl::uninitialized("sparse_blas", function, error_str);
            case CUSPARSE_STATUS_INVALID_VALUE:
            case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
                throw oneapi::mkl::invalid_argument("sparse_blas", function, error_str);
            default: throw oneapi::mkl::exception("sparse_blas", function, error_str);
        }
    }
}

#define CUSPARSE_ERR_FUNC(func, ...)         \
    do {                                     \
        auto status = func(__VA_ARGS__);     \
        detail::check_status(status, #func); \
    } while (0)

} // namespace oneapi::mkl::sparse::cusparse::detail

#endif // _ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_ERROR_HPP_
