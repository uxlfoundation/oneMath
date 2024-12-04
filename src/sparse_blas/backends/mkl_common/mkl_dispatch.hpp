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

#ifndef _ONEMATH_SRC_SPARSE_BLAS_BACKENDS_MKL_COMMON_MKL_DISPATCH_HPP_
#define _ONEMATH_SRC_SPARSE_BLAS_BACKENDS_MKL_COMMON_MKL_DISPATCH_HPP_

/// Convert \p value_type to template type argument and use it to call \p op_functor.
#define DISPATCH_ONEMATH_OPERATION(function_name, value_type, op_functor, ...)                     \
    switch (value_type) {                                                                          \
        case detail::data_type::real_fp32: return op_functor<float>(__VA_ARGS__);                  \
        case detail::data_type::real_fp64: return op_functor<double>(__VA_ARGS__);                 \
        case detail::data_type::complex_fp32: return op_functor<std::complex<float>>(__VA_ARGS__); \
        case detail::data_type::complex_fp64:                                                      \
            return op_functor<std::complex<double>>(__VA_ARGS__);                                  \
        default:                                                                                   \
            throw oneapi::math::exception(                                                         \
                "sparse_blas", function_name,                                                      \
                "Internal error: unsupported type " + data_type_to_str(value_type));               \
    }

#endif // _ONEMATH_SRC_SPARSE_BLAS_BACKENDS_MKL_COMMON_MKL_DISPATCH_HPP_
