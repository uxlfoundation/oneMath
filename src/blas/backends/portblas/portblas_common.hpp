/*******************************************************************************
* Copyright Codeplay Software
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

#ifndef _PORTBLAS_COMMON_HPP_
#define _PORTBLAS_COMMON_HPP_

#include "portblas.hpp"
#include "oneapi/math/types.hpp"
#include "oneapi/math/exceptions.hpp"

#include <tuple>
#include <utility>

namespace oneapi {
namespace math {
namespace blas {
namespace portblas {

namespace detail {
// portBLAS handle type. Constructed with sycl::queue.
using handle_t = ::blas::SB_Handle;

// portBLAS buffer iterator. Constructed with sycl::buffer<ElemT,1>
template <typename ElemT>
using buffer_iterator_t = ::blas::BufferIterator<ElemT>;

// sycl complex data type (experimental)
template <typename ElemT>
using sycl_complex_t = sycl::ext::oneapi::experimental::complex<ElemT>;

/** A trait for obtaining equivalent portBLAS API types from oneMath API
 *  types.
 * 
 *  @tparam InputT is the oneMath type.
 *  portblas_type<InputT>::type should be the equivalent portBLAS type.
**/
template <typename InputT>
struct portblas_type;

#define DEF_PORTBLAS_TYPE(onemath_t, portblas_t) \
    template <>                                  \
    struct portblas_type<onemath_t> {            \
        using type = portblas_t;                 \
    };

DEF_PORTBLAS_TYPE(sycl::queue, handle_t)
DEF_PORTBLAS_TYPE(int64_t, int64_t)
DEF_PORTBLAS_TYPE(sycl::half, sycl::half)
DEF_PORTBLAS_TYPE(float, float)
DEF_PORTBLAS_TYPE(double, double)
DEF_PORTBLAS_TYPE(oneapi::math::transpose, char)
DEF_PORTBLAS_TYPE(oneapi::math::uplo, char)
DEF_PORTBLAS_TYPE(oneapi::math::side, char)
DEF_PORTBLAS_TYPE(oneapi::math::diag, char)
DEF_PORTBLAS_TYPE(std::complex<float>, sycl_complex_t<float>)
DEF_PORTBLAS_TYPE(std::complex<double>, sycl_complex_t<double>)
// Passthrough of portBLAS arg types for more complex wrapping.
DEF_PORTBLAS_TYPE(::blas::gemm_batch_type_t, ::blas::gemm_batch_type_t)

#undef DEF_PORTBLAS_TYPE

template <typename ElemT>
struct portblas_type<sycl::buffer<ElemT, 1>> {
    using type = buffer_iterator_t<ElemT>;
};

template <typename ElemT>
struct portblas_type<ElemT*> {
    using type = ElemT*;
};

// USM Complex
template <typename ElemT>
struct portblas_type<std::complex<ElemT>*> {
    using type = sycl_complex_t<ElemT>*;
};

template <typename ElemT>
struct portblas_type<const std::complex<ElemT>*> {
    using type = const sycl_complex_t<ElemT>*;
};

template <>
struct portblas_type<std::vector<sycl::event>> {
    using type = std::vector<sycl::event>;
};

/** Convert a oneMath argument to the type required for portBLAS.
 *  
 *  @tparam InputT The oneMath type.
 *  @param input The value of the oneMath type.
 *  @return The portBLAS value with appropriate type.
**/
template <typename InputT>
inline typename portblas_type<InputT>::type convert_to_portblas_type(InputT& input) {
    return typename portblas_type<InputT>::type(input);
}

template <>
inline char convert_to_portblas_type<oneapi::math::transpose>(oneapi::math::transpose& trans) {
    if (trans == oneapi::math::transpose::nontrans) {
        return 'n';
    }
    else if (trans == oneapi::math::transpose::trans) {
        return 't';
    }
    else { // trans == oneapi::math::transpose::conjtrans
        return 'c';
    }
}

template <>
inline char convert_to_portblas_type<oneapi::math::uplo>(oneapi::math::uplo& upper_lower) {
    if (upper_lower == oneapi::math::uplo::upper) {
        return 'u';
    }
    else {
        return 'l';
    }
}

template <>
inline char convert_to_portblas_type<oneapi::math::side>(oneapi::math::side& left_right) {
    if (left_right == oneapi::math::side::left) {
        return 'l';
    }
    else {
        return 'r';
    }
}

template <>
inline char convert_to_portblas_type<oneapi::math::diag>(oneapi::math::diag& unit_diag) {
    if (unit_diag == oneapi::math::diag::unit) {
        return 'u';
    }
    else {
        return 'n';
    }
}

template <typename... ArgT>
inline auto convert_to_portblas_type(ArgT... args) {
    return std::make_tuple(convert_to_portblas_type(args)...);
}

/** Throw an unsupported_device exception if a certain argument type is found in
 * the argument pack.
 *  
 *  @tparam CheckT is type to look for a template parameter pack.
 *  @tparam AspectVal is the device aspect required to support CheckT.
**/
template <typename CheckT, sycl::aspect AspectVal>
struct throw_if_unsupported_by_device {
    /** Operator to throw if unsupported.
 * 
 *  @tparam ArgTs The argument types to check.
 *  @param The message to include in the exception.
 *  @param q is the sycl::queue.
 *  @param args are the remaining args to check for CheckT in.
**/
    template <typename... ArgTs>
    void operator()(const std::string& message, sycl::queue q, ArgTs... args) {
        static constexpr bool checkTypeInPack = (std::is_same_v<CheckT, ArgTs> || ...);
        if (checkTypeInPack) {
            if (!q.get_info<sycl::info::queue::device>().has(AspectVal)) {
                throw math::unsupported_device("blas", message,
                                               q.get_info<sycl::info::queue::device>());
            }
        }
    }
};

} // namespace detail

#define CALL_PORTBLAS_FN(portBLASFunc, ...)                                                     \
    if constexpr (is_column_major()) {                                                          \
        detail::throw_if_unsupported_by_device<sycl::buffer<double>, sycl::aspect::fp64>{}(     \
            " portBLAS function requiring fp64 support", __VA_ARGS__);                          \
        detail::throw_if_unsupported_by_device<sycl::buffer<sycl::half>, sycl::aspect::fp16>{}( \
            " portBLAS function requiring fp16 support", __VA_ARGS__);                          \
        auto args = detail::convert_to_portblas_type(__VA_ARGS__);                              \
        auto fn = [](auto&&... targs) {                                                         \
            portBLASFunc(std::forward<decltype(targs)>(targs)...);                              \
        };                                                                                      \
        try {                                                                                   \
            std::apply(fn, args);                                                               \
        }                                                                                       \
        catch (const ::blas::unsupported_exception& e) {                                        \
            throw unimplemented("blas", e.what());                                              \
        }                                                                                       \
    }                                                                                           \
    else {                                                                                      \
        throw unimplemented("blas", "portBLAS function");                                       \
    }

#define CALL_PORTBLAS_USM_FN(portblasFunc, ...)                                   \
    if constexpr (is_column_major()) {                                            \
        detail::throw_if_unsupported_by_device<double, sycl::aspect::fp64>{}(     \
            " portBLAS function requiring fp64 support", __VA_ARGS__);            \
        detail::throw_if_unsupported_by_device<sycl::half, sycl::aspect::fp16>{}( \
            " portBLAS function requiring fp16 support", __VA_ARGS__);            \
        auto args = detail::convert_to_portblas_type(__VA_ARGS__);                \
        auto fn = [](auto&&... targs) {                                           \
            return portblasFunc(std::forward<decltype(targs)>(targs)...).back();  \
        };                                                                        \
        try {                                                                     \
            return std::apply(fn, args);                                          \
        }                                                                         \
        catch (const ::blas::unsupported_exception& e) {                          \
            throw unimplemented("blas", e.what());                                \
        }                                                                         \
    }                                                                             \
    else {                                                                        \
        throw unimplemented("blas", "portBLAS function");                         \
    }

} // namespace portblas
} // namespace blas
} // namespace math
} // namespace oneapi

#endif // _PORTBLAS_COMMON_HPP_
