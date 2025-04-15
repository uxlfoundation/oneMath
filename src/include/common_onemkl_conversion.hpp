/*******************************************************************************
* Copyright Codeplay Software Ltd
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

#ifndef _ONEMATH_SRC_INCLUDE_COMMON_ONEMKL_TYPES_CONVERSION_HPP_
#define _ONEMATH_SRC_INCLUDE_COMMON_ONEMKL_TYPES_CONVERSION_HPP_

// The file is used to convert oneMath types to Intel(R) oneMKL types for all the common types shared across domains.
// The file assumes that the common types are identical between the 2 libraries, except for their namespace.

#include <mkl/exceptions.hpp>
#include <mkl/types.hpp>

#include "oneapi/math/types.hpp"
#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/lapack/exceptions.hpp"

namespace oneapi {
namespace math {
namespace detail {

inline auto get_onemkl_transpose(oneapi::math::transpose* param_ptr) {
    return reinterpret_cast<oneapi::mkl::transpose*>(param_ptr);
}
inline auto get_onemkl_transpose(oneapi::math::transpose param) {
    return *get_onemkl_transpose(&param);
}

inline auto get_onemkl_uplo(oneapi::math::uplo* param_ptr) {
    return reinterpret_cast<oneapi::mkl::uplo*>(param_ptr);
}
inline auto get_onemkl_uplo(oneapi::math::uplo param) {
    return *get_onemkl_uplo(&param);
}

inline auto get_onemkl_diag(oneapi::math::diag* param_ptr) {
    return reinterpret_cast<oneapi::mkl::diag*>(param_ptr);
}
inline auto get_onemkl_diag(oneapi::math::diag param) {
    return *get_onemkl_diag(&param);
}

inline auto get_onemkl_side(oneapi::math::side* param_ptr) {
    return reinterpret_cast<oneapi::mkl::side*>(param_ptr);
}
inline auto get_onemkl_side(oneapi::math::side param) {
    return *get_onemkl_side(&param);
}

inline auto get_onemkl_offset(oneapi::math::offset param) {
    return *reinterpret_cast<oneapi::mkl::offset*>(&param);
}

inline auto get_onemkl_layout(oneapi::math::layout param) {
    return *reinterpret_cast<oneapi::mkl::layout*>(&param);
}

inline auto get_onemkl_index_base(oneapi::math::index_base param) {
    return *reinterpret_cast<oneapi::mkl::index_base*>(&param);
}

inline auto get_onemkl_job(oneapi::math::job param) {
    return *reinterpret_cast<oneapi::mkl::job*>(&param);
}

inline auto get_onemkl_jobsvd(oneapi::math::jobsvd param) {
    return *reinterpret_cast<oneapi::mkl::jobsvd*>(&param);
}

inline auto get_onemkl_generate(oneapi::math::generate param) {
    return *reinterpret_cast<oneapi::mkl::generate*>(&param);
}

inline auto get_onemkl_compz(oneapi::math::compz param) {
    return *reinterpret_cast<oneapi::mkl::compz*>(&param);
}

inline auto get_onemkl_direct(oneapi::math::direct param) {
    return *reinterpret_cast<oneapi::mkl::direct*>(&param);
}

inline auto get_onemkl_storev(oneapi::math::storev param) {
    return *reinterpret_cast<oneapi::mkl::storev*>(&param);
}

inline auto get_onemkl_rangev(oneapi::math::rangev param) {
    return *reinterpret_cast<oneapi::mkl::rangev*>(&param);
}

inline auto get_onemkl_order(oneapi::math::order param) {
    return *reinterpret_cast<oneapi::mkl::order*>(&param);
}

} // namespace detail
} // namespace math
} // namespace oneapi

// Rethrow Intel(R) oneMKL exceptions as oneMath exceptions
#define RETHROW_ONEMKL_EXCEPTIONS(EXPRESSION)                 \
    do {                                                      \
        try {                                                 \
            EXPRESSION;                                       \
        }                                                     \
        catch (const oneapi::mkl::unsupported_device& e) {    \
            throw oneapi::math::unsupported_device(e.what()); \
        }                                                     \
        catch (const oneapi::mkl::host_bad_alloc& e) {        \
            throw oneapi::math::host_bad_alloc(e.what());     \
        }                                                     \
        catch (const oneapi::mkl::device_bad_alloc& e) {      \
            throw oneapi::math::device_bad_alloc(e.what());   \
        }                                                     \
        catch (const oneapi::mkl::unimplemented& e) {         \
            throw oneapi::math::unimplemented(e.what());      \
        }                                                     \
        catch (const oneapi::mkl::invalid_argument& e) {      \
            throw oneapi::math::invalid_argument(e.what());   \
        }                                                     \
        catch (const oneapi::mkl::uninitialized& e) {         \
            throw oneapi::math::uninitialized(e.what());      \
        }                                                     \
        catch (const oneapi::mkl::computation_error& e) {     \
            throw oneapi::math::computation_error(e.what());  \
        }                                                     \
        catch (const oneapi::mkl::batch_error& e) {           \
            throw oneapi::math::batch_error(e.what());        \
        }                                                     \
        catch (const oneapi::mkl::exception& e) {             \
            throw oneapi::math::exception(e.what());          \
        }                                                     \
    } while (0)

#define RETHROW_ONEMKL_EXCEPTIONS_RET(EXPRESSION)     \
    do {                                              \
        RETHROW_ONEMKL_EXCEPTIONS(return EXPRESSION); \
    } while (0)

// Rethrow Intel(R) oneMKL LAPCK exceptions as oneMath LAPACK exceptions
#define RETHROW_ONEMKL_LAPACK_EXCEPTIONS(EXPRESSION)                                              \
    do {                                                                                          \
        try {                                                                                     \
            EXPRESSION;                                                                           \
        }                                                                                         \
        catch (const oneapi::mkl::unsupported_device& e) {                                        \
            throw oneapi::math::unsupported_device(e.what());                                     \
        }                                                                                         \
        catch (const oneapi::mkl::host_bad_alloc& e) {                                            \
            throw oneapi::math::host_bad_alloc(e.what());                                         \
        }                                                                                         \
        catch (const oneapi::mkl::device_bad_alloc& e) {                                          \
            throw oneapi::math::device_bad_alloc(e.what());                                       \
        }                                                                                         \
        catch (const oneapi::mkl::unimplemented& e) {                                             \
            throw oneapi::math::unimplemented(e.what());                                          \
        }                                                                                         \
        catch (const oneapi::mkl::uninitialized& e) {                                             \
            throw oneapi::math::uninitialized(e.what());                                          \
        }                                                                                         \
        catch (const oneapi::mkl::lapack::invalid_argument& e) {                                  \
            throw oneapi::math::lapack::invalid_argument(e.what(), e.info(), e.detail());         \
        }                                                                                         \
        catch (const oneapi::mkl::lapack::computation_error& e) {                                 \
            throw oneapi::math::lapack::computation_error(e.what(), e.info());                    \
        }                                                                                         \
        catch (const oneapi::mkl::lapack::batch_error& e) {                                       \
            throw oneapi::math::lapack::batch_error(e.what(), e.info(), e.ids(), e.exceptions()); \
        }                                                                                         \
        catch (const oneapi::mkl::exception& e) {                                                 \
            throw oneapi::math::exception(e.what());                                              \
        }                                                                                         \
    } while (0)

#define RETHROW_ONEMKL_LAPACK_EXCEPTIONS_RET(EXPRESSION)     \
    do {                                                     \
        RETHROW_ONEMKL_LAPACK_EXCEPTIONS(return EXPRESSION); \
    } while (0)

#endif // _ONEMATH_SRC_INCLUDE_COMMON_ONEMKL_TYPES_CONVERSION_HPP_
