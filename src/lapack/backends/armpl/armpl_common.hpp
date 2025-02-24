/*******************************************************************************
* Copyright 2025 SiPearl
* Copyright 2020-2021 Intel Corporation
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

#ifndef _ARMPL_COMMON_HPP_
#define _ARMPL_COMMON_HPP_

#define __fp16    _Float16
#define INTEGER64 1

#include <sycl/sycl.hpp>
#include <complex>

#include "armpl.h"

#include "oneapi/math/lapack/detail/armpl/onemath_lapack_armpl.hpp"
#include "oneapi/math/types.hpp"

#define GET_MULTI_PTR template get_multi_ptr<sycl::access::decorated::yes>().get_raw()

namespace oneapi {
namespace math {
namespace lapack {
namespace armpl {

template <typename K, typename H, typename F>
static inline auto host_task_internal(H& cgh, F f, int) -> decltype(cgh.host_task(f)) {
    return cgh.host_task(f);
}

template <typename K, typename H, typename F>
static inline void host_task_internal(H& cgh, F f, long) {
#ifndef __SYCL_DEVICE_ONLY__
    cgh.template single_task<K>(f);
#endif
}

template <typename K, typename H, typename F>
static inline void host_task(H& cgh, F f) {
    (void)host_task_internal<K>(cgh, f, 0);
}

inline char get_operation(oneapi::math::transpose trn) {
    switch (trn) {
        case oneapi::math::transpose::nontrans: return 'N';
        case oneapi::math::transpose::trans: return 'T';
        case oneapi::math::transpose::conjtrans: return 'C';
        default: throw "Wrong transpose Operation.";
    }
}

inline char get_fill_mode(oneapi::math::uplo ul) {
    switch (ul) {
        case oneapi::math::uplo::upper: return 'U';
        case oneapi::math::uplo::lower: return 'L';
        default: throw "Wrong fill mode.";
    }
}

inline char get_side_mode(oneapi::math::side lr) {
    switch (lr) {
        case oneapi::math::side::left: return 'L';
        case oneapi::math::side::right: return 'R';
        default: throw "Wrong side mode.";
    }
}

inline char get_generate(oneapi::math::generate qp) {
    switch (qp) {
        case oneapi::math::generate::Q: return 'Q';
        case oneapi::math::generate::P: return 'P';
        default: throw "Wrong generate.";
    }
}

inline char get_job(oneapi::math::job jobz) {
    switch (jobz) {
        case oneapi::math::job::N: return 'N';
        case oneapi::math::job::V: return 'V';
        default: throw "Wrong jobz.";
    }
}

inline char get_jobsvd(oneapi::math::jobsvd job) {
    switch (job) {
        case oneapi::math::jobsvd::N: return 'N';
        case oneapi::math::jobsvd::A: return 'A';
        case oneapi::math::jobsvd::O: return 'O';
        case oneapi::math::jobsvd::S: return 'S';
        default: throw "Wrong job.";
    }
}

inline char get_diag(oneapi::math::diag diag) {
    switch (diag) {
        case oneapi::math::diag::N: return 'N';
        case oneapi::math::diag::U: return 'U';
        default: throw "Wrong diag.";
    }
}

/*converting std::complex<T> to cu<T>Complex*/
/*converting sycl::half to __half*/
template <typename T>
struct ArmEquivalentType {
    using Type = T;
};
template <>
struct ArmEquivalentType<std::complex<float>> {
    using Type = armpl_singlecomplex_t;
};
template <>
struct ArmEquivalentType<std::complex<double>> {
    using Type = armpl_doublecomplex_t;
};

template <typename T>
constexpr bool is_complex = false;
template <typename T>
constexpr bool is_complex<std::complex<T>> = true;
template <>
inline constexpr bool is_complex<armpl_singlecomplex_t> = true;
template <>
inline constexpr bool is_complex<armpl_doublecomplex_t> = true;

template <typename T>
constexpr auto cast_to_int_if_complex(const T& alpha) {
    if constexpr (is_complex<T>) {
        return static_cast<std::int64_t>((*((T*)&alpha)));
    }
    else {
        return (std::int64_t)alpha;
    }
}

class armpl_lapacke_error : virtual public std::runtime_error {
protected:
    // Lapacke errors are already reported by a printf in lapacke_xerbla, so this may be redundant.
    inline std::string lapacke_error_message(std::int64_t info) {
        if (info == LAPACK_WORK_MEMORY_ERROR) {
            return std::string("Not enough memory to allocate work array\n");
        }
        else if (info == LAPACK_TRANSPOSE_MEMORY_ERROR) {
            return std::string("Not enough memory to transpose matrix\n");
        }
        else if (info < 0) {
            return std::string("Wrong parameter number " + std::to_string(-info));
        }
        else {
            return std::string("Runtime error\n");
        }
    }

public:
    explicit armpl_lapacke_error(std::string func, std::int64_t result)
            : std::runtime_error("Arm Performance Libraries backend: LAPACKE error in " + func +
                                 ": " + std::string(lapacke_error_message(result))) {}

    virtual ~armpl_lapacke_error() throw() {}
};

} // namespace armpl
} // namespace lapack
} // namespace math
} // namespace oneapi

#endif //_ARMPL_COMMON_HPP_
