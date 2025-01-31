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
#include <string>

#include "armpl.h"

#include "oneapi/math/blas/detail/armpl/onemath_blas_armpl.hpp"
#include "oneapi/math/types.hpp"
#include "oneapi/math/exceptions.hpp"
#include "runtime_support_helper.hpp"

#define GET_MULTI_PTR template get_multi_ptr<sycl::access::decorated::yes>().get_raw()

typedef enum { CblasRowOffset = 171, CblasColOffset = 172, CblasFixOffset = 173 } CBLAS_OFFSET;

namespace oneapi {
namespace math {
namespace blas {
namespace armpl {

inline int check_armpl_version(armpl_int_t major_req, armpl_int_t minor_req, armpl_int_t build_req,
                               const char* message) {
    armpl_int_t major, minor, build;
    char* tag;
    armplversion(&major, &minor, &build, (const char**)&tag);
    if (major > major_req) {
        return 0;
    }
    else if (major == major_req && minor > minor_req) {
        return 0;
    }
    else if (major == major_req && minor == minor_req && build >= build_req) {
        return 0;
    }
    throw oneapi::math::unimplemented("blas", "version support", message);
}

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

// Conversion functions to traditional Fortran characters.
inline const char fortran_char(transpose t) {
    switch (t) {
        case transpose::trans: return 'T';
        case transpose::conjtrans: return 'C';
        case transpose::nontrans: return 'N';
    }
}

inline const char fortran_char(offset o) {
    switch (o) {
        case offset::row: return 'R';
        case offset::column: return 'C';
        case offset::fix: return 'F';
    }
}

inline const char fortran_char(uplo u) {
    switch (u) {
        case uplo::lower: return 'L';
        case uplo::upper: return 'U';
    }
}

inline const char fortran_char(diag d) {
    switch (d) {
        case diag::unit: return 'U';
        case diag::nonunit: return 'N';
    }
}

inline const char fortran_char(side s) {
    switch (s) {
        case side::right: return 'R';
        case side::left: return 'L';
    }
}

// Conversion functions to CBLAS enums.
inline CBLAS_TRANSPOSE cblas_convert(transpose t) {
    switch (t) {
        case transpose::trans: return CblasTrans;
        case transpose::conjtrans: return CblasConjTrans;
        case transpose::nontrans: return CblasNoTrans;
    }
}

inline CBLAS_UPLO cblas_convert(uplo u) {
    switch (u) {
        case uplo::lower: return CblasLower;
        case uplo::upper: return CblasUpper;
    }
}

inline CBLAS_DIAG cblas_convert(diag d) {
    switch (d) {
        case diag::unit: return CblasUnit;
        case diag::nonunit: return CblasNonUnit;
    }
}

inline CBLAS_SIDE cblas_convert(side s) {
    switch (s) {
        case side::right: return CblasRight;
        case side::left: return CblasLeft;
    }
}

inline CBLAS_OFFSET cblas_convert(oneapi::math::offset o) {
    switch (o) {
        case oneapi::math::offset::fix: return CblasFixOffset;
        case oneapi::math::offset::column: return CblasColOffset;
        case oneapi::math::offset::row: return CblasRowOffset;
    }
}

template <typename offset_type>
inline offset offset_convert(offset_type off_kind) {
    return offset::F;
}

template <>
inline offset offset_convert(CBLAS_OFFSET off_kind) {
    switch (off_kind) {
        case CblasFixOffset: return offset::F;
        case CblasColOffset: return offset::C;
        case CblasRowOffset: return offset::R;
    }
}

template <typename transpose_type>
inline bool isNonTranspose(transpose_type trans) {
    throw unimplemented("armpl", "isnontranspoe", "");
    return false;
}

template <>
inline bool isNonTranspose(CBLAS_TRANSPOSE trans) {
    return trans == CblasNoTrans;
}

template <>
inline bool isNonTranspose(transpose trans) {
    return trans == transpose::nontrans;
}

template <typename T_src, typename T_dest, typename transpose_type>
static inline void copy_mat(T_src& src, CBLAS_LAYOUT layout, transpose_type trans, int64_t row,
                            int64_t col, int64_t ld, T_dest off, T_dest*& dest) {
    int64_t Jend = isNonTranspose(trans) ? col : row;
    int64_t Iend = isNonTranspose(trans) ? row : col;
    if (layout == CblasRowMajor)
        std::swap(Jend, Iend);
    for (size_t j = 0; j < Jend; ++j) {
        for (size_t i = 0; i < Iend; ++i) {
            dest[i + ld * j] = (T_dest)src[i + ld * j] - off;
        }
    }
}

template <typename T_src, typename T_dest, typename T_off, typename offset_type>
static inline void copy_mat(T_src* src, CBLAS_LAYOUT layout, int64_t row, int64_t col, int64_t ld,
                            offset_type off_kind, T_off off, T_dest* dest) {
    using T_data = typename std::remove_reference<decltype(dest[0])>::type;
    T_data tmp;

    int64_t Jend = (layout == CblasColMajor) ? col : row;
    int64_t Iend = (layout == CblasColMajor) ? row : col;

    if (offset_convert(off_kind) == offset::F) {
        tmp = off[0];
        for (size_t j = 0; j < Jend; ++j) {
            for (size_t i = 0; i < Iend; ++i) {
                dest[i + ld * j] = tmp + (T_data)src[i + ld * j];
            }
        }
    }
    else if (((offset_convert(off_kind) == offset::C) && (layout == CblasColMajor)) ||
             ((offset_convert(off_kind) == offset::R) && (layout == CblasRowMajor))) {
        for (size_t j = 0; j < Jend; ++j) {
            for (size_t i = 0; i < Iend; ++i) {
                tmp = off[i];
                dest[i + ld * j] = tmp + (T_data)src[i + ld * j];
            }
        }
    }
    else {
        for (size_t j = 0; j < Jend; ++j) {
            tmp = off[j];
            for (size_t i = 0; i < Iend; ++i) {
                dest[i + ld * j] = tmp + (T_data)src[i + ld * j];
            }
        }
    }
}

inline offset column_to_row(offset o) {
    return (o == offset::C) ? offset::R : (o == offset::R) ? offset::C : offset::F;
}

static inline bool is_int8(int v) {
    return (v >= -128) && (v < 128);
}

template <typename T>
constexpr bool is_complex = false;
template <typename T>
constexpr bool is_complex<std::complex<T>> = true;

template <typename T>
constexpr auto cast_to_void_if_complex(const T& alpha) {
    if constexpr (is_complex<T>) {
        return static_cast<const void*>(&alpha);
    }
    else {
        return alpha;
    }
}

template <typename T>
struct ArmPLEquivalentType {
    using Type = T;
};

template <>
struct ArmPLEquivalentType<std::complex<float>> {
    using Type = armpl_singlecomplex_t;
};
template <>
struct ArmPLEquivalentType<std::complex<double>> {
    using Type = armpl_doublecomplex_t;
};

// Help function to handle the error code of the interleaving batch function
inline void handle_interleave_errors(armpl_status_t& info, std::string s) {
    if (info != ARMPL_STATUS_SUCCESS) {
        std::string str_info_exception;
        if (info == ARMPL_STATUS_INPUT_PARAMETER_ERROR) {
            str_info_exception = "ARMPL_STATUS_INPUT_PARAMETER_ERROR";
        }
        else if (info == ARMPL_STATUS_EXECUTION_FAILURE) {
            str_info_exception = "ARMPL_STATUS_EXECUTION_FAILURE";
        }
        else {
            str_info_exception = "UNKNOWN";
        }
        throw oneapi::math::batch_error("blas", s, str_info_exception);
    }
}

} // namespace armpl
} // namespace blas
} // namespace math
} // namespace oneapi

#endif //_ARMPL_COMMON_HPP_
