/*******************************************************************************
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

#ifndef ONEMATH_BLAS_HELPER_HPP
#define ONEMATH_BLAS_HELPER_HPP

#include "cblas.h"

#include "oneapi/math/types.hpp"

typedef enum { CblasRowOffset = 101, CblasColOffset = 102, CblasFixOffset = 103 } CBLAS_OFFSET;

/**
 * Helper methods for converting between onemath types and their CBLAS
 * equivalents.
 */

inline CBLAS_TRANSPOSE convert_to_cblas_trans(oneapi::math::transpose trans) {
    if (trans == oneapi::math::transpose::trans)
        return CBLAS_TRANSPOSE::CblasTrans;
    else if (trans == oneapi::math::transpose::conjtrans)
        return CBLAS_TRANSPOSE::CblasConjTrans;
    else
        return CBLAS_TRANSPOSE::CblasNoTrans;
}

inline CBLAS_UPLO convert_to_cblas_uplo(oneapi::math::uplo is_upper) {
    return is_upper == oneapi::math::uplo::upper ? CBLAS_UPLO::CblasUpper : CBLAS_UPLO::CblasLower;
}

inline CBLAS_DIAG convert_to_cblas_diag(oneapi::math::diag is_unit) {
    return is_unit == oneapi::math::diag::unit ? CBLAS_DIAG::CblasUnit : CBLAS_DIAG::CblasNonUnit;
}

inline CBLAS_SIDE convert_to_cblas_side(oneapi::math::side is_left) {
    return is_left == oneapi::math::side::left ? CBLAS_SIDE::CblasLeft : CBLAS_SIDE::CblasRight;
}

inline CBLAS_OFFSET convert_to_cblas_offset(oneapi::math::offset offsetc) {
    if (offsetc == oneapi::math::offset::fix)
        return CBLAS_OFFSET::CblasFixOffset;
    else if (offsetc == oneapi::math::offset::column)
        return CBLAS_OFFSET::CblasColOffset;
    else
        return CBLAS_OFFSET::CblasRowOffset;
}

inline CBLAS_LAYOUT convert_to_cblas_layout(oneapi::math::layout is_column) {
    return is_column == oneapi::math::layout::col_major ? CBLAS_LAYOUT::CblasColMajor
                                                        : CBLAS_LAYOUT::CblasRowMajor;
}

static const CBLAS_TRANSPOSE fcblastrans[] = { CblasNoTrans, CblasTrans, CblasConjTrans };

static const CBLAS_UPLO fcblasuplo[] = { CblasUpper, CblasLower };

static const CBLAS_SIDE fcblasside[] = { CblasLeft, CblasRight };

static const CBLAS_DIAG fcblasdiag[] = { CblasNonUnit, CblasUnit };

static const CBLAS_TRANSPOSE fcblastrans_r[] = { CblasTrans, CblasNoTrans, CblasNoTrans };

static const CBLAS_TRANSPOSE fcblastrans_r2[] = { CblasTrans, CblasNoTrans, CblasConjTrans };

static const CBLAS_TRANSPOSE fcblastrans_c[] = { CblasConjTrans, CblasNoTrans, CblasNoTrans };

static const CBLAS_OFFSET fcblasoffset[] = { CblasColOffset, CblasRowOffset, CblasFixOffset };

#endif // ONEMATH_BLAS_HELPER_HPP
