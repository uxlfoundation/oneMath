/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef ONEMKL_BLAS_HELPER_HPP
#define ONEMKL_BLAS_HELPER_HPP

#include "cblas.h"

#include "oneapi/mkl/types.hpp"

typedef enum { CblasFixOffset = 101, CblasColOffset = 102, CblasRowOffset = 103 } CBLAS_OFFSET;

/**
 * Helper methods for converting between onemkl types and their BLAS
 * equivalents.
 */

inline CBLAS_TRANSPOSE convert_to_cblas_trans(oneapi::mkl::transpose trans) {
    if (trans == oneapi::mkl::transpose::trans)
        return CBLAS_TRANSPOSE::CblasTrans;
    else if (trans == oneapi::mkl::transpose::conjtrans)
        return CBLAS_TRANSPOSE::CblasConjTrans;
    else
        return CBLAS_TRANSPOSE::CblasNoTrans;
}

inline CBLAS_UPLO convert_to_cblas_uplo(oneapi::mkl::uplo is_upper) {
    return is_upper == oneapi::mkl::uplo::upper ? CBLAS_UPLO::CblasUpper : CBLAS_UPLO::CblasLower;
}

inline CBLAS_DIAG convert_to_cblas_diag(oneapi::mkl::diag is_unit) {
    return is_unit == oneapi::mkl::diag::unit ? CBLAS_DIAG::CblasUnit : CBLAS_DIAG::CblasNonUnit;
}

inline CBLAS_SIDE convert_to_cblas_side(oneapi::mkl::side is_left) {
    return is_left == oneapi::mkl::side::left ? CBLAS_SIDE::CblasLeft : CBLAS_SIDE::CblasRight;
}

inline CBLAS_OFFSET convert_to_cblas_offset(oneapi::mkl::offset offsetc) {
    if (offsetc == oneapi::mkl::offset::fix)
        return CBLAS_OFFSET::CblasFixOffset;
    else if (offsetc == oneapi::mkl::offset::column)
        return CBLAS_OFFSET::CblasColOffset;
    else
        return CBLAS_OFFSET::CblasRowOffset;
}

#endif // ONEMKL_BLAS_HELPER_HPP
