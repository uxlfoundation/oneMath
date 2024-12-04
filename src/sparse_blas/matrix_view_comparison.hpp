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

#ifndef _ONEMATH_SRC_SPARSE_BLAS_MATRIX_VIEW_COMPARISON_HPP_
#define _ONEMATH_SRC_SPARSE_BLAS_MATRIX_VIEW_COMPARISON_HPP_

#include "oneapi/math/sparse_blas/matrix_view.hpp"

inline bool operator==(const oneapi::math::sparse::matrix_view& lhs,
                       const oneapi::math::sparse::matrix_view& rhs) {
    return lhs.type_view == rhs.type_view && lhs.uplo_view == rhs.uplo_view &&
           lhs.diag_view == rhs.diag_view;
}

inline bool operator!=(const oneapi::math::sparse::matrix_view& lhs,
                       const oneapi::math::sparse::matrix_view& rhs) {
    return !(lhs == rhs);
}

#endif // _ONEMATH_SRC_SPARSE_BLAS_MATRIX_VIEW_COMPARISON_HPP_