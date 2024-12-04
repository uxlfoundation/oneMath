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

#ifndef _ONEMATH_SPARSE_BLAS_TYPES_HPP_
#define _ONEMATH_SPARSE_BLAS_TYPES_HPP_

#include "oneapi/math/types.hpp"
#include "matrix_view.hpp"
#include "detail/handles.hpp"
#include "detail/operation_types.hpp"

/**
 * @file Include and define the sparse types that are common between Intel(R) oneMKL API and oneMath API.
*/

namespace oneapi {
namespace math {
namespace sparse {

enum class matrix_property {
    symmetric,
    sorted,
    sorted_by_rows,
};

enum class spmm_alg {
    default_alg,
    no_optimize_alg,
    coo_alg1,
    coo_alg2,
    coo_alg3,
    coo_alg4,
    csr_alg1,
    csr_alg2,
    csr_alg3,
};

enum class spmv_alg {
    default_alg,
    no_optimize_alg,
    coo_alg1,
    coo_alg2,
    csr_alg1,
    csr_alg2,
    csr_alg3,
};

enum class spsv_alg {
    default_alg,
    no_optimize_alg,
};

} // namespace sparse
} // namespace math
} // namespace oneapi

#endif // _ONEMATH_SPARSE_BLAS_TYPES_HPP_
