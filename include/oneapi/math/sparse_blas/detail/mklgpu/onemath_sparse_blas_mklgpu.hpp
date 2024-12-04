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

#ifndef _ONEMATH_SPARSE_BLAS_DETAIL_MKLGPU_ONEMATH_SPARSE_BLAS_MKLGPU_HPP_
#define _ONEMATH_SPARSE_BLAS_DETAIL_MKLGPU_ONEMATH_SPARSE_BLAS_MKLGPU_HPP_

#include "oneapi/math/detail/export.hpp"
#include "oneapi/math/sparse_blas/detail/helper_types.hpp"
#include "oneapi/math/sparse_blas/types.hpp"

namespace oneapi::math::sparse::mklgpu {

namespace detail = oneapi::math::sparse::detail;

#include "oneapi/math/sparse_blas/detail/onemath_sparse_blas_backends.hxx"

} // namespace oneapi::math::sparse::mklgpu

#endif // _ONEMATH_SPARSE_BLAS_DETAIL_MKLGPU_ONEMATH_SPARSE_BLAS_MKLGPU_HPP_
