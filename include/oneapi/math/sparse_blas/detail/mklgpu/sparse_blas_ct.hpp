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

#ifndef _ONEMATH_SPARSE_BLAS_DETAIL_MKLGPU_SPARSE_BLAS_CT_HPP_
#define _ONEMATH_SPARSE_BLAS_DETAIL_MKLGPU_SPARSE_BLAS_CT_HPP_

#include "oneapi/math/detail/backends.hpp"
#include "oneapi/math/detail/backend_selector.hpp"

#include "onemath_sparse_blas_mklgpu.hpp"

namespace oneapi {
namespace math {
namespace sparse {

#define BACKEND mklgpu
#include "oneapi/math/sparse_blas/detail/sparse_blas_ct.hxx"
#undef BACKEND

} //namespace sparse
} //namespace math
} //namespace oneapi

#endif // _ONEMATH_SPARSE_BLAS_DETAIL_MKLGPU_SPARSE_BLAS_CT_HPP_
