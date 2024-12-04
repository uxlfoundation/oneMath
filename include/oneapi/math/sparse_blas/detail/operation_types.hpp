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

#ifndef _ONEMATH_SPARSE_BLAS_DETAIL_OPERATION_TYPES_HPP_
#define _ONEMATH_SPARSE_BLAS_DETAIL_OPERATION_TYPES_HPP_

namespace oneapi::math::sparse {

// Each backend can create its own descriptor type or re-use the native descriptor types that will be reinterpret_cast'ed to the types below

struct spmm_descr;
using spmm_descr_t = spmm_descr*;

struct spmv_descr;
using spmv_descr_t = spmv_descr*;

struct spsv_descr;
using spsv_descr_t = spsv_descr*;

} // namespace oneapi::math::sparse

#endif // _ONEMATH_SPARSE_BLAS_DETAIL_OPERATION_TYPES_HPP_
