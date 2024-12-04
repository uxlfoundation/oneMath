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

#ifndef _ONEMATH_SPARSE_BLAS_DETAIL_HANDLES_HPP_
#define _ONEMATH_SPARSE_BLAS_DETAIL_HANDLES_HPP_

namespace oneapi::math::sparse {

// Each backend can create its own handle type or re-use the native handle types that will be reinterpret_cast'ed to the types below

struct dense_matrix_handle;
using dense_matrix_handle_t = dense_matrix_handle*;

struct dense_vector_handle;
using dense_vector_handle_t = dense_vector_handle*;

struct matrix_handle;
using matrix_handle_t = matrix_handle*;

} // namespace oneapi::math::sparse

#endif // _ONEMATH_SPARSE_BLAS_DETAIL_HANDLES_HPP_
