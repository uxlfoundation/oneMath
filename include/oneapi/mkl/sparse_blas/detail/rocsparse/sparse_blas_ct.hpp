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

#ifndef _ONEMKL_SPARSE_BLAS_DETAIL_ROCSPARSE_SPARSE_BLAS_CT_HPP_
#define _ONEMKL_SPARSE_BLAS_DETAIL_ROCSPARSE_SPARSE_BLAS_CT_HPP_

#include "oneapi/mkl/detail/backends.hpp"
#include "oneapi/mkl/detail/backend_selector.hpp"

#include "onemkl_sparse_blas_rocsparse.hpp"

namespace oneapi {
namespace mkl {
namespace sparse {

#define BACKEND rocsparse
#include "oneapi/mkl/sparse_blas/detail/sparse_blas_ct.hxx"
#undef BACKEND

} //namespace sparse
} //namespace mkl
} //namespace oneapi

#endif // _ONEMKL_SPARSE_BLAS_DETAIL_ROCSPARSE_SPARSE_BLAS_CT_HPP_
