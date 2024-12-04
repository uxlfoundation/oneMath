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

#ifndef _ONEMATH_SPARSE_BLAS_HPP_
#define _ONEMATH_SPARSE_BLAS_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/detail/config.hpp"

#ifdef ONEMATH_ENABLE_MKLCPU_BACKEND
#include "sparse_blas/detail/mklcpu/sparse_blas_ct.hpp"
#endif
#ifdef ONEMATH_ENABLE_MKLGPU_BACKEND
#include "sparse_blas/detail/mklgpu/sparse_blas_ct.hpp"
#endif
#ifdef ONEMATH_ENABLE_CUSPARSE_BACKEND
#include "sparse_blas/detail/cusparse/sparse_blas_ct.hpp"
#endif
#ifdef ONEMATH_ENABLE_ROCSPARSE_BACKEND
#include "sparse_blas/detail/rocsparse/sparse_blas_ct.hpp"
#endif

#include "sparse_blas/detail/sparse_blas_rt.hpp"

#endif // _ONEMATH_SPARSE_BLAS_HPP_
