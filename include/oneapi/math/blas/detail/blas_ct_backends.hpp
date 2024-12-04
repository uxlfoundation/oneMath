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

#ifndef _BLAS_CT_BACKENDS_HPP__
#define _BLAS_CT_BACKENDS_HPP__

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <complex>
#include <cstdint>

#include "oneapi/math/types.hpp"
#include "oneapi/math/detail/backend_selector.hpp"

namespace oneapi {
namespace math {
namespace blas {
namespace column_major {

#define BACKEND mklcpu
#include "blas_ct_backends.hxx"
#undef BACKEND
#define BACKEND mklgpu
#include "blas_ct_backends.hxx"
#undef BACKEND
#define BACKEND cublas
#include "blas_ct_backends.hxx"
#undef BACKEND
#define BACKEND rocblas
#include "blas_ct_backends.hxx"
#undef BACKEND
#define BACKEND netlib
#include "blas_ct_backends.hxx"
#undef BACKEND
#define BACKEND portblas
#include "blas_ct_backends.hxx"
#undef BACKEND

} //namespace column_major
namespace row_major {

#define BACKEND mklcpu
#include "blas_ct_backends.hxx"
#undef BACKEND
#define BACKEND mklgpu
#include "blas_ct_backends.hxx"
#undef BACKEND
#define BACKEND cublas
#include "blas_ct_backends.hxx"
#undef BACKEND
#define BACKEND rocblas
#include "blas_ct_backends.hxx"
#undef BACKEND
#define BACKEND netlib
#include "blas_ct_backends.hxx"
#undef BACKEND
#define BACKEND portblas
#include "blas_ct_backends.hxx"
#undef BACKEND

} //namespace row_major
} //namespace blas
} //namespace math
} //namespace oneapi

#endif //_BLAS_CT_BACKENDS_HPP__
