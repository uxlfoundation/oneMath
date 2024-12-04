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

#ifndef _ONEMATH_BLAS_HPP_
#define _ONEMATH_BLAS_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <complex>
#include <cstdint>

#include "oneapi/math/detail/config.hpp"
#include "oneapi/math/types.hpp"

#include "oneapi/math/detail/get_device_id.hpp"

#include "oneapi/math/blas/detail/blas_loader.hpp"
#ifdef ONEMATH_ENABLE_CUBLAS_BACKEND
#include "oneapi/math/blas/detail/cublas/blas_ct.hpp"
#endif
#ifdef ONEMATH_ENABLE_ROCBLAS_BACKEND
#include "oneapi/math/blas/detail/rocblas/blas_ct.hpp"
#endif
#ifdef ONEMATH_ENABLE_MKLCPU_BACKEND
#include "oneapi/math/blas/detail/mklcpu/blas_ct.hpp"
#endif
#ifdef ONEMATH_ENABLE_MKLGPU_BACKEND
#include "oneapi/math/blas/detail/mklgpu/blas_ct.hpp"
#endif
#ifdef ONEMATH_ENABLE_NETLIB_BACKEND
#include "oneapi/math/blas/detail/netlib/blas_ct.hpp"
#endif
#ifdef ONEMATH_ENABLE_PORTBLAS_BACKEND
#include "oneapi/math/blas/detail/portblas/blas_ct.hpp"
#endif

namespace oneapi {
namespace math {
namespace blas {
namespace column_major {

#include "blas.hxx"

} //namespace column_major
namespace row_major {

#include "blas.hxx"

} //namespace row_major
} //namespace blas
} //namespace math
} //namespace oneapi

#endif //_ONEMATH_BLAS_LOADER_HPP_
