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

#ifndef _DETAIL_MKLGPU_BLAS_CT_HPP__
#define _DETAIL_MKLGPU_BLAS_CT_HPP__

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <complex>
#include <cstdint>

#include "oneapi/math/types.hpp"
#include "oneapi/math/detail/backends.hpp"

#include "oneapi/math/blas/detail/blas_ct_backends.hpp"
#include "oneapi/math/blas/detail/mklgpu/onemath_blas_mklgpu.hpp"

namespace oneapi {
namespace math {
namespace blas {
namespace column_major {

#define MAJOR column_major
#include "blas_ct.hxx"
#undef MAJOR

} //namespace column_major
namespace row_major {

#define MAJOR row_major
#include "blas_ct.hxx"
#undef MAJOR

} //namespace row_major
} //namespace blas
} //namespace math
} //namespace oneapi

#endif //_DETAIL_MKLGPU_BLAS_CT_HPP_
