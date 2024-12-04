/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Copyright 2022 Intel Corporation
*
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

#ifndef _ONEMATH_LAPACK_ROCSOLVER_HPP_
#define _ONEMATH_LAPACK_ROCSOLVER_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <complex>
#include <cstdint>
#include <string>

#include "oneapi/math/types.hpp"
#include "oneapi/math/detail/export.hpp"

namespace oneapi {
namespace math {
namespace lapack {
namespace rocsolver {

#include "onemath_lapack_rocsolver.hxx"

} // namespace rocsolver
} // namespace lapack
} // namespace math
} // namespace oneapi

#endif //_ONEMATH_LAPACK_ROCSOLVER_HPP_
