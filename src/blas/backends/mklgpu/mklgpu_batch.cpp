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

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/blas/detail/mklgpu/onemath_blas_mklgpu.hpp"

#include "oneapi/math/exceptions.hpp"
#include "../mkl_common/mkl_blas_backend.hpp"

namespace oneapi {
namespace math {
namespace blas {
namespace mklgpu {
namespace column_major {

namespace blas_major = ::oneapi::mkl::blas::column_major;
#include "../mkl_common/mkl_batch.cxx"

} // namespace column_major
namespace row_major {

namespace blas_major = ::oneapi::mkl::blas::row_major;
#include "../mkl_common/mkl_batch.cxx"

} // namespace row_major
} // namespace mklgpu
} // namespace blas
} // namespace math
} // namespace oneapi
