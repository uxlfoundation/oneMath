/*******************************************************************************
* Copyright Codeplay Software
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

#include "portblas_common.hpp"
#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/blas/detail/portblas/onemath_blas_portblas.hpp"

namespace oneapi {
namespace math {
namespace blas {
namespace portblas {

using real_t = double;

namespace column_major {

#define COLUMN_MAJOR
constexpr bool is_column_major() {
    return true;
}
#include "portblas_level2.cxx"
#undef COLUMN_MAJOR

} // namespace column_major
namespace row_major {

#define ROW_MAJOR
constexpr bool is_column_major() {
    return false;
}
#include "portblas_level2.cxx"
#undef ROW_MAJOR

} // namespace row_major
} // namespace portblas
} // namespace blas
} // namespace math
} // namespace oneapi
