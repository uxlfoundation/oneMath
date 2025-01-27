/*******************************************************************************
* Copyright 2025 SiPearl
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

#include <sycl/sycl.hpp>

#include "oneapi/math/exceptions.hpp"
#include "armpl_common.hpp"
#include "oneapi/math/blas/detail/armpl/onemath_blas_armpl.hpp"

namespace oneapi {
namespace math {
namespace blas {
namespace armpl {
namespace column_major {

#define COLUMN_MAJOR
#define MAJOR            CblasColMajor
#define MAJOR_MINOR_TEXT "for column major layout"
#include "armpl_level3.cxx"
#undef MAJOR_MINOR_TEXT
#undef COLUMN_MAJOR
#undef MAJOR

} // namespace column_major
namespace row_major {

#define ROW_MAJOR
#define MAJOR            CblasRowMajor
#define MAJOR_MINOR_TEXT "for row major layout"
#include "armpl_level3.cxx"
#undef MAJOR_MINOR_TEXT
#undef ROW_MAJOR
#undef MAJOR

} // namespace row_major
} // namespace armpl
} // namespace blas
} // namespace math
} // namespace oneapi
