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

#include "generic_common.hpp"
#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/blas/detail/generic/onemath_blas_generic.hpp"

namespace oneapi {
namespace math {
namespace blas {
namespace generic {

using real_t = float;

namespace column_major {

#define COLUMN_MAJOR
constexpr bool is_column_major() {
    return true;
}
#include "generic_level1.cxx"

void dot(sycl::queue&, std::int64_t, sycl::buffer<sycl::half, 1>&, std::int64_t,
         sycl::buffer<sycl::half, 1>&, std::int64_t, sycl::buffer<sycl::half, 1>&) {
    throw unimplemented("blas", "dot", "for sycl::half");
}

void dot(sycl::queue&, std::int64_t, sycl::buffer<bfloat16, 1>&, std::int64_t,
         sycl::buffer<bfloat16, 1>&, std::int64_t, sycl::buffer<bfloat16, 1>&) {
    throw unimplemented("blas", "dot", "for bfloat16");
}

sycl::event dot(sycl::queue&, std::int64_t, const sycl::half*, std::int64_t, const sycl::half*,
                std::int64_t, sycl::half*, const std::vector<sycl::event>&) {
    throw unimplemented("blas", "dot", "for sycl::half");
}

sycl::event dot(sycl::queue&, std::int64_t, const bfloat16*, std::int64_t, const bfloat16*,
                std::int64_t, bfloat16*, const std::vector<sycl::event>&) {
    throw unimplemented("blas", "dot", "for bfloat16");
}
#undef COLUMN_MAJOR

} // namespace column_major
namespace row_major {

#define ROW_MAJOR
constexpr bool is_column_major() {
    return false;
}
#include "generic_level1.cxx"

void dot(sycl::queue&, std::int64_t, sycl::buffer<sycl::half, 1>&, std::int64_t,
         sycl::buffer<sycl::half, 1>&, std::int64_t, sycl::buffer<sycl::half, 1>&) {
    throw unimplemented("blas", "dot", "for sycl::half");
}

void dot(sycl::queue&, std::int64_t, sycl::buffer<bfloat16, 1>&, std::int64_t,
         sycl::buffer<bfloat16, 1>&, std::int64_t, sycl::buffer<bfloat16, 1>&) {
    throw unimplemented("blas", "dot", "for bfloat16");
}

sycl::event dot(sycl::queue&, std::int64_t, const sycl::half*, std::int64_t, const sycl::half*,
                std::int64_t, sycl::half*, const std::vector<sycl::event>&) {
    throw unimplemented("blas", "dot", "for sycl::half");
}

sycl::event dot(sycl::queue&, std::int64_t, const bfloat16*, std::int64_t, const bfloat16*,
                std::int64_t, bfloat16*, const std::vector<sycl::event>&) {
    throw unimplemented("blas", "dot", "for bfloat16");
}
#undef ROW_MAJOR

} // namespace row_major
} // namespace generic
} // namespace blas
} // namespace math
} // namespace oneapi
