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

#ifndef _ONEMATH_TYPES_HPP_
#define _ONEMATH_TYPES_HPP_

#ifdef __HIPSYCL__
#include "oneapi/math/bfloat16.hpp"
#endif

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

namespace oneapi {
namespace math {

#ifndef __HIPSYCL__
using bfloat16 = sycl::ext::oneapi::bfloat16;
#endif

// BLAS flag types.
enum class transpose : char { nontrans = 0, trans = 1, conjtrans = 3, N = 0, T = 1, C = 3 };

enum class uplo : char { upper = 0, lower = 1, U = 0, L = 1 };

enum class diag : char { nonunit = 0, unit = 1, N = 0, U = 1 };

enum class side : char { left = 0, right = 1, L = 0, R = 1 };

enum class offset : char { row = 0, column = 1, fix = 2, R = 0, C = 1, F = 2 };

enum class layout : char { row_major = 0, col_major = 1, R = 0, C = 1 };

enum class index_base : char {
    zero = 0,
    one = 1,
};

// LAPACK flag types.
enum class job : char {
    novec = 0,
    vec = 1,
    updatevec = 2,
    allvec = 3,
    somevec = 4,
    overwritevec = 5,
    N = 0,
    V = 1,
    U = 2,
    A = 3,
    S = 4,
    O = 5
};
enum class jobsvd : char {
    novec = 0,
    vectors = 1,
    vectorsina = 2,
    somevec = 3,
    N = 0,
    A = 1,
    O = 2,
    S = 3
};
enum class generate : char { q = 0, p = 1, none = 2, both = 3, Q = 0, P = 1, N = 2, V = 3 };
enum class compz : char {
    novectors = 0,
    vectors = 1,
    initvectors = 2,
    N = 0,
    V = 1,
    I = 2,
};
enum class direct : char {
    forward = 0,
    backward = 1,
    F = 0,
    B = 1,
};
enum class storev : char {
    columnwise = 0,
    rowwise = 1,
    C = 0,
    R = 1,
};
enum class rangev : char {
    all = 0,
    values = 1,
    indices = 2,
    A = 0,
    V = 1,
    I = 2,
};
enum class order : char {
    block = 0,
    entire = 1,
    B = 0,
    E = 1,
};

} //namespace math
} //namespace oneapi

#endif //_ONEMATH_TYPES_HPP_
