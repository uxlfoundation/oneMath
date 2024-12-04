/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "allocator_helper.hpp"
#include "cblas.h"
#include "oneapi/math/detail/config.hpp"
#include "oneapi/math.hpp"
#include "onemath_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

using namespace sycl;
using std::vector;

extern std::vector<sycl::device*> devices;

namespace {

template <typename fp>
int test(device* dev, oneapi::math::layout layout) {
    // Prepare data.
    int64_t m, n;
    int64_t lda, ldb;
    int64_t stride_a, stride_b;
    oneapi::math::transpose trans;
    fp alpha;

    stride_a = 1 + std::rand() % 50;
    stride_b = 1 + std::rand() % 50;
    m = 1 + std::rand() % 50;
    n = 1 + std::rand() % 50;
    lda = stride_a * (std::max(m, n) - 1) + 1;
    ldb = stride_b * (std::max(m, n) - 1) + 1;
    alpha = rand_scalar<fp>();
    trans = rand_trans<fp>();

    int64_t size_a, size_b;

    switch (layout) {
        case oneapi::math::layout::col_major:
            size_a = lda * n;
            size_b = (trans == oneapi::math::transpose::nontrans) ? ldb * n : ldb * m;
            break;
        case oneapi::math::layout::row_major:
            size_a = lda * m;
            size_b = (trans == oneapi::math::transpose::nontrans) ? ldb * m : ldb * n;
            break;
        default: break;
    }

    vector<fp, allocator_helper<fp, 64>> A(size_a), B(size_b), B_ref(size_b);

    rand_matrix(A.data(), layout, oneapi::math::transpose::nontrans, m, n, lda);
    rand_matrix(B.data(), layout, trans, m, n, ldb);
    copy_matrix(B.data(), oneapi::math::layout::col_major, oneapi::math::transpose::nontrans,
                size_b, 1, size_b, B_ref.data());

    // Call reference OMATCOPY2.
    int64_t m_ref = m;
    int64_t n_ref = n;
    int64_t lda_ref = lda;
    int64_t ldb_ref = ldb;
    int64_t stride_a_ref = stride_a;
    int64_t stride_b_ref = stride_b;
    omatcopy2_ref(layout, trans, m_ref, n_ref, alpha, A.data(), lda_ref, stride_a_ref, B_ref.data(),
                  ldb_ref, stride_b_ref);

    // Call DPC++ OMATCOPY2

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during OMATCOPY2:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<fp, 1> A_buffer(A.data(), range<1>(A.size()));
    buffer<fp, 1> B_buffer(B.data(), range<1>(B.size()));

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::math::layout::col_major:
                oneapi::math::blas::column_major::omatcopy2(main_queue, trans, m, n, alpha,
                                                            A_buffer, lda, stride_a, B_buffer, ldb,
                                                            stride_b);
                break;
            case oneapi::math::layout::row_major:
                oneapi::math::blas::row_major::omatcopy2(main_queue, trans, m, n, alpha, A_buffer,
                                                         lda, stride_a, B_buffer, ldb, stride_b);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::omatcopy2,
                                        trans, m, n, alpha, A_buffer, lda, stride_a, B_buffer, ldb,
                                        stride_b);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::omatcopy2, trans,
                                        m, n, alpha, A_buffer, lda, stride_a, B_buffer, ldb,
                                        stride_b);
                break;
            default: break;
        }
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during OMATCOPY2:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of OMATCOPY2:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    auto B_accessor = B_buffer.get_host_access(read_only);
    bool good = check_equal_matrix(B_accessor, B_ref, oneapi::math::layout::col_major, size_b, 1,
                                   size_b, 10, std::cout);

    return (int)good;
}

class Omatcopy2Tests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {};

TEST_P(Omatcopy2Tests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

TEST_P(Omatcopy2Tests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

TEST_P(Omatcopy2Tests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

TEST_P(Omatcopy2Tests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Omatcopy2TestSuite, Omatcopy2Tests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
