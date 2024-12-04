/*******************************************************************************
* Copyright 2022 Intel Corporation
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
#include <type_traits>

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
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during OMATCOPY:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.
    int64_t m, n;
    int64_t lda, ldb;
    oneapi::math::transpose trans;
    fp alpha;
    int64_t i, tmp;

    m = 1 + std::rand() % 50;
    n = 1 + std::rand() % 50;
    lda = std::max(m, n);
    ldb = std::max(m, n);
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

    auto ua = usm_allocator<fp, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp, decltype(ua)> A(ua), B(ua), B_ref(ua);

    A.resize(size_a);
    B.resize(size_b);
    B_ref.resize(size_b);

    rand_matrix(A, oneapi::math::layout::col_major, oneapi::math::transpose::nontrans, size_a, 1,
                size_a);
    rand_matrix(B, oneapi::math::layout::col_major, oneapi::math::transpose::nontrans, size_b, 1,
                size_b);
    copy_matrix(B, oneapi::math::layout::col_major, oneapi::math::transpose::nontrans, size_b, 1,
                size_b, B_ref);

    // Call reference OMATCOPY.
    int m_ref = (int)m;
    int n_ref = (int)n;
    int lda_ref = (int)lda;
    int ldb_ref = (int)ldb;
    omatcopy_ref(layout, trans, m_ref, n_ref, alpha, A.data(), lda_ref, B_ref.data(), ldb_ref);

    // Call DPC++ OMATCOPY
    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::math::layout::col_major:
                done = oneapi::math::blas::column_major::omatcopy(
                    main_queue, trans, m, n, alpha, &A[0], lda, &B[0], ldb, dependencies);
                break;
            case oneapi::math::layout::row_major:
                done = oneapi::math::blas::row_major::omatcopy(
                    main_queue, trans, m, n, alpha, &A[0], lda, &B[0], ldb, dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::omatcopy,
                                        trans, m, n, alpha, &A[0], lda, &B[0], ldb, dependencies);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::omatcopy, trans,
                                        m, n, alpha, &A[0], lda, &B[0], ldb, dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during OMATCOPY:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of OMATCOPY:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good = check_equal_matrix(B, B_ref, oneapi::math::layout::col_major, size_b, 1, size_b, 10,
                                   std::cout);

    return (int)good;
}

class OmatcopyUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {};

TEST_P(OmatcopyUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

TEST_P(OmatcopyUsmTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

TEST_P(OmatcopyUsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

TEST_P(OmatcopyUsmTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(OmatcopyUsmTestSuite, OmatcopyUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
