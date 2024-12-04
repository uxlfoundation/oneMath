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
    oneapi::math::transpose trans;
    fp alpha;
    int64_t i, tmp;

    m = 1 + std::rand() % 50;
    n = 1 + std::rand() % 50;
    lda = std::max(m, n);
    ldb = std::max(m, n);
    alpha = rand_scalar<fp>();
    trans = rand_trans<fp>();

    int64_t size_a, size_b, size;
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
    size = std::max(size_a, size_b);

    vector<fp, allocator_helper<fp, 64>> AB(size), AB_ref(size);

    rand_matrix(AB, oneapi::math::layout::col_major, oneapi::math::transpose::nontrans, size, 1,
                size);
    copy_matrix(AB, oneapi::math::layout::col_major, oneapi::math::transpose::nontrans, size, 1,
                size, AB_ref);

    // Call reference IMATCOPY.
    int m_ref = (int)m;
    int n_ref = (int)n;
    int lda_ref = (int)lda;
    int ldb_ref = (int)ldb;
    imatcopy_ref(layout, trans, m_ref, n_ref, alpha, AB_ref.data(), lda_ref, ldb_ref);

    // Call DPC++ IMATCOPY

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during IMATCOPY:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<fp, 1> AB_buffer(AB.data(), range<1>(AB.size()));

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::math::layout::col_major:
                oneapi::math::blas::column_major::imatcopy(main_queue, trans, m, n, alpha,
                                                           AB_buffer, lda, ldb);
                break;
            case oneapi::math::layout::row_major:
                oneapi::math::blas::row_major::imatcopy(main_queue, trans, m, n, alpha, AB_buffer,
                                                        lda, ldb);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::imatcopy,
                                        trans, m, n, alpha, AB_buffer, lda, ldb);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::imatcopy, trans,
                                        m, n, alpha, AB_buffer, lda, ldb);
                break;
            default: break;
        }
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during IMATCOPY:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of IMATCOPY:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    auto AB_accessor = AB_buffer.get_host_access(read_only);
    bool good = check_equal_matrix(AB_accessor, AB_ref, oneapi::math::layout::col_major, size, 1,
                                   size, 10, std::cout);

    return (int)good;
}

class ImatcopyTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {};

TEST_P(ImatcopyTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

TEST_P(ImatcopyTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

TEST_P(ImatcopyTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

TEST_P(ImatcopyTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(ImatcopyTestSuite, ImatcopyTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
