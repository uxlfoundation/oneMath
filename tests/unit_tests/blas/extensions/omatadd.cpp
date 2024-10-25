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
#include "oneapi/mkl/detail/config.hpp"
#include "oneapi/mkl.hpp"
#include "onemkl_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

using namespace sycl;
using std::vector;

extern std::vector<sycl::device*> devices;

namespace {

template <typename fp>
int test(device* dev, oneapi::mkl::layout layout) {
    // Prepare data.
    int64_t m, n;
    int64_t lda, ldb, ldc;
    oneapi::mkl::transpose transa, transb;
    fp alpha, beta;
    int64_t i, tmp;

    m = 1 + std::rand() % 50;
    n = 1 + std::rand() % 50;
    lda = std::max(m, n);
    ldb = std::max(m, n);
    ldc = std::max(m, n);
    alpha = rand_scalar<fp>();
    beta = rand_scalar<fp>();
    transa = rand_trans<fp>();
    transb = rand_trans<fp>();

    int64_t size_a, size_b, size_c;

    switch (layout) {
        case oneapi::mkl::layout::col_major:
            size_a = (transa == oneapi::mkl::transpose::nontrans) ? lda * n : lda * m;
            size_b = (transb == oneapi::mkl::transpose::nontrans) ? ldb * n : ldb * m;
            size_c = ldc * n;
            break;
        case oneapi::mkl::layout::row_major:
            size_a = (transa == oneapi::mkl::transpose::nontrans) ? lda * m : lda * n;
            size_b = (transb == oneapi::mkl::transpose::nontrans) ? ldb * m : ldb * n;
            size_c = ldc * m;
            break;
        default: break;
    }

    vector<fp, allocator_helper<fp, 64>> A(size_a), B(size_b), C(size_c), C_ref(size_c);

    rand_matrix(A.data(), oneapi::mkl::layout::col_major, oneapi::mkl::transpose::nontrans, size_a,
                1, size_a);
    rand_matrix(B.data(), oneapi::mkl::layout::col_major, oneapi::mkl::transpose::nontrans, size_b,
                1, size_b);
    rand_matrix(C.data(), oneapi::mkl::layout::col_major, oneapi::mkl::transpose::nontrans, size_c,
                1, size_c);
    copy_matrix(C.data(), oneapi::mkl::layout::col_major, oneapi::mkl::transpose::nontrans, size_c,
                1, size_c, C_ref.data());

    // Call reference OMATADD.
    int m_ref = (int)m;
    int n_ref = (int)n;
    int lda_ref = (int)lda;
    int ldb_ref = (int)ldb;
    int ldc_ref = (int)ldc;
    omatadd_ref(layout, transa, transb, m_ref, n_ref, alpha, A.data(), lda_ref, beta, B.data(),
                ldb_ref, C_ref.data(), ldc_ref);

    // Call DPC++ OMATADD

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during OMATADD:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<fp, 1> A_buffer(A.data(), range<1>(A.size()));
    buffer<fp, 1> B_buffer(B.data(), range<1>(B.size()));
    buffer<fp, 1> C_buffer(C.data(), range<1>(C.size()));

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                oneapi::mkl::blas::column_major::omatadd(main_queue, transa, transb, m, n, alpha,
                                                         A_buffer, lda, beta, B_buffer, ldb,
                                                         C_buffer, ldc);
                break;
            case oneapi::mkl::layout::row_major:
                oneapi::mkl::blas::row_major::omatadd(main_queue, transa, transb, m, n, alpha,
                                                      A_buffer, lda, beta, B_buffer, ldb, C_buffer,
                                                      ldc);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::omatadd,
                                        transa, transb, m, n, alpha, A_buffer, lda, beta, B_buffer,
                                        ldb, C_buffer, ldc);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::omatadd, transa,
                                        transb, m, n, alpha, A_buffer, lda, beta, B_buffer, ldb,
                                        C_buffer, ldc);
                break;
            default: break;
        }
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during OMATADD:\n" << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of OMATADD:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    auto C_accessor = C_buffer.get_host_access(read_only);
    bool good = check_equal_matrix(C_accessor, C_ref, oneapi::mkl::layout::col_major, size_c, 1,
                                   size_c, 10, std::cout);

    return (int)good;
}

class OmataddTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::mkl::layout>> {};

TEST_P(OmataddTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

TEST_P(OmataddTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

TEST_P(OmataddTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

TEST_P(OmataddTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(OmataddTestSuite, OmataddTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::col_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
