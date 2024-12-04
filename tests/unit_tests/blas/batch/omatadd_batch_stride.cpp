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
int test(device* dev, oneapi::math::layout layout, int64_t batch_size) {
    // Prepare data.
    int64_t m, n;
    int64_t lda, ldb, ldc;
    oneapi::math::transpose transa, transb;
    fp alpha, beta;
    int64_t i, tmp;

    batch_size = 1 + std::rand() % 20;
    m = 1 + std::rand() % 50;
    n = 1 + std::rand() % 50;
    lda = std::max(m, n);
    ldb = std::max(m, n);
    ldc = std::max(m, n);
    alpha = rand_scalar<fp>();
    beta = rand_scalar<fp>();
    transa = rand_trans<fp>();
    transb = rand_trans<fp>();

    int64_t stride_a, stride_b, stride_c;

    switch (layout) {
        case oneapi::math::layout::col_major:
            stride_a = (transa == oneapi::math::transpose::nontrans) ? lda * n : lda * m;
            stride_b = (transb == oneapi::math::transpose::nontrans) ? ldb * n : ldb * m;
            stride_c = ldc * n;
            break;
        case oneapi::math::layout::row_major:
            stride_a = (transa == oneapi::math::transpose::nontrans) ? lda * m : lda * n;
            stride_b = (transb == oneapi::math::transpose::nontrans) ? ldb * m : ldb * n;
            stride_c = ldc * m;
            break;
        default: break;
    }

    vector<fp, allocator_helper<fp, 64>> A(stride_a * batch_size), B(stride_b * batch_size),
        C(stride_c * batch_size), C_ref(stride_c * batch_size);

    rand_matrix(A.data(), oneapi::math::layout::col_major, oneapi::math::transpose::nontrans,
                stride_a * batch_size, 1, stride_a * batch_size);
    rand_matrix(B.data(), oneapi::math::layout::col_major, oneapi::math::transpose::nontrans,
                stride_b * batch_size, 1, stride_b * batch_size);
    rand_matrix(C.data(), oneapi::math::layout::col_major, oneapi::math::transpose::nontrans,
                stride_c * batch_size, 1, stride_c * batch_size);
    copy_matrix(C.data(), oneapi::math::layout::col_major, oneapi::math::transpose::nontrans,
                stride_c * batch_size, 1, stride_c * batch_size, C_ref.data());

    // Call reference OMATADD_BATCH_STRIDE.
    int m_ref = (int)m;
    int n_ref = (int)n;
    int lda_ref = (int)lda;
    int ldb_ref = (int)ldb;
    int ldc_ref = (int)ldc;
    int batch_size_ref = (int)batch_size;
    for (i = 0; i < batch_size_ref; i++) {
        omatadd_ref(layout, transa, transb, m_ref, n_ref, alpha, A.data() + stride_a * i, lda_ref,
                    beta, B.data() + stride_b * i, ldb_ref, C_ref.data() + stride_c * i, ldc_ref);
    }

    // Call DPC++ OMATADD_BATCH_STRIDE

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during OMATADD_BATCH_STRIDE:\n"
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
            case oneapi::math::layout::col_major:
                oneapi::math::blas::column_major::omatadd_batch(
                    main_queue, transa, transb, m, n, alpha, A_buffer, lda, stride_a, beta,
                    B_buffer, ldb, stride_b, C_buffer, ldc, stride_c, batch_size);
                break;
            case oneapi::math::layout::row_major:
                oneapi::math::blas::row_major::omatadd_batch(
                    main_queue, transa, transb, m, n, alpha, A_buffer, lda, stride_a, beta,
                    B_buffer, ldb, stride_b, C_buffer, ldc, stride_c, batch_size);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::omatadd_batch,
                                        transa, transb, m, n, alpha, A_buffer, lda, stride_a, beta,
                                        B_buffer, ldb, stride_b, C_buffer, ldc, stride_c,
                                        batch_size);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::omatadd_batch,
                                        transa, transb, m, n, alpha, A_buffer, lda, stride_a, beta,
                                        B_buffer, ldb, stride_b, C_buffer, ldc, stride_c,
                                        batch_size);
                break;
            default: break;
        }
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during OMATADD_BATCH_STRIDE:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of OMATADD_BATCH_STRIDE:\n"
                  << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    auto C_accessor = C_buffer.get_host_access(read_only);
    bool good = check_equal_matrix(C_accessor, C_ref, oneapi::math::layout::col_major,
                                   stride_c * batch_size, 1, stride_c * batch_size, 10, std::cout);

    return (int)good;
}

class OmataddBatchStrideTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {};

TEST_P(OmataddBatchStrideTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(OmataddBatchStrideTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(OmataddBatchStrideTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(OmataddBatchStrideTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

INSTANTIATE_TEST_SUITE_P(OmataddBatchStrideTestSuite, OmataddBatchStrideTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
