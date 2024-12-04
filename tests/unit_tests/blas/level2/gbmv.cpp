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

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "cblas.h"
#include "oneapi/math.hpp"
#include "oneapi/math/detail/config.hpp"
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
int test(device* dev, oneapi::math::layout layout, oneapi::math::transpose transa, int m, int n,
         int kl, int ku, fp alpha, fp beta, int incx, int incy, int lda) {
    // Prepare data.
    int x_len = outer_dimension(transa, m, n);
    int y_len = inner_dimension(transa, m, n);

    vector<fp> x, y, y_ref, A;

    rand_vector(x, x_len, incx);
    rand_vector(y, y_len, incy);
    y_ref = y;
    rand_matrix(A, layout, oneapi::math::transpose::nontrans, m, n, lda);

    // Call Reference GBMV.
    const int m_ref = m, n_ref = n, incx_ref = incx, incy_ref = incy, lda_ref = lda;
    int kl_ref = kl, ku_ref = ku;
    using fp_ref = typename ref_type_info<fp>::type;

    ::gbmv(convert_to_cblas_layout(layout), convert_to_cblas_trans(transa), &m_ref, &n_ref, &kl_ref,
           &ku_ref, (fp_ref*)&alpha, (fp_ref*)A.data(), &lda_ref, (fp_ref*)x.data(), &incx_ref,
           (fp_ref*)&beta, (fp_ref*)y_ref.data(), &incy_ref);

    // Call DPC++ GBMV.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GBMV:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<fp, 1> x_buffer = make_buffer(x);
    buffer<fp, 1> y_buffer = make_buffer(y);
    buffer<fp, 1> A_buffer = make_buffer(A);

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::math::layout::col_major:
                oneapi::math::blas::column_major::gbmv(main_queue, transa, m, n, kl, ku, alpha,
                                                       A_buffer, lda, x_buffer, incx, beta,
                                                       y_buffer, incy);
                break;
            case oneapi::math::layout::row_major:
                oneapi::math::blas::row_major::gbmv(main_queue, transa, m, n, kl, ku, alpha,
                                                    A_buffer, lda, x_buffer, incx, beta, y_buffer,
                                                    incy);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::gbmv, transa,
                                        m, n, kl, ku, alpha, A_buffer, lda, x_buffer, incx, beta,
                                        y_buffer, incy);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::gbmv, transa, m,
                                        n, kl, ku, alpha, A_buffer, lda, x_buffer, incx, beta,
                                        y_buffer, incy);
                break;
            default: break;
        }
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during GBMV:\n" << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of GBMV:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    auto y_accessor = y_buffer.get_host_access(read_only);
    bool good = check_equal_vector(y_accessor, y_ref, y_len, incy, std::max<int>(m, n), std::cout);

    return (int)good;
}

class GbmvTests : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {
};

TEST_P(GbmvTests, RealSinglePrecision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::transpose::nontrans, 25, 30, 5, 7, alpha, beta, 2,
                                  3, 42));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::transpose::nontrans, 25, 30, 5, 7, alpha, beta, -2,
                                  -3, 42));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::transpose::nontrans, 25, 30, 5, 7, alpha, beta, 1,
                                  1, 42));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::transpose::trans, 25, 30, 5, 7, alpha, beta, 2, 3,
                                  42));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::transpose::trans, 25, 30, 5, 7, alpha, beta, -2, -3,
                                  42));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::transpose::trans, 25, 30, 5, 7, alpha, beta, 1, 1,
                                  42));
}
TEST_P(GbmvTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    double alpha(2.0);
    double beta(3.0);
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::transpose::nontrans, 25, 30, 5, 7, alpha, beta, 2,
                                   3, 42));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::transpose::nontrans, 25, 30, 5, 7, alpha, beta, -2,
                                   -3, 42));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::transpose::nontrans, 25, 30, 5, 7, alpha, beta, 1,
                                   1, 42));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::transpose::trans, 25, 30, 5, 7, alpha, beta, 2, 3,
                                   42));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::transpose::trans, 25, 30, 5, 7, alpha, beta, -2,
                                   -3, 42));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::transpose::trans, 25, 30, 5, 7, alpha, beta, 1, 1,
                                   42));
}
TEST_P(GbmvTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0, -0.5);
    std::complex<float> beta(3.0, -1.5);
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::math::transpose::nontrans, 25, 30, 5, 7,
                                                alpha, beta, 2, 3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::math::transpose::nontrans, 25, 30, 5, 7,
                                                alpha, beta, -2, -3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::math::transpose::nontrans, 25, 30, 5, 7,
                                                alpha, beta, 1, 1, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::math::transpose::trans, 25, 30, 5, 7, alpha,
                                                beta, 2, 3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::math::transpose::trans, 25, 30, 5, 7, alpha,
                                                beta, -2, -3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::math::transpose::trans, 25, 30, 5, 7, alpha,
                                                beta, 1, 1, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::math::transpose::conjtrans, 25, 30, 5, 7,
                                                alpha, beta, 2, 3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::math::transpose::conjtrans, 25, 30, 5, 7,
                                                alpha, beta, -2, -3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::math::transpose::conjtrans, 25, 30, 5, 7,
                                                alpha, beta, 1, 1, 42));
}
TEST_P(GbmvTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    std::complex<double> alpha(2.0, -0.5);
    std::complex<double> beta(3.0, -1.5);
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::math::transpose::nontrans, 25, 30, 5, 7,
                                                 alpha, beta, 2, 3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::math::transpose::nontrans, 25, 30, 5, 7,
                                                 alpha, beta, -2, -3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::math::transpose::nontrans, 25, 30, 5, 7,
                                                 alpha, beta, 1, 1, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::math::transpose::trans, 25, 30, 5, 7,
                                                 alpha, beta, 2, 3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::math::transpose::trans, 25, 30, 5, 7,
                                                 alpha, beta, -2, -3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::math::transpose::trans, 25, 30, 5, 7,
                                                 alpha, beta, 1, 1, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::math::transpose::conjtrans, 25, 30, 5, 7,
                                                 alpha, beta, 2, 3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::math::transpose::conjtrans, 25, 30, 5, 7,
                                                 alpha, beta, -2, -3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::math::transpose::conjtrans, 25, 30, 5, 7,
                                                 alpha, beta, 1, 1, 42));
}

INSTANTIATE_TEST_SUITE_P(GbmvTestSuite, GbmvTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
