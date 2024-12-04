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
int test(device* dev, oneapi::math::layout layout, oneapi::math::side left_right,
         oneapi::math::uplo upper_lower, oneapi::math::transpose transa,
         oneapi::math::diag unit_nonunit, int m, int n, int lda, int ldb, fp alpha) {
    // Prepare data.
    vector<fp, allocator_helper<fp, 64>> A, B, B_ref;
    if (left_right == oneapi::math::side::right)
        rand_matrix(A, layout, transa, n, n, lda);
    else
        rand_matrix(A, layout, transa, m, m, lda);

    rand_matrix(B, layout, oneapi::math::transpose::nontrans, m, n, ldb);
    B_ref = B;

    // Call Reference TRMM.
    const int m_ref = m, n_ref = n;
    const int lda_ref = lda, ldb_ref = ldb;

    using fp_ref = typename ref_type_info<fp>::type;

    ::trmm(convert_to_cblas_layout(layout), convert_to_cblas_side(left_right),
           convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
           convert_to_cblas_diag(unit_nonunit), &m_ref, &n_ref, (fp_ref*)&alpha, (fp_ref*)A.data(),
           &lda_ref, (fp_ref*)B_ref.data(), &ldb_ref);

    // Call DPC++ TRMM.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during TRMM:\n"
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
                oneapi::math::blas::column_major::trmm(main_queue, left_right, upper_lower, transa,
                                                       unit_nonunit, m, n, alpha, A_buffer, lda,
                                                       B_buffer, ldb);
                break;
            case oneapi::math::layout::row_major:
                oneapi::math::blas::row_major::trmm(main_queue, left_right, upper_lower, transa,
                                                    unit_nonunit, m, n, alpha, A_buffer, lda,
                                                    B_buffer, ldb);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::trmm,
                                        left_right, upper_lower, transa, unit_nonunit, m, n, alpha,
                                        A_buffer, lda, B_buffer, ldb);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::trmm, left_right,
                                        upper_lower, transa, unit_nonunit, m, n, alpha, A_buffer,
                                        lda, B_buffer, ldb);
                break;
            default: break;
        }
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during TRMM:\n" << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of TRMM:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    auto B_accessor = B_buffer.get_host_access(read_only);
    bool good =
        check_equal_matrix(B_accessor, B_ref, layout, m, n, ldb, 10 * std::max(m, n), std::cout);

    return (int)good;
}

class TrmmTests : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {
};

TEST_P(TrmmTests, RealSinglePrecision) {
    float alpha(2.0);
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::side::left, oneapi::math::uplo::lower,
                                  oneapi::math::transpose::nontrans, oneapi::math::diag::unit, 72,
                                  27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::side::right, oneapi::math::uplo::lower,
                                  oneapi::math::transpose::nontrans, oneapi::math::diag::unit, 72,
                                  27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::side::left, oneapi::math::uplo::lower,
                                  oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit,
                                  72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::side::right, oneapi::math::uplo::lower,
                                  oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit,
                                  72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::side::left, oneapi::math::uplo::upper,
                                  oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit,
                                  72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::side::right, oneapi::math::uplo::upper,
                                  oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit,
                                  72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::side::left, oneapi::math::uplo::lower,
                                  oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 72,
                                  27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::side::right, oneapi::math::uplo::lower,
                                  oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 72,
                                  27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::side::left, oneapi::math::uplo::upper,
                                  oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 72,
                                  27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::side::right, oneapi::math::uplo::upper,
                                  oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 72,
                                  27, 101, 102, alpha));
}
TEST_P(TrmmTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    double alpha(2.0);
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::side::left, oneapi::math::uplo::lower,
                                   oneapi::math::transpose::nontrans, oneapi::math::diag::unit, 72,
                                   27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::side::right, oneapi::math::uplo::lower,
                                   oneapi::math::transpose::nontrans, oneapi::math::diag::unit, 72,
                                   27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::side::left, oneapi::math::uplo::lower,
                                   oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit,
                                   72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::side::right, oneapi::math::uplo::lower,
                                   oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit,
                                   72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::side::left, oneapi::math::uplo::upper,
                                   oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit,
                                   72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::side::right, oneapi::math::uplo::upper,
                                   oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit,
                                   72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::side::left, oneapi::math::uplo::lower,
                                   oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 72,
                                   27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::side::right, oneapi::math::uplo::lower,
                                   oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 72,
                                   27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::side::left, oneapi::math::uplo::upper,
                                   oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 72,
                                   27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::side::right, oneapi::math::uplo::upper,
                                   oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 72,
                                   27, 101, 102, alpha));
}
TEST_P(TrmmTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0, -0.5);
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::math::side::left, oneapi::math::uplo::lower,
                                                oneapi::math::transpose::nontrans,
                                                oneapi::math::diag::unit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::right,
        oneapi::math::uplo::lower, oneapi::math::transpose::nontrans, oneapi::math::diag::unit, 72,
        27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::left,
        oneapi::math::uplo::lower, oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit,
        72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::right,
        oneapi::math::uplo::lower, oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit,
        72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::left,
        oneapi::math::uplo::upper, oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit,
        72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::right,
        oneapi::math::uplo::upper, oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit,
        72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::left,
        oneapi::math::uplo::lower, oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 72,
        27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::right,
        oneapi::math::uplo::lower, oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 72,
        27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::left,
        oneapi::math::uplo::upper, oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 72,
        27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::right,
        oneapi::math::uplo::upper, oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 72,
        27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::left,
        oneapi::math::uplo::lower, oneapi::math::transpose::conjtrans, oneapi::math::diag::nonunit,
        72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::right,
        oneapi::math::uplo::lower, oneapi::math::transpose::conjtrans, oneapi::math::diag::nonunit,
        72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::left,
        oneapi::math::uplo::upper, oneapi::math::transpose::conjtrans, oneapi::math::diag::nonunit,
        72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::right,
        oneapi::math::uplo::upper, oneapi::math::transpose::conjtrans, oneapi::math::diag::nonunit,
        72, 27, 101, 102, alpha));
}
TEST_P(TrmmTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    std::complex<double> alpha(2.0, -0.5);
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::left,
        oneapi::math::uplo::lower, oneapi::math::transpose::nontrans, oneapi::math::diag::unit, 72,
        27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::right,
        oneapi::math::uplo::lower, oneapi::math::transpose::nontrans, oneapi::math::diag::unit, 72,
        27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::left,
        oneapi::math::uplo::lower, oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit,
        72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::right,
        oneapi::math::uplo::lower, oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit,
        72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::left,
        oneapi::math::uplo::upper, oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit,
        72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::right,
        oneapi::math::uplo::upper, oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit,
        72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::left,
        oneapi::math::uplo::lower, oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 72,
        27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::right,
        oneapi::math::uplo::lower, oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 72,
        27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::left,
        oneapi::math::uplo::upper, oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 72,
        27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::right,
        oneapi::math::uplo::upper, oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 72,
        27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::left,
        oneapi::math::uplo::lower, oneapi::math::transpose::conjtrans, oneapi::math::diag::nonunit,
        72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::right,
        oneapi::math::uplo::lower, oneapi::math::transpose::conjtrans, oneapi::math::diag::nonunit,
        72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::left,
        oneapi::math::uplo::upper, oneapi::math::transpose::conjtrans, oneapi::math::diag::nonunit,
        72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::side::right,
        oneapi::math::uplo::upper, oneapi::math::transpose::conjtrans, oneapi::math::diag::nonunit,
        72, 27, 101, 102, alpha));
}

INSTANTIATE_TEST_SUITE_P(TrmmTestSuite, TrmmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
