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
int test(device* dev, oneapi::math::layout layout, oneapi::math::uplo upper_lower,
         oneapi::math::transpose transa, oneapi::math::diag unit_nonunit, int n, int k, int incx,
         int lda) {
    // Prepare data.
    vector<fp> x, x_ref, A;
    rand_vector(x, n, incx);
    x_ref = x;
    rand_tbsv_matrix(A, layout, upper_lower, transa, n, k, lda);

    // Call Reference TBSV.
    const int n_ref = n, incx_ref = incx, lda_ref = lda;
    const int k_ref = k;
    using fp_ref = typename ref_type_info<fp>::type;

    ::tbsv(convert_to_cblas_layout(layout), convert_to_cblas_uplo(upper_lower),
           convert_to_cblas_trans(transa), convert_to_cblas_diag(unit_nonunit), &n_ref, &k_ref,
           (fp_ref*)A.data(), &lda_ref, (fp_ref*)x_ref.data(), &incx_ref);

    // Call DPC++ TBSV.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during TBSV:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<fp, 1> x_buffer = make_buffer(x);
    buffer<fp, 1> A_buffer = make_buffer(A);

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::math::layout::col_major:
                oneapi::math::blas::column_major::tbsv(main_queue, upper_lower, transa,
                                                       unit_nonunit, n, k, A_buffer, lda, x_buffer,
                                                       incx);
                break;
            case oneapi::math::layout::row_major:
                oneapi::math::blas::row_major::tbsv(main_queue, upper_lower, transa, unit_nonunit,
                                                    n, k, A_buffer, lda, x_buffer, incx);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::tbsv,
                                        upper_lower, transa, unit_nonunit, n, k, A_buffer, lda,
                                        x_buffer, incx);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::tbsv,
                                        upper_lower, transa, unit_nonunit, n, k, A_buffer, lda,
                                        x_buffer, incx);
                break;
            default: break;
        }
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during TBSV:\n" << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of TBSV:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    auto x_accessor = x_buffer.get_host_access(read_only);
    bool good = check_equal_trsv_vector(x_accessor, x_ref, n, incx, n, std::cout);

    return (int)good;
}

class TbsvTests : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {
};

TEST_P(TbsvTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::uplo::lower, oneapi::math::transpose::nontrans,
                                  oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::uplo::upper, oneapi::math::transpose::nontrans,
                                  oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::uplo::lower, oneapi::math::transpose::trans,
                                  oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::uplo::upper, oneapi::math::transpose::trans,
                                  oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::uplo::lower, oneapi::math::transpose::nontrans,
                                  oneapi::math::diag::nonunit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::uplo::upper, oneapi::math::transpose::nontrans,
                                  oneapi::math::diag::nonunit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::uplo::lower, oneapi::math::transpose::trans,
                                  oneapi::math::diag::nonunit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::uplo::upper, oneapi::math::transpose::trans,
                                  oneapi::math::diag::nonunit, 30, 5, 2, 42));
}
TEST_P(TbsvTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::uplo::lower, oneapi::math::transpose::nontrans,
                                   oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::uplo::upper, oneapi::math::transpose::nontrans,
                                   oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::uplo::lower, oneapi::math::transpose::trans,
                                   oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::uplo::upper, oneapi::math::transpose::trans,
                                   oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::uplo::lower, oneapi::math::transpose::nontrans,
                                   oneapi::math::diag::nonunit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::uplo::upper, oneapi::math::transpose::nontrans,
                                   oneapi::math::diag::nonunit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::uplo::lower, oneapi::math::transpose::trans,
                                   oneapi::math::diag::nonunit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::uplo::upper, oneapi::math::transpose::trans,
                                   oneapi::math::diag::nonunit, 30, 5, 2, 42));
}
TEST_P(TbsvTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::nontrans, oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::nontrans, oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::trans, oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::trans, oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::conjtrans, oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::conjtrans, oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::conjtrans, oneapi::math::diag::nonunit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::conjtrans, oneapi::math::diag::nonunit, 30, 5, 2, 42));
}
TEST_P(TbsvTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::nontrans, oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::nontrans, oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::trans, oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::trans, oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::conjtrans, oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::conjtrans, oneapi::math::diag::unit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::conjtrans, oneapi::math::diag::nonunit, 30, 5, 2, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::conjtrans, oneapi::math::diag::nonunit, 30, 5, 2, 42));
}

INSTANTIATE_TEST_SUITE_P(TbsvTestSuite, TbsvTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
