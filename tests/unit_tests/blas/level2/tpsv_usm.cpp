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
int test(device* dev, oneapi::math::layout layout, oneapi::math::uplo upper_lower,
         oneapi::math::transpose transa, oneapi::math::diag unit_nonunit, int n, int incx) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during TPSV:\n"
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
    auto ua = usm_allocator<fp, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp, decltype(ua)> x(ua), A(ua);
    rand_vector(x, n, incx);
    rand_tpsv_matrix(A, layout, upper_lower, transa, n);

    auto x_ref = x;

    // Call Reference TPSV.
    const int n_ref = n, incx_ref = incx;
    using fp_ref = typename ref_type_info<fp>::type;

    ::tpsv(convert_to_cblas_layout(layout), convert_to_cblas_uplo(upper_lower),
           convert_to_cblas_trans(transa), convert_to_cblas_diag(unit_nonunit), &n_ref,
           (fp_ref*)A.data(), (fp_ref*)x_ref.data(), &incx_ref);

    // Call DPC++ TPSV.

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::math::layout::col_major:
                done = oneapi::math::blas::column_major::tpsv(main_queue, upper_lower, transa,
                                                              unit_nonunit, n, A.data(), x.data(),
                                                              incx, dependencies);
                break;
            case oneapi::math::layout::row_major:
                done = oneapi::math::blas::row_major::tpsv(main_queue, upper_lower, transa,
                                                           unit_nonunit, n, A.data(), x.data(),
                                                           incx, dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::tpsv,
                                        upper_lower, transa, unit_nonunit, n, A.data(), x.data(),
                                        incx, dependencies);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::tpsv,
                                        upper_lower, transa, unit_nonunit, n, A.data(), x.data(),
                                        incx, dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during TPSV:\n" << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of TPSV:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good = check_equal_trsv_vector(x, x_ref, n, incx, n, std::cout);

    return (int)good;
}

class TpsvUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {};

TEST_P(TpsvUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::uplo::lower, oneapi::math::transpose::nontrans,
                                  oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::uplo::upper, oneapi::math::transpose::nontrans,
                                  oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::uplo::lower, oneapi::math::transpose::trans,
                                  oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::uplo::upper, oneapi::math::transpose::trans,
                                  oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::uplo::lower, oneapi::math::transpose::nontrans,
                                  oneapi::math::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::uplo::upper, oneapi::math::transpose::nontrans,
                                  oneapi::math::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::uplo::lower, oneapi::math::transpose::trans,
                                  oneapi::math::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::math::uplo::upper, oneapi::math::transpose::trans,
                                  oneapi::math::diag::nonunit, 30, 2));
}
TEST_P(TpsvUsmTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::uplo::lower, oneapi::math::transpose::nontrans,
                                   oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::uplo::upper, oneapi::math::transpose::nontrans,
                                   oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::uplo::lower, oneapi::math::transpose::trans,
                                   oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::uplo::upper, oneapi::math::transpose::trans,
                                   oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::uplo::lower, oneapi::math::transpose::nontrans,
                                   oneapi::math::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::uplo::upper, oneapi::math::transpose::nontrans,
                                   oneapi::math::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::uplo::lower, oneapi::math::transpose::trans,
                                   oneapi::math::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::math::uplo::upper, oneapi::math::transpose::trans,
                                   oneapi::math::diag::nonunit, 30, 2));
}
TEST_P(TpsvUsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::nontrans, oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::nontrans, oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::trans, oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::trans, oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::conjtrans, oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::conjtrans, oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::conjtrans, oneapi::math::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::conjtrans, oneapi::math::diag::nonunit, 30, 2));
}
TEST_P(TpsvUsmTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::nontrans, oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::nontrans, oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::trans, oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::trans, oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::conjtrans, oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::conjtrans, oneapi::math::diag::unit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::trans, oneapi::math::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::conjtrans, oneapi::math::diag::nonunit, 30, 2));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::conjtrans, oneapi::math::diag::nonunit, 30, 2));
}

INSTANTIATE_TEST_SUITE_P(TpsvUsmTestSuite, TpsvUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
