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

template <typename fp, typename fp_scalar>
int test(device* dev, oneapi::math::layout layout, oneapi::math::uplo upper_lower,
         oneapi::math::transpose trans, int n, int k, int lda, int ldb, int ldc, fp alpha,
         fp_scalar beta) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during HER2K:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    fp alpha_row(alpha.real(), -alpha.imag());

    // Prepare data.
    auto ua = usm_allocator<fp, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp, decltype(ua)> A(ua), B(ua), C(ua);
    rand_matrix(A, layout, trans, n, k, lda);
    rand_matrix(B, layout, trans, n, k, ldb);
    rand_matrix(C, layout, oneapi::math::transpose::nontrans, n, n, ldc);

    auto C_ref = C;

    // Call Reference HER2K.
    const int n_ref = n, k_ref = k;
    const int lda_ref = lda, ldb_ref = ldb, ldc_ref = ldc;

    using fp_ref = typename ref_type_info<fp>::type;
    using fp_scalar_ref = typename ref_type_info<fp_scalar>::type;

    ::her2k(convert_to_cblas_layout(layout), convert_to_cblas_uplo(upper_lower),
            convert_to_cblas_trans(trans), &n_ref, &k_ref, (fp_ref*)&alpha, (fp_ref*)A.data(),
            &lda_ref, (fp_ref*)B.data(), &ldb_ref, (fp_scalar_ref*)&beta, (fp_ref*)C_ref.data(),
            &ldc_ref);

    // Call DPC++ HER2K.

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::math::layout::col_major:
                done = oneapi::math::blas::column_major::her2k(main_queue, upper_lower, trans, n, k,
                                                               alpha, A.data(), lda, B.data(), ldb,
                                                               beta, C.data(), ldc, dependencies);
                break;
            case oneapi::math::layout::row_major:
                done = oneapi::math::blas::row_major::her2k(main_queue, upper_lower, trans, n, k,
                                                            alpha, A.data(), lda, B.data(), ldb,
                                                            beta, C.data(), ldc, dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::her2k,
                                        upper_lower, trans, n, k, alpha, A.data(), lda, B.data(),
                                        ldb, beta, C.data(), ldc, dependencies);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::her2k,
                                        upper_lower, trans, n, k, alpha, A.data(), lda, B.data(),
                                        ldb, beta, C.data(), ldc, dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during HER2K:\n" << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of HER2K:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good = check_equal_matrix(C, C_ref, layout, n, n, ldc, 10 * std::max(n, k), std::cout);

    return (int)good;
}

class Her2kUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {};

TEST_P(Her2kUsmTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0, -0.5);
    float beta(1.0);
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::nontrans, 72, 27, 101, 102, 103, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::nontrans, 72, 27, 101, 102, 103, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::conjtrans, 72, 27, 101, 102, 103, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::conjtrans, 72, 27, 101, 102, 103, alpha, beta)));
}
TEST_P(Her2kUsmTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    std::complex<double> alpha(2.0, -0.5);
    double beta(1.0);
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::nontrans, 72, 27, 101, 102, 103, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::nontrans, 72, 27, 101, 102, 103, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::lower,
        oneapi::math::transpose::conjtrans, 72, 27, 101, 102, 103, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::uplo::upper,
        oneapi::math::transpose::conjtrans, 72, 27, 101, 102, 103, alpha, beta)));
}

INSTANTIATE_TEST_SUITE_P(Her2kUsmTestSuite, Her2kUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
