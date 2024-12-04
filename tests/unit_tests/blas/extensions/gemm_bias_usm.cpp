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

template <typename Ts, typename Ta, typename Tb, typename Tc>
int test(device* dev, oneapi::math::layout layout, oneapi::math::transpose transa,
         oneapi::math::transpose transb, oneapi::math::offset offsetc, int m, int n, int k, int lda,
         int ldb, int ldc, Ts alpha, Ts beta) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMM_BIAS:\n"
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
    auto ua = usm_allocator<Ta, usm::alloc::shared, 64>(cxt, *dev);
    auto ub = usm_allocator<Tb, usm::alloc::shared, 64>(cxt, *dev);
    auto uc = usm_allocator<Tc, usm::alloc::shared, 64>(cxt, *dev);
    vector<Ta, decltype(ua)> A(ua);
    vector<Tb, decltype(ub)> B(ub);
    vector<Tc, decltype(uc)> C(uc), C_ref(uc), co(uc);

    Ta ao = rand_scalar<Ta>();
    Tb bo = rand_scalar<Tb>();

    rand_matrix(A, layout, transa, m, k, lda);
    rand_matrix(B, layout, transb, k, n, ldb);
    rand_matrix(C, layout, oneapi::math::transpose::nontrans, m, n, ldc);
    if (offsetc == oneapi::math::offset::fix)
        rand_matrix(co, oneapi::math::layout::col_major, oneapi::math::transpose::nontrans, 1, 1,
                    1);
    if (offsetc == oneapi::math::offset::column)
        rand_matrix(co, oneapi::math::layout::col_major, oneapi::math::transpose::nontrans, m, 1,
                    m);
    if (offsetc == oneapi::math::offset::row)
        rand_matrix(co, oneapi::math::layout::col_major, oneapi::math::transpose::nontrans, n, 1,
                    n);

    C_ref.resize(C.size());
    for (int i = 0; i < C.size(); i++)
        C_ref[i] = C[i];

    // Call Reference GEMM_BIAS.
    const int m_ref = m, n_ref = n, k_ref = k;
    const int lda_ref = lda, ldb_ref = ldb, ldc_ref = ldc;

    using Ts_ref = typename ref_type_info<Ts>::type;
    using Ta_ref = typename ref_type_info<Ta>::type;
    using Tb_ref = typename ref_type_info<Tb>::type;
    using Tc_ref = typename ref_type_info<Tc>::type;

    ::gemm_bias(convert_to_cblas_layout(layout), convert_to_cblas_trans(transa),
                convert_to_cblas_trans(transb), convert_to_cblas_offset(offsetc), &m_ref, &n_ref,
                &k_ref, (Ts_ref*)&alpha, (Ta_ref*)A.data(), &lda_ref, (Ta_ref*)&ao,
                (Tb_ref*)B.data(), &ldb_ref, (Tb_ref*)&bo, (Ts_ref*)&beta, (Tc_ref*)C_ref.data(),
                &ldc_ref, (Tc_ref*)co.data());

    // Call DPC++ GEMM_BIAS.

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::math::layout::col_major:
                done = oneapi::math::blas::column_major::gemm_bias(
                    main_queue, transa, transb, offsetc, m, n, k, alpha, A.data(), lda, ao,
                    B.data(), ldb, bo, beta, C.data(), ldc, co.data(), dependencies);
                break;
            case oneapi::math::layout::row_major:
                done = oneapi::math::blas::row_major::gemm_bias(
                    main_queue, transa, transb, offsetc, m, n, k, alpha, A.data(), lda, ao,
                    B.data(), ldb, bo, beta, C.data(), ldc, co.data(), dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::gemm_bias,
                                        transa, transb, offsetc, m, n, k, alpha, A.data(), lda, ao,
                                        B.data(), ldb, bo, beta, C.data(), ldc, co.data(),
                                        dependencies);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::gemm_bias,
                                        transa, transb, offsetc, m, n, k, alpha, A.data(), lda, ao,
                                        B.data(), ldb, bo, beta, C.data(), ldc, co.data(),
                                        dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during GEMM_BIAS:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of GEMM_BIAS:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good = check_equal_matrix(C, C_ref, layout, m, n, ldc, 10 * k, std::cout);

    return (int)good;
}

class GemmBiasUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {};

TEST_P(GemmBiasUsmTests, Int8Int8Int32Precision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
}

TEST_P(GemmBiasUsmTests, Int8Uint8Int32Precision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
}

TEST_P(GemmBiasUsmTests, Uint8Int8Int32Precision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
}

TEST_P(GemmBiasUsmTests, Uint8Uint8Int32Precision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
}

INSTANTIATE_TEST_SUITE_P(GemmBiasUsmTestSuite, GemmBiasUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
